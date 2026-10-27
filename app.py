
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Tuesday AI V27", page_icon="🥇", layout="wide")
st.title("🥇 Tuesday AI V27 — Fibonacci Risk Audit")
st.caption("Research only | XAUUSD M15 | No live orders")

with st.sidebar:
    capital = st.number_input("Starting balance ($)", min_value=100.0, value=1000.0, step=100.0)
    risk = st.slider("Risk per trade (%)", 0.1, 1.0, 0.5, 0.1)
    spread = st.number_input("Spread ($/oz)", min_value=0.0, value=0.30, step=0.05)
    slip = st.number_input("Slippage per side ($/oz)", min_value=0.0, value=0.10, step=0.05)
    commission = st.number_input("Round-trip commission ($/oz)", min_value=0.0, value=0.10, step=0.05)
    pivot = st.slider("Confirmed pivot bars", 2, 12, 5)
    daily_limit = st.slider("Daily realized loss limit (%)", 0.5, 5.0, 2.0, 0.5)
    streak_limit = st.slider("Pause after consecutive losses", 2, 6, 3)
    cooldown = st.slider("Pause duration (M15 bars)", 4, 96, 24, 4)
    min_stop = st.number_input("Minimum stop ($/oz)", min_value=0.1, value=1.0)
    rr = st.number_input("Reward/risk target", min_value=1.0, max_value=4.0, value=2.0, step=0.25)

st.info("Upload original XAUUSD OHLC candle data, NOT V26/V25 trade results.")
file = st.file_uploader("Upload XAUUSD M1/M15 OHLC CSV or TXT", type=["csv", "txt"])

def load_data(raw):
    frame = None
    for sep in [None, ",", "\t", ";"]:
        try:
            t = pd.read_csv(BytesIO(raw), sep=sep, engine="python")
            if len(t.columns) >= 5:
                frame = t
                break
        except Exception:
            pass

    if frame is None:
        raise ValueError("Cannot read OHLC file")

    frame.columns = [str(x).lower().strip().strip("<>") for x in frame.columns]

    if "datetime" in frame.columns:
        dates = pd.to_datetime(frame["datetime"], errors="coerce")
    elif "timestamp" in frame.columns:
        dates = pd.to_datetime(frame["timestamp"], errors="coerce")
    elif "date" in frame.columns and "time" in frame.columns:
        dates = pd.to_datetime(
            frame["date"].astype(str) + " " + frame["time"].astype(str),
            errors="coerce"
        )
    elif "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="coerce")
    else:
        raise ValueError("Need datetime or date/time columns")

    d = frame[["open", "high", "low", "close"]].copy()
    d.index = pd.DatetimeIndex(dates)
    d = d.loc[~d.index.isna()]

    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna().sort_index()
    d = d[~d.index.duplicated(keep="last")]
    d = d.resample("15min", label="right", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()

    d = d[
        (d.high >= d[["open", "close", "low"]].max(axis=1)) &
        (d.low <= d[["open", "close", "high"]].min(axis=1))
    ]

    if len(d) < 400:
        raise ValueError("At least 400 M15 candles required")

    return d


def backtest(d, zlow, zhigh, buy_only, split_idx):
    df = d.copy()
    df["ema50"] = df.close.ewm(
        span=50, adjust=False, min_periods=50
    ).mean()
    df["ema200"] = df.close.ewm(
        span=200, adjust=False, min_periods=200
    ).mean()

    trades = []
    confirmed = []
    position = None
    pending = None
    balance = capital
    peak = capital
    loss_streak = 0
    paused_until = -1
    day = None
    day_start = capital
    blocked_day = False

    for i in range(len(df)):
        row = df.iloc[i]
        ts = df.index[i]

        if i == split_idx:
            position = None
            pending = None
            balance = capital
            peak = capital
            loss_streak = 0
            paused_until = -1
            day = None
            blocked_day = False

        if day != ts.date():
            day = ts.date()
            day_start = balance
            blocked_day = False

        if i >= 2 * pivot:
            j = i - pivot
            window = df.iloc[j-pivot:j+pivot+1]

            if (
                df.high.iloc[j] == window.high.max()
                and (window.high == df.high.iloc[j]).sum() == 1
            ):
                confirmed.append(("H", float(df.high.iloc[j]), j))

            if (
                df.low.iloc[j] == window.low.min()
                and (window.low == df.low.iloc[j]).sum() == 1
            ):
                confirmed.append(("L", float(df.low.iloc[j]), j))

        if position is not None:
            side = position["side"]
            stop_hit = (
                row.low <= position["stop"]
                if side == "BUY"
                else row.high >= position["stop"]
            )
            tp_hit = (
                row.high >= position["target"]
                if side == "BUY"
                else row.low <= position["target"]
            )

            if stop_hit or tp_hit:
                intended = (
                    position["stop"] if stop_hit
                    else position["target"]
                )

                if stop_hit:
                    intended = (
                        min(intended, row.open)
                        if side == "BUY"
                        else max(intended, row.open)
                    )

                fill = (
                    intended - (spread / 2 + slip)
                    if side == "BUY"
                    else intended + (spread / 2 + slip)
                )

                pnl = (
                    (fill - position["entry"])
                    if side == "BUY"
                    else (position["entry"] - fill)
                ) * position["qty"] - commission * position["qty"]

                balance += pnl
                peak = max(peak, balance)

                trades.append({
                    **position,
                    "exit_time": ts,
                    "exit": fill,
                    "net_pnl": pnl,
                    "equity": balance,
                    "drawdown_pct": (peak-balance)/peak*100,
                    "reason": "Stop" if stop_hit else "Target",
                    "sample": (
                        "Training"
                        if position["entry_bar"] < split_idx
                        else "Out of sample"
                    )
                })

                loss_streak = loss_streak + 1 if pnl < 0 else 0

                if loss_streak >= streak_limit:
                    paused_until = i + cooldown
                    loss_streak = 0

                position = None

        if balance <= day_start * (1-daily_limit/100):
            blocked_day = True

        if pending is not None:
            order = pending
            pending = None

            if (
                position is None
                and not blocked_day
                and i >= paused_until
                and balance > 0
            ):
                side = order["side"]
                entry = float(row.open) + (
                    spread/2 + slip
                ) * (1 if side == "BUY" else -1)

                stop = order["stop"]
                dist = (
                    entry-stop if side == "BUY"
                    else stop-entry
                )

                if dist >= min_stop:
                    target = entry + rr * dist * (
                        1 if side == "BUY" else -1
                    )
                    qty = (
                        balance*risk/100 /
                        (dist+spread/2+slip+commission)
                    )

                    position = {
                        "entry_time": ts,
                        "entry_bar": i,
                        "side": side,
                        "entry": entry,
                        "stop": stop,
                        "target": target,
                        "qty": qty,
                        "zone": order["zone"]
                    }

                    sh = (
                        row.low <= stop if side == "BUY"
                        else row.high >= stop
                    )
                    th = (
                        row.high >= target if side == "BUY"
                        else row.low <= target
                    )

                    if sh or th:
                        intended = stop if sh else target

                        if sh:
                            intended = (
                                min(intended, row.open)
                                if side == "BUY"
                                else max(intended, row.open)
                            )

                        fill = (
                            intended-(spread/2+slip)
                            if side == "BUY"
                            else intended+(spread/2+slip)
                        )

                        pnl = (
                            (fill-entry)
                            if side == "BUY"
                            else (entry-fill)
                        )*qty-commission*qty

                        balance += pnl
                        peak = max(peak, balance)

                        trades.append({
                            **position,
                            "exit_time": ts,
                            "exit": fill,
                            "net_pnl": pnl,
                            "equity": balance,
                            "drawdown_pct": (peak-balance)/peak*100,
                            "reason": "Stop" if sh else "Target",
                            "sample": (
                                "Training"
                                if i < split_idx
                                else "Out of sample"
                            )
                        })

                        loss_streak = (
                            loss_streak+1 if pnl < 0 else 0
                        )

                        if loss_streak >= streak_limit:
                            paused_until = i+cooldown
                            loss_streak = 0

                        position = None

        if balance <= day_start*(1-daily_limit/100):
            blocked_day = True

        if (
            position is not None
            or blocked_day
            or i < 201
            or i < paused_until
        ):
            continue

        if not np.isfinite(row.ema200):
            continue

        highs = [
            (v, j) for typ, v, j in confirmed
            if typ == "H"
        ]
        lows = [
            (v, j) for typ, v, j in confirmed
            if typ == "L"
        ]

        if not highs or not lows:
            continue

        h, hj = highs[-1]
        l, lj = lows[-1]

        if h <= l:
            continue

        if (
            lj < hj
            and row.ema50 > row.ema200
            and row.close > row.ema200
        ):
            lower = h-(h-l)*zhigh
            upper = h-(h-l)*zlow

            if (
                row.low <= upper
                and row.high >= lower
                and row.close > row.open
                and row.close > lower
            ):
                pending = {
                    "side": "BUY",
                    "stop": l-max(0.2, 0.05*(h-l)),
                    "zone": f"{zlow:.3f}-{zhigh:.3f}"
                }

        elif (
            not buy_only
            and hj < lj
            and row.ema50 < row.ema200
            and row.close < row.ema200
        ):
            lower = l+(h-l)*zlow
            upper = l+(h-l)*zhigh

            if (
                row.high >= lower
                and row.low <= upper
                and row.close < row.open
                and row.close < upper
            ):
                pending = {
                    "side": "SELL",
                    "stop": h+max(0.2, 0.05*(h-l)),
                    "zone": f"{zlow:.3f}-{zhigh:.3f}"
                }

    return pd.DataFrame(trades)


def metrics(t):
    if t.empty:
        return {
            "Trades": 0,
            "Win rate %": 0.0,
            "Net P&L $": 0.0,
            "Profit factor": None,
            "Max DD %": 0.0
        }

    wins = t.loc[t.net_pnl > 0, "net_pnl"].sum()
    losses = -t.loc[t.net_pnl < 0, "net_pnl"].sum()

    return {
        "Trades": len(t),
        "Win rate %": round(
            100*(t.net_pnl > 0).mean(), 2
        ),
        "Net P&L $": round(t.net_pnl.sum(), 2),
        "Profit factor": (
            round(wins/losses, 2)
            if losses > 0 else None
        ),
        "Max DD %": round(
            t.drawdown_pct.max(), 2
        )
    }


if file:
    try:
        df = load_data(file.getvalue())
        st.success(
            f"Loaded {len(df):,} M15 bars: "
            f"{df.index.min()} to {df.index.max()}"
        )

        split = int(len(df)*0.7)

        st.caption(
            f"Chronological 70/30 split at {df.index[split]}. "
            "Test balances reset at split. "
            "Indicators use prior candles for warm-up."
        )

        if st.button(
            "Run V27 Fibonacci comparison",
            type="primary"
        ):
            summary = []

            for zone, z1, z2 in [
                ("Golden Zone", 0.5, 0.618),
                ("Deep Retracement", 0.618, 0.786)
            ]:
                for mode, buy_only in [
                    ("BUY only", True),
                    ("BUY + SELL", False)
                ]:
                    trades = backtest(
                        df, z1, z2, buy_only, split
                    )

                    for sample in [
                        "Training", "Out of sample"
                    ]:
                        sub = (
                            trades[trades["sample"] == sample]
                            if not trades.empty else trades
                        )
                        summary.append({
                            "Zone": zone,
                            "Mode": mode,
                            "Sample": sample,
                            **metrics(sub)
                        })

                    if not trades.empty:
                        st.download_button(
                            f"Download {zone} / {mode} trades CSV",
                            trades.drop(
                                columns=["entry_bar"]
                            ).to_csv(index=False),
                            file_name=(
                                "Tuesday_AI_V27_"
                                + zone.replace(" ", "_")
                                + "_"
                                + mode.replace(" ", "_").replace("+", "and")
                                + ".csv"
                            ),
                            mime="text/csv"
                        )

            st.subheader("Comparison — all 8 groups")
            st.dataframe(
                pd.DataFrame(summary),
                hide_index=True,
                use_container_width=True
            )

            st.warning(
                "August data and V26 audit have already "
                "been reviewed. Validate final rules on "
                "new unseen months before live trading."
            )

            st.info(
                "Assumptions: quantities measured in oz; "
                "no broker lot-step or margin model. "
                "Conservative same-bar stop-first fills. "
                "Loss limits use realized equity only."
            )

    except Exception as e:
        st.error(
            f"Input issue: {type(e).__name__}: {e}"
        )
