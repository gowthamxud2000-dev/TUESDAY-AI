
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(
    page_title="Tuesday AI V28",
    page_icon="🥇",
    layout="wide"
)

st.title("🥇 Tuesday AI V28")
st.subheader("Golden Zone BUY + SELL | Backtest Audit")
st.caption("Research only | XAUUSD M15 | No live orders")

with st.sidebar:
    st.header("⚙️ Risk Settings")

    capital = st.number_input(
        "Starting Balance ($)",
        min_value=100.0,
        value=2500.0,
        step=100.0
    )

    risk_pct = st.number_input(
        "Risk per Trade (%)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1
    )

    daily_limit = st.number_input(
        "Daily Loss Limit ($)",
        min_value=1.0,
        value=40.0
    )

    spread = st.number_input(
        "Spread ($/oz)",
        min_value=0.0,
        value=0.30,
        step=0.05
    )

    slippage = st.number_input(
        "Slippage per Side ($/oz)",
        min_value=0.0,
        value=0.10,
        step=0.05
    )

    commission = st.number_input(
        "Round Trip Commission ($/oz)",
        min_value=0.0,
        value=0.0,
        step=0.05
    )

    rr = st.number_input(
        "Reward Risk Ratio",
        min_value=1.0,
        max_value=4.0,
        value=2.0,
        step=0.25
    )

    pivot = st.slider(
        "Confirmed Pivot Bars",
        2, 12, 5
    )

    split = st.slider(
        "Training Percentage",
        50, 90, 70
    )

st.info(
    "Upload original XAUUSD OHLC historical data. "
    "Do not upload old backtest results for a fresh test."
)

file = st.file_uploader(
    "Upload XAUUSD CSV or TXT",
    type=["csv", "txt"]
)


def load_data(raw):
    df = None

    for sep in [None, ",", ";", "\t"]:
        try:
            t = pd.read_csv(
                BytesIO(raw),
                sep=sep,
                engine="python",
                header=None
            )
            if t.shape[1] >= 5:
                df = t
                break
        except Exception:
            continue

    if df is None:
        raise ValueError("Cannot read OHLC file")

    first = df.iloc[0].astype(str).str.lower().tolist()

    if any("open" in x for x in first):
        df = pd.read_csv(BytesIO(raw), sep=None, engine="python")
        df.columns = [
            str(c).lower().strip().replace("<", "").replace(">", "")
            for c in df.columns
        ]

        if "datetime" in df.columns:
            dates = pd.to_datetime(df["datetime"], errors="coerce")
        elif "timestamp" in df.columns:
            dates = pd.to_datetime(df["timestamp"], errors="coerce")
        elif "date" in df.columns and "time" in df.columns:
            dates = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str),
                errors="coerce"
            )
        elif "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce")
        else:
            raise ValueError("Missing date/time columns")

        needed = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in needed):
            raise ValueError("Missing OHLC columns")

        d = df[needed].copy()
    else:
        if df.shape[1] >= 6:
            dates = pd.to_datetime(
                df.iloc[:, 0].astype(str) + " " +
                df.iloc[:, 1].astype(str),
                errors="coerce"
            )
            start = 2
        else:
            dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            start = 1

        d = df.iloc[:, start:start + 4].copy()
        d.columns = ["open", "high", "low", "close"]

    d.index = pd.DatetimeIndex(dates)
    d = d.loc[~d.index.isna()]

    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna().sort_index()
    d = d[~d.index.duplicated(keep="last")]

    # M1 data is converted to M15.
    # Existing M15 data is also normalized.
    d = d.resample(
        "15min", label="right", closed="left"
    ).agg({
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


def run_backtest(data):
    d = data.copy()

    d["ema50"] = d.close.ewm(
        span=50, adjust=False, min_periods=50
    ).mean()

    d["ema200"] = d.close.ewm(
        span=200, adjust=False, min_periods=200
    ).mean()

    split_idx = int(len(d) * split / 100)

    balance = capital
    peak = capital
    trades = []
    equity_points = []
    position = None
    pending = None
    pivots = []
    day = None
    day_start = capital
    blocked = False

    for i in range(200, len(d)):
        row = d.iloc[i]
        ts = d.index[i]

        # Preserve equity across training/test boundary.
        if i == split_idx:
            position = None
            pending = None
            pivots = []

        if day != ts.date():
            day = ts.date()
            day_start = balance
            blocked = False

        sample = (
            "Training" if i < split_idx
            else "Out-of-Sample"
        )

        # Only use pivots that have become confirmed.
        j = i - pivot
        if j >= pivot:
            window = d.iloc[j - pivot:j + pivot + 1]

            if d.high.iloc[j] == window.high.max():
                if (window.high == d.high.iloc[j]).sum() == 1:
                    pivots.append(
                        ("H", float(d.high.iloc[j]), j)
                    )

            if d.low.iloc[j] == window.low.min():
                if (window.low == d.low.iloc[j]).sum() == 1:
                    pivots.append(
                        ("L", float(d.low.iloc[j]), j)
                    )

        pivots = pivots[-30:]

        # Manage open position.
        if position is not None:
            side = position["side"]
            stop = position["stop"]
            target = position["target"]

            if side == "BUY":
                hit_stop = row.low <= stop
                hit_target = row.high >= target
            else:
                hit_stop = row.high >= stop
                hit_target = row.low <= target

            if hit_stop or hit_target:
                # Conservative: stop wins if both touched.
                reason = "Stop" if hit_stop else "Target"
                intended = stop if hit_stop else target

                cost = spread / 2 + slippage

                if side == "BUY":
                    exit_price = intended - cost
                    gross = (
                        exit_price - position["entry"]
                    ) * position["qty"]
                else:
                    exit_price = intended + cost
                    gross = (
                        position["entry"] - exit_price
                    ) * position["qty"]

                pnl = gross - commission * position["qty"]
                balance += pnl
                peak = max(peak, balance)

                dd = (
                    (peak - balance) / peak * 100
                    if peak > 0 else 0
                )

                trades.append({
                    "entry_time": position["time"],
                    "side": side,
                    "entry": position["entry"],
                    "stop": stop,
                    "target": target,
                    "qty_oz": position["qty"],
                    "exit_time": ts,
                    "exit": exit_price,
                    "net_pnl": pnl,
                    "equity": balance,
                    "drawdown_pct": dd,
                    "reason": reason,
                    "sample": position["sample"]
                })

                position = None

                if balance <= day_start - daily_limit:
                    blocked = True

        equity_points.append({
            "time": ts,
            "equity": balance
        })

        if blocked or position is not None:
            continue

        if balance <= 0:
            break

        # Create a new setup from confirmed swing pivots.
        if pending is None and len(pivots) >= 2:
            a, b = pivots[-2], pivots[-1]

            if a[0] != b[0]:
                high = max(a[1], b[1])
                low = min(a[1], b[1])
                width = high - low

                if width > 0:
                    if a[0] == "L" and b[0] == "H":
                        side = "BUY"
                    elif a[0] == "H" and b[0] == "L":
                        side = "SELL"
                    else:
                        side = None

                    if side:
                        pending = {
                            "side": side,
                            "high": high,
                            "low": low,
                            "created": i
                        }

        if pending is None:
            continue

        if i - pending["created"] > 40:
            pending = None
            continue

        side = pending["side"]
        high = pending["high"]
        low = pending["low"]
        width = high - low

        if side == "BUY":
            zone_high = high - width * 0.500
            zone_low = high - width * 0.618
            valid_trend = row.ema50 > row.ema200
            touched = (
                row.low <= zone_high and
                row.high >= zone_low
            )
            stop = low
        else:
            zone_low = low + width * 0.500
            zone_high = low + width * 0.618
            valid_trend = row.ema50 < row.ema200
            touched = (
                row.high >= zone_low and
                row.low <= zone_high
            )
            stop = high

        if not valid_trend or not touched:
            continue

        # Enter on next candle's open, not the signal candle.
        if i + 1 >= len(d):
            continue

        next_row = d.iloc[i + 1]
        next_ts = d.index[i + 1]

        if i + 1 == split_idx:
            pending = None
            continue

        cost = spread / 2 + slippage

        if side == "BUY":
            entry = next_row.open + cost
            distance = entry - stop
            target = entry + rr * distance
        else:
            entry = next_row.open - cost
            distance = stop - entry
            target = entry - rr * distance

        if distance <= 0:
            pending = None
            continue

        available_daily_risk = max(
            0.0, daily_limit - max(0.0, day_start - balance)
        )

        risk_dollars = min(
            balance * risk_pct / 100,
            available_daily_risk
        )

        # Include estimated exit costs in risk sizing.
        qty = risk_dollars / (
            distance + cost + commission
        )

        if qty <= 0:
            pending = None
            continue

        position = {
            "side": side,
            "entry": float(entry),
            "stop": float(stop),
            "target": float(target),
            "qty": float(qty),
            "time": next_ts,
            "sample": (
                "Training" if i + 1 < split_idx
                else "Out-of-Sample"
            )
        }

        pending = None

    # Close any remaining position at final available close.
    if position is not None:
        final_row = d.iloc[-1]
        side = position["side"]
        cost = spread / 2 + slippage

        if side == "BUY":
            exit_price = final_row.close - cost
            gross = (
                exit_price - position["entry"]
            ) * position["qty"]
        else:
            exit_price = final_row.close + cost
            gross = (
                position["entry"] - exit_price
            ) * position["qty"]

        pnl = gross - commission * position["qty"]
        balance += pnl
        peak = max(peak, balance)

        trades.append({
            "entry_time": position["time"],
            "side": side,
            "entry": position["entry"],
            "stop": position["stop"],
            "target": position["target"],
            "qty_oz": position["qty"],
            "exit_time": d.index[-1],
            "exit": exit_price,
            "net_pnl": pnl,
            "equity": balance,
            "drawdown_pct": (
                (peak - balance) / peak * 100
                if peak > 0 else 0
            ),
            "reason": "End of Data",
            "sample": position["sample"]
        })

        equity_points.append({
            "time": d.index[-1],
            "equity": balance
        })

    return (
        pd.DataFrame(trades),
        pd.DataFrame(equity_points),
        balance
    )


if file is not None:
    try:
        data = load_data(file.getvalue())

        st.success(
            f"Loaded {len(data):,} M15 candles"
        )

        st.write(
            "Data period:",
            str(data.index.min()),
            "to",
            str(data.index.max())
        )

        st.line_chart(data[["close"]])

        if st.button("🚀 Run V28 Backtest"):
            trades, curve, final_balance = run_backtest(data)

            if trades.empty:
                st.warning("No completed trades found.")
            else:
                wins = int((trades.net_pnl > 0).sum())
                total = len(trades)
                pnl = float(trades.net_pnl.sum())
                win_rate = wins / total * 100

                max_dd = (
                    (curve.equity.cummax() - curve.equity)
                    / curve.equity.cummax()
                    * 100
                ).max()

                st.header("📊 Backtest Results")

                c1, c2, c3 = st.columns(3)
                c1.metric("Trades", total)
                c2.metric("Win Rate", f"{win_rate:.1f}%")
                c3.metric("Net P&L", f"${pnl:.2f}")

                c4, c5, c6 = st.columns(3)
                c4.metric(
                    "Final Balance",
                    f"${final_balance:.2f}"
                )
                c5.metric(
                    "Max Drawdown",
                    f"{max_dd:.2f}%"
                )
                c6.metric(
                    "Return",
                    f"{pnl / capital * 100:.2f}%"
                )

                st.subheader("📈 Continuous Equity")
                st.line_chart(
                    curve.set_index("time")["equity"]
                )

                st.subheader("Training vs Out-of-Sample")

                summary = trades.groupby("sample").agg(
                    trades=("net_pnl", "count"),
                    net_pnl=("net_pnl", "sum")
                )

                st.dataframe(summary)

                st.subheader("🔍 Equity Audit")

                expected = capital + pnl
                difference = final_balance - expected

                st.write("Expected Balance:", round(expected, 2))
                st.write("Actual Balance:", round(final_balance, 2))
                st.write("Difference:", round(difference, 8))

                if abs(difference) < 0.000001:
                    st.success(
                        "PASS: Equity and total P&L reconcile."
                    )
                else:
                    st.error(
                        "FAIL: Equity reconciliation mismatch."
                    )

                st.subheader("📋 Trades")
                st.dataframe(trades)

                st.download_button(
                    "⬇️ Download V28 Audited CSV",
                    trades.to_csv(index=False).encode("utf-8"),
                    file_name="Tuesday_AI_V28_audited.csv",
                    mime="text/csv"
                )

                st.warning(
                    "Experimental strategy. No profitability "
                    "or funded-account compliance is guaranteed. "
                    "Daily loss limits are approximate because "
                    "gaps and slippage can exceed planned risk."
                )

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload original historical OHLC data to begin.")
