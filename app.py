
import streamlit as st
import pandas as pd
import numpy as np
import io
import math

st.set_page_config(
    page_title="Tuesday AI V31",
    page_icon="🥇",
    layout="wide"
)

st.title("🥇 Tuesday AI V31")
st.caption("XAUUSD M15 | Manual BUY / SELL / WAIT")

st.warning(
    "Research and demo use only. Signals are not guaranteed. "
    "No live feed or automatic order execution."
)

# =====================================
# FUNDED ACCOUNT SETTINGS
# =====================================

st.sidebar.header("⚙️ Funded Account")

balance = st.sidebar.number_input(
    "Starting balance ($)",
    min_value=100.0,
    value=2500.0,
    step=100.0
)

daily_limit = st.sidebar.number_input(
    "Daily loss limit ($)",
    min_value=1.0,
    value=40.0
)

risk_pct = st.sidebar.number_input(
    "Risk per trade (%)",
    min_value=0.05,
    max_value=2.0,
    value=0.25,
    step=0.05
)

rr = st.sidebar.number_input(
    "Risk reward ratio",
    min_value=1.0,
    max_value=5.0,
    value=2.0,
    step=0.25
)

spread = st.sidebar.number_input(
    "Spread (gold price units)",
    min_value=0.0,
    value=0.25,
    step=0.05
)

slippage = st.sidebar.number_input(
    "Slippage per side (price units)",
    min_value=0.0,
    value=0.10,
    step=0.05
)

commission = st.sidebar.number_input(
    "Commission per lot round trip ($)",
    min_value=0.0,
    value=7.0
)

contract_size = st.sidebar.number_input(
    "Ounces per lot",
    min_value=1.0,
    value=100.0
)

lot_step = st.sidebar.number_input(
    "Lot step",
    min_value=0.001,
    value=0.01,
    step=0.001,
    format="%.3f"
)

max_lots = st.sidebar.number_input(
    "Maximum lots",
    min_value=0.01,
    value=1.0
)

max_trades = st.sidebar.slider(
    "Maximum trades per day",
    1, 10, 3
)

max_hold = st.sidebar.slider(
    "Maximum holding candles",
    4, 96, 32
)

st.sidebar.info(
    "Check broker contract size, minimum lot, "
    "commission, and funded-account rules."
)

# =====================================
# CSV LOADER
# =====================================

def load_csv(upload):
    raw = upload.getvalue()
    text = raw.decode(
        "utf-8-sig", errors="replace"
    )

    first_line = text.splitlines()[0]
    sep = (
        ";" if first_line.count(";") >
        first_line.count(",") else ","
    )

    df = pd.read_csv(
        io.StringIO(text), sep=sep
    )

    df.columns = [
        str(c).strip().lower()
        .replace("<", "").replace(">", "")
        for c in df.columns
    ]

    required = {"open", "high", "low", "close"}

    if not required.issubset(df.columns):
        df = pd.read_csv(
            io.StringIO(text),
            sep=sep,
            header=None
        )

        if df.shape[1] >= 7:
            df = df.iloc[:, :7].copy()
            df.columns = [
                "date", "time", "open",
                "high", "low", "close", "volume"
            ]
        elif df.shape[1] == 6:
            df.columns = [
                "datetime", "open", "high",
                "low", "close", "volume"
            ]
        elif df.shape[1] == 5:
            df.columns = [
                "datetime", "open", "high",
                "low", "close"
            ]
        else:
            raise ValueError(
                "CSV must contain OHLC data."
            )

    if "datetime" in df.columns:
        dt = df["datetime"].astype(str)
    elif "timestamp" in df.columns:
        dt = df["timestamp"].astype(str)
    elif "date" in df.columns and "time" in df.columns:
        dt = (
            df["date"].astype(str)
            + " "
            + df["time"].astype(str)
        )
    elif "date" in df.columns:
        dt = df["date"].astype(str)
    else:
        raise ValueError(
            "Datetime column not found."
        )

    dates = pd.to_datetime(
        dt, errors="coerce"
    )

    if dates.notna().mean() < 0.8:
        dates = pd.to_datetime(
            dt,
            format="%Y.%m.%d %H:%M",
            errors="coerce"
        )

    if dates.notna().mean() < 0.8:
        dates = pd.to_datetime(
            dt,
            format="%Y%m%d %H%M%S",
            errors="coerce"
        )

    df["datetime"] = dates

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(
            df[c], errors="coerce"
        )

    df = df.dropna(
        subset=[
            "datetime", "open",
            "high", "low", "close"
        ]
    )

    df = df[
        (df["low"] > 0)
        & (df["high"] >= df["low"])
        & (
            df["high"] >=
            df[["open", "close"]].max(axis=1)
        )
        & (
            df["low"] <=
            df[["open", "close"]].min(axis=1)
        )
    ]

    df = df.sort_values("datetime")
    df = df.drop_duplicates("datetime")

    return df.reset_index(drop=True)


# =====================================
# INDICATORS
# =====================================

def indicators(df):
    d = df.copy()

    d["ema50"] = d["close"].ewm(
        span=50, adjust=False
    ).mean()

    d["ema200"] = d["close"].ewm(
        span=200, adjust=False
    ).mean()

    previous = d["close"].shift(1)

    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - previous).abs(),
        (d["low"] - previous).abs()
    ], axis=1).max(axis=1)

    d["atr"] = tr.rolling(14).mean()

    d["swing_high"] = (
        d["high"].shift(1)
        .rolling(40).max()
    )

    d["swing_low"] = (
        d["low"].shift(1)
        .rolling(40).min()
    )

    width = (
        d["swing_high"] - d["swing_low"]
    )

    d["buy_low"] = (
        d["swing_high"] - width * 0.65
    )
    d["buy_high"] = (
        d["swing_high"] - width * 0.50
    )

    d["sell_low"] = (
        d["swing_low"] + width * 0.50
    )
    d["sell_high"] = (
        d["swing_low"] + width * 0.65
    )

    return d


# =====================================
# SIGNAL ENGINE
# =====================================

def signal_engine(d, i):
    if i < 200:
        return "WAIT", "Insufficient history"

    r = d.iloc[i]
    p = d.iloc[i - 1]

    if pd.isna(r["atr"]) or r["atr"] <= 0:
        return "WAIT", "ATR unavailable"

    bullish = (
        r["ema50"] > r["ema200"]
        and r["close"] > r["ema50"]
        and r["ema50"] > p["ema50"]
    )

    bearish = (
        r["ema50"] < r["ema200"]
        and r["close"] < r["ema50"]
        and r["ema50"] < p["ema50"]
    )

    buy_zone = (
        r["low"] <= r["buy_high"]
        and r["high"] >= r["buy_low"]
    )

    sell_zone = (
        r["high"] >= r["sell_low"]
        and r["low"] <= r["sell_high"]
    )

    buy_candle = (
        r["close"] > r["open"]
        and r["close"] > p["close"]
    )

    sell_candle = (
        r["close"] < r["open"]
        and r["close"] < p["close"]
    )

    if bullish and buy_zone and buy_candle:
        return (
            "BUY",
            "EMA uptrend + Golden Zone + confirmation"
        )

    if bearish and sell_zone and sell_candle:
        return (
            "SELL",
            "EMA downtrend + Golden Zone + confirmation"
        )

    return "WAIT", "No confirmed setup"


# =====================================
# TRADE SIMULATION
# =====================================

def run_backtest(d, start, end):
    trades = []
    equity = float(balance)
    daily_pnl = {}
    daily_count = {}
    i = max(200, start)

    while i < min(end - 1, len(d) - 1):
        sig, reason = signal_engine(d, i)

        if sig == "WAIT":
            i += 1
            continue

        entry_i = i + 1
        er = d.iloc[entry_i]
        day = str(er["datetime"].date())

        day_pnl = daily_pnl.get(day, 0.0)
        count = daily_count.get(day, 0)

        if count >= max_trades:
            i += 1
            continue

        remaining = daily_limit + min(
            day_pnl, 0.0
        )

        if remaining <= 0:
            i += 1
            continue

        atr = float(d.iloc[i]["atr"])
        distance = atr * 1.5

        if distance <= 0:
            i += 1
            continue

        half_spread = spread / 2

        if sig == "BUY":
            entry = (
                float(er["open"])
                + half_spread + slippage
            )
            stop = entry - distance
            target = entry + distance * rr
        else:
            entry = (
                float(er["open"])
                - half_spread - slippage
            )
            stop = entry + distance
            target = entry - distance * rr

        risk_budget = min(
            equity * risk_pct / 100,
            remaining * 0.80
        )

        # Conservative loss includes exit costs.
        cost_per_lot = (
            (distance + spread / 2 + slippage)
            * contract_size + commission
        )

        raw_lots = risk_budget / cost_per_lot

        lots = (
            math.floor(
                raw_lots / lot_step + 1e-9
            ) * lot_step
        )

        lots = min(
            lots,
            math.floor(max_lots / lot_step)
            * lot_step
        )

        if lots < lot_step:
            i += 1
            continue

        exit_price = None
        exit_i = None
        outcome = "TIME"

        last_j = min(
            entry_i + max_hold - 1,
            end - 1,
            len(d) - 1
        )

        for j in range(entry_i, last_j + 1):
            bar = d.iloc[j]

            if sig == "BUY":
                hit_sl = bar["low"] <= stop
                hit_tp = bar["high"] >= target
            else:
                hit_sl = bar["high"] >= stop
                hit_tp = bar["low"] <= target

            # Conservative if both touched.
            if hit_sl:
                exit_price = stop
                exit_i = j
                outcome = "SL"
                break

            if hit_tp:
                exit_price = target
                exit_i = j
                outcome = "TP"
                break

        if exit_price is None:
            exit_i = last_j
            exit_price = float(
                d.iloc[exit_i]["close"]
            )

        if sig == "BUY":
            effective_exit = (
                exit_price - half_spread - slippage
            )
            gross = (
                effective_exit - entry
            ) * contract_size * lots
        else:
            effective_exit = (
                exit_price + half_spread + slippage
            )
            gross = (
                entry - effective_exit
            ) * contract_size * lots

        net = gross - commission * lots

        equity += net

        daily_pnl[day] = (
            daily_pnl.get(day, 0.0) + net
        )

        daily_count[day] = (
            daily_count.get(day, 0) + 1
        )

        trades.append({
            "signal_time": d.iloc[i]["datetime"],
            "entry_time": er["datetime"],
            "exit_time": d.iloc[exit_i]["datetime"],
            "side": sig,
            "entry": round(entry, 3),
            "stop": round(stop, 3),
            "target": round(target, 3),
            "exit": round(effective_exit, 3),
            "lots": round(lots, 4),
            "outcome": outcome,
            "net_pnl": round(net, 2),
            "equity": round(equity, 2),
            "reason": reason
        })

        # Only one position at a time.
        i = exit_i + 1

    return pd.DataFrame(trades)


# =====================================
# PERFORMANCE REPORT
# =====================================

def report(trades, name):
    st.subheader(name)

    if trades.empty:
        st.info("No trades found for this period.")
        return

    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] < 0]

    total = len(trades)
    win_rate = len(wins) / total * 100
    net = trades["net_pnl"].sum()

    gross_profit = wins["net_pnl"].sum()
    gross_loss = abs(losses["net_pnl"].sum())

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf")
    )

    curve = pd.concat([
        pd.Series([balance]),
        trades["equity"].reset_index(drop=True)
    ], ignore_index=True)

    peak = curve.cummax()
    drawdown = peak - curve

    max_dd = drawdown.max()

    c1, c2, c3 = st.columns(3)

    c1.metric("Trades", total)
    c2.metric("Win Rate", f"{win_rate:.1f}%")
    c3.metric("Net P&L", f"${net:.2f}")

    c4, c5, c6 = st.columns(3)

    c4.metric("Profit Factor", f"{pf:.2f}")
    c5.metric("Max Drawdown", f"${max_dd:.2f}")
    c6.metric(
        "Final Equity",
        f"${balance + net:.2f}"
    )

    st.line_chart(
        curve.reset_index(drop=True)
    )

    st.dataframe(
        trades, use_container_width=True
    )

    csv = trades.to_csv(index=False)

    st.download_button(
        f"⬇️ Download {name} CSV",
        data=csv,
        file_name=(
            name.replace(" ", "_") + ".csv"
        ),
        mime="text/csv"
    )

    if net <= 0 or pf < 1.2:
        st.error(
            "Validation failed: not suitable "
            "for live funded trading."
        )
    else:
        st.warning(
            "Positive historical results do not "
            "prove future profitability. "
            "Further demo validation required."
        )


# =====================================
# APP UI
# =====================================

st.header("📂 Upload XAUUSD M15 CSV")

uploaded = st.file_uploader(
    "Upload historical OHLC CSV",
    type=["csv", "txt"]
)

if uploaded is not None:
    try:
        df = load_csv(uploaded)

        if len(df) < 250:
            st.error(
                "At least 250 candles required."
            )
            st.stop()

        d = indicators(df)

        st.success(
            f"Loaded {len(d):,} candles"
        )

        st.write(
            "Period:",
            str(d["datetime"].min()),
            "to",
            str(d["datetime"].max())
        )

        st.line_chart(
            d.set_index("datetime")[
                ["close", "ema50", "ema200"]
            ].tail(300)
        )

        st.header("🚦 Latest Signal")

        latest = len(d) - 1
        sig, reason = signal_engine(d, latest)

        if sig == "BUY":
            st.success("🟢 BUY SETUP")
        elif sig == "SELL":
            st.error("🔴 SELL SETUP")
        else:
            st.warning("🟡 WAIT")

        st.write("Reason:", reason)
        st.caption(
            "Signal uses the latest uploaded candle. "
            "Only use completed M15 candles. "
            "This is not a live market signal."
        )

        st.header("📊 Independent Backtest")

        st.info(
            "Rules are fixed. September is the "
            "development period; October is the "
            "held-out test period. Do not tune "
            "parameters using October results."
        )

        months = d["datetime"].dt.to_period("M")

        train_mask = (
            months.astype(str) == "2026-09"
        )
        test_mask = (
            months.astype(str) == "2026-10"
        )

        if train_mask.any() and test_mask.any():
            train_indices = np.where(
                train_mask.to_numpy()
            )[0]

            test_indices = np.where(
                test_mask.to_numpy()
            )[0]

            train = run_backtest(
                d,
                int(train_indices[0]),
                int(train_indices[-1]) + 1
            )

            test = run_backtest(
                d,
                int(test_indices[0]),
                int(test_indices[-1]) + 1
            )

            report(
                train,
                "September 2026 Development"
            )

            report(
                test,
                "October 2026 Independent Test"
            )

        else:
            st.warning(
                "For independent validation, "
                "upload one CSV containing both "
                "September and October 2026 M15 data."
            )

            if st.button("Run full-period backtest"):
                trades = run_backtest(
                    d, 200, len(d)
                )

                report(
                    trades,
                    "Full Period Backtest"
                )

    except Exception as e:
        st.error(f"CSV processing error: {e}")

else:
    st.info(
        "Upload XAUUSD M15 historical CSV "
        "to begin analysis."
    )

st.divider()

st.caption(
    "Tuesday AI V31 | Manual research assistant | "
    "No automatic orders"
)
