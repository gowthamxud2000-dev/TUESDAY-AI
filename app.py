import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tuesday AI V43", layout="wide")
st.title("Tuesday AI V43 | Trend Pullback Lab")
st.warning("Synthetic demo only. No live prices, orders, or guaranteed profits.")

with st.sidebar:
    seed = st.number_input("Random seed", 1, 999999, 43)
    days = st.slider("Demo days", 10, 35, 18)
    starting = st.number_input("Starting balance ($)", min_value=100.0, value=2500.0)
    risk = st.number_input("Risk per trade ($)", min_value=0.5, value=4.0)
    daily_cap = st.number_input("Daily loss budget ($)", min_value=1.0, value=40.0)
    contract = st.number_input("Ounces per lot", min_value=1.0, value=100.0)
    minimum = st.number_input("Minimum lot", min_value=0.01, value=0.01, step=0.01)
    step = st.number_input("Lot increment", min_value=0.01, value=0.01, step=0.01)
    spread = st.number_input("Spread ($/oz)", min_value=0.0, value=0.30)
    slip = st.number_input("Slippage per side ($/oz)", min_value=0.0, value=0.10)
    commission = st.number_input("Round-trip commission ($/lot)", min_value=0.0, value=7.0)


@st.cache_data
def make_data(seed, days):
    rng = np.random.default_rng(int(seed))
    n = int(days) * 1440
    t = pd.date_range("2026-09-01", periods=n, freq="min")
    drift = 0.035 * np.sin(np.arange(n) / 700)
    changes = rng.normal(0, 0.31, n) + drift
    close = 3400 + np.cumsum(changes)
    op = np.r_[3400.0, close[:-1]]
    high = np.maximum(op, close) + rng.uniform(0.02, 0.18, n)
    low = np.minimum(op, close) - rng.uniform(0.02, 0.18, n)
    return pd.DataFrame({
        "time": t, "open": op, "high": high,
        "low": low, "close": close
    })


def prepare(raw):
    d = raw.copy().reset_index(drop=True)
    m = d.set_index("time").resample(
        "5min", closed="left", label="right"
    ).agg({"close": "last"}).dropna()

    m["fast"] = m.close.ewm(span=50, adjust=False).mean()
    m["slow"] = m.close.ewm(span=200, adjust=False).mean()
    m["trend"] = np.select([
        (m.close > m.fast) & (m.fast > m.slow),
        (m.close < m.fast) & (m.fast < m.slow)
    ], [1, -1], 0)

    d["known_at"] = d.time + pd.Timedelta(minutes=1)
    known = m[["trend"]].reset_index().rename(
        columns={"time": "m5_end"}
    )
    d = pd.merge_asof(
        d, known, left_on="known_at",
        right_on="m5_end", direction="backward"
    )
    d["trend"] = d.trend.fillna(0)
    d["ema"] = d.close.ewm(span=20, adjust=False).mean()
    tr = pd.concat([
        d.high - d.low,
        (d.high - d.close.shift()).abs(),
        (d.low - d.close.shift()).abs()
    ], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()

    prior_touch = (
        (d.low.shift(1) <= d.ema.shift(1))
        & (d.close.shift(1) < d.ema.shift(1))
    )
    prior_sell = (
        (d.high.shift(1) >= d.ema.shift(1))
        & (d.close.shift(1) > d.ema.shift(1))
    )
    buy = (
        (d.trend == 1) & prior_touch
        & (d.close > d.ema)
        & (d.close > d.high.shift(1))
    )
    sell = (
        (d.trend == -1) & prior_sell
        & (d.close < d.ema)
        & (d.close < d.low.shift(1))
    )
    d["signal"] = np.select([buy, sell], [1, -1], 0)
    d.loc[:999, "signal"] = 0
    return d


def exit_on_bar(side, opening, low, high, stop, target):
    if side * (opening - stop) <= 0:
        return opening, "STOP_GAP"
    if side * (opening - target) >= 0:
        return target, "TARGET_OPEN"

    stop_hit = low <= stop if side == 1 else high >= stop
    target_hit = high >= target if side == 1 else low <= target

    if stop_hit:
        return stop, "BOTH_STOP_FIRST" if target_hit else "STOP"
    if target_hit:
        return target, "TARGET"
    return None, None


def test(d, start, end):
    equity = float(starting)
    day_open = {}
    locked_days = set()
    trades = []
    i = max(start, 1000)

    while i + 1 < end:
        side = int(d.at[i, "signal"])
        if side == 0 or not np.isfinite(d.at[i, "atr"]) or equity <= 0:
            i += 1
            continue

        k = i + 1
        day = d.at[k, "time"].date()
        day_open.setdefault(day, equity)
        remaining = daily_cap - max(0, day_open[day] - equity)

        if day in locked_days or remaining <= 1e-8:
            i += 1
            continue

        dist = max(1.5 * float(d.at[i, "atr"]), 0.10)
        cost = spread / 2 + slip
        per_lot = dist * contract + commission
        budget = min(risk, remaining, equity * 0.01)
        lots = round(
            float(np.floor((budget / per_lot) / step + 1e-9) * step), 8
        )
        if lots < minimum:
            i += 1
            continue

        entry = float(d.at[k, "open"]) + side * cost
        stop = entry - side * dist
        target = entry + side * 2 * dist
        daily_floor = day_open[day] - daily_cap
        daily_stop = entry + side * (
            daily_floor - equity + commission * lots
        ) / (contract * lots)

        effective_stop = (
            max(stop, daily_stop) if side == 1
            else min(stop, daily_stop)
        )
        daily_stop_active = side * (effective_stop - stop) > 1e-10
        j_exit = end - 1
        exit_fill = float(d.at[j_exit, "close"]) - side * cost
        reason = "SEGMENT_END"

        for j in range(k, end):
            adj = side * cost
            opening = float(d.at[j, "open"]) - adj
            low = float(d.at[j, "low"]) - adj
            high = float(d.at[j, "high"]) - adj
            fill, bar_reason = exit_on_bar(
                side, opening, low, high, effective_stop, target
            )
            if fill is not None:
                exit_fill = fill
                j_exit = j
                reason = (
                    "DAILY_LOCK_" + bar_reason
                    if daily_stop_active and "STOP" in bar_reason
                    else bar_reason
                )
                break

            if j + 1 < end and d.at[j + 1, "time"].date() != day:
                exit_fill = float(d.at[j, "close"]) - adj
                j_exit = j
                reason = "DAY_END"
                break

        pnl = (
            side * (exit_fill - entry) * contract * lots
            - commission * lots
        )
        equity += pnl
        if equity <= daily_floor + 1e-8:
            locked_days.add(day)

        trades.append({
            "entry_time": d.at[k, "time"],
            "exit_time": d.at[j_exit, "time"],
            "side": "BUY" if side == 1 else "SELL",
            "lots": lots,
            "entry": entry,
            "sl": stop,
            "tp": target,
            "exit": exit_fill,
            "reason": reason,
            "pnl": pnl,
            "equity": equity,
            "day": str(day),
            "daily_drawdown": max(0, day_open[day] - equity)
        })
        i = j_exit + 1

    return pd.DataFrame(trades)


def report(title, trades):
    st.subheader(title)
    if trades.empty:
        st.info("No trades under these settings.")
        return

    a, b, c = st.columns(3)
    a.metric("Trades", len(trades))
    b.metric("Win rate", f"{(trades.pnl > 0).mean() * 100:.1f}%")
    c.metric("Net P/L", f"${trades.pnl.sum():.2f}")

    breaches = int((trades.daily_drawdown > daily_cap + 1e-8).sum())
    ambiguous = int(trades.reason.str.contains("BOTH_STOP_FIRST").sum())
    locked_count = int(trades.loc[
        trades.daily_drawdown >= daily_cap - 1e-8, "day"
    ].nunique())

    st.write("Closed-trade daily budget breaches:", breaches)
    st.write("Stop-first ambiguous candles:", ambiguous)
    st.write("Days reaching the daily loss budget:", locked_count)
    st.line_chart(trades.set_index("exit_time").equity)

    with st.expander("Trade audit"):
        st.dataframe(trades, use_container_width=True)


if st.button("Run V43 Demo Backtest", type="primary"):
    d = prepare(make_data(seed, days))
    split = int(len(d) * 0.7)
    st.success(f"{len(d):,} synthetic M1 candles generated")
    report("Development 70%", test(d, 0, split))
    report("Holdout 30%", test(d, split, len(d)))

    label = {1: "BUY", -1: "SELL", 0: "WAIT"}
    st.write("Latest synthetic signal:", label[int(d.signal.iloc[-1])])
    st.info(
        "No real historical validation. Daily loss exits use "
        "executable candle quotes and include commission. "
        "Gaps can exceed the daily budget; reaching it blocks "
        "further entries that day. Constant costs and stop-first "
        "ambiguous candles are modeling assumptions."
    )
else:
    st.info("Tap Run V43 Demo Backtest. No downloads needed.")

# END OF APP
