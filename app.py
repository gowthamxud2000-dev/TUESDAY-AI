
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tuesday AI V42", layout="wide")
st.title("Tuesday AI V42 | Synthetic Strategy Audit")
st.warning("Research only. Synthetic prices are not historical XAUUSD. No live signals or orders.")

with st.sidebar:
    seed = st.number_input("Random seed", 1, 999999, 42)
    days = st.slider("Synthetic days", 10, 35, 18)
    initial = st.number_input("Starting balance ($)", min_value=100.0, value=2500.0)
    daily_cap = st.number_input("Daily loss budget ($)", min_value=1.0, value=40.0)
    risk_budget = st.number_input("Risk per trade ($)", min_value=0.5, value=5.0)
    contract = st.number_input("Ounces per lot", min_value=1.0, value=100.0)
    min_lot = st.number_input("Minimum lot", min_value=0.01, value=0.01, step=0.01)
    lot_step = st.number_input("Lot step", min_value=0.01, value=0.01, step=0.01)
    spread = st.number_input("Spread ($/oz)", min_value=0.0, value=0.30, step=0.05)
    slip = st.number_input("Slippage per side ($/oz)", min_value=0.0, value=0.10, step=0.05)
    commission = st.number_input("Round-trip commission ($/lot)", min_value=0.0, value=7.0)

@st.cache_data
def synthetic(seed, days):
    rng = np.random.default_rng(int(seed))
    n = int(days) * 1440
    idx = pd.date_range("2026-09-01", periods=n, freq="min")
    drift = 0.045 * np.sin(np.arange(n) / 950)
    delta = rng.normal(0, 0.33, n) + drift
    close = 3400 + np.cumsum(delta)
    op = np.r_[3400., close[:-1]]
    high = np.maximum(op, close) + rng.uniform(0.03, 0.23, n)
    low = np.minimum(op, close) - rng.uniform(0.03, 0.23, n)
    return pd.DataFrame(dict(time=idx, open=op, high=high, low=low, close=close))

def features(raw):
    d = raw.copy().sort_values("time").reset_index(drop=True)
    # Time is candle OPEN; a 5m bar is only available at its close.
    m = d.set_index("time").resample(
        "5min", closed="left", label="right"
    ).agg({"close": "last"}).dropna()
    m["e50"] = m.close.ewm(span=50, adjust=False).mean()
    m["e200"] = m.close.ewm(span=200, adjust=False).mean()
    m["trend"] = np.select([
        (m.close > m.e50) & (m.e50 > m.e200),
        (m.close < m.e50) & (m.e50 < m.e200)
    ], [1, -1], 0)
    d = pd.merge_asof(
        d, m[["trend"]].reset_index(),
        on="time", direction="backward"
    )
    d["trend"] = d.trend.fillna(0)
    d["ema20"] = d.close.ewm(span=20, adjust=False).mean()
    d["hi20"] = d.high.shift(1).rolling(20).max()
    d["lo20"] = d.low.shift(1).rolling(20).min()
    tr = pd.concat([
        d.high-d.low,
        (d.high-d.close.shift()).abs(),
        (d.low-d.close.shift()).abs()
    ], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    d["swing_hi"] = d.high.shift(1).rolling(120).max()
    d["swing_lo"] = d.low.shift(1).rolling(120).min()
    d["breakout"] = np.select([
        (d.trend==1) & (d.close>d.hi20) & (d.close>d.ema20),
        (d.trend==-1) & (d.close<d.lo20) & (d.close<d.ema20)
    ], [1, -1], 0)
    span = d.swing_hi-d.swing_lo
    d["fib"] = np.select([
        (d.trend==1) & (span>0) &
        (d.close>=d.swing_hi-.618*span) &
        (d.close<=d.swing_hi-.5*span) &
        (d.close>d.open),
        (d.trend==-1) & (span>0) &
        (d.close>=d.swing_lo+.5*span) &
        (d.close<=d.swing_lo+.618*span) &
        (d.close<d.open)
    ], [1, -1], 0)
    d.loc[:999, ["breakout", "fib"]] = 0
    return d

def backtest(d, column, first, last, start_balance):
    equity = float(start_balance)
    trades = []
    day_start = {}
    i = max(1000, first)
    while i < min(last, len(d)-1):
        signal = int(d.at[i, column])
        if not signal or not np.isfinite(d.at[i, "atr"]):
            i += 1
            continue
        entry_i = i+1
        if entry_i >= last:
            break
        day = d.at[entry_i, "time"].date()
        if day not in day_start:
            day_start[day] = equity
        remaining = daily_cap - (day_start[day] - equity)
        if remaining <= 0 or equity <= 0:
            i += 1
            continue
        distance = max(float(d.at[i, "atr"])*1.5, .10)
        # Conservative normal-stop budget, not a gap guarantee.
        per_lot = (distance+spread+2*slip)*contract+commission
        allowed = min(risk_budget, remaining, equity*.01)
        lots = round(float(np.floor(
            (allowed/per_lot)/lot_step+1e-9
        )*lot_step), 8)
        if lots < min_lot:
            i += 1
            continue
        entry = float(d.at[entry_i, "open"]) + signal*(spread/2+slip)
        stop = entry-signal*distance
        target = entry+signal*2*distance
        exit_i = min(last-1, len(d)-1)
        exit_fill = float(d.at[exit_i, "close"])-signal*(spread/2+slip)
        reason = "SEGMENT_END"
        for j in range(entry_i, min(last,len(d))):
            adjust = signal*(spread/2+slip)
            low = float(d.at[j, "low"])-adjust
            high = float(d.at[j, "high"])-adjust
            stop_hit = low <= stop if signal==1 else high >= stop
            target_hit = high >= target if signal==1 else low <= target
            if stop_hit:
                open_exit = float(d.at[j, "open"])-adjust
                exit_fill = min(stop,open_exit) if signal==1 else max(stop,open_exit)
                exit_i = j
                reason = "STOP_FIRST" if target_hit else "STOP"
                break
            if target_hit:
                exit_fill = target
                exit_i = j
                reason = "TARGET"
                break
            if j+1 < last and d.at[j+1, "time"].date()!=day:
                exit_fill = float(d.at[j, "close"])-adjust
                exit_i = j
                reason = "DAY_END"
                break
        pnl = signal*(exit_fill-entry)*contract*lots-commission*lots
        equity += pnl
        trades.append(dict(
            entry_time=d.at[entry_i, "time"],
            exit_time=d.at[exit_i, "time"],
            side="BUY" if signal==1 else "SELL",
            lots=lots, entry=entry, sl=stop, tp=target,
            exit=exit_fill, reason=reason, pnl=pnl,
            equity=equity, day=day,
            daily_loss=max(0,day_start[day]-equity)
        ))
        i = exit_i+1
    return pd.DataFrame(trades)

def report(name, t):
    st.subheader(name)
    if t.empty:
        st.info("No trades at these risk/lot settings.")
        return
    wins = (t.pnl>0).mean()*100
    a,b,c,d = st.columns(4)
    a.metric("Trades",len(t))
    b.metric("Win rate",f"{wins:.1f}%")
    c.metric("Net P/L",f"${t.pnl.sum():,.2f}")
    d.metric("Worst daily P/L",f"${t.groupby('day').pnl.sum().min():,.2f}")
    st.write(
        "Days exceeding daily loss budget:",
        int((t.groupby("day").pnl.sum() < -daily_cap-1e-7).sum())
    )
    st.write(
        "Stop-first ambiguous bars:",
        int((t.reason=="STOP_FIRST").sum())
    )
    st.line_chart(t.set_index("exit_time").equity)
    with st.expander("Trade-level audit"):
        st.dataframe(t, use_container_width=True)

if st.button("Generate Demo & Compare Strategies", type="primary"):
    raw = synthetic(seed, days)
    d = features(raw)
    split = int(len(d)*.7)
    st.success(f"{len(d):,} synthetic M1 candles generated")
    st.caption(
        "70% development / 30% chronological holdout. "
        "Indicators use earlier history for warm-up."
    )
    summary = []
    for strategy, col in [
        ("Breakout","breakout"),
        ("Fibonacci Golden Zone","fib")
    ]:
        for segment, a, b in [
            ("Development",0,split),
            ("Holdout",split,len(d))
        ]:
            t = backtest(d,col,a,b,initial)
            report(strategy+" - "+segment,t)
            summary.append(dict(
                strategy=strategy,
                segment=segment,
                trades=len(t),
                win_rate=round((t.pnl>0).mean()*100,2)
                    if len(t) else None,
                net_pnl=round(t.pnl.sum(),2)
                    if len(t) else 0
            ))
    st.subheader("Comparison")
    st.dataframe(pd.DataFrame(summary),use_container_width=True)
    st.info(
        "Golden Zone uses a simplified rolling range, "
        "not confirmed swing Fibonacci. Synthetic holdout "
        "is not independent real-market validation. "
        "Daily losses can exceed the planned budget."
    )
else:
    st.info("Tap Generate Demo & Compare Strategies. No file needed.")
