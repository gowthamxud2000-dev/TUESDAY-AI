
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tuesday AI V42 Audit", layout="wide")
st.title("Tuesday AI V42 | Audit Fix")
st.warning("Synthetic research only. No live trading.")

with st.sidebar:
    seed = st.number_input("Random seed", 1, 999999, 42)
    days = st.slider("Demo days", 10, 35, 18)
    balance = st.number_input("Starting balance ($)", min_value=100.0, value=2500.0)
    daily_limit = st.number_input("Daily loss budget ($)", min_value=1.0, value=40.0)
    risk = st.number_input("Risk per trade ($)", min_value=0.5, value=5.0)
    contract = st.number_input("Ounces per lot", min_value=1.0, value=100.0)
    minlot = st.number_input("Minimum lot", min_value=0.01, value=0.01)
    step = st.number_input("Lot step", min_value=0.01, value=0.01)
    spread = st.number_input("Spread ($/oz)", min_value=0.0, value=0.30)
    slip = st.number_input("Slippage per side ($/oz)", min_value=0.0, value=0.10)
    commission = st.number_input("Commission per lot ($)", min_value=0.0, value=7.0)

@st.cache_data
def make_data(seed, days):
    rng = np.random.default_rng(int(seed))
    n = days * 1440
    t = pd.date_range("2026-09-01", periods=n, freq="min")
    delta = rng.normal(0, .33, n) + .045*np.sin(np.arange(n)/950)
    close = 3400 + np.cumsum(delta)
    op = np.r_[3400., close[:-1]]
    hi = np.maximum(op, close) + rng.uniform(.03, .23, n)
    lo = np.minimum(op, close) - rng.uniform(.03, .23, n)
    return pd.DataFrame({
        "time":t, "open":op, "high":hi,
        "low":lo, "close":close
    })

def prepare(raw):
    d = raw.copy()
    m = d.set_index("time").resample(
        "5min", closed="left", label="right"
    ).agg({"close":"last"}).dropna()
    m["e50"] = m.close.ewm(span=50, adjust=False).mean()
    m["e200"] = m.close.ewm(span=200, adjust=False).mean()
    m["trend"] = np.select([
        (m.close>m.e50)&(m.e50>m.e200),
        (m.close<m.e50)&(m.e50<m.e200)
    ], [1,-1], 0)
    d["known_at"] = d.time + pd.Timedelta(minutes=1)
    d = pd.merge_asof(
        d, m[["trend"]].reset_index().rename(
            columns={"time":"m5_time"}
        ),
        left_on="known_at",
        right_on="m5_time",
        direction="backward"
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
    d["swhi"] = d.high.shift(1).rolling(120).max()
    d["swlo"] = d.low.shift(1).rolling(120).min()
    span = d.swhi-d.swlo
    d["breakout"] = np.select([
        (d.trend==1)&(d.close>d.hi20)&(d.close>d.ema20),
        (d.trend==-1)&(d.close<d.lo20)&(d.close<d.ema20)
    ], [1,-1], 0)
    d["fib"] = np.select([
        (d.trend==1)&(span>0)&
        (d.close>=d.swhi-.618*span)&
        (d.close<=d.swhi-.5*span)&
        (d.close>d.open),
        (d.trend==-1)&(span>0)&
        (d.close>=d.swlo+.5*span)&
        (d.close<=d.swlo+.618*span)&
        (d.close<d.open)
    ], [1,-1], 0)
    d.loc[:999, ["breakout","fib"]] = 0
    return d

def backtest(d, strategy, first, last):
    equity = float(balance)
    trades = []
    day_start = {}
    locked = set()
    i = max(first, 1000)

    while i+1 < last:
        side = int(d.at[i,strategy])
        if side == 0 or not np.isfinite(d.at[i,"atr"]):
            i += 1
            continue

        k = i+1
        day = d.at[k,"time"].date()
        if day not in day_start:
            day_start[day] = equity

        remaining = daily_limit - max(
            0, day_start[day]-equity
        )
        if day in locked or remaining <= 0 or equity <= 0:
            locked.add(day)
            i += 1
            continue

        dist = max(float(d.at[i,"atr"])*1.5, .10)
        cost = spread/2 + slip
        perlot = (dist+2*cost)*contract+commission
        budget = min(risk, remaining, equity*.01)
        lots = round(float(np.floor(
            (budget/perlot)/step+1e-10
        )*step), 8)
        if lots < minlot:
            i += 1
            continue

        entry = float(d.at[k,"open"])+side*cost
        sl = entry-side*dist
        tp = entry+side*2*dist
        exit_k = last-1
        exit_price = float(d.at[exit_k,"close"])-side*cost
        reason = "SEGMENT_END"

        for j in range(k,last):
            op = float(d.at[j,"open"])-side*cost
            lo = float(d.at[j,"low"])-side*cost
            hi = float(d.at[j,"high"])-side*cost

            stop_hit = lo<=sl if side==1 else hi>=sl
            tp_hit = hi>=tp if side==1 else lo<=tp

            if stop_hit:
                exit_price = min(sl,op) if side==1 else max(sl,op)
                exit_k = j
                reason = "BOTH_STOP_FIRST" if tp_hit else "STOP"
                break
            if tp_hit:
                exit_price = tp
                exit_k = j
                reason = "TARGET"
                break
            if j+1<last and d.at[j+1,"time"].date()!=day:
                exit_price = float(d.at[j,"close"])-side*cost
                exit_k = j
                reason = "DAY_END"
                break

        pnl = side*(exit_price-entry)*contract*lots-commission*lots
        equity += pnl
        drawdown = max(0,day_start[day]-equity)
        if drawdown >= daily_limit-1e-8:
            locked.add(day)

        trades.append({
            "entry_time":d.at[k,"time"],
            "exit_time":d.at[exit_k,"time"],
            "side":"BUY" if side==1 else "SELL",
            "lots":lots,
            "entry":entry,
            "sl":sl,
            "tp":tp,
            "exit":exit_price,
            "reason":reason,
            "pnl":pnl,
            "equity":equity,
            "day":day,
            "daily_drawdown":drawdown
        })
        i = exit_k+1

    return pd.DataFrame(trades)

def report(name, trades):
    st.subheader(name)
    if trades.empty:
        st.info("No executable trades.")
        return {
            "Strategy":name,"Trades":0,
            "Win %":0,"Net $":0,"Cap breaches":0
        }

    a,b,c = st.columns(3)
    a.metric("Trades",len(trades))
    b.metric("Win rate",f"{(trades.pnl>0).mean()*100:.1f}%")
    c.metric("Net P/L",f"${trades.pnl.sum():.2f}")

    breaches = int(
        (trades.daily_drawdown>daily_limit+1e-7).sum()
    )
    st.write("Daily loss breaches:",breaches)
    st.write(
        "Stop-first ambiguous bars:",
        int((trades.reason=="BOTH_STOP_FIRST").sum())
    )
    with st.expander("Trade audit"):
        st.dataframe(trades,use_container_width=True)
        st.line_chart(trades.set_index("exit_time").equity)

    return {
        "Strategy":name,
        "Trades":len(trades),
        "Win %":round((trades.pnl>0).mean()*100,2),
        "Net $":round(trades.pnl.sum(),2),
        "Cap breaches":breaches
    }

if st.button("Run V42 Audit Fix",type="primary"):
    data = make_data(int(seed),days)
    d = prepare(data)
    split = int(len(d)*.7)
    st.success(f"{len(d):,} synthetic M1 candles generated")

    results = []
    for name,col in [
        ("Breakout","breakout"),
        ("Fibonacci Golden Zone","fib")
    ]:
        for segment,a,b in [
            ("Development",0,split),
            ("Holdout",split,len(d))
        ]:
            trades = backtest(d,col,a,b)
            results.append(
                report(name+" - "+segment,trades)
            )

    st.subheader("Strategy Comparison")
    st.dataframe(pd.DataFrame(results),use_container_width=True)

    st.subheader("Latest Synthetic Signals")
    labels = {1:"BUY",-1:"SELL",0:"WAIT"}
    st.write("Breakout:",labels[int(d.breakout.iloc[-1])])
    st.write("Fibonacci:",labels[int(d.fib.iloc[-1])])

    st.info(
        "Synthetic validation only. Fibonacci uses a "
        "rolling range approximation. Daily loss "
        "breaches are checked on closed trades, "
        "not intratrade equity. Gaps can exceed $40."
    )
else:
    st.info("Press Run V42 Audit Fix. No CSV needed.")
