
import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tuesday AI V41 Fixed", layout="wide")
st.title("Tuesday AI V41 | Demo + CSV Backtest")
st.warning("Research/demo only. Synthetic data is not real XAUUSD history or a live signal.")

with st.sidebar:
    balance = st.number_input("Starting balance ($)", min_value=100.0, value=2500.0)
    daily_limit = st.number_input("Daily loss limit ($)", min_value=1.0, value=40.0)
    risk = st.number_input("Risk per trade ($)", min_value=0.5, value=5.0)
    contract = st.number_input("Ounces per lot", min_value=1.0, value=100.0)
    min_lot = st.number_input("Minimum lot", min_value=0.01, value=0.01, step=0.01)
    step = st.number_input("Lot step", min_value=0.01, value=0.01, step=0.01)
    spread = st.number_input("Spread ($/oz)", min_value=0.0, value=0.30)
    slip = st.number_input("Slippage per side ($/oz)", min_value=0.0, value=0.10)
    commission = st.number_input("Round-trip commission ($/lot)", min_value=0.0, value=7.0)

@st.cache_data
def demo_data(seed, days):
    rng = np.random.default_rng(seed)
    n = days * 1440
    times = pd.date_range("2026-09-01", periods=n, freq="min")
    increments = rng.normal(0, 0.38, n) + np.sin(np.arange(n)/500)*0.035
    close = 3400 + np.cumsum(increments)
    op = np.r_[3400, close[:-1]]
    high = np.maximum(op, close) + rng.uniform(0.02, 0.22, n)
    low = np.minimum(op, close) - rng.uniform(0.02, 0.22, n)
    return pd.DataFrame({"time":times,"open":op,"high":high,"low":low,"close":close})

def load_upload(upload):
    raw = upload.getvalue()
    if upload.name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = [n for n in z.namelist() if n.lower().endswith((".csv", ".txt"))]
            if not names:
                raise ValueError("No CSV/TXT inside ZIP")
            raw = z.read(names[0])
    text = raw.decode("utf-8-sig", errors="replace")
    first = text.splitlines()[0]
    sep = ";" if first.count(";") > first.count(",") else ","
    header = any(x in first.lower() for x in ("open", "high", "close", "date", "time"))
    df = pd.read_csv(io.StringIO(text), sep=sep, header=0 if header else None)
    if not header:
        if df.shape[1] < 5:
            raise ValueError("Expected time, open, high, low, close")
        df = df.iloc[:, :5]
        df.columns = ["time", "open", "high", "low", "close"]
    else:
        df.columns = [str(x).lower().strip().replace("<", "").replace(">", "") for x in df.columns]
        df = df.rename(columns={"datetime":"time", "timestamp":"time"})
        if "date" in df and "time" in df:
            df["time"] = df["date"].astype(str) + " " + df["time"].astype(str)
        elif "date" in df and "time" not in df:
            df = df.rename(columns={"date":"time"})
    required = ["time", "open", "high", "low", "close"]
    if any(c not in df.columns for c in required):
        raise ValueError("Missing time/open/high/low/close columns")
    t = df.time.astype(str).str.strip()
    dt = pd.to_datetime(t, format="%Y%m%d %H%M%S", errors="coerce")
    df["time"] = dt.fillna(pd.to_datetime(t, errors="coerce"))
    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).sort_values("time").drop_duplicates("time")
    df = df[(df.high >= df[["open","close"]].max(axis=1)) & (df.low <= df[["open","close"]].min(axis=1))]
    return df[required].reset_index(drop=True)

def indicators(data):
    d = data.copy().sort_values("time").reset_index(drop=True)
    m = d.set_index("time").resample("5min", closed="left", label="right").agg({"close":"last"}).dropna()
    m["e50"] = m.close.ewm(span=50, adjust=False).mean()
    m["e200"] = m.close.ewm(span=200, adjust=False).mean()
    m["trend"] = np.where((m.close > m.e50) & (m.e50 > m.e200), 1, np.where((m.close < m.e50) & (m.e50 < m.e200), -1, 0))
    # At the open of each minute, only completed M5 candles are available.
    d = pd.merge_asof(d, m[["trend"]].reset_index(), on="time", direction="backward")
    d["ema20"] = d.close.ewm(span=20, adjust=False).mean()
    d["high20"] = d.high.shift(1).rolling(20).max()
    d["low20"] = d.low.shift(1).rolling(20).min()
    tr = pd.concat([d.high-d.low, (d.high-d.close.shift()).abs(), (d.low-d.close.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    d["signal"] = "WAIT"
    buy = (d.trend == 1) & (d.close > d.high20) & (d.close > d.ema20)
    sell = (d.trend == -1) & (d.close < d.low20) & (d.close < d.ema20)
    d.loc[buy, "signal"] = "BUY"
    d.loc[sell, "signal"] = "SELL"
    d.loc[:999, "signal"] = "WAIT"
    return d

def backtest(d, start, cutoff=None):
    equity = float(start)
    trades = []
    day_start = {}
    i = 1000
    while i < len(d)-1:
        row = d.iloc[i]
        nxt = d.iloc[i+1]
        if cutoff is not None and nxt.time < cutoff:
            i += 1
            continue
        if row.signal not in ("BUY", "SELL") or not np.isfinite(row.atr) or row.atr <= 0:
            i += 1
            continue
        day = nxt.time.date()
        if day not in day_start:
            day_start[day] = equity
        remaining = daily_limit - (day_start[day] - equity)
        if remaining <= 0:
            i += 1
            continue
        direction = 1 if row.signal == "BUY" else -1
        dist = max(float(row.atr)*1.5, 0.10)
        per_lot = (dist + spread + 2*slip)*contract + commission
        lots = round(float(np.floor(min(risk, remaining, equity*0.01)/per_lot/step)*step), 8)
        if lots < min_lot:
            i += 1
            continue
        entry = float(nxt.open) + direction*(spread/2 + slip)
        sl = entry - direction*dist
        tp = entry + direction*2*dist
        exit_i = len(d)-1
        exit_fill = float(d.iloc[-1].close) - direction*(spread/2+slip)
        reason = "END"
        for j in range(i+1, len(d)):
            bar = d.iloc[j]
            # Quote-mid candle converted to executable exit-side quote.
            adjustment = direction*(spread/2+slip)
            stop_hit = (bar.low-adjustment <= sl) if direction == 1 else (bar.high-adjustment >= sl)
            target_hit = (bar.high-adjustment >= tp) if direction == 1 else (bar.low-adjustment <= tp)
            if stop_hit:
                # Conservative stop-first assumption for ambiguous candles.
                exit_fill = sl
                exit_i = j
                reason = "STOP"
                break
            if target_hit:
                exit_fill = tp
                exit_i = j
                reason = "TARGET"
                break
            if j+1 < len(d) and d.iloc[j+1].time.date() != day:
                exit_fill = float(bar.close)-adjustment
                exit_i = j
                reason = "DAY_END"
                break
        pnl = direction*(exit_fill-entry)*contract*lots - commission*lots
        equity += pnl
        trades.append({"entry_time":nxt.time, "exit_time":d.iloc[exit_i].time, "side":row.signal, "lot":lots, "entry":round(entry,3), "sl":round(sl,3), "tp":round(tp,3), "exit":round(exit_fill,3), "reason":reason, "pnl":round(pnl,2), "equity":round(equity,2)})
        i = exit_i + 1
    return pd.DataFrame(trades)

def report(name, trades, initial):
    st.subheader(name)
    if trades.empty:
        st.info("No trades. Minimum lot or risk settings may prevent entries.")
        return
    a,b,c = st.columns(3)
    a.metric("Trades", len(trades))
    b.metric("Win rate", f"{(trades.pnl>0).mean()*100:.1f}%")
    c.metric("Net P/L", f"${trades.pnl.sum():.2f}")
    st.line_chart(pd.concat([pd.Series([initial]), trades.equity.reset_index(drop=True)], ignore_index=True))
    st.dataframe(trades, use_container_width=True)

mode = st.radio("Data source", ["Generate demo data", "Upload historical ZIP/CSV"])
if mode == "Generate demo data":
    seed = st.number_input("Random seed", min_value=1, value=41)
    days = st.slider("Demo days", min_value=8, max_value=35, value=16)
    if st.button("Generate Demo Data & Run Backtest", type="primary"):
        data = demo_data(int(seed), days)
        d = indicators(data)
        split = int(len(d)*0.7)
        # Indicators use past data only. Trades do not cross the split.
        train = backtest(d.iloc[:split].reset_index(drop=True), balance)
        holdout = backtest(d, balance, cutoff=d.time.iloc[split])
        st.success(f"Generated {len(data):,} synthetic candles")
        report("Training (70%)", train, balance)
        report("Holdout (30%)", holdout, balance)
        st.write("Latest historical synthetic signal:", d.signal.iloc[-1])
else:
    files = st.file_uploader("Upload M1 CSV or ZIP", type=["csv", "txt", "zip"], accept_multiple_files=True)
    if files and st.button("Run Historical Backtest", type="primary"):
        try:
            data = pd.concat([load_upload(f) for f in files], ignore_index=True)
            data = data.sort_values("time").drop_duplicates("time").reset_index(drop=True)
            if len(data) < 1200:
                st.error("At least 1,200 M1 candles required")
            else:
                d = indicators(data)
                split = int(len(d)*0.7)
                report("Development 70%", backtest(d.iloc[:split].reset_index(drop=True), balance), balance)
                report("Holdout 30%", backtest(d, balance, cutoff=d.time.iloc[split]), balance)
                st.caption(f"Coverage: {data.time.min()} to {data.time.max()}")
        except Exception as e:
            st.error(f"Data error: {e}")

st.caption("Breakout demo strategy, not the frozen Fibonacci Golden Zone strategy. No broker orders. Daily limit controls planned risk; losses may exceed it due to gaps.")
