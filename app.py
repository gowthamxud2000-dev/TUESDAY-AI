
import io
import zipfile
import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tuesday AI V34", page_icon="🥇", layout="wide")
st.title("🥇 Tuesday AI V34 — Risk Diagnostic")
st.caption("XAUUSD M15 | $100 personal account | Manual trading only")
st.warning("Research backtest only. No live prices or automatic orders.")

with st.sidebar:
    st.header("Account")
    balance0 = st.number_input("Starting balance ($)", 10.0, 100000.0, 100.0)
    risk = st.slider("Risk per trade (%)", 0.1, 2.0, 0.5, 0.1)
    daily = st.slider("Daily loss limit (%)", 0.5, 10.0, 2.0, 0.5)
    max_trades = st.slider("Max trades per day", 1, 10, 3)
    min_lot = st.number_input("Minimum lot", 0.001, 1.0, 0.01, format="%.3f")
    lot_step = st.number_input("Lot step", 0.001, 1.0, 0.01, format="%.3f")
    contract = st.number_input("Ounces per lot", 1.0, 1000.0, 100.0)
    spread = st.number_input("Spread ($/oz)", 0.0, 10.0, 0.30)
    slip = st.number_input("Slippage per side ($/oz)", 0.0, 5.0, 0.10)
    commission = st.number_input("Round-trip commission ($/lot)", 0.0, 50.0, 7.0)
    st.caption("Check broker contract size, spread and commission.")

def load_data(f):
    raw = f.getvalue()
    if f.name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = [n for n in z.namelist()
                     if n.lower().endswith((".csv", ".txt"))]
            if not names:
                raise ValueError("ZIP has no CSV/TXT")
            raw = z.read(names[0])

    s = raw.decode("utf-8-sig", errors="replace")
    first = s.splitlines()[0].strip()
    if first[:8].isdigit():
        sep = ";" if ";" in first else ","
        d = pd.read_csv(io.StringIO(s), sep=sep, header=None)
        d = d.iloc[:, :5]
        d.columns = ["time", "open", "high", "low", "close"]
        d["time"] = pd.to_datetime(
            d.time.astype(str), format="%Y%m%d %H%M%S",
            errors="coerce"
        )
    else:
        d = pd.read_csv(io.StringIO(s), sep=None, engine="python")
        d.columns = [str(c).lower().strip().strip("<>")
                     for c in d.columns]
        d = d.rename(columns={
            "datetime": "time", "timestamp": "time"
        })
        if "date" in d and "time" in d:
            d["time"] = pd.to_datetime(
                d.date.astype(str) + " " + d.time.astype(str),
                errors="coerce"
            )
        else:
            d["time"] = pd.to_datetime(d.time, errors="coerce")

    for c in ["open", "high", "low", "close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["time", "open", "high", "low", "close"])
    return d.sort_values("time").drop_duplicates("time").set_index("time")[
        ["open", "high", "low", "close"]
    ]

def prepare(d, tf):
    if tf == "M1":
        counts = d.open.resample("15min", closed="left", label="right").count()
        d = d.resample("15min", closed="left", label="right").agg({
            "open": "first", "high": "max",
            "low": "min", "close": "last"
        })
        d = d[counts == 15].dropna()
    else:
        d = d.copy()
        d.index = d.index + pd.Timedelta(minutes=15)
    return d

def features(d):
    x = d.copy()
    x["ema50"] = x.close.ewm(span=50, adjust=False).mean()
    x["ema200"] = x.close.ewm(span=200, adjust=False).mean()
    prev = x.close.shift(1)
    tr = pd.concat([
        x.high - x.low,
        (x.high - prev).abs(),
        (x.low - prev).abs()
    ], axis=1).max(axis=1)
    x["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    x["hi"] = x.high.shift(1).rolling(32).max()
    x["lo"] = x.low.shift(1).rolling(32).min()
    w = x.hi - x.lo
    x["buy"] = (
        (x.ema50 > x.ema200) &
        (x.low <= x.hi - 0.5*w) &
        (x.high >= x.hi - 0.618*w) &
        (x.close > x.open) &
        (x.close > x.ema50)
    )
    x["sell"] = (
        (x.ema50 < x.ema200) &
        (x.high >= x.lo + 0.5*w) &
        (x.low <= x.lo + 0.618*w) &
        (x.close < x.open) &
        (x.close < x.ema50)
    )
    x.iloc[:200, x.columns.get_indexer(["buy", "sell"])] = False
    return x

def test(x, start, end, label):
    st.subheader(label)
    balance = float(balance0)
    rows, rejects, equity = [], [], []
    day = None
    day_start = balance
    trades_today = 0
    loss_streak = 0
    i = max(200, int(x.index.searchsorted(start)))
    last = int(x.index.searchsorted(end, side="right")) - 1
    last = min(last, len(x)-1)

    while i < last:
        b = x.iloc[i]
        t = x.index[i]
        if day != t.date():
            day = t.date()
            day_start = balance
            trades_today = 0
            loss_streak = 0

        if not (bool(b.buy) or bool(b.sell)):
            i += 1
            continue

        side = "BUY" if b.buy else "SELL"
        reason = ""
        if trades_today >= max_trades:
            reason = "MAX_DAILY_TRADES"
        elif loss_streak >= 2:
            reason = "CONSECUTIVE_LOSSES"
        elif balance <= day_start * (1-daily/100):
            reason = "DAILY_LOSS_STOP"

        raw = float(x.iloc[i+1].open)
        atr = float(b.atr)
        if not np.isfinite(atr) or atr <= 0:
            i += 1
            continue

        if side == "BUY":
            entry = raw + spread/2 + slip
            sl = min(float(b.lo), float(b.low)) - 0.2*atr
            dist = entry-sl
        else:
            entry = raw - spread/2 - slip
            sl = max(float(b.hi), float(b.high)) + 0.2*atr
            dist = sl-entry

        if dist <= 0 or dist > 4*atr:
            reason = reason or "STOP_DISTANCE_FILTER"

        # Conservative estimated loss includes adverse exit costs.
        per_lot = (dist + spread/2 + slip)*contract + commission
        min_risk = max(0, per_lot*min_lot)
        budget = balance*risk/100
        remaining = max(0, balance-day_start*(1-daily/100))

        if min_risk > budget and not reason:
            reason = "MIN_LOT_RISK_TOO_HIGH"

        if min_risk > remaining and not reason:
            reason = "DAILY_BUDGET_TOO_SMALL"

        lot = 0.0
        if not reason and per_lot > 0:
            steps = math.floor((budget/per_lot)/lot_step + 1e-9)
            lot = round(steps*lot_step, 6)
            if lot < min_lot:
                reason = "LOT_STEP_REJECTED"
            elif lot*per_lot > remaining:
                reason = "DAILY_BUDGET_TOO_SMALL"

        if reason:
            rejects.append({
                "time": t, "side": side, "reason": reason,
                "sl_distance": round(dist, 3),
                "min_lot_risk": round(min_risk, 2),
                "risk_budget": round(budget, 2)
            })
            i += 1
            continue

        tp = entry+2*dist if side == "BUY" else entry-2*dist
        j = i+1
        raw_exit = float(x.iloc[j].close)
        outcome = "END"

        while j <= last:
            q = x.iloc[j]
            hit_sl = q.low <= sl if side == "BUY" else q.high >= sl
            hit_tp = q.high >= tp if side == "BUY" else q.low <= tp

            # Stop first when both hit in same candle.
            if hit_sl:
                raw_exit, outcome = sl, "SL"
                break
            if hit_tp:
                raw_exit, outcome = tp, "TP"
                break
            if j-i >= 32 or j == last:
                raw_exit, outcome = float(q.close), "TIME"
                break
            j += 1

        exit_price = (
            raw_exit-spread/2-slip if side == "BUY"
            else raw_exit+spread/2+slip
        )
        pnl = (
            (exit_price-entry) if side == "BUY"
            else (entry-exit_price)
        )*contract*lot - commission*lot

        balance += pnl
        trades_today += 1
        loss_streak = loss_streak+1 if pnl < 0 else 0

        rows.append({
            "entry_time": x.index[i+1],
            "exit_time": x.index[j],
            "side": side, "lot": lot,
            "entry": round(entry, 3),
            "sl": round(sl, 3), "tp": round(tp, 3),
            "exit": round(exit_price, 3),
            "outcome": outcome,
            "pnl": round(pnl, 2),
            "balance": round(balance, 2)
        })
        equity.append({"time": x.index[j], "balance": balance})
        i = j+1

    r = pd.DataFrame(rows)
    rejected = pd.DataFrame(rejects)

    signals = len(r)+len(rejected)
    a, b, c = st.columns(3)
    a.metric("Signals evaluated", signals)
    b.metric("Executed trades", len(r))
    c.metric("Rejected signals", len(rejected))

    if not rejected.empty:
        st.write("### Why signals were rejected")
        st.bar_chart(rejected.reason.value_counts())
        st.dataframe(rejected, use_container_width=True)
        st.download_button(
            f"Download {label} diagnostic CSV",
            rejected.to_csv(index=False),
            file_name=f"V34_{label}_rejected.csv",
            mime="text/csv"
        )

    if r.empty:
        st.warning("No executable trades at the configured risk.")
    else:
        wins = r[r.pnl > 0]
        gross_loss = abs(r.loc[r.pnl < 0, "pnl"].sum())
        pf = wins.pnl.sum()/gross_loss if gross_loss else float("inf")
        eq = pd.DataFrame(equity).set_index("time")
        dd = (eq.balance/eq.balance.cummax()-1).min()*100

        a, b, c = st.columns(3)
        a.metric("Win rate", f"{len(wins)/len(r)*100:.1f}%")
        b.metric("Net P&L", f"${r.pnl.sum():.2f}")
        c.metric("Final balance", f"${balance:.2f}")
        st.write(f"Profit factor: {pf:.2f}")
        st.write(f"Closed-equity drawdown: {dd:.2f}%")
        st.line_chart(eq)
        st.dataframe(r, use_container_width=True)
        st.download_button(
            f"Download {label} trades CSV",
            r.to_csv(index=False),
            file_name=f"V34_{label}_trades.csv",
            mime="text/csv"
        )

st.header("📂 Upload historical XAUUSD data")
uploads = st.file_uploader(
    "Upload September and October M1 ZIP or M15 CSV",
    type=["zip", "csv", "txt"],
    accept_multiple_files=True
)
tf = st.radio("Uploaded timeframe", ["M1", "M15"], index=1, horizontal=True)

if uploads:
    try:
        data = pd.concat([load_data(f) for f in uploads])
        data = data.sort_index()
        data = data[~data.index.duplicated(keep="first")]
        x = features(prepare(data, tf))
        st.success(f"Loaded {len(x):,} M15 candles")

        if len(x) < 250:
            st.warning("Not enough M15 candles for EMA200.")
        else:
            latest = x.iloc[-1]
            signal = "BUY" if latest.buy else "SELL" if latest.sell else "WAIT"
            st.metric("Latest historical signal", signal)
            st.line_chart(x[["close", "ema50", "ema200"]].tail(200))

            # Compute indicators on combined data, then split by dates.
            years = sorted(set(x.index.year))
            year = 2026 if 2026 in years else years[-1]
            sept_start = pd.Timestamp(year=year, month=9, day=1)
            sept_end = pd.Timestamp(year=year, month=9, day=30, hour=23, minute=59)
            oct_start = pd.Timestamp(year=year, month=10, day=1)
            oct_end = pd.Timestamp(year=year, month=10, day=31, hour=23, minute=59)

            if (x.index.month == 9).any():
                test(x, sept_start, sept_end, "September")
            if (x.index.month == 10).any():
                test(x, oct_start, oct_end, "October")

            if not (x.index.month.isin([9, 10])).any():
                test(x, x.index[0], x.index[-1], "Uploaded_Data")

            st.info(
                "October is not a fresh independent test if "
                "strategy rules were adjusted using October results."
            )
    except Exception as e:
        st.error(f"Processing error: {e}")
else:
    st.info("Upload historical candles to run diagnostics.")

st.caption(
    "V34 | Personal account | Historical simulation only. "
    "Margin, floating drawdown and gaps are not fully modeled."
            )
            
