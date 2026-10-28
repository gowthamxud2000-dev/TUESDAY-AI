
import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tuesday AI V40", layout="wide")
st.title("Tuesday AI V40 | XAUUSD M1 + M5")
st.caption("Demo research only. No broker connection or live orders. Times use CSV timestamps.")

with st.sidebar:
    st.header("Risk & execution")
    balance = st.number_input("Starting balance ($)", min_value=1.0, value=2500.0)
    daily_limit = st.number_input("Daily loss limit ($)", min_value=1.0, value=40.0)
    risk_pct = st.number_input("Risk per trade (%)", min_value=0.05, max_value=5.0, value=0.25, step=0.05)
    lot_min = st.number_input("Minimum lot", min_value=0.01, value=0.01, step=0.01)
    lot_step = st.number_input("Lot step", min_value=0.01, value=0.01, step=0.01)
    max_lot = st.number_input("Maximum lot", min_value=0.01, value=1.0, step=0.01)
    contract = st.number_input("Gold ounces per 1.0 lot (verify broker)", min_value=1.0, value=100.0)
    spread = st.number_input("Spread in XAUUSD price units", min_value=0.0, value=0.30, step=0.05)
    slip = st.number_input("Slippage per side in price units", min_value=0.0, value=0.10, step=0.05)
    commission = st.number_input("Round-trip commission ($ per 1 lot)", min_value=0.0, value=7.0)
    rr = st.number_input("Reward/risk target", min_value=1.0, max_value=5.0, value=2.0, step=0.25)

st.info("Upload raw HISTDATA M1 CSV/TXT or monthly ZIP files. Data must contain Date, Time, Open, High, Low, Close. Use one consistent broker time zone.")
files = st.file_uploader("Upload XAUUSD M1 files", type=["csv", "txt", "zip"], accept_multiple_files=True)

def read_one(data):
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(io.BytesIO(data), sep=sep, header=None, engine="python", on_bad_lines="skip")
            if df.shape[1] >= 5:
                return df
        except Exception:
            pass
    raise ValueError("Unsupported file layout")

def parse_data(uploaded):
    parts = []
    for f in uploaded:
        raw = f.getvalue()
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                members = [n for n in z.namelist() if n.lower().endswith((".csv", ".txt")) and not n.startswith("__MACOSX")]
                for n in members:
                    parts.append(read_one(z.read(n)))
        else:
            parts.append(read_one(raw))
    if not parts:
        raise ValueError("No CSV/TXT data found")
    frames = []
    for d in parts:
        d = d.dropna(how="all").copy()
        if d.shape[1] >= 7:
            dates = d.iloc[:, 0].astype(str).str.strip()
            times = d.iloc[:, 1].astype(str).str.strip().str.zfill(6)
            dt = pd.to_datetime(dates + " " + times, format="%Y%m%d %H%M%S", errors="coerce")
            if dt.notna().mean() < 0.5:
                dt = pd.to_datetime(dates + " " + d.iloc[:, 1].astype(str), errors="coerce")
            ohlc = d.iloc[:, 2:6]
        else:
            dt = pd.to_datetime(d.iloc[:, 0].astype(str).str.strip(), errors="coerce")
            if dt.notna().mean() < 0.5:
                dt = pd.to_datetime(d.iloc[:, 0].astype(str).str.strip(), format="%Y%m%d %H%M%S", errors="coerce")
            ohlc = d.iloc[:, 1:5]
        temp = pd.DataFrame(ohlc.to_numpy(), columns=["open", "high", "low", "close"])
        temp["time"] = dt.to_numpy()
        for c in ["open", "high", "low", "close"]:
            temp[c] = pd.to_numeric(temp[c], errors="coerce")
        frames.append(temp)
    out = pd.concat(frames, ignore_index=True).dropna().drop_duplicates("time").sort_values("time").set_index("time")
    out = out[(out.high >= out[["open", "close", "low"]].max(axis=1)) & (out.low <= out[["open", "close", "high"]].min(axis=1))]
    return out

def ema(x, n):
    return x.ewm(span=n, adjust=False, min_periods=n).mean()

def signals(m1):
    x = m1.copy()
    x["e20"] = ema(x.close, 20)
    x["e50"] = ema(x.close, 50)
    x["atr"] = pd.concat([
        (x.high - x.low),
        (x.high - x.close.shift()).abs(),
        (x.low - x.close.shift()).abs()
    ], axis=1).max(axis=1).rolling(14, min_periods=14).mean()

    # Completed M5 bars only, aligned with M1 candle timestamps.
    m5 = x[["open", "high", "low", "close"]].resample(
        "5min", label="right", closed="left"
    ).agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last"
    }).dropna()
    m5["fast"] = ema(m5.close, 20)
    m5["slow"] = ema(m5.close, 50)
    aligned = m5[["close", "fast", "slow"]].reindex(x.index, method="ffill")
    x[["m5close", "m5fast", "m5slow"]] = aligned

    prevhigh = x.high.shift(1).rolling(5, min_periods=5).max()
    prevlow = x.low.shift(1).rolling(5, min_periods=5).min()

    x["buy"] = (
        (x.m5fast > x.m5slow) &
        (x.m5close > x.m5fast) &
        (x.e20 > x.e50) &
        (x.close > prevhigh) &
        (x.close > x.e20)
    )
    x["sell"] = (
        (x.m5fast < x.m5slow) &
        (x.m5close < x.m5fast) &
        (x.e20 < x.e50) &
        (x.close < prevlow) &
        (x.close < x.e20)
    )
    return x

def backtest(x):
    trades = []
    equity = balance
    pos = None
    locked = set()

    for i in range(1, len(x)):
        bar = x.iloc[i]
        prev = x.iloc[i - 1]
        t = x.index[i]
        day = t.date()

        if pos is not None:
            side = pos["signal"]
            stop = pos["sl"]
            target = pos["tp"]

            if side == "BUY":
                exit_price = (
                    min(bar.open, stop) if bar.open <= stop
                    else stop if bar.low <= stop
                    else target if bar.high >= target
                    else None
                )
            else:
                exit_price = (
                    max(bar.open, stop) if bar.open >= stop
                    else stop if bar.high >= stop
                    else target if bar.low <= target
                    else None
                )

            if exit_price is not None:
                direction = 1 if side == "BUY" else -1
                pnl = (
                    direction * (exit_price - pos["entry_raw"]) * pos["lot"] * contract
                    - (spread + 2 * slip) * pos["lot"] * contract
                    - commission * pos["lot"]
                )
                equity += pnl
                trades.append({
                    **pos,
                    "exit_time": t,
                    "exit": exit_price,
                    "pnl": round(pnl, 2),
                    "equity": round(equity, 2),
                    "result": "WIN" if pnl > 0 else "LOSS"
                })
                pos = None

        todays = sum(
            tr["pnl"] for tr in trades
            if pd.Timestamp(tr["exit_time"]).date() == day
        )
        if todays <= -daily_limit:
            locked.add(day)

        if pos is None and day not in locked and pd.notna(prev.atr) and pd.notna(prev.m5slow):
            side = "BUY" if prev.buy else ("SELL" if prev.sell else None)
            if side:
                entry = bar.open
                distance = max(float(prev.atr) * 1.5, spread + 2 * slip + 0.01)
                stop = entry - distance if side == "BUY" else entry + distance
                target = entry + rr * distance if side == "BUY" else entry - rr * distance

                available = daily_limit + min(todays, 0)
                budget = min(equity * risk_pct / 100, available)
                cost_per_lot = (
                    distance * contract
                    + (spread + 2 * slip) * contract
                    + commission
                )
                steps = np.floor((budget / cost_per_lot + 1e-9) / lot_step)
                lot = round(steps * lot_step, 8)
                lot = min(lot, max_lot)

                if lot >= lot_min and equity > 0 and budget > 0:
                    pos = {
                        "entry_time": t,
                        "signal": side,
                        "entry_raw": float(entry),
                        "entry": round(float(entry), 3),
                        "sl": round(float(stop), 3),
                        "tp": round(float(target), 3),
                        "lot": lot,
                        "risk_budget": round(budget, 2)
                    }

    return pd.DataFrame(trades), equity, pos, locked

if files:
    try:
        m1 = parse_data(files)
        if len(m1) < 500:
            st.error("Need at least 500 M1 candles for M5 EMA warm-up.")
            st.stop()

        x = signals(m1)
        st.success(f"Loaded {len(m1):,} candles: {m1.index.min()} to {m1.index.max()}")
        st.warning("M1 data is candle-level, not ticks. Intrabar fill order and actual broker spread cannot be verified. Stops may exceed $40 on gaps; this is not a guaranteed loss cap.")

        latest = x.iloc[-1]
        state = "BUY" if latest.buy else ("SELL" if latest.sell else "WAIT")
        st.metric("Latest CLOSED M1 candle signal (not a live signal)", state)
        st.caption("Signal at latest candle close; hypothetical entry at the next candle open.")

        trades, final_equity, open_pos, locked = backtest(x)

        if trades.empty:
            st.warning("No trades passed the filters and risk sizing. Check data, contract size and lot settings.")
        else:
            wins = (trades.pnl > 0).sum()
            net = trades.pnl.sum()
            a, b, c, d = st.columns(4)
            a.metric("Closed trades", len(trades))
            b.metric("Win rate", f"{100 * wins / len(trades):.1f}%")
            c.metric("Net P&L", f"${net:,.2f}")
            d.metric("Realized ending balance", f"${final_equity:,.2f}")

            curve = pd.concat([
                pd.Series([balance], index=[m1.index[0]]),
                pd.Series(trades.equity.to_numpy(), index=pd.to_datetime(trades.exit_time))
            ]).sort_index()
            dd = (curve.cummax() - curve).max()
            st.metric("Closed-equity max drawdown", f"${dd:,.2f}")
            st.line_chart(curve)

            st.subheader("Independent month-by-month results")
            trades["month"] = pd.to_datetime(trades.entry_time).dt.to_period("M").astype(str)
            monthly = trades.groupby("month").agg(
                trades=("pnl", "size"),
                net_pnl=("pnl", "sum"),
                win_rate=("pnl", lambda s: round(100 * (s > 0).mean(), 1))
            )
            st.dataframe(monthly)
            st.caption("Parameters are frozen for every month. Do not tune using October then call October independent. Test on new unseen data afterward.")

            st.subheader("Trades")
            st.dataframe(trades, hide_index=True)
            st.download_button(
                "Download V40 trades CSV",
                trades.to_csv(index=False).encode(),
                "Tuesday_AI_V40_trades.csv",
                "text/csv"
            )
            st.caption(f"Days with realized-loss lock: {len(locked)}. The lock resets at CSV midnight, not necessarily broker session midnight.")

        if open_pos:
            st.warning(f"One position remains open at end of data: {open_pos['signal']}. Unrealized P&L not included in final equity.")

    except Exception as e:
        st.error(f"Input/data error: {e}")
else:
    st.write("Upload September and October 2026 HISTDATA XAUUSD M1 ZIPs to start.")
