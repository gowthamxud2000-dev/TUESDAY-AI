
import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Tuesday AI V33",
    page_icon="🥇",
    layout="wide"
)

st.title("🥇 Tuesday AI V33 — Personal Account")
st.caption("XAUUSD M15 | Fibonacci | EMA | Manual trading")
st.warning("Experimental strategy. No guaranteed profits or live orders.")

with st.sidebar:
    st.header("Account Settings")
    initial = st.number_input(
        "Starting balance ($)", min_value=10.0, value=100.0
    )
    risk_pct = st.slider("Risk per trade (%)", 0.1, 2.0, 0.5, 0.1)
    daily_stop = st.slider("Daily loss stop (%)", 0.5, 10.0, 2.0, 0.5)
    max_trades = st.slider("Maximum trades per day", 1, 10, 3)
    max_losses = st.slider("Consecutive loss stop", 1, 5, 2)

    st.header("Trading Costs")
    spread = st.number_input("Spread ($/oz)", 0.0, 10.0, 0.30)
    slip = st.number_input("Slippage per side ($/oz)", 0.0, 5.0, 0.10)
    commission = st.number_input(
        "Round-trip commission ($/lot)", 0.0, 50.0, 7.0
    )
    contract = st.number_input("Ounces per lot", 1.0, 1000.0, 100.0)
    min_lot = st.number_input(
        "Minimum lot", 0.001, 1.0, 0.01, format="%.3f"
    )
    lot_step = st.number_input(
        "Lot step", 0.001, 1.0, 0.01, format="%.3f"
    )

def read_data(file):
    raw = file.getvalue()

    if file.name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = [
                n for n in z.namelist()
                if n.lower().endswith((".csv", ".txt"))
            ]
            if not names:
                raise ValueError("ZIP has no CSV/TXT file")
            raw = z.read(names[0])

    text = raw.decode("utf-8-sig", errors="replace")
    first = text.splitlines()[0].strip()

    if first[:8].isdigit() and (";" in first or "," in first):
        sep = ";" if ";" in first else ","
        d = pd.read_csv(io.StringIO(text), sep=sep, header=None)
        d = d.iloc[:, :5]
        d.columns = ["time", "open", "high", "low", "close"]
        d["time"] = pd.to_datetime(
            d.time.astype(str),
            format="%Y%m%d %H%M%S",
            errors="coerce"
        )
    else:
        d = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        d.columns = [
            str(c).lower().strip().strip("<>")
            for c in d.columns
        ]
        d = d.rename(columns={
            "datetime": "time",
            "timestamp": "time",
            "date_time": "time"
        })

        if "date" in d and "time" in d:
            d["time"] = pd.to_datetime(
                d.date.astype(str) + " " + d.time.astype(str),
                errors="coerce"
            )
        elif "time" in d:
            d["time"] = pd.to_datetime(d.time, errors="coerce")
        else:
            raise ValueError("Time column missing")

    for c in ["open", "high", "low", "close"]:
        if c not in d:
            raise ValueError("Missing column: " + c)
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=["time", "open", "high", "low", "close"])
    d = d.sort_values("time").drop_duplicates("time")
    d = d.set_index("time")

    return d[["open", "high", "low", "close"]]

def make_m15(d, timeframe):
    if timeframe == "M1":
        counts = d.open.resample(
            "15min", label="right", closed="left"
        ).count()
        x = d.resample(
            "15min", label="right", closed="left"
        ).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        })
        return x[counts == 15].dropna()

    x = d.copy()
    x.index = x.index + pd.Timedelta(minutes=15)
    return x

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
    x["hi32"] = x.high.shift(1).rolling(32).max()
    x["lo32"] = x.low.shift(1).rolling(32).min()

    width = x.hi32 - x.lo32

    x["buy_low"] = x.hi32 - 0.618 * width
    x["buy_high"] = x.hi32 - 0.5 * width
    x["sell_low"] = x.lo32 + 0.5 * width
    x["sell_high"] = x.lo32 + 0.618 * width

    x["buy"] = (
        (x.ema50 > x.ema200) &
        (x.low <= x.buy_high) &
        (x.high >= x.buy_low) &
        (x.close > x.open) &
        (x.close > x.ema50) &
        (x.atr > 0)
    )

    x["sell"] = (
        (x.ema50 < x.ema200) &
        (x.high >= x.sell_low) &
        (x.low <= x.sell_high) &
        (x.close < x.open) &
        (x.close < x.ema50) &
        (x.atr > 0)
    )

    x.loc[x.index[:200], ["buy", "sell"]] = False
    return x

def lot_for_risk(balance, entry, stop):
    per_lot = (
        (abs(entry - stop) + spread + 2 * slip) * contract
        + commission
    )
    budget = balance * risk_pct / 100

    if per_lot <= 0:
        return 0.0

    steps = np.floor((budget / per_lot) / lot_step)
    lot = round(float(steps * lot_step), 6)

    return lot if lot >= min_lot else 0.0

def backtest(x):
    balance = float(initial)
    records = []
    equity = [{"time": x.index[0], "equity": balance}]

    current_day = None
    day_start = balance
    day_entries = 0
    streak = 0
    i = 200

    while i < len(x) - 1:
        bar = x.iloc[i]
        now = x.index[i]

        if current_day != now.date():
            current_day = now.date()
            day_start = balance
            day_entries = 0
            streak = 0

        if (
            not (bool(bar.buy) or bool(bar.sell))
            or day_entries >= max_trades
            or streak >= max_losses
            or balance <= day_start * (1 - daily_stop / 100)
        ):
            i += 1
            continue

        side = "BUY" if bool(bar.buy) else "SELL"
        raw_entry = float(x.iloc[i + 1].open)
        atr = float(bar.atr)

        if not np.isfinite(atr) or atr <= 0:
            i += 1
            continue

        if side == "BUY":
            stop = min(float(bar.lo32), float(bar.low)) - 0.2 * atr
            entry = raw_entry + spread / 2 + slip
            distance = entry - stop
        else:
            stop = max(float(bar.hi32), float(bar.high)) + 0.2 * atr
            entry = raw_entry - spread / 2 - slip
            distance = stop - entry

        if distance <= 0 or distance > 4 * atr:
            i += 1
            continue

        target = (
            entry + 2 * distance
            if side == "BUY"
            else entry - 2 * distance
        )

        lot = lot_for_risk(balance, entry, stop)

        if lot == 0:
            i += 1
            continue

        estimated_loss = (
            (distance + spread / 2 + slip) * contract * lot
            + commission * lot
        )

        remaining_daily = max(
            0, balance - day_start * (1 - daily_stop / 100)
        )

        if estimated_loss > remaining_daily:
            i += 1
            continue

        day_entries += 1
        j = i + 1
        reason = "END"
        raw_exit = float(x.iloc[j].close)

        while j < len(x):
            b = x.iloc[j]

            if side == "BUY":
                hit_sl = b.low <= stop
                hit_tp = b.high >= target
            else:
                hit_sl = b.high >= stop
                hit_tp = b.low <= target

            if hit_sl:
                raw_exit = stop
                reason = "SL"
                break

            if hit_tp:
                raw_exit = target
                reason = "TP"
                break

            if j - (i + 1) >= 31:
                raw_exit = float(b.close)
                reason = "TIME"
                break

            j += 1

        if j >= len(x):
            j = len(x) - 1
            raw_exit = float(x.iloc[j].close)

        exit_price = (
            raw_exit - spread / 2 - slip
            if side == "BUY"
            else raw_exit + spread / 2 + slip
        )

        pnl = (
            (exit_price - entry)
            if side == "BUY"
            else (entry - exit_price)
        ) * contract * lot - commission * lot

        balance += pnl
        streak = streak + 1 if pnl < 0 else 0

        records.append({
            "entry_time": x.index[i + 1],
            "exit_time": x.index[j],
            "side": side,
            "entry": round(entry, 3),
            "sl": round(stop, 3),
            "tp": round(target, 3),
            "lots": lot,
            "exit": round(exit_price, 3),
            "reason": reason,
            "pnl": round(pnl, 2),
            "balance": round(balance, 2)
        })

        equity.append({
            "time": x.index[j],
            "equity": balance
        })

        i = j + 1

    return pd.DataFrame(records), pd.DataFrame(equity)

def show_report(label, data):
    st.subheader(label)

    if len(data) < 250:
        st.info("Need at least 250 M15 candles.")
        return

    trades, curve = backtest(data)

    if trades.empty:
        st.warning(
            "No executable trades. Minimum lot may exceed "
            "the $100 account risk budget. Signal: WAIT."
        )
        return

    wins = trades[trades.pnl > 0]
    losses = trades[trades.pnl < 0]
    pnl = trades.pnl.sum()

    dd = (
        curve.equity / curve.equity.cummax() - 1
    ).min() * 100

    gross_loss = abs(losses.pnl.sum())
    pf = (
        wins.pnl.sum() / gross_loss
        if gross_loss > 0
        else float("inf")
    )

    a, b, c, d = st.columns(4)
    a.metric("Trades", len(trades))
    b.metric("Win rate", f"{len(wins)/len(trades)*100:.1f}%")
    c.metric("Net P&L", f"${pnl:.2f}")
    d.metric("Closed-equity DD", f"{dd:.2f}%")

    st.write(f"Profit factor: {pf:.2f}")
    st.write(f"Final balance: ${initial + pnl:.2f}")

    st.line_chart(curve.set_index("time").equity)
    st.dataframe(trades, use_container_width=True)

    st.download_button(
        "Download " + label + " CSV",
        trades.to_csv(index=False).encode(),
        file_name=label.lower().replace(" ", "_") + "_v33.csv",
        mime="text/csv"
    )

st.header("📂 Upload XAUUSD Historical Data")

files = st.file_uploader(
    "Upload M1 ZIP/CSV or M15 CSV",
    type=["zip", "csv", "txt"],
    accept_multiple_files=True
)

timeframe = st.radio(
    "Uploaded timeframe",
    ["M1", "M15"],
    horizontal=True
)

if files:
    try:
        frames = [read_data(f) for f in files]
        combined = pd.concat(frames).sort_index()
        combined = combined[
            ~combined.index.duplicated(keep="first")
        ]

        m15 = make_m15(combined, timeframe)
        x = features(m15)

        st.success(f"Loaded {len(x):,} M15 candles")

        last = x.iloc[-1]
        signal = (
            "BUY" if last.buy
            else "SELL" if last.sell
            else "WAIT"
        )

        st.subheader("🚦 Historical Signal")
        st.metric("BUY / SELL / WAIT", signal)
        st.caption(
            "Based on uploaded historical candles, "
            "not live market prices."
        )

        st.line_chart(
            x[["close", "ema50", "ema200"]].tail(200)
        )

        sep = x[x.index.month == 9]
        octo = x[x.index.month == 10]

        if not sep.empty and not octo.empty:
            st.info(
                "September development and October holdout. "
                "October is independent only if rules were "
                "not tuned using October results."
            )
            show_report("September Development", sep)
            show_report("October Holdout", octo)
        else:
            show_report("Uploaded Data", x)

    except Exception as e:
        st.error(f"File processing error: {e}")
else:
    st.info("Upload historical XAUUSD files to start.")

st.divider()
st.caption(
    "Tuesday AI V33 | Personal account | "
    "Research and manual trading only"
)
