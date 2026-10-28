
import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile

st.set_page_config(page_title="Tuesday AI V36", layout="wide")
st.title("Tuesday AI V36")
st.caption("XAUUSD M15 | Risk-Safe Independent Backtest")

st.warning("Research only. No automatic trading or guaranteed profits.")

balance = st.number_input("Starting balance ($)", 1.0, value=100.0)
risk_pct = st.number_input("Risk per trade (%)", 0.1, 5.0, 1.0)
daily_limit = st.number_input("Daily loss limit ($)", 0.1, value=3.0)
min_lot = st.number_input("Minimum lot", 0.01, value=0.01)
contract = st.number_input("Ounces per 1 lot", 1.0, value=100.0)
spread = st.number_input("Spread in gold price units", 0.0, value=0.30)
slippage = st.number_input("Slippage per side", 0.0, value=0.10)
commission = st.number_input("Round-trip commission per lot ($)", 0.0, value=7.0)

st.caption("Check lot contract size, spread and commission with your broker.")

files = st.file_uploader(
    "Upload September and October M1 ZIP or M15 CSV",
    type=["csv", "txt", "zip"],
    accept_multiple_files=True
)

def load_one(name, data):
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            frames = []
            for item in z.namelist():
                if item.lower().endswith((".csv", ".txt")):
                    frames.append(load_one(item, z.read(item)))
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    raw = pd.read_csv(
        io.BytesIO(data), sep=None, engine="python",
        header=None, on_bad_lines="skip"
    )

    if raw.empty:
        return pd.DataFrame()

    first = str(raw.iloc[0, 0]).lower()
    if first in ["date", "datetime", "time", "timestamp"]:
        raw = raw.iloc[1:].reset_index(drop=True)

    if raw.shape[1] >= 7:
        raw = raw.iloc[:, :7]
        raw.columns = ["date", "clock", "open", "high", "low", "close", "volume"]
        raw["time"] = pd.to_datetime(
            raw["date"].astype(str) + " " + raw["clock"].astype(str),
            errors="coerce"
        )
    elif raw.shape[1] >= 5:
        raw = raw.iloc[:, :min(raw.shape[1], 6)]
        raw.columns = ["time", "open", "high", "low", "close"] + (
            ["volume"] if raw.shape[1] == 6 else []
        )
        raw["time"] = pd.to_datetime(raw["time"], errors="coerce")
    else:
        raise ValueError("CSV needs time, open, high, low, close columns.")

    for c in ["open", "high", "low", "close"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    return raw.dropna(subset=["time", "open", "high", "low", "close"])

def prepare(files):
    frames = [load_one(f.name, f.getvalue()) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("time").sort_values("time").set_index("time")

    df = df[["open", "high", "low", "close"]].resample("15min").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last"
    }).dropna()

    df["ema50"] = df.close.ewm(span=50, adjust=False).mean()
    df["ema200"] = df.close.ewm(span=200, adjust=False).mean()

    prev = df.close.shift(1)
    tr = pd.concat([
        df.high - df.low,
        (df.high - prev).abs(),
        (df.low - prev).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(14).mean()
    df["swing_low"] = df.low.shift(1).rolling(10).min()
    df["swing_high"] = df.high.shift(1).rolling(10).max()
    return df.dropna()

def backtest(df):
    equity = balance
    trades = []
    rejected = []
    daily_loss = {}
    i = 1

    while i < len(df) - 1:
        r = df.iloc[i]
        p = df.iloc[i - 1]
        t = df.index[i]

        side = None
        if r.ema50 > r.ema200 and p.close <= p.ema50 and r.close > r.ema50:
            side = "BUY"
        elif r.ema50 < r.ema200 and p.close >= p.ema50 and r.close < r.ema50:
            side = "SELL"

        if side is None:
            i += 1
            continue

        entry = float(df.iloc[i + 1].open)
        stop = float(r.swing_low if side == "BUY" else r.swing_high)

        if (side == "BUY" and stop >= entry) or (
            side == "SELL" and stop <= entry
        ):
            rejected.append([t, side, "Invalid stop", 0])
            i += 1
            continue

        distance = abs(entry - stop)
        target = entry + 2 * distance if side == "BUY" else entry - 2 * distance

        max_risk = min(equity * risk_pct / 100, balance * risk_pct / 100)
        estimated_risk = (
            (distance + spread + 2 * slippage) * contract * min_lot
            + commission * min_lot
        )

        if estimated_risk > max_risk:
            rejected.append([t, side, "Minimum lot risk too high", estimated_risk])
            i += 1
            continue

        day = str(t.date())
        if daily_loss.get(day, 0) >= daily_limit:
            rejected.append([t, side, "Daily loss limit", estimated_risk])
            i += 1
            continue

        exit_price = None
        exit_time = None
        outcome = None
        j = i + 1

        while j < len(df):
            candle = df.iloc[j]
            if side == "BUY":
                hit_sl = candle.low <= stop
                hit_tp = candle.high >= target
            else:
                hit_sl = candle.high >= stop
                hit_tp = candle.low <= target

            if hit_sl or hit_tp:
                # Conservative: stop wins if both touched.
                exit_price = stop if hit_sl else target
                exit_time = df.index[j]
                outcome = "SL" if hit_sl else "TP"
                break
            j += 1

        if exit_price is None:
            exit_price = float(df.iloc[-1].close)
            exit_time = df.index[-1]
            outcome = "END"
            j = len(df) - 1

        direction = 1 if side == "BUY" else -1
        gross = (exit_price - entry) * direction * contract * min_lot
        costs = (spread + 2 * slippage) * contract * min_lot + commission * min_lot
        pnl = gross - costs

        equity += pnl
        if pnl < 0:
            daily_loss[day] = daily_loss.get(day, 0) + abs(pnl)

        trades.append([
            t, exit_time, side, entry, stop, target,
            outcome, round(pnl, 2), round(equity, 2)
        ])

        if equity <= 0:
            break
        i = j + 1

    trade_cols = [
        "entry_time", "exit_time", "side", "entry",
        "sl", "tp", "result", "pnl", "equity"
    ]
    reject_cols = ["time", "side", "reason", "estimated_risk"]

    return (
        pd.DataFrame(trades, columns=trade_cols),
        pd.DataFrame(rejected, columns=reject_cols)
    )

if files:
    try:
        df = prepare(files)
        st.success(f"Loaded {len(df)} M15 candles")
        st.write("Period:", df.index.min(), "to", df.index.max())

        if st.button("Run V36 Independent Backtest"):
            if len(df) < 250:
                st.error("Not enough data.")
            else:
                trades, rejected = backtest(df)

                st.subheader("Results")
                a, b, c = st.columns(3)
                a.metric("Trades", len(trades))
                b.metric("Rejected", len(rejected))
                c.metric(
                    "Net P&L",
                    f"${trades.pnl.sum():.2f}" if len(trades) else "$0.00"
                )

                if len(trades):
                    st.metric("Win rate", f"{(trades.pnl > 0).mean()*100:.1f}%")
                    st.line_chart(trades.set_index("exit_time")["equity"])
                    st.dataframe(trades)
                    st.download_button(
                        "Download V36 Trades CSV",
                        trades.to_csv(index=False),
                        "V36_trades.csv",
                        "text/csv"
                    )

                if len(rejected):
                    st.dataframe(rejected)
                    st.download_button(
                        "Download V36 Rejected CSV",
                        rejected.to_csv(index=False),
                        "V36_rejected.csv",
                        "text/csv"
                    )

                if not len(trades):
                    st.warning("WAIT: No risk-safe trades found.")

    except Exception as e:
        st.error(f"Data error: {e}")
else:
    st.info("Upload historical data to begin.")
