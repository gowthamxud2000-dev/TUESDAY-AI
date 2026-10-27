
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Tuesday AI V19", page_icon="🥇", layout="wide")
st.title("🥇 Tuesday AI V19")
st.caption("XAUUSD Smart Signal Engine | Historical Backtesting")
st.warning("Research only. Signals are not verified live trading recommendations.")

uploaded = st.file_uploader("Upload XAUUSD M1 or M15 CSV", type=["csv", "txt"])

with st.sidebar:
    st.header("Strategy Settings")
    rr = st.slider("Risk / Reward", 1.0, 5.0, 2.0, 0.5)
    atr_mult = st.slider("ATR Stop Multiplier", 1.0, 4.0, 1.5, 0.5)
    breakout_len = st.slider("Breakout Lookback", 5, 40, 20)
    pullback_atr = st.slider("Pullback Distance (ATR)", 0.1, 1.0, 0.4, 0.1)
    spread = st.number_input("Spread in price units", 0.0, 10.0, 0.30, 0.05)
    st.caption("Spread is an estimate; commissions and slippage are excluded.")

def load_data(file):
    file.seek(0)
    df = pd.read_csv(file, sep=None, engine="python")
    df.columns = [str(c).strip().lower().replace("<", "").replace(">", "") for c in df.columns]
    if "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            errors="coerce"
        )
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        raise ValueError("CSV needs date/time or datetime columns.")

    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    df = df.sort_values("datetime").drop_duplicates("datetime")
    df = df.set_index("datetime")
    df = df[(df.high >= df[["open", "close", "low"]].max(axis=1)) &
            (df.low <= df[["open", "close"]].min(axis=1))]
    return df

def indicators(df):
    d = df.copy()
    d["ema50"] = d.close.ewm(span=50, adjust=False).mean()
    d["ema200"] = d.close.ewm(span=200, adjust=False).mean()
    prev = d.close.shift(1)
    tr = pd.concat([
        d.high - d.low,
        (d.high - prev).abs(),
        (d.low - prev).abs()
    ], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    d["resistance"] = d.high.shift(1).rolling(breakout_len).max()
    d["support"] = d.low.shift(1).rolling(breakout_len).min()

    up = d.ema50 > d.ema200
    down = d.ema50 < d.ema200
    buy_break = up & (d.close > d.resistance) & (d.close > d.open)
    sell_break = down & (d.close < d.support) & (d.close < d.open)
    buy_pull = up & (d.low <= d.ema50 + d.atr * pullback_atr) & (d.close > d.ema50) & (d.close > d.open)
    sell_pull = down & (d.high >= d.ema50 - d.atr * pullback_atr) & (d.close < d.ema50) & (d.close < d.open)

    d["signal"] = np.select(
        [buy_break | buy_pull, sell_break | sell_pull],
        ["BUY", "SELL"], default="WAIT"
    )
    d.iloc[:200, d.columns.get_loc("signal")] = "WAIT"
    return d

def backtest(d):
    trades = []
    i = 200
    while i < len(d) - 1:
        signal = d.signal.iloc[i]
        atr = d.atr.iloc[i]
        if signal == "WAIT" or not np.isfinite(atr) or atr <= 0:
            i += 1
            continue

        entry_bar = i + 1
        direction = 1 if signal == "BUY" else -1
        entry = float(d.open.iloc[entry_bar]) + direction * spread / 2
        risk = atr * atr_mult
        sl = entry - direction * risk
        tp = entry + direction * risk * rr
        exit_price = None
        exit_idx = None
        outcome = "OPEN"

        for j in range(entry_bar, len(d)):
            high = float(d.high.iloc[j])
            low = float(d.low.iloc[j])
            if direction == 1:
                hit_sl, hit_tp = low <= sl, high >= tp
            else:
                hit_sl, hit_tp = high >= sl, low <= tp

            # Conservative: SL first if both touched in one candle
            if hit_sl:
                exit_price, outcome = sl, "SL"
            elif hit_tp:
                exit_price, outcome = tp, "TP"

            if exit_price is not None:
                exit_idx = j
                break

        if exit_idx is None:
            exit_idx = len(d) - 1
            exit_price = float(d.close.iloc[exit_idx]) - direction * spread / 2
            outcome = "END"

        pnl = direction * (exit_price - entry)
        trades.append({
            "signal_time": d.index[i],
            "entry_time": d.index[entry_bar],
            "exit_time": d.index[exit_idx],
            "side": signal,
            "entry": round(entry, 3),
            "sl": round(sl, 3),
            "tp": round(tp, 3),
            "exit": round(exit_price, 3),
            "outcome": outcome,
            "pnl_price": round(pnl, 3),
            "r_multiple": round(pnl / risk, 4)
        })
        i = exit_idx + 1
    return pd.DataFrame(trades)

if uploaded:
    try:
        raw = load_data(uploaded)
        timeframe = st.selectbox("Backtest timeframe", ["M15", "M1"])
        if timeframe == "M15":
            data = raw.resample("15min", label="left", closed="left").agg({
                "open": "first", "high": "max",
                "low": "min", "close": "last"
            }).dropna()
            # Avoid including an incomplete final 15-minute candle
            if len(data) and raw.index.max() < data.index[-1] + pd.Timedelta(minutes=14):
                data = data.iloc[:-1]
        else:
            data = raw.copy()

        if len(data) < 250:
            st.error("Need at least 250 candles.")
            st.stop()

        d = indicators(data)
        trades = backtest(d)
        st.success(f"Loaded {len(d):,} candles")
        st.subheader("Latest Historical Signal")
        latest = d.iloc[-1]
        st.metric("Signal", latest.signal)
        st.metric("Last Close", f"{latest.close:.2f}")

        if trades.empty:
            st.info("No trades found. Try another dataset or settings.")
        else:
            wins = trades[trades.pnl_price > 0]
            losses = trades[trades.pnl_price < 0]
            winrate = len(wins) / len(trades) * 100
            gross_profit = wins.pnl_price.sum()
            gross_loss = abs(losses.pnl_price.sum())
            pf = gross_profit / gross_loss if gross_loss else float("inf")
            equity = trades.r_multiple.cumsum()
            dd = (equity - equity.cummax().clip(lower=0)).min()

            st.subheader("Backtest Results")
            a, b, c, e = st.columns(4)
            a.metric("Trades", len(trades))
            b.metric("Win Rate", f"{winrate:.1f}%")
            c.metric("Profit Factor", f"{pf:.2f}")
            e.metric("Max Drawdown", f"{dd:.2f} R")
            st.metric("Total Return (R units)", f"{trades.r_multiple.sum():.2f} R")
            st.line_chart(equity.reset_index(drop=True))
            st.dataframe(trades, use_container_width=True)
            st.download_button(
                "Download V19 Trades CSV",
                trades.to_csv(index=False),
                "Tuesday_AI_V19_trades.csv",
                "text/csv"
            )
    except Exception as exc:
        st.error(f"Data error: {exc}")
else:
    st.info("Upload historical gold candles to begin V19 backtesting.")
