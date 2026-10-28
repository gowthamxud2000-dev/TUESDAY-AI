
import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile

st.set_page_config(
    page_title="Tuesday AI V29",
    page_icon="🥇",
    layout="wide"
)

st.title("🥇 Tuesday AI V29")
st.subheader("XAUUSD | Manual BUY / SELL / WAIT")
st.caption("Rule-based research assistant | No automatic orders")

st.warning(
    "Signals are experimental, not guaranteed profitable. "
    "Use completed candles and paper-test before live trading."
)

# ---------- DATA LOADER ----------

def read_text_data(raw):
    text = raw.decode("utf-8-sig", errors="replace")
    lines = [x for x in text.splitlines() if x.strip()]

    if not lines:
        raise ValueError("File is empty")

    first = lines[0]
    sep = ";" if ";" in first else (
        "\t" if "\t" in first else ","
    )

    df = pd.read_csv(
        io.StringIO(text),
        sep=sep,
        header=None,
        low_memory=False
    )

    # HistData formats:
    # Date Time Open High Low Close Volume
    # DateTime Open High Low Close Volume
    # Generic header CSV

    if len(df.columns) >= 7:
        dt = (
            df.iloc[:, 0].astype(str) + " " +
            df.iloc[:, 1].astype(str)
        )
        price_cols = [2, 3, 4, 5]

    elif len(df.columns) >= 5:
        dt = df.iloc[:, 0].astype(str)
        price_cols = [1, 2, 3, 4]

    else:
        raise ValueError("Expected date and OHLC columns")

    dates = pd.to_datetime(
        dt, errors="coerce", format="mixed"
    )

    if dates.notna().sum() == 0:
        dates = pd.to_datetime(
            dt, errors="coerce", dayfirst=True
        )

    out = pd.DataFrame({
        "datetime": dates,
        "open": pd.to_numeric(
            df.iloc[:, price_cols[0]], errors="coerce"
        ),
        "high": pd.to_numeric(
            df.iloc[:, price_cols[1]], errors="coerce"
        ),
        "low": pd.to_numeric(
            df.iloc[:, price_cols[2]], errors="coerce"
        ),
        "close": pd.to_numeric(
            df.iloc[:, price_cols[3]], errors="coerce"
        )
    })

    out = out.dropna()
    out = out.drop_duplicates("datetime")
    out = out.sort_values("datetime")
    out = out.set_index("datetime")

    valid = (
        (out["high"] >= out[["open", "close", "low"]].max(axis=1))
        & (out["low"] <= out[["open", "close", "high"]].min(axis=1))
        & (out["low"] > 0)
    )

    return out.loc[valid]


def load_data(upload):
    raw = upload.getvalue()

    if upload.name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = [
                n for n in z.namelist()
                if n.lower().endswith((".csv", ".txt"))
            ]
            if not names:
                raise ValueError("ZIP has no CSV/TXT")
            raw = z.read(names[0])

    return read_text_data(raw)


# ---------- INDICATORS ----------

def add_indicators(df):
    df = df.copy()

    df["ema50"] = df["close"].ewm(
        span=50, adjust=False
    ).mean()

    df["ema200"] = df["close"].ewm(
        span=200, adjust=False
    ).mean()

    prev_close = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    return df


# ---------- SIGNAL ENGINE ----------

def analyze(df, lookback, rr, sl_atr):
    if len(df) < max(250, lookback + 5):
        return {
            "signal": "WAIT",
            "reason": "Not enough completed M15 candles"
        }

    d = add_indicators(df)
    last = d.iloc[-1]

    price = float(last["close"])
    atr = float(last["atr"])

    if not np.isfinite(atr) or atr <= 0:
        return {
            "signal": "WAIT",
            "reason": "Invalid volatility data"
        }

    # Swing levels exclude current candle
    history = d.iloc[-lookback-1:-1]

    swing_high = float(history["high"].max())
    swing_low = float(history["low"].min())

    swing_range = swing_high - swing_low

    if swing_range <= 0:
        return {
            "signal": "WAIT",
            "reason": "Invalid swing range"
        }

    # Bullish retracement from swing high
    buy_z1 = swing_high - swing_range * 0.618
    buy_z2 = swing_high - swing_range * 0.50

    # Bearish retracement from swing low
    sell_z1 = swing_low + swing_range * 0.50
    sell_z2 = swing_low + swing_range * 0.618

    bullish_trend = (
        last["ema50"] > last["ema200"]
        and price > last["ema200"]
    )

    bearish_trend = (
        last["ema50"] < last["ema200"]
        and price < last["ema200"]
    )

    bullish_candle = (
        last["close"] > last["open"]
        and last["close"] > d.iloc[-2]["high"]
    )

    bearish_candle = (
        last["close"] < last["open"]
        and last["close"] < d.iloc[-2]["low"]
    )

    # Confirmation candle may close outside zone.
    # Require its range to overlap Golden Zone.
    buy_zone = (
        last["low"] <= buy_z2
        and last["high"] >= buy_z1
    )

    sell_zone = (
        last["high"] >= sell_z1
        and last["low"] <= sell_z2
    )

    result = {
        "signal": "WAIT",
        "reason": "No fully confirmed setup",
        "price": price,
        "ema50": float(last["ema50"]),
        "ema200": float(last["ema200"]),
        "atr": atr,
        "buy_zone": (buy_z1, buy_z2),
        "sell_zone": (sell_z1, sell_z2),
        "trend": (
            "UP" if bullish_trend else
            "DOWN" if bearish_trend else "SIDEWAYS"
        )
    }

    if bullish_trend and buy_zone and bullish_candle:
        entry = price
        sl = min(
            float(last["low"]),
            buy_z1
        ) - atr * sl_atr

        risk = entry - sl

        if risk > 0:
            result.update({
                "signal": "BUY",
                "reason": "Uptrend + Golden Zone + bullish break",
                "entry": entry,
                "sl": sl,
                "tp": entry + risk * rr
            })

    elif bearish_trend and sell_zone and bearish_candle:
        entry = price
        sl = max(
            float(last["high"]),
            sell_z2
        ) + atr * sl_atr

        risk = sl - entry

        if risk > 0:
            result.update({
                "signal": "SELL",
                "reason": "Downtrend + Golden Zone + bearish break",
                "entry": entry,
                "sl": sl,
                "tp": entry - risk * rr
            })

    return result


# ---------- SIDEBAR ----------

st.sidebar.header("⚙️ Strategy Settings")

lookback = st.sidebar.number_input(
    "Swing lookback candles",
    min_value=20,
    max_value=200,
    value=60
)

rr = st.sidebar.slider(
    "Risk Reward",
    min_value=1.0,
    max_value=4.0,
    value=2.0,
    step=0.25
)

sl_atr = st.sidebar.slider(
    "Extra SL buffer (ATR)",
    min_value=0.0,
    max_value=2.0,
    value=0.3,
    step=0.1
)

st.sidebar.info(
    "These are experimental defaults, "
    "not independently validated settings."
)

# ---------- UPLOAD ----------

st.header("📂 Upload Gold Data")

uploaded = st.file_uploader(
    "Upload XAUUSD M1 or M15 CSV / TXT / ZIP",
    type=["csv", "txt", "zip"]
)

source_tf = st.selectbox(
    "Uploaded data timeframe",
    ["M15", "M1"]
)

st.caption(
    "M1 files are converted to M15. "
    "M15 data is used directly."
)

if uploaded is not None:
    try:
        df = load_data(uploaded)

        if df.empty:
            st.error("No valid OHLC rows found")
            st.stop()

        if source_tf == "M1":
            counts = df["close"].resample("15min").count()

            df = df.resample("15min").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last"
            })

            # Require 15 observed minute bars
            df = df.loc[counts == 15].dropna()

        else:
            df = df.dropna()

        # Last bar may still be forming.
        # Remove it conservatively.
        if len(df) > 0:
            df = df.iloc[:-1]

        if df.empty:
            st.error("No completed candles available")
            st.stop()

        st.success(
            f"{len(df):,} completed candles loaded"
        )

        st.write(
            "Last analyzed candle:",
            str(df.index[-1])
        )

        result = analyze(
            df,
            lookback=int(lookback),
            rr=float(rr),
            sl_atr=float(sl_atr)
        )

        st.header("🧠 Tuesday AI Signal")

        signal = result["signal"]

        if signal == "BUY":
            st.success("🟢 BUY SIGNAL")

        elif signal == "SELL":
            st.error("🔴 SELL SIGNAL")

        else:
            st.warning("🟡 WAIT — NO TRADE")

        st.write("Reason:", result["reason"])

        if "price" in result:
            a, b, c = st.columns(3)

            a.metric(
                "Gold Close",
                f'{result["price"]:.2f}'
            )

            b.metric(
                "Trend",
                result["trend"]
            )

            c.metric(
                "ATR",
                f'{result["atr"]:.2f}'
            )

            st.write(
                "EMA 50:",
                round(result["ema50"], 2)
            )

            st.write(
                "EMA 200:",
                round(result["ema200"], 2)
            )

            bz = result["buy_zone"]
            sz = result["sell_zone"]

            st.write(
                "BUY Golden Zone:",
                f"{bz[0]:.2f} – {bz[1]:.2f}"
            )

            st.write(
                "SELL Golden Zone:",
                f"{sz[0]:.2f} – {sz[1]:.2f}"
            )

        if signal in ["BUY", "SELL"]:
            st.subheader("📍 Proposed Trade Levels")

            x, y, z = st.columns(3)

            x.metric(
                "Entry",
                f'{result["entry"]:.2f}'
            )

            y.metric(
                "Stop Loss",
                f'{result["sl"]:.2f}'
            )

            z.metric(
                "Take Profit",
                f'{result["tp"]:.2f}'
            )

            st.write(f"Risk Reward: 1:{rr:.2f}")

            st.warning(
                "Entry is based on historical candle close. "
                "Actual fill price and spread may differ."
            )

        st.header("📈 Gold Chart")

        chart = add_indicators(df)

        st.line_chart(
            chart[["close", "ema50", "ema200"]].tail(250)
        )

        st.header("📋 Recent Candles")
        st.dataframe(df.tail(15))

        st.caption(
            "Historical data is not live. "
            "This version does not include execution costs "
            "or an independent signal-performance audit."
        )

    except Exception as e:
        st.error(f"Data error: {e}")
        st.info(
            "Check the CSV format and selected timeframe."
        )

else:
    st.info("Upload historical XAUUSD data to begin.")

st.divider()
st.caption("Tuesday AI V29 | Manual Trading Research")
