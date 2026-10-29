import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tuesday AI V43", layout="wide")
st.title("Tuesday AI V43 | Trend Pullback Lab")
st.warning("Synthetic demo only. No live prices, orders, or guaranteed profits.")

with st.sidebar:
    seed = st.number_input("Random seed", 1, 999999, 43)
    days = st.slider("Demo days", 10, 35, 18)
    starting = st.number_input(
        "Starting balance ($)", min_value=100.0, value=2500.0
    )
    risk = st.number_input(
        "Risk per trade ($)", min_value=0.5, value=4.0
    )
    daily_cap = st.number_input(
        "Daily loss budget ($)", min_value=1.0, value=40.0
    )
    contract = st.number_input(
        "Ounces per lot", min_value=1.0, value=100.0
    )
    minimum = st.number_input(
        "Minimum lot", min_value=0.01, value=0.01, step=0.01
    )
    step = st.number_input(
        "Lot increment", min_value=0.01, value=0.01, step=0.01
    )
    spread = st.number_input(
        "Spread ($/oz)", min_value=0.0, value=0.30
    )
    slip = st.number_input(
        "Slippage per side ($/oz)", min_value=0.0, value=0.10
    )
    commission = st.number_input(
        "Round-trip commission ($/lot)",
        min_value=0.0,
        value=7.0,
    )


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
    return pd.DataFrame(
        {
            "time": t,
            "open": op,
            "high": high,
            "low": low,
            "close": close,
        }
    )


def prepare(raw):
    d = raw.copy().reset_index(drop=True)

    # M5 information becomes available only after the M5 candle closes.
    m = (
        d.set_index("time")
        .resample("5min", closed="left", label="right")
        .agg({"close": "last"})
        .dropna()
    )
    m["fast"] = m.close.ewm(span=50, adjust=False).mean()
    m["slow"] = m.close.ewm(span=200, adjust=False).mean()
    m["trend"] = np.select(
        [
            (m.close > m.fast) & (m.fast > m.slow),
            (m.close < m.fast) & (m.fast < m.slow),
        ],
        [1, -1],
        0,
    )

    d["known_at"] = d.time + pd.Timedelta(minutes=1)
    d = pd.merge_asof(
        d,
        m[["trend"]].reset_index().rename(
            columns={"time": "m5_end"}
        ),
        left_on="known_at",
        right_on="m5_end",
        direction="backward",
    )
    d["trend"] = d.trend.fillna(0)
    d["ema"] = d.close.ewm(span=20, adjust=False).mean()

    tr = pd.concat(
        [
            d.high - d.low,
            (d.high - d.close.shift()).abs(),
            (d.low - d.close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
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
        (d.trend == 1)
        & prior_touch
        & (d.close > d.ema)
        & (d.close > d.high.shift(1))
    )
    sell = (
        (d.trend == -1)
        & prior_sell
        & (d.close
