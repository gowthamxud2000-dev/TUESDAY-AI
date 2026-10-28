
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Tuesday AI V35", page_icon="📈", layout="wide")
st.title("📈 Tuesday AI V35")
st.caption("XAUUSD.r | Personal account | Manual trading only")

# Broker specifications
CONTRACT_SIZE = 100.0
MIN_LOT = 0.01
LOT_STEP = 0.01

st.sidebar.header("Account & Risk")
balance = st.sidebar.number_input(
    "Starting balance ($)", min_value=1.0, value=100.0
)
risk_pct = st.sidebar.number_input(
    "Risk per trade (%)", min_value=0.1,
    max_value=1.0, value=1.0, step=0.1
)
spread = st.sidebar.number_input(
    "Spread (gold price units)", min_value=0.0,
    value=0.22, step=0.01
)
slippage = st.sidebar.number_input(
    "Slippage (price units per side)", min_value=0.0,
    value=0.05, step=0.01
)
commission = st.sidebar.number_input(
    "Commission ($ per lot, round trip)",
    min_value=0.0, value=0.0, step=0.5
)

risk_budget = balance * risk_pct / 100

st.metric("Risk budget per trade", f"${risk_budget:.2f}")
st.info(
    "0.01 lot = 1 ounce exposure. "
    "A $5 gold price movement = approximately $5 P/L "
    "before trading costs."
)

def indicators(df):
    df = df.copy()
    df["ema50"] = df["close"].ewm(
        span=50, adjust=False
    ).mean()
    df["ema200"] = df["close"].ewm(
        span=200, adjust=False
    ).mean()
    df["prev_high"] = df["high"].shift(1).rolling(20).max()
    df["prev_low"] = df["low"].shift(1).rolling(20).min()
    return df

def signal(row):
    if pd.isna(row["ema200"]) or pd.isna(row["prev_high"]):
        return "WAIT"
    if row["ema50"] > row["ema200"] and row["close"] > row["prev_high"]:
        return "BUY"
    if row["ema50"] < row["ema200"] and row["close"] < row["prev_low"]:
        return "SELL"
    return "WAIT"

def risk_check(entry, stop, account_balance):
    distance = abs(entry - stop)
    cost = (
        spread + 2 * slippage
    ) * CONTRACT_SIZE * MIN_LOT
    cost += commission * MIN_LOT

    estimated_loss = distance * CONTRACT_SIZE * MIN_LOT + cost
    allowed = account_balance * risk_pct / 100

    return estimated_loss, estimated_loss <= allowed

st.header("1. Manual BUY / SELL / WAIT")

direction = st.selectbox(
    "Signal", ["WAIT", "BUY", "SELL"]
)
entry = st.number_input(
    "Entry price", min_value=0.01, value=4030.0
)
stop = st.number_input(
    "Stop loss", min_value=0.01, value=4027.0
)
target = st.number_input(
    "Take profit", min_value=0.01, value=4036.0
)

if st.button("Check trade"):
    valid_direction = (
        (direction == "BUY" and stop < entry < target)
        or (direction == "SELL" and target < entry < stop)
    )

    if direction == "WAIT" or not valid_direction:
        st.warning("WAIT — invalid or missing trade setup.")
    else:
        loss, allowed = risk_check(entry, stop, balance)
        st.write(f"Estimated risk: ${loss:.2f}")
        if allowed:
            st.success(f"{direction} — risk check passed")
            st.warning(
                "Risk approval is not a confirmed profitable signal."
            )
        else:
            st.error("WAIT — minimum lot risk exceeds budget.")

st.header("2. Historical CSV Backtest")

uploaded = st.file_uploader(
    "Upload XAUUSD M15 OHLC CSV",
    type=["csv"]
)

def load_data(file):
    raw = pd.read_csv(file)
    raw.columns = [
        str(c).strip().lower() for c in raw.columns
    ]

    aliases = {
        "open": ["open", "<open>"],
        "high": ["high", "<high>"],
        "low": ["low", "<low>"],
        "close": ["close", "<close>"]
    }

    for name, options in aliases.items():
        found = next(
            (c for c in options if c in raw.columns), None
        )
        if found is None:
            raise ValueError(f"Missing column: {name}")
        raw[name] = pd.to_numeric(raw[found], errors="coerce")

    if "datetime" in raw.columns:
        raw["time"] = pd.to_datetime(
            raw["datetime"], errors="coerce"
        )
    elif "date" in raw.columns and "time" in raw.columns:
        raw["time"] = pd.to_datetime(
            raw["date"].astype(str) + " " +
            raw["time"].astype(str), errors="coerce"
        )
    elif "<date>" in raw.columns and "<time>" in raw.columns:
        raw["time"] = pd.to_datetime(
            raw["<date>"].astype(str) + " " +
            raw["<time>"].astype(str), errors="coerce"
        )
    else:
        raise ValueError("CSV needs date/time columns.")

    raw = raw.dropna(
        subset=["time", "open", "high", "low", "close"]
    )
    raw = raw.sort_values("time").drop_duplicates("time")
    raw = raw.set_index("time")

    if len(raw) < 250:
        raise ValueError("Need at least 250 candles.")

    return raw

def backtest(data, initial_balance):
    df = indicators(data)
    equity = initial_balance
    trades = []
    rejected = []
    last_exit = -1

    # Signal uses completed candle i.
    # Entry is at next candle open.
    for i in range(250, len(df) - 1):
        if i <= last_exit:
            continue

        row = df.iloc[i]
        side = signal(row)
        if side == "WAIT":
            continue

        next_bar = df.iloc[i + 1]
        raw_entry = float(next_bar["open"])

        # Previous 10 candles define stop.
        history = df.iloc[i-9:i+1]
        if side == "BUY":
            stop = float(history["low"].min())
            entry_price = raw_entry + spread / 2 + slippage
            distance = entry_price - stop
            target_price = entry_price + 2 * distance
        else:
            stop = float(history["high"].max())
            entry_price = raw_entry - spread / 2 - slippage
            distance = stop - entry_price
            target_price = entry_price - 2 * distance

        if distance <= 0:
            continue

        loss, allowed = risk_check(
            entry_price, stop, equity
        )

        if not allowed:
            rejected.append({
                "time": df.index[i+1],
                "side": side,
                "reason": "Minimum lot risk too high",
                "estimated_risk": round(loss, 2)
            })
            continue

        exit_price = None
        exit_index = None
        outcome = None

        for j in range(i + 1, len(df)):
            bar = df.iloc[j]
            if side == "BUY":
                hit_sl = bar["low"] <= stop
                hit_tp = bar["high"] >= target_price
            else:
                hit_sl = bar["high"] >= stop
                hit_tp = bar["low"] <= target_price

            # Conservative: SL wins if both touched.
            if hit_sl:
                exit_price = stop
                outcome = "SL"
            elif hit_tp:
                exit_price = target_price
                outcome = "TP"

            if exit_price is not None:
                exit_index = j
                break

        if exit_price is None:
            exit_index = len(df) - 1
            exit_price = float(df.iloc[-1]["close"])
            outcome = "End of data"

        if side == "BUY":
            pnl = (exit_price - entry_price)
        else:
            pnl = (entry_price - exit_price)

        pnl = pnl * CONTRACT_SIZE * MIN_LOT
        pnl -= commission * MIN_LOT

        equity += pnl
        last_exit = exit_index

        trades.append({
            "entry_time": df.index[i+1],
            "exit_time": df.index[exit_index],
            "side": side,
            "entry": round(entry_price, 2),
            "stop": round(stop, 2),
            "target": round(target_price, 2),
            "outcome": outcome,
            "pnl": round(pnl, 2),
            "equity": round(equity, 2)
        })

        if equity <= 0:
            break

    return pd.DataFrame(trades), pd.DataFrame(rejected)

if uploaded is not None:
    try:
        data = load_data(uploaded)
        st.success(f"Loaded {len(data)} candles")
        st.write(
            "Data range:",
            str(data.index.min()),
            "to",
            str(data.index.max())
        )

        split = int(len(data) * 0.7)
        train = data.iloc[:split]
        test = data.iloc[max(0, split-250):]

        st.subheader("Independent validation")
        st.caption(
            "70% earlier candles = development sample; "
            "30% later candles = test sample. "
            "250 warm-up candles included."
        )

        train_trades, train_rejected = backtest(
            train, balance
        )
        test_trades, test_rejected = backtest(
            test, balance
        )

        for title, trades, rejected in [
            ("Development", train_trades, train_rejected),
            ("Out-of-sample", test_trades, test_rejected)
        ]:
            st.subheader(title)
            st.metric("Trades", len(trades))
            st.metric("Rejected signals", len(rejected))

            if not trades.empty:
                total = trades["pnl"].sum()
                wins = (trades["pnl"] > 0).sum()
                win_rate = wins / len(trades) * 100

                st.metric("Net P/L", f"${total:.2f}")
                st.metric("Win rate", f"{win_rate:.1f}%")
                st.line_chart(
                    trades.set_index("exit_time")["equity"]
                )
                st.dataframe(trades)
                st.download_button(
                    f"Download {title} trades",
                    trades.to_csv(index=False),
                    file_name=(
                        "V35_" +
                        title.replace("-", "_") +
                        "_trades.csv"
                    ),
                    mime="text/csv"
                )
            else:
                st.warning(
                    "No trades passed the risk filter."
                )

            if not rejected.empty:
                st.dataframe(rejected)
                st.download_button(
                    f"Download {title} rejected",
                    rejected.to_csv(index=False),
                    file_name=(
                        "V35_" +
                        title.replace("-", "_") +
                        "_rejected.csv"
                    ),
                    mime="text/csv"
                )

    except Exception as error:
        st.error(f"CSV error: {error}")

st.warning(
    "Research tool only. No automatic orders. "
    "Backtests do not guarantee profits. "
    "Verify broker costs and data before live trading."
)
