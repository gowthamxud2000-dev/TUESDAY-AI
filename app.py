
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Tuesday AI V32",
    page_icon="🥇",
    layout="wide"
)

st.title("🥇 Tuesday AI V32")
st.caption("XAUUSD M15 | BUY / SELL / WAIT | Manual Assistant")
st.warning(
    "Research and demo use only. No live price feed. "
    "Signals are not guaranteed."
)

# ACCOUNT SETTINGS
st.sidebar.header("Funded Account")
balance = st.sidebar.number_input(
    "Account balance ($)", min_value=100.0,
    value=2500.0, step=100.0
)
daily_limit = st.sidebar.number_input(
    "Daily loss limit ($)", min_value=1.0,
    value=40.0
)
risk_pct = st.sidebar.number_input(
    "Risk per trade (%)", min_value=0.1,
    max_value=2.0, value=0.5, step=0.1
)
spread = st.sidebar.number_input(
    "Estimated spread ($/oz)", min_value=0.0,
    value=0.30, step=0.05
)
slippage = st.sidebar.number_input(
    "Slippage ($/oz per side)", min_value=0.0,
    value=0.10, step=0.05
)
commission = st.sidebar.number_input(
    "Commission ($ per lot round trip)",
    min_value=0.0, value=7.0
)
lot_contract = st.sidebar.number_input(
    "Ounces per lot", min_value=1.0,
    value=100.0
)
st.sidebar.caption(
    "Check broker contract size, costs and "
    "funded-account rules before use."
)

st.header("📂 Upload XAUUSD M15 CSV")
uploaded = st.file_uploader(
    "Upload historical OHLC CSV", type=["csv"]
)

def load_csv(file):
    df = pd.read_csv(file, sep=None, engine="python")
    df.columns = [
        str(c).strip().lower().replace("<", "")
        .replace(">", "").replace(" ", "_")
        for c in df.columns
    ]

    # Recognize multiple timestamp formats
    time_names = [
        "datetime", "timestamp", "time",
        "date_time", "gmt_time"
    ]
    time_col = next(
        (c for c in time_names if c in df.columns),
        None
    )

    if time_col:
        dt = pd.to_datetime(
            df[time_col], errors="coerce"
        )
    elif "date" in df.columns and "time" in df.columns:
        dt = pd.to_datetime(
            df["date"].astype(str) + " " +
            df["time"].astype(str),
            errors="coerce"
        )
    elif "date" in df.columns:
        dt = pd.to_datetime(
            df["date"], errors="coerce"
        )
    else:
        raise ValueError(
            "Date/time column missing. "
            "Expected time, datetime, or date."
        )

    df["datetime"] = dt

    for col in ["open", "high", "low", "close"]:
        matches = [
            c for c in df.columns
            if c.lower() == col
        ]
        if not matches:
            raise ValueError(
                f"Missing OHLC column: {col}"
            )
        df[col] = pd.to_numeric(
            df[matches[0]], errors="coerce"
        )

    df = df.dropna(
        subset=["datetime", "open", "high",
                "low", "close"]
    )
    df = df.sort_values("datetime")
    df = df.drop_duplicates("datetime")
    df = df.set_index("datetime")

    valid = (
        (df["high"] >= df[["open", "close", "low"]].max(axis=1)) &
        (df["low"] <= df[["open", "close", "high"]].min(axis=1)) &
        (df["low"] > 0)
    )
    return df.loc[valid].copy()

def prepare(df):
    df = df.copy()
    df["ema50"] = df["close"].ewm(
        span=50, adjust=False, min_periods=50
    ).mean()
    df["ema200"] = df["close"].ewm(
        span=200, adjust=False, min_periods=200
    ).mean()

    # Previous 40 candles only: no future leakage
    prev_high = df["high"].shift(1).rolling(40).max()
    prev_low = df["low"].shift(1).rolling(40).min()
    width = prev_high - prev_low

    df["buy_zone_low"] = prev_high - 0.618 * width
    df["buy_zone_high"] = prev_high - 0.5 * width
    df["sell_zone_low"] = prev_low + 0.5 * width
    df["sell_zone_high"] = prev_low + 0.618 * width

    bull = (
        (df["ema50"] > df["ema200"]) &
        (df["close"] > df["ema200"]) &
        (df["low"] <= df["buy_zone_high"]) &
        (df["close"] >= df["buy_zone_low"]) &
        (df["close"] > df["open"])
    )
    bear = (
        (df["ema50"] < df["ema200"]) &
        (df["close"] < df["ema200"]) &
        (df["high"] >= df["sell_zone_low"]) &
        (df["close"] <= df["sell_zone_high"]) &
        (df["close"] < df["open"])
    )

    df["signal"] = np.select(
        [bull, bear], ["BUY", "SELL"],
        default="WAIT"
    )
    return df

def backtest(df, balance, daily_limit,
             risk_pct, spread, slippage,
             commission, lot_contract):
    trades = []
    equity = balance
    start_balance = balance
    current_day = None
    day_pnl = 0.0
    blocked = False
    i = 200
    cost = spread + 2 * slippage

    while i < len(df) - 1:
        bar = df.iloc[i]
        entry_bar = df.iloc[i + 1]
        day = entry_bar.name.date()

        if day != current_day:
            current_day = day
            day_pnl = 0.0
            blocked = False

        if equity <= 0:
            break

        if blocked or bar["signal"] == "WAIT":
            i += 1
            continue

        side = bar["signal"]
        entry = float(entry_bar["open"])
        zone = (
            bar["buy_zone_low"] if side == "BUY"
            else bar["sell_zone_high"]
        )
        stop = float(zone)
        distance = (
            entry - stop if side == "BUY"
            else stop - entry
        )

        if not np.isfinite(distance) or distance <= cost:
            i += 1
            continue

        # Stop opening new trades if insufficient
        # daily risk budget remains.
        remaining = daily_limit + day_pnl
        risk_budget = min(
            equity * risk_pct / 100,
            max(0.0, remaining)
        )
        if risk_budget <= 0:
            blocked = True
            i += 1
            continue

        risk_per_lot = (
            (distance + cost) * lot_contract
            + commission
        )
        lots = np.floor(
            (risk_budget / risk_per_lot) * 100
        ) / 100

        if lots < 0.01:
            i += 1
            continue

        target = (
            entry + 2 * distance if side == "BUY"
            else entry - 2 * distance
        )

        # Exit on stop/target. Same-bar ambiguity:
        # conservatively assume stop is hit first.
        exit_price = None
        reason = None
        exit_i = None

        for j in range(i + 1, len(df)):
            candle = df.iloc[j]

            if side == "BUY":
                hit_sl = candle["low"] <= stop
                hit_tp = candle["high"] >= target
            else:
                hit_sl = candle["high"] >= stop
                hit_tp = candle["low"] <= target

            if hit_sl:
                exit_price = stop
                reason = "SL"
            elif hit_tp:
                exit_price = target
                reason = "TP"
            elif j == len(df) - 1:
                exit_price = float(candle["close"])
                reason = "END"

            if exit_price is not None:
                exit_i = j
                break

        if exit_i is None:
            break

        gross_move = (
            exit_price - entry if side == "BUY"
            else entry - exit_price
        )
        pnl = (
            (gross_move - cost) * lot_contract
            - commission
        ) * lots

        equity += pnl

        # Attribute P&L to exit day
        exit_day = df.index[exit_i].date()
        if exit_day != current_day:
            current_day = exit_day
            day_pnl = 0.0
        day_pnl += pnl
        if day_pnl <= -daily_limit:
            blocked = True

        trades.append({
            "entry_time": entry_bar.name,
            "exit_time": df.index[exit_i],
            "side": side,
            "entry": round(entry, 3),
            "sl": round(stop, 3),
            "tp": round(target, 3),
            "lots": lots,
            "exit_reason": reason,
            "pnl": round(pnl, 2),
            "equity": round(equity, 2)
        })
        i = exit_i + 1

    return pd.DataFrame(trades)

if uploaded is not None:
    try:
        df = load_csv(uploaded)

        if len(df) < 250:
            st.error("Need at least 250 M15 candles.")
            st.stop()

        df = prepare(df)
        st.success(
            f"CSV loaded: {len(df)} candles"
        )
        st.write(
            "Data range:",
            str(df.index.min()),
            "to",
            str(df.index.max())
        )

        st.subheader("📊 Gold Chart")
        st.line_chart(
            df[["close", "ema50", "ema200"]].tail(200)
        )

        st.subheader("🚦 Latest Historical Signal")
        latest = df.iloc[-1]
        signal = latest["signal"]

        if signal == "BUY":
            st.success("🟢 BUY setup detected")
        elif signal == "SELL":
            st.error("🔴 SELL setup detected")
        else:
            st.warning("🟡 WAIT — no valid setup")

        st.caption(
            "Signal is from the last uploaded candle, "
            "not the live market. Use completed candles."
        )

        st.subheader("🧪 Independent Backtest")
        split = int(len(df) * 0.7)
        train = df.iloc[:split].copy()
        test = df.iloc[split:].copy()

        st.write(
            "Training period:",
            str(train.index.min()),
            "to", str(train.index.max())
        )
        st.write(
            "Test period:",
            str(test.index.min()),
            "to", str(test.index.max())
        )

        if st.button("▶ Run Backtest", type="primary"):
            # Warm-up indicators from earlier data.
            # Test execution starts at index 200.
            train_trades = backtest(
                train, balance, daily_limit,
                risk_pct, spread, slippage,
                commission, lot_contract
            )
            test_input = df.iloc[
                max(0, split - 200):
            ].copy()
            test_trades = backtest(
                test_input, balance, daily_limit,
                risk_pct, spread, slippage,
                commission, lot_contract
            )

            # Keep only test-period entries
            test_trades = test_trades[
                test_trades["entry_time"] >=
                df.index[split]
            ].copy() if not test_trades.empty else test_trades

            for label, trades in [
                ("Training", train_trades),
                ("Independent test", test_trades)
            ]:
                st.markdown(f"### {label}")
                if trades.empty:
                    st.warning("No trades found.")
                    continue

                total = len(trades)
                wins = (trades["pnl"] > 0).sum()
                net = trades["pnl"].sum()
                winrate = 100 * wins / total
                eq = balance + trades["pnl"].cumsum()
                peak = pd.concat([
                    pd.Series([balance]), eq
                ], ignore_index=True).cummax()
                path = pd.concat([
                    pd.Series([balance]), eq
                ], ignore_index=True)
                dd = (peak - path).max()

                c1, c2, c3 = st.columns(3)
                c1.metric("Trades", total)
                c2.metric("Win rate", f"{winrate:.1f}%")
                c3.metric("Net P&L", f"${net:.2f}")
                st.metric("Max drawdown ($)", f"{dd:.2f}")
                st.dataframe(trades, use_container_width=True)
                st.download_button(
                    f"Download {label} trades CSV",
                    trades.to_csv(index=False),
                    file_name=(
                        "Tuesday_AI_V32_" +
                        label.replace(" ", "_") + ".csv"
                    ),
                    mime="text/csv"
                )

            st.info(
                "Backtest is experimental. It assumes "
                "historical candle fills and does not "
                "verify prop-firm intraday equity rules, "
                "market gaps or broker execution."
            )

    except Exception as e:
        st.error(f"CSV processing error: {e}")

st.caption(
    "Tuesday AI V32 | Manual research assistant "
    "| No automatic trading"
)
