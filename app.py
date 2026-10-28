
import io
import zipfile
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Tuesday AI V39",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Tuesday AI V39")
st.caption("XAUUSD M1 + M5 | Demo Scalping Assistant")
st.warning("Demo/backtest only. No automatic live orders or guaranteed profit.")

# ---------------- SETTINGS ----------------

st.sidebar.header("⚙️ Demo Account")

balance = st.sidebar.number_input(
    "Starting balance ($)",
    min_value=100.0,
    value=2500.0,
    step=100.0
)

risk_usd = st.sidebar.number_input(
    "Maximum risk per trade ($)",
    min_value=1.0,
    value=10.0,
    step=1.0
)

daily_limit = st.sidebar.number_input(
    "Maximum daily loss ($)",
    min_value=1.0,
    value=40.0,
    step=1.0
)

spread = st.sidebar.number_input(
    "Spread in gold price ($)",
    min_value=0.0,
    value=0.30,
    step=0.05
)

commission = st.sidebar.number_input(
    "Round-trip commission per 1.00 lot ($)",
    min_value=0.0,
    value=7.0,
    step=0.5
)

min_lot = 0.01
lot_step = 0.01
contract_size = 100.0

st.sidebar.info(
    "Assumption: 1.00 lot = 100 oz. "
    "Verify with your broker."
)

# ---------------- DATA LOADING ----------------

def read_csv_bytes(raw):
    try:
        df = pd.read_csv(
            io.BytesIO(raw),
            sep=None,
            engine="python",
            header=None
        )
    except Exception:
        return None

    if df.empty:
        return None

    first = str(df.iloc[0, 0]).lower()

    if "date" in first or "time" in first:
        df = pd.read_csv(
            io.BytesIO(raw),
            sep=None,
            engine="python"
        )
        df.columns = [
            str(c).strip().lower()
            for c in df.columns
        ]

        if "datetime" in df.columns:
            dt = pd.to_datetime(
                df["datetime"],
                errors="coerce"
            )
        elif "date" in df.columns and "time" in df.columns:
            dt = pd.to_datetime(
                df["date"].astype(str) + " " +
                df["time"].astype(str),
                errors="coerce"
            )
        elif "date" in df.columns:
            dt = pd.to_datetime(
                df["date"],
                errors="coerce"
            )
        else:
            return None

        required = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in required):
            return None

        out = df[required].copy()
        out.index = dt

    else:
        if df.shape[1] < 5:
            return None

        # HistData format:
        # YYYYMMDD HHMMSS,OPEN,HIGH,LOW,CLOSE,VOLUME
        dt = pd.to_datetime(
            df.iloc[:, 0].astype(str).str.strip(),
            format="%Y%m%d %H%M%S",
            errors="coerce"
        )

        if dt.isna().all() and df.shape[1] >= 6:
            dt = pd.to_datetime(
                df.iloc[:, 0].astype(str) + " " +
                df.iloc[:, 1].astype(str),
                errors="coerce"
            )
            price_start = 2
        else:
            price_start = 1

        if df.shape[1] < price_start + 4:
            return None

        out = df.iloc[
            :, price_start:price_start + 4
        ].copy()

        out.columns = ["open", "high", "low", "close"]
        out.index = dt

    for c in ["open", "high", "low", "close"]:
        out[c] = pd.to_numeric(
            out[c],
            errors="coerce"
        )

    out = out[
        ~out.index.isna()
    ].dropna()

    out = out.sort_index()
    out = out[
        ~out.index.duplicated(keep="first")
    ]

    return out


def load_files(files):
    frames = []

    for file in files:
        raw = file.getvalue()

        if file.name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    for name in z.namelist():
                        if name.lower().endswith((".csv", ".txt")):
                            frame = read_csv_bytes(
                                z.read(name)
                            )
                            if frame is not None:
                                frames.append(frame)
            except Exception as e:
                st.error(f"ZIP error: {e}")

        else:
            frame = read_csv_bytes(raw)
            if frame is not None:
                frames.append(frame)

    if not frames:
        return None

    data = pd.concat(frames)
    data = data.sort_index()
    data = data[
        ~data.index.duplicated(keep="first")
    ]

    return data


# ---------------- INDICATORS ----------------

def indicators(df):
    df = df.copy()

    df["ema20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    previous_close = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(14).mean()

    df["signal"] = "WAIT"

    buy = (
        (df["ema20"] > df["ema50"]) &
        (df["close"] > df["ema20"]) &
        (df["low"] <= df["ema20"]) &
        (df["close"] > df["open"])
    )

    sell = (
        (df["ema20"] < df["ema50"]) &
        (df["close"] < df["ema20"]) &
        (df["high"] >= df["ema20"]) &
        (df["close"] < df["open"])
    )

    df.loc[buy, "signal"] = "BUY"
    df.loc[sell, "signal"] = "SELL"

    return df


def to_m5(df):
    return df.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()


# ---------------- BACKTEST ----------------

def backtest(df, timeframe):
    df = indicators(df)

    trades = []
    rejected = []
    equity = balance
    daily_pnl = {}
    last_exit = -1

    for i in range(60, len(df) - 1):

        if i <= last_exit:
            continue

        row = df.iloc[i]
        signal = row["signal"]

        if signal == "WAIT":
            continue

        entry_bar = df.iloc[i + 1]
        entry_time = df.index[i + 1]

        day = entry_time.date()
        day_pnl = daily_pnl.get(day, 0.0)

        if day_pnl <= -daily_limit:
            rejected.append({
                "time": entry_time,
                "reason": "Daily loss limit"
            })
            continue

        atr = row["atr"]

        if pd.isna(atr) or atr <= 0:
            rejected.append({
                "time": entry_time,
                "reason": "Invalid ATR"
            })
            continue

        # Entry at next candle open, with spread
        if signal == "BUY":
            entry = entry_bar["open"] + spread / 2
            sl = entry - atr * 1.5
            tp = entry + atr * 3.0
        else:
            entry = entry_bar["open"] - spread / 2
            sl = entry + atr * 1.5
            tp = entry - atr * 3.0

        sl_distance = abs(entry - sl)

        remaining_daily = daily_limit + day_pnl
        allowed_risk = min(
            risk_usd,
            remaining_daily,
            equity
        )

        if allowed_risk <= 0:
            rejected.append({
                "time": entry_time,
                "reason": "Risk limit"
            })
            continue

        min_risk = (
            sl_distance * contract_size * min_lot
            + commission * min_lot
        )

        if min_risk > allowed_risk:
            rejected.append({
                "time": entry_time,
                "reason": "Minimum lot exceeds risk"
            })
            continue

        raw_lot = allowed_risk / (
            sl_distance * contract_size
            + commission
        )

        lot = np.floor(
            raw_lot / lot_step
        ) * lot_step

        lot = round(lot, 2)

        if lot < min_lot:
            rejected.append({
                "time": entry_time,
                "reason": "Lot below minimum"
            })
            continue

        exit_price = None
        exit_time = None
        result = None

        # Exit search: maximum 20 candles
        end = min(i + 21, len(df))

        for j in range(i + 1, end):
            candle = df.iloc[j]

            if signal == "BUY":
                stop_hit = candle["low"] <= sl
                target_hit = candle["high"] >= tp
            else:
                stop_hit = candle["high"] >= sl
                target_hit = candle["low"] <= tp

            # Conservative: SL first if both hit
            if stop_hit:
                exit_price = sl
                result = "SL"
            elif target_hit:
                exit_price = tp
                result = "TP"

            if result is not None:
                exit_time = df.index[j]
                last_exit = j
                break

        if result is None:
            j = end - 1
            candle = df.iloc[j]

            if signal == "BUY":
                exit_price = candle["close"] - spread / 2
            else:
                exit_price = candle["close"] + spread / 2

            exit_time = df.index[j]
            last_exit = j
            result = "TIME"

        if signal == "BUY":
            gross = (
                exit_price - entry
            ) * contract_size * lot
        else:
            gross = (
                entry - exit_price
            ) * contract_size * lot

        fee = commission * lot
        pnl = gross - fee

        equity += pnl

        exit_day = exit_time.date()
        daily_pnl[exit_day] = (
            daily_pnl.get(exit_day, 0.0) + pnl
        )

        trades.append({
            "timeframe": timeframe,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "signal": signal,
            "entry": round(entry, 3),
            "sl": round(sl, 3),
            "tp": round(tp, 3),
            "lot": lot,
            "result": result,
            "pnl": round(pnl, 2),
            "equity": round(equity, 2)
        })

        if equity <= 0:
            break

    return pd.DataFrame(trades), pd.DataFrame(rejected)


# ---------------- USER INTERFACE ----------------

st.subheader("📂 Upload XAUUSD Historical Data")

files = st.file_uploader(
    "Upload September + October M1 ZIP / CSV",
    type=["zip", "csv", "txt"],
    accept_multiple_files=True
)

if files:
    data = load_files(files)

    if data is None or data.empty:
        st.error("Valid OHLC data not found.")
        st.stop()

    st.success(
        f"Loaded {len(data):,} candles"
    )

    st.write(
        "Date range:",
        data.index.min(),
        "to",
        data.index.max()
    )

    # Detect data interval
    intervals = data.index.to_series().diff()
    median_interval = intervals.median()

    if median_interval > pd.Timedelta(minutes=1):
        st.warning(
            "Input may not be M1 data. "
            "Check uploaded file."
        )

    m1 = data.copy()
    m5 = to_m5(m1)

    st.subheader("📊 Latest Signals")

    c1, c2 = st.columns(2)

    for column, name, frame in [
        (c1, "M1", m1),
        (c2, "M5", m5)
    ]:
        result = indicators(frame)
        latest = result.iloc[-1]

        with column:
            st.metric(
                f"{name} Signal",
                latest["signal"]
            )
            st.write(
                "Latest close:",
                round(latest["close"], 2)
            )

    if st.button("🚀 Run V39 Backtest"):

        with st.spinner("Backtesting M1 and M5..."):
            m1_trades, m1_rejected = backtest(m1, "M1")
            m5_trades, m5_rejected = backtest(m5, "M5")

        summary = []

        for name, trades, rejected in [
            ("M1", m1_trades, m1_rejected),
            ("M5", m5_trades, m5_rejected)
        ]:
            if trades.empty:
                summary.append({
                    "Timeframe": name,
                    "Trades": 0,
                    "Win Rate %": 0,
                    "Net PnL $": 0,
                    "Final Balance $": balance,
                    "Rejected": len(rejected)
                })
                continue

            wins = (trades["pnl"] > 0).sum()

            summary.append({
                "Timeframe": name,
                "Trades": len(trades),
                "Win Rate %": round(
                    wins / len(trades) * 100, 2
                ),
                "Net PnL $": round(
                    trades["pnl"].sum(), 2
                ),
                "Final Balance $": round(
                    balance + trades["pnl"].sum(), 2
                ),
                "Rejected": len(rejected)
            })

        st.subheader("📈 M1 vs M5 Results")
        st.dataframe(
            pd.DataFrame(summary),
            use_container_width=True
        )

        for name, trades, rejected in [
            ("M1", m1_trades, m1_rejected),
            ("M5", m5_trades, m5_rejected)
        ]:
            st.subheader(f"{name} Trade Audit")

            if not trades.empty:
                st.dataframe(trades)

                st.download_button(
                    f"Download V39 {name} Trades",
                    trades.to_csv(index=False),
                    file_name=f"V39_{name}_trades.csv",
                    mime="text/csv"
                )

                st.line_chart(
                    trades.set_index("exit_time")["equity"]
                )
            else:
                st.warning(
                    f"No executed {name} trades."
                )

            if not rejected.empty:
                st.download_button(
                    f"Download V39 {name} Rejected",
                    rejected.to_csv(index=False),
                    file_name=f"V39_{name}_rejected.csv",
                    mime="text/csv"
                )

        st.info(
            "Backtest is experimental. "
            "Spread, commission, intrabar execution, "
            "slippage and broker specifications "
            "can change real results. "
            "This is not independent validation."
        )

else:
    st.info(
        "Upload your XAUUSD M1 historical files to begin."
    )
