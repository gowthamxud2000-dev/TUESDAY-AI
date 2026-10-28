
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(
    page_title="Tuesday AI V37",
    page_icon="📈",
    layout="wide"
)

st.title("Tuesday AI V37 — XAUUSD M1 Scalping")
st.caption("Experimental backtest assistant. Manual trading only.")

with st.sidebar:
    balance = st.number_input(
        "Starting balance ($)", min_value=1.0, value=100.0
    )
    risk_pct = st.number_input(
        "Risk per trade (%)", 0.1, 5.0, 1.0, 0.1
    )
    min_lot = st.number_input(
        "Minimum lot", min_value=0.001,
        value=0.01, format="%.3f"
    )
    contract = st.number_input(
        "Ounces per lot (check broker)",
        min_value=1.0, value=100.0
    )
    spread = st.number_input(
        "Spread (price units)",
        min_value=0.0, value=0.30
    )
    slippage = st.number_input(
        "Slippage per side",
        min_value=0.0, value=0.10
    )
    commission = st.number_input(
        "Round-trip commission per lot ($)",
        min_value=0.0, value=7.0
    )
    rr = st.number_input(
        "Reward/Risk", min_value=1.0, value=1.5
    )
    max_daily_loss = st.number_input(
        "Daily loss cap ($)", min_value=0.1, value=3.0
    )
    max_trades = st.number_input(
        "Max trades per day", min_value=1, value=3
    )

st.info("Upload XAUUSD M1 CSV. M5 and M15 will be generated automatically.")

upload = st.file_uploader(
    "Upload M1 CSV", type=["csv", "txt"]
)

def load_data(file):
    raw = file.getvalue()
    df = None

    for sep in [",", ";", "\t"]:
        try:
            temp = pd.read_csv(BytesIO(raw), sep=sep)
            cols = {
                str(c).lower().strip(): c
                for c in temp.columns
            }

            if all(
                k in cols
                for k in ["open", "high", "low", "close"]
            ):
                df = temp.rename(
                    columns={
                        cols[k]: k
                        for k in ["open", "high", "low", "close"]
                    }
                )

                if "datetime" in cols:
                    dates = df[cols["datetime"]]
                elif "timestamp" in cols:
                    dates = df[cols["timestamp"]]
                elif "date" in cols:
                    dates = df[cols["date"]].astype(str)
                    if "time" in cols:
                        dates += " " + df[cols["time"]].astype(str)
                else:
                    raise ValueError("Date/Time missing")

                df["dt"] = pd.to_datetime(
                    dates, errors="coerce"
                )
                break
        except Exception:
            pass

    if df is None:
        for sep in [";", ","]:
            try:
                temp = pd.read_csv(
                    BytesIO(raw), sep=sep, header=None
                )
                if temp.shape[1] >= 5:
                    df = temp.iloc[:, :5].copy()
                    df.columns = [
                        "dt", "open", "high", "low", "close"
                    ]
                    df["dt"] = pd.to_datetime(
                        df["dt"].astype(str).str.strip(),
                        format="%Y%m%d %H%M%S",
                        errors="coerce"
                    )
                    if df["dt"].notna().sum() > 10:
                        break
            except Exception:
                pass

    if df is None:
        raise ValueError("CSV format not recognized")

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.dropna()
        .drop_duplicates("dt")
        .sort_values("dt")
        .set_index("dt")
    )

    if len(df) < 500:
        raise ValueError("Minimum 500 M1 candles needed")

    return df


def resample(df, rule):
    return df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()


def signals(m1):
    m5 = resample(m1, "5min")
    m15 = resample(m1, "15min")

    m15["ema50"] = m15.close.ewm(
        span=50, adjust=False
    ).mean()
    m15["ema200"] = m15.close.ewm(
        span=200, adjust=False
    ).mean()

    m15["trend"] = np.where(
        m15.ema50 > m15.ema200, 1, -1
    )

    m5["prior_high"] = (
        m5.high.shift(1).rolling(10).max()
    )
    m5["prior_low"] = (
        m5.low.shift(1).rolling(10).min()
    )

    m5["bos"] = np.select(
        [
            m5.close > m5.prior_high,
            m5.close < m5.prior_low
        ],
        [1, -1],
        default=0
    )

    trend = m15["trend"].shift(1).reindex(
        m1.index, method="ffill"
    )
    bos = m5["bos"].shift(1).reindex(
        m1.index, method="ffill"
    )

    prev_low = m1.low.shift(1).rolling(10).min()
    prev_high = m1.high.shift(1).rolling(10).max()

    buy = (
        (trend == 1) &
        (bos == 1) &
        (m1.low < prev_low) &
        (m1.close > prev_low) &
        (m1.close > m1.open)
    )

    sell = (
        (trend == -1) &
        (bos == -1) &
        (m1.high > prev_high) &
        (m1.close < prev_high) &
        (m1.close < m1.open)
    )

    out = m1.copy()
    out["signal"] = np.select(
        [buy, sell], [1, -1], default=0
    )
    return out


def backtest(data):
    trades = []
    rejected = []
    equity = balance
    current_day = None
    day_start = balance
    count = 0
    i = 220

    while i < len(data) - 1:
        candle = data.iloc[i]
        ts = data.index[i]

        if current_day != ts.date():
            current_day = ts.date()
            day_start = equity
            count = 0

        side = int(candle.signal)

        if side == 0:
            i += 1
            continue

        if count >= max_trades or day_start - equity >= max_daily_loss:
            rejected.append((ts, "Daily limit"))
            i += 1
            continue

        entry_raw = float(data.iloc[i + 1].open)

        stop = (
            float(candle.low - 0.10)
            if side == 1
            else float(candle.high + 0.10)
        )

        entry = entry_raw + side * (
            spread / 2 + slippage
        )
        stop_fill = stop - side * (
            spread / 2 + slippage
        )

        risk_price = (entry - stop_fill) * side

        if risk_price <= 0:
            rejected.append((ts, "Invalid stop"))
            i += 1
            continue

        risk_dollars = (
            risk_price * contract * min_lot
            + commission * min_lot
        )

        remaining_daily = max_daily_loss - (
            day_start - equity
        )

        if risk_dollars > min(
            equity * risk_pct / 100,
            remaining_daily
        ):
            rejected.append((ts, "Risk exceeds limit"))
            i += 1
            continue

        target = entry + side * rr * risk_price

        exit_index = None
        exit_price = None
        outcome = None

        for j in range(i + 1, len(data)):
            bar = data.iloc[j]

            if side == 1:
                hit_sl = bar.low <= stop
                hit_tp = bar.high >= target + spread / 2
            else:
                hit_sl = bar.high >= stop
                hit_tp = bar.low <= target - spread / 2

            if hit_sl or hit_tp:
                outcome = "SL" if hit_sl else "TP"
                exit_index = j
                exit_price = (
                    stop_fill
                    if hit_sl
                    else target - side * (
                        spread / 2 + slippage
                    )
                )
                break

        if exit_index is None:
            exit_index = len(data) - 1
            outcome = "END"
            exit_price = (
                float(data.iloc[-1].close)
                - side * (spread / 2 + slippage)
            )

        pnl = (
            (exit_price - entry)
            * side * contract * min_lot
            - commission * min_lot
        )

        equity += pnl
        count += 1

        trades.append({
            "time": ts,
            "side": "BUY" if side == 1 else "SELL",
            "entry": round(entry, 3),
            "stop": round(stop, 3),
            "target": round(target, 3),
            "exit_time": data.index[exit_index],
            "exit_reason": outcome,
            "pnl": round(pnl, 2),
            "equity": round(equity, 2)
        })

        i = exit_index + 1

    return (
        pd.DataFrame(trades),
        pd.DataFrame(rejected, columns=["time", "reason"])
    )


if upload:
    try:
        df = load_data(upload)
        st.write(f"M1 candles: {len(df):,}")

        gaps = (
            df.index.to_series().diff()
            > pd.Timedelta(minutes=1)
        ).sum()

        if gaps:
            st.warning(
                f"{gaps} gaps detected. Check data quality."
            )

        data = signals(df)
        latest = data.iloc[-2]

        signal = {
            1: "🟢 BUY Setup",
            -1: "🔴 SELL Setup",
            0: "🟡 WAIT"
        }[int(latest.signal)]

        st.subheader(signal)
        st.caption("Historical signal, not live market data.")

        trades, rejected = backtest(data)

        if len(trades):
            st.metric("Trades", len(trades))
            st.metric(
                "Net P/L",
                f"${trades.pnl.sum():.2f}"
            )
            st.metric(
                "Win Rate",
                f"{(trades.pnl > 0).mean() * 100:.1f}%"
            )
            st.metric(
                "Ending Balance",
                f"${trades.equity.iloc[-1]:.2f}"
            )

            st.line_chart(
                trades.set_index("exit_time").equity
            )
            st.dataframe(trades)

            st.download_button(
                "Download V37 Trades",
                trades.to_csv(index=False),
                "V37_trades.csv",
                "text/csv"
            )
        else:
            st.warning("No trades passed risk rules. WAIT.")

        st.write("Rejected setups:", len(rejected))

        if len(rejected):
            st.dataframe(rejected.reason.value_counts())
            st.download_button(
                "Download V37 Rejected",
                rejected.to_csv(index=False),
                "V37_rejected.csv",
                "text/csv"
            )

        st.warning(
            "Experimental strategy. M1 OHLC backtesting "
            "cannot reproduce real tick execution. "
            "Validate on unseen data before demo trading."
        )

    except Exception as e:
        st.error(f"CSV error: {e}")
else:
    st.write("Upload XAUUSD M1 CSV to start.")
