
import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Tuesday AI V41",
    layout="wide"
)

st.title("Tuesday AI V41")
st.caption("XAUUSD M1 + M5 | Audit | Independent Backtest")
st.warning("Research only. No live trading or broker orders.")

with st.sidebar:
    st.header("Account & Risk")
    balance = st.number_input(
        "Starting balance ($)", min_value=1.0,
        value=2500.0
    )
    daily_limit = st.number_input(
        "Daily loss limit ($)", min_value=1.0,
        value=40.0
    )
    risk_pct = st.number_input(
        "Risk per trade (%)", min_value=0.05,
        max_value=2.0, value=0.25, step=0.05
    )
    min_lot = st.number_input(
        "Minimum lot", min_value=0.01,
        value=0.01, step=0.01
    )
    lot_step = st.number_input(
        "Lot step", min_value=0.01,
        value=0.01, step=0.01
    )
    contract = st.number_input(
        "Ounces per lot", min_value=1.0,
        value=100.0
    )
    spread = st.number_input(
        "Spread ($/oz)", min_value=0.0,
        value=0.30, step=0.05
    )
    slippage = st.number_input(
        "Slippage per side ($/oz)", min_value=0.0,
        value=0.10, step=0.05
    )
    commission = st.number_input(
        "Round-trip commission ($/lot)",
        min_value=0.0, value=7.0
    )
    st.caption("Use actual broker costs and contract size.")

def normalize(df):
    df = df.copy()
    df.columns = [
        str(c).strip().lower()
        .replace("<", "").replace(">", "")
        .replace(" ", "_")
        for c in df.columns
    ]
    return df

def read_candles(raw):
    text = raw.decode("utf-8-sig", errors="replace")
    first = text.splitlines()[0]
    sep = ";" if first.count(";") > first.count(",") else ","

    has_header = any(
        x in first.lower()
        for x in ["open", "high", "close", "date", "time"]
    )

    if has_header:
        df = normalize(pd.read_csv(io.StringIO(text), sep=sep))
    else:
        df = pd.read_csv(
            io.StringIO(text), sep=sep, header=None
        )
        if df.shape[1] < 5:
            raise ValueError("Invalid OHLC file")
        names = ["time", "open", "high", "low", "close", "volume"]
        df.columns = names[:df.shape[1]]

    if "datetime" in df:
        df = df.rename(columns={"datetime": "time"})
    if "timestamp" in df:
        df = df.rename(columns={"timestamp": "time"})

    if "date" in df and "time" in df:
        sample = str(df["time"].iloc[0])
        if ":" in sample and len(sample) <= 8:
            df["time"] = (
                df["date"].astype(str) + " " +
                df["time"].astype(str)
            )

    if "time" not in df and "date" in df:
        df["time"] = df["date"]

    required = ["time", "open", "high", "low", "close"]
    if any(c not in df for c in required):
        raise ValueError("Missing OHLC columns")

    t = df["time"].astype(str).str.strip()
    dt = pd.to_datetime(
        t, format="%Y%m%d %H%M%S", errors="coerce"
    )
    dt = dt.fillna(pd.to_datetime(t, errors="coerce"))
    df["time"] = dt

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required)
    df = df[
        (df.high >= df.low) &
        (df.high >= df[["open", "close"]].max(axis=1)) &
        (df.low <= df[["open", "close"]].min(axis=1))
    ]
    return df[required].sort_values("time").drop_duplicates("time")

def load_files(files):
    frames = []
    errors = []
    for f in files:
        try:
            raw = f.getvalue()
            if f.name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    for name in z.namelist():
                        if name.lower().endswith((".csv", ".txt")):
                            frames.append(read_candles(z.read(name)))
            else:
                frames.append(read_candles(raw))
        except Exception as e:
            errors.append(f"{f.name}: {e}")

    for e in errors:
        st.error(e)

    if not frames:
        return pd.DataFrame()

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

def make_signals(data):
    d = data.copy().set_index("time")

    m5 = d.resample(
        "5min", closed="left", label="right"
    ).agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last"
    }).dropna()

    m5["ema50"] = m5.close.ewm(
        span=50, adjust=False
    ).mean()
    m5["ema200"] = m5.close.ewm(
        span=200, adjust=False
    ).mean()

    m5 = m5.rename(columns={"close": "m5_close"})

    d = pd.merge_asof(
        d.reset_index().sort_values("time"),
        m5[["m5_close", "ema50", "ema200"]]
        .reset_index().sort_values("time"),
        on="time", direction="backward"
    )

    d["ema20"] = d.close.ewm(
        span=20, adjust=False
    ).mean()

    d["prior_high"] = d.high.shift(1).rolling(20).max()
    d["prior_low"] = d.low.shift(1).rolling(20).min()

    tr = pd.concat([
        d.high - d.low,
        (d.high - d.close.shift(1)).abs(),
        (d.low - d.close.shift(1)).abs()
    ], axis=1).max(axis=1)

    d["atr"] = tr.rolling(14).mean()

    up = (
        (d.m5_close > d.ema50) &
        (d.ema50 > d.ema200)
    )
    down = (
        (d.m5_close < d.ema50) &
        (d.ema50 < d.ema200)
    )

    d["signal"] = "WAIT"

    d.loc[
        up & (d.close > d.prior_high) &
        (d.close > d.ema20), "signal"
    ] = "BUY"

    d.loc[
        down & (d.close < d.prior_low) &
        (d.close < d.ema20), "signal"
    ] = "SELL"

    d.loc[d.index < 1000, "signal"] = "WAIT"

    return d

def backtest(data, start_balance, start_time=None):
    d = make_signals(data)
    equity = float(start_balance)
    trades = []
    day_start = {}
    locked = set()
    i = 1000

    while i < len(d) - 1:
        row = d.iloc[i]

        if start_time is not None:
            if row.time < pd.Timestamp(start_time):
                i += 1
                continue

        side = row.signal
        if side not in ("BUY", "SELL"):
            i += 1
            continue

        nxt = d.iloc[i + 1]
        day = nxt.time.date()

        if start_time is not None:
            if nxt.time < pd.Timestamp(start_time):
                i += 1
                continue

        if day in locked:
            i += 1
            continue

        if day not in day_start:
            day_start[day] = equity

        remaining = equity - (
            day_start[day] - daily_limit
        )

        if remaining <= 0:
            locked.add(day)
            i += 1
            continue

        if not np.isfinite(row.atr) or row.atr <= 0:
            i += 1
            continue

        direction = 1 if side == "BUY" else -1
        distance = max(float(row.atr) * 1.5, 0.10)

        # Levels are based on the next candle's
        # simulated entry fill.
        entry_mid = float(nxt.open)
        entry = entry_mid + direction * (
            spread / 2 + slippage
        )

        sl = entry - direction * distance
        tp = entry + direction * distance * 2

        # Reserve conservative risk including costs.
        per_lot_risk = (
            (distance + spread + 2 * slippage) *
            contract + commission
        )
        budget = min(
            equity * risk_pct / 100, remaining
        )
        lot = np.floor(
            budget / per_lot_risk / lot_step
        ) * lot_step
        lot = round(float(lot), 8)

        if lot < min_lot:
            i += 1
            continue

        exit_mid = float(d.iloc[-1].close)
        exit_i = len(d) - 1
        reason = "END_OF_DATA"

        # SL/TP are executable-price levels.
        # Convert candle mid-price to exit-side price.
        for j in range(i + 1, len(d)):
            bar = d.iloc[j]
            exit_adjust = direction * (
                spread / 2 + slippage
            )

            if side == "BUY":
                stop_hit = bar.low - exit_adjust <= sl
                target_hit = bar.high - exit_adjust >= tp
            else:
                stop_hit = bar.high - exit_adjust >= sl
                target_hit = bar.low - exit_adjust <= tp

            if stop_hit:
                exit_mid = sl + exit_adjust
                exit_i = j
                reason = (
                    "BOTH_HIT_STOP_FIRST"
                    if target_hit else "STOP"
                )
                break

            if target_hit:
                exit_mid = tp + exit_adjust
                exit_i = j
                reason = "TARGET"
                break

            if j + 1 < len(d):
                if d.iloc[j + 1].time.date() != day:
                    exit_mid = float(bar.close)
                    exit_i = j
                    reason = "DAY_END"
                    break

        exit_fill = exit_mid - direction * (
            spread / 2 + slippage
        )

        pnl = (
            direction * (exit_fill - entry) *
            contract * lot -
            commission * lot
        )

        equity += pnl

        if equity <= day_start[day] - daily_limit:
            locked.add(day)

        trades.append({
            "entry_time": nxt.time,
            "exit_time": d.iloc[exit_i].time,
            "signal": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "exit": exit_fill,
            "lot": lot,
            "pnl": pnl,
            "equity": equity,
            "reason": reason
        })

        i = exit_i + 1

    return pd.DataFrame(trades), d

def show_report(trades, name):
    st.subheader(name)

    if trades.empty:
        st.warning("No trades generated.")
        return

    a, b, c = st.columns(3)
    a.metric("Trades", len(trades))
    b.metric(
        "Win rate",
        f"{(trades.pnl > 0).mean() * 100:.2f}%"
    )
    c.metric(
        "Net P&L",
        f"${trades.pnl.sum():,.2f}"
    )

    gross_win = trades.loc[
        trades.pnl > 0, "pnl"
    ].sum()
    gross_loss = -trades.loc[
        trades.pnl < 0, "pnl"
    ].sum()

    pf = (
        gross_win / gross_loss
        if gross_loss > 0 else np.nan
    )

    st.write("Profit factor:", round(pf, 3))
    st.write(
        "Maximum drawdown ($):",
        round(
            (
                trades.equity.cummax() -
                trades.equity
            ).max(), 2
        )
    )

    st.line_chart(
        trades.set_index("exit_time")["equity"]
    )
    st.dataframe(trades, use_container_width=True)

    st.download_button(
        "Download " + name + " CSV",
        trades.to_csv(index=False),
        file_name=name.replace(" ", "_") + ".csv",
        mime="text/csv"
    )

st.header("1. V40 Trade Audit")

audit_file = st.file_uploader(
    "Upload Tuesday_AI_V40_audit_trades.csv",
    type=["csv"],
    key="audit"
)

v40 = None

if audit_file is not None:
    try:
        v40 = normalize(pd.read_csv(audit_file))

        if "side" in v40 and "signal" not in v40:
            v40 = v40.rename(columns={"side": "signal"})

        if "pnl" not in v40:
            st.error("P&L column missing.")
            v40 = None
        else:
            v40["pnl"] = pd.to_numeric(
                v40.pnl, errors="coerce"
            )

            st.metric("V40 Trades", len(v40))
            st.metric(
                "V40 Reported Win Rate",
                f"{(v40.pnl > 0).mean() * 100:.2f}%"
            )
            st.metric(
                "V40 Reported P&L",
                f"${v40.pnl.sum():,.2f}"
            )

            st.warning(
                "Reported results are not verified "
                "until candle-path and signal audit."
            )

    except Exception as e:
        st.error(str(e))
        v40 = None

st.divider()
st.header("2. Upload XAUUSD M1 Historical Data")

files = st.file_uploader(
    "September + October 2026 ZIP/CSV",
    type=["zip", "csv", "txt"],
    accept_multiple_files=True,
    key="history"
)

if files:
    data = load_files(files)

    if not data.empty:
        st.success(
            f"{len(data):,} candles loaded"
        )
        st.write(
            "Data coverage:",
            data.time.min(),
            "to",
            data.time.max()
        )

        gaps = data.time.diff().dt.total_seconds() / 60
        st.write(
            "Gaps greater than 1 minute:",
            int((gaps > 1).sum())
        )

        if v40 is not None:
            needed = [
                "entry_time", "exit_time",
                "signal", "entry", "sl", "tp"
            ]

            if all(c in v40 for c in needed):
                st.subheader("V40 Candle Path Audit")
                check = v40.copy()

                check["entry_time"] = pd.to_datetime(
                    check.entry_time, errors="coerce"
                )
                check["exit_time"] = pd.to_datetime(
                    check.exit_time, errors="coerce"
                )

                for c in ["entry", "sl", "tp"]:
                    check[c] = pd.to_numeric(
                        check[c], errors="coerce"
                    )

                results = []
                candles = data.set_index("time")

                for _, tr in check.iterrows():
                    if pd.isna(tr.entry_time) or pd.isna(tr.exit_time):
                        results.append("INVALID_TIME")
                        continue

                    bars = candles.loc[
                        (candles.index >= tr.entry_time) &
                        (candles.index <= tr.exit_time)
                    ]

                    if bars.empty:
                        results.append("MISSING_DATA")
                        continue

                    if tr.signal == "BUY":
                        sl_hit = bars.low <= tr.sl
                        tp_hit = bars.high >= tr.tp
                    elif tr.signal == "SELL":
                        sl_hit = bars.high >= tr.sl
                        tp_hit = bars.low <= tr.tp
                    else:
                        results.append("INVALID_SIDE")
                        continue

                    sl_indices = np.flatnonzero(sl_hit.to_numpy())
                    tp_indices = np.flatnonzero(tp_hit.to_numpy())

                    if not len(sl_indices) and not len(tp_indices):
                        status = "NO_LEVEL_HIT"
                    elif not len(sl_indices):
                        status = "TP_FIRST"
                    elif not len(tp_indices):
                        status = "SL_FIRST"
                    elif sl_indices[0] == tp_indices[0]:
                        status = "SAME_BAR_AMBIGUOUS"
                    elif sl_indices[0] < tp_indices[0]:
                        status = "SL_FIRST"
                    else:
                        status = "TP_FIRST"

                    results.append(status)

                check["path_check"] = results
                st.dataframe(
                    check.path_check.value_counts()
                )
                st.download_button(
                    "Download V41 V40 Audit",
                    check.to_csv(index=False),
                    "V41_V40_audit.csv",
                    mime="text/csv"
                )
                st.caption(
                    "Candle path is diagnostic, not "
                    "proof of actual broker execution."
                )

        st.divider()
        st.header("3. Independent Backtest")

        sep = data[
            (data.time >= "2026-09-01") &
            (data.time < "2026-10-01")
        ].copy()

        octo = data[
            (data.time >= "2026-10-01") &
            (data.time < "2026-11-01")
        ].copy()

        st.write("September M1 candles:", len(sep))
        st.write("October M1 candles:", len(octo))

        if st.button(
            "Run V41 Independent Backtest",
            type="primary"
        ):
            if len(sep) < 1100 or len(octo) < 1100:
                st.error(
                    "Need more September and October M1 data."
                )
            else:
                with st.spinner("Running tests..."):
                    sep_trades, sep_signals = backtest(
                        sep, balance
                    )

                    # September data is used only for
                    # indicator warm-up before October.
                    # October starts with fresh balance.
                    combined = data[
                        (data.time >= "2026-09-01") &
                        (data.time < "2026-11-01")
                    ].reset_index(drop=True)

                    oct_trades, oct_signals = backtest(
                        combined,
                        balance,
                        start_time="2026-10-01"
                    )

                show_report(
                    sep_trades,
                    "September Development"
                )
                show_report(
                    oct_trades,
                    "October Independent"
                )

                st.subheader("Latest historical signal")
                latest = oct_signals.iloc[-1]
                st.write("Signal:", latest.signal)
                st.write("Candle time:", latest.time)

                st.warning(
                    "Historical signal only. "
                    "Not a live trading instruction."
                )

st.divider()
st.header("4. V41 Safety Summary")
st.info(
    "BUY/SELL/WAIT signals are experimental. "
    "No live execution. The $40 loss lock "
    "limits planned risk, but gaps or slippage "
    "can cause larger realized losses."
)
