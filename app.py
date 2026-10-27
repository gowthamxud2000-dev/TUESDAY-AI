
import streamlit as st

st.set_page_config(
    page_title="Tuesday AI",
    page_icon="🥇",
    layout="wide"
)

st.title("🥇 Tuesday AI")
st.subheader("Your XAUUSD Gold Trading Assistant")

st.success("Welcome to your trading workspace!")

st.warning(
    "Starter version: no live market feed or verified signals yet. "
    "Always check your risk before trading."
)

st.header("📊 Gold Price")
price = st.number_input(
    "Enter current XAUUSD price",
    min_value=0.0,
    value=4000.0,
    step=1.0
)
st.write(f"Reference price: {price:.2f}")

st.header("📚 Upload Trading Books")
books = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if books:
    st.success(f"{len(books)} PDF file(s) selected.")

st.header("🎯 Risk : Reward Calculator")

entry = st.number_input("Entry price", min_value=0.0, value=4000.0)
stop = st.number_input("Stop-loss price", min_value=0.0, value=3990.0)
target = st.number_input("Take-profit price", min_value=0.0, value=4020.0)

risk = abs(entry - stop)
reward = abs(target - entry)

if risk > 0:
    st.metric("Risk : Reward", f"1 : {reward / risk:.2f}")
else:
    st.info("Enter a different stop-loss price.")

st.header("🤖 Signal Status")
st.info(
    "WAIT — Live data and strategy validation are not connected yet. "
    "This is not a live trading signal."
)

st.caption("Tuesday AI | Manual trading only | No automatic order execution")


# TUESDAY AI V2 - PDF Reading Engine
import pymupdf

st.header("📚 PDF Strategy Reader")

if books:
    for book in books:
        st.subheader(book.name)
        try:
            pdf = pymupdf.open(stream=book.getvalue(), filetype="pdf")
            extracted_text = "\n".join(
                page.get_text() for page in pdf
            )
            pdf.close()

            if extracted_text.strip():
                st.success("PDF text extracted successfully!")
                st.text_area(
                    "Extracted trading strategy content",
                    extracted_text,
                    height=300,
                    key=book.name
                )
            else:
                st.warning("No readable text found in this PDF.")
        except Exception as e:
            st.error(f"Could not read PDF: {e}")

# TUESDAY AI V3 - Strategy Analyzer

st.header("🧠 Trading Strategy Analyzer")

if books:
    for book in books:
        st.subheader(f"Analysis: {book.name}")

        try:
            doc = pymupdf.open(
                stream=book.getvalue(),
                filetype="pdf"
            )
            content = "\n".join(
                page.get_text() for page in doc
            ).lower()
            doc.close()

            strategies = {
                "EMA Trend Pullback": [
                    "ema", "pullback"
                ],
                "Breakout and Retest": [
                    "breakout", "retest"
                ],
                "Liquidity Sweep and BOS": [
                    "liquidity", "bos"
                ]
            }

            for name, keywords in strategies.items():
                if all(word in content for word in keywords):
                    st.success(f"✅ Found: {name}")
                else:
                    st.info(f"Not detected: {name}")

            st.caption(
                "Keyword matching only. "
                "No strategy validation or live signals."
            )

        except Exception as e:
            st.error(f"Analysis error: {e}")

# TUESDAY AI V4 - Strategy Rules Extractor
import re

st.header("🎯 PDF Entry, SL & TP Rules")

if books:
    for book in books:
        st.subheader(f"Rules: {book.name}")

        try:
            doc = pymupdf.open(
                stream=book.getvalue(),
                filetype="pdf"
            )
            text = "\n".join(
                page.get_text() for page in doc
            )
            doc.close()

            patterns = {
                "Entry": r"\bentry\b",
                "Stop Loss": r"\bstop[\s-]?loss\b|\bSL\b",
                "Take Profit": r"\btake[\s-]?profit\b|\bTP\b",
                "Confirmation": r"\bconfirm(?:ation|ed)?\b",
            }

            lines = text.splitlines()

            for label, pattern in patterns.items():
                matches = [
                    line.strip()
                    for line in lines
                    if re.search(pattern, line, re.I)
                ]

                st.markdown(f"### {label}")

                if matches:
                    for line in matches[:10]:
                        st.write("• " + line)
                else:
                    st.info("No matching rules found.")

            st.caption(
                "Extracted text only. Rules are not "
                "validated or converted into live signals."
            )

        except Exception as e:
            st.error(f"PDF processing error: {e}")

# TUESDAY AI V5 - XAUUSD Historical Chart Analyzer
import pandas as pd

st.header("📈 XAUUSD Chart Analysis V5")
st.caption("Historical OHLC data only - not live trading signals.")

chart_file = st.file_uploader(
    "Upload XAUUSD OHLC CSV",
    type=["csv"],
    key="v5_chart"
)

if chart_file is not None:
    try:
        df = pd.read_csv(chart_file)
        df.columns = df.columns.str.strip().str.lower()

        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            for c in required:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df = df.dropna(subset=required).copy()

            if len(df) < 200:
                st.warning("At least 200 candles required.")
            elif (
                (df["high"] < df[["open", "close", "low"]].max(axis=1)).any()
                or (df["low"] > df[["open", "close", "high"]].min(axis=1)).any()
            ):
                st.error("Invalid OHLC candle values.")
            else:
                df["ema50"] = df["close"].ewm(
                    span=50, adjust=False
                ).mean()
                df["ema200"] = df["close"].ewm(
                    span=200, adjust=False
                ).mean()

                latest = df.iloc[-1]
                price = float(latest["close"])
                ema50 = float(latest["ema50"])
                ema200 = float(latest["ema200"])

                if price > ema50 > ema200:
                    st.success("Trend: BULLISH")
                elif price < ema50 < ema200:
                    st.error("Trend: BEARISH")
                else:
                    st.warning("Trend: MIXED / WAIT")

                st.metric("Last CSV Close", f"{price:.2f}")
                st.metric("EMA 50", f"{ema50:.2f}")
                st.metric("EMA 200", f"{ema200:.2f}")

                st.line_chart(
                    df[["close", "ema50", "ema200"]].tail(100)
                )
                st.info(
                    "Trend classification only. No verified "
                    "BUY/SELL entry, SL or TP generated."
                )
    except Exception as e:
        st.error(f"Chart analysis error: {e}")
  
# TUESDAY AI V6 - Experimental Signal Engine
st.header("🤖 XAUUSD Signal Engine V6")
st.caption("Historical CSV analysis only. Not live trading advice.")

if chart_file is not None:
    try:
        chart_file.seek(0)
        data = pd.read_csv(chart_file)
        data.columns = data.columns.str.strip().str.lower()

        required = ["open", "high", "low", "close"]

        if not all(c in data.columns for c in required):
            st.error("CSV must contain open, high, low, close.")
        else:
            for c in required:
                data[c] = pd.to_numeric(data[c], errors="coerce")

            data = data.dropna(subset=required).copy()

            valid = (
                (data["high"] >= data[["open", "close", "low"]].max(axis=1))
                & (data["low"] <= data[["open", "close", "high"]].min(axis=1))
            )

            if len(data) < 220 or not valid.all():
                st.warning("Need 220 valid OHLC candles.")
            else:
                data["ema50"] = data["close"].ewm(
                    span=50, adjust=False
                ).mean()
                data["ema200"] = data["close"].ewm(
                    span=200, adjust=False
                ).mean()

                # Use previous 20 completed candles for levels
                previous = data.iloc[-21:-1]
                candle = data.iloc[-1]

                resistance = previous["high"].max()
                support = previous["low"].min()

                close = float(candle["close"])
                high = float(candle["high"])
                low = float(candle["low"])
                ema50 = float(candle["ema50"])
                ema200 = float(candle["ema200"])

                bullish = close > ema50 > ema200
                bearish = close < ema50 < ema200

                buy_breakout = bullish and close > resistance
                sell_breakdown = bearish and close < support

                # Pullback must touch EMA50 and close back
                buy_pullback = (
                    bullish and low <= ema50 and close > ema50
                )
                sell_pullback = (
                    bearish and high >= ema50 and close < ema50
                )

                signal = "WAIT"
                setup = "No confirmed setup"
                entry = close
                stop = None
                target = None

                if buy_breakout or buy_pullback:
                    signal = "BUY"
                    setup = (
                        "Breakout" if buy_breakout else "EMA Pullback"
                    )
                    stop = min(low, float(previous["low"].tail(5).min()))
                    risk = entry - stop
                    if risk > 0:
                        target = entry + 2 * risk
                    else:
                        signal = "WAIT"

                elif sell_breakdown or sell_pullback:
                    signal = "SELL"
                    setup = (
                        "Breakdown" if sell_breakdown else "EMA Pullback"
                    )
                    stop = max(high, float(previous["high"].tail(5).max()))
                    risk = stop - entry
                    if risk > 0:
                        target = entry - 2 * risk
                    else:
                        signal = "WAIT"

                st.subheader(f"Historical Signal: {signal}")
                st.write(f"Setup: {setup}")

                if signal != "WAIT" and target is not None:
                    st.metric("Example Entry", f"{entry:.2f}")
                    st.metric("Example SL", f"{stop:.2f}")
                    st.metric("Example TP (2R)", f"{target:.2f}")
                    st.info(
                        "Experimental setup only. No spread, "
                        "slippage, volatility or news filters."
                    )
                else:
                    st.warning("WAIT - No qualifying setup.")

                st.caption(
                    "This engine uses hard-coded EMA and price "
                    "rules, not the uploaded PDF strategies. "
                    "Backtesting is required."
                )

    except Exception as e:
                st.error(f"Signal engine error: {e}")
# TUESDAY AI V7 - PDF Strategy Comparison

st.header("📚 Tuesday AI V7 - PDF Strategy Comparison")

v7_files = st.file_uploader(
    "Upload 2 or more trading strategy PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    key="v7_pdf_upload"
)

strategy_keywords = {
    "EMA Trend Pullback": [
        "ema", "moving average", "pullback"
    ],
    "Breakout and Retest": [
        "breakout", "retest", "resistance", "support"
    ],
    "Liquidity Sweep and BOS": [
        "liquidity", "sweep", "bos", "break of structure"
    ]
}

if v7_files:
    if len(v7_files) < 2:
        st.warning("Upload at least 2 PDFs to compare.")
    else:
        import fitz

        comparison = []

        for pdf_file in v7_files:
            try:
                pdf_file.seek(0)
                doc = fitz.open(
                    stream=pdf_file.read(),
                    filetype="pdf"
                )

                content = " ".join(
                    page.get_text() for page in doc
                ).lower()

                doc.close()

                result = {"PDF": pdf_file.name}

                for strategy, keywords in strategy_keywords.items():
                    matches = sum(
                        1 for word in keywords
                        if word in content
                    )
                    result[strategy] = matches

                comparison.append(result)

            except Exception as e:
                st.error(
                    f"Error reading {pdf_file.name}: {e}"
                )

        if comparison:
            comparison_df = pd.DataFrame(comparison)

            st.subheader("📊 Strategy Comparison")
            st.dataframe(
                comparison_df,
                use_container_width=True
            )

            st.info(
                "Scores count keyword matches, not strategy "
                "accuracy, profitability or win rate."
            )

            st.success("PDF comparison completed!")        

# TUESDAY AI V8 - PDF Trading Rule Extractor

st.header("🧠 Tuesday AI V8 - PDF Rule Extractor")
st.caption("Extracts possible trading rules from PDF text.")

v8_pdf = st.file_uploader(
    "Upload a trading strategy PDF",
    type=["pdf"],
    key="v8_pdf_upload"
)

if v8_pdf is not None:
    import fitz
    import re

    try:
        v8_pdf.seek(0)

        with fitz.open(
            stream=v8_pdf.read(),
            filetype="pdf"
        ) as doc:
            pdf_text = "\n".join(
                page.get_text() for page in doc
            )

        categories = {
            "Entry Rules": [
                "entry", "enter", "buy", "sell"
            ],
            "Stop Loss Rules": [
                "stop loss", "stop-loss", "sl"
            ],
            "Take Profit Rules": [
                "take profit", "take-profit", "target"
            ],
            "Confirmation Rules": [
                "confirmation", "confirm", "candle close"
            ]
        }

        lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in pdf_text.splitlines()
        ]

        lines = [
            line for line in lines
            if len(line) >= 8
        ]

        for category, keywords in categories.items():
            st.subheader(category)

            matches = [
                line for line in lines
                if any(
                    re.search(
                        r"\b" + re.escape(word) + r"\b",
                        line,
                        flags=re.IGNORECASE
                    )
                    for word in keywords
                )
            ]

            if matches:
                for line in dict.fromkeys(matches[:15]):
                    st.write("• " + line)
            else:
                st.info("No matching rules found.")

        st.success("PDF rule extraction completed!")

        st.caption(
            "These are keyword-matched PDF lines, "
            "not verified trading instructions. "
            "Scanned PDFs may require OCR."
        )

    except Exception as e:
        st.error(f"V8 PDF extraction error: {e}")

# TUESDAY AI V9 - Strategy Rule Checklist

st.header("🧠 Tuesday AI V9 - Strategy Rule Checklist")
st.caption("Checks whether key trading rules appear in PDF text.")

v9_pdf = st.file_uploader(
    "Upload a trading strategy PDF for V9",
    type=["pdf"],
    key="v9_pdf_upload"
)

if v9_pdf is not None:
    import fitz
    import re

    try:
        v9_pdf.seek(0)

        with fitz.open(
            stream=v9_pdf.read(),
            filetype="pdf"
        ) as doc:
            pdf_text = "\n".join(
                page.get_text() for page in doc
            )

        text_lower = re.sub(
            r"\s+", " ", pdf_text.lower()
        )

        rule_categories = {
            "Entry": [
                r"\bentry\b", r"\benter\b",
                r"\bbuy\b", r"\bsell\b"
            ],
            "Stop Loss": [
                r"\bstop[\s-]*loss\b",
                r"\bstop\b", r"\bsl\b"
            ],
            "Take Profit": [
                r"\btake[\s-]*profit\b",
                r"\btarget\b", r"\btp\b",
                r"\brisk[\s/-]*reward\b"
            ],
            "Confirmation": [
                r"\bconfirm\w*\b",
                r"\bcandle[\s-]*close\b",
                r"\bretest\b"
            ]
        }

        results = []

        for category, patterns in rule_categories.items():
            found = any(
                re.search(pattern, text_lower)
                for pattern in patterns
            )

            results.append({
                "Rule": category,
                "Status": "Mentioned" if found else "Not found"
            })

        st.subheader("📋 PDF Rule Checklist")
        st.dataframe(
            pd.DataFrame(results),
            use_container_width=True,
            hide_index=True
        )

        missing = [
            item["Rule"]
            for item in results
            if item["Status"] == "Not found"
        ]

        if missing:
            st.warning(
                "No matching keywords found for: "
                + ", ".join(missing)
            )
        else:
            st.success(
                "All four rule categories have keyword matches!"
            )

        st.info(
            "Mentioned does not mean the rule is complete, "
            "correct, or suitable for trading."
        )

    except Exception as e:
        st.error(f"V9 error: {e}")

# TUESDAY AI V10 - PDF and Chart Rule Matching

st.header("🤖 Tuesday AI V10 - PDF + Chart Matching")
st.caption("Experimental historical rule matching only.")

v10_pdf = st.file_uploader(
    "Upload EMA strategy PDF",
    type=["pdf"],
    key="v10_pdf"
)

v10_csv = st.file_uploader(
    "Upload historical XAUUSD OHLC CSV",
    type=["csv"],
    key="v10_csv"
)

if v10_pdf is not None and v10_csv is not None:
    import fitz

    try:
        v10_pdf.seek(0)

        with fitz.open(
            stream=v10_pdf.read(),
            filetype="pdf"
        ) as doc:
            pdf_text = " ".join(
                page.get_text() for page in doc
            ).lower()

        v10_csv.seek(0)
        prices = pd.read_csv(v10_csv)
        prices.columns = prices.columns.str.strip().str.lower()

        required = ["open", "high", "low", "close"]

        if not all(c in prices.columns for c in required):
            st.error("CSV needs open, high, low, close columns.")

        else:
            for col in required:
                prices[col] = pd.to_numeric(
                    prices[col], errors="coerce"
                )

            prices = prices.dropna(subset=required).copy()

            valid = (
                (prices["high"] >= prices[
                    ["open", "close", "low"]
                ].max(axis=1))
                &
                (prices["low"] <= prices[
                    ["open", "close", "high"]
                ].min(axis=1))
            )

            if len(prices) < 220 or not valid.all():
                st.warning("Need at least 220 valid OHLC candles.")

            else:
                prices["ema50"] = prices["close"].ewm(
                    span=50, adjust=False
                ).mean()

                prices["ema200"] = prices["close"].ewm(
                    span=200, adjust=False
                ).mean()

                last = prices.iloc[-1]
                previous = prices.iloc[-2]

                pdf_has_ema = (
                    "ema" in pdf_text
                    and "pullback" in pdf_text
                )

                bullish = (
                    last["ema50"] > last["ema200"]
                    and last["close"] > last["ema50"]
                )

                bearish = (
                    last["ema50"] < last["ema200"]
                    and last["close"] < last["ema50"]
                )

                buy_match = (
                    pdf_has_ema
                    and bullish
                    and last["low"] <= last["ema50"]
                    and last["close"] > last["open"]
                )

                sell_match = (
                    pdf_has_ema
                    and bearish
                    and last["high"] >= last["ema50"]
                    and last["close"] < last["open"]
                )

                st.subheader("📋 PDF Strategy Check")

                st.write(
                    "EMA Pullback keywords:",
                    "Found" if pdf_has_ema else "Not found"
                )

                st.write(
                    "Previous candle close:",
                    round(float(previous["close"]), 2)
                )

                if buy_match:
                    st.success("Historical BUY rule match")

                elif sell_match:
                    st.error("Historical SELL rule match")

                else:
                    st.warning("WAIT - No rule match")

                st.info(
                    "Uses predefined EMA rules when PDF "
                    "keywords are found. Does not interpret "
                    "all PDF instructions or validate trades."
                )

    except Exception as e:
        st.error(f"V10 analysis error: {e}")


# TUESDAY AI V11 - HISTORICAL BACKTESTING
st.divider()
st.header("📊 TUESDAY AI V11 - Backtesting")

backtest_file = st.file_uploader(
    "Upload historical OHLC CSV",
    type=["csv"],
    key="v11_backtest_csv"
)

if backtest_file is not None:
    try:
        import pandas as pd

        data = pd.read_csv(backtest_file)
        data.columns = data.columns.str.strip().str.lower()

        required = ["open", "high", "low", "close"]

        if not all(c in data.columns for c in required):
            st.error("CSV needs open, high, low, close columns.")
        else:
            data[required] = data[required].apply(
                pd.to_numeric, errors="coerce"
            )
            data = data.dropna(subset=required).copy()

            if len(data) < 220:
                st.warning("Need at least 220 candles.")
            else:
                data["ema50"] = data["close"].ewm(
                    span=50, adjust=False
                ).mean()
                data["ema200"] = data["close"].ewm(
                    span=200, adjust=False
                ).mean()

                trades = []
                risk_points = st.number_input(
                    "SL distance (price units)",
                    min_value=0.01,
                    value=5.0,
                    key="v11_risk"
                )
                reward_ratio = st.number_input(
                    "Reward / Risk",
                    min_value=0.5,
                    value=2.0,
                    key="v11_rr"
                )

                i = 220
                while i < len(data) - 1:
                    candle = data.iloc[i]
                    direction = None

                    if (
                        candle["ema50"] > candle["ema200"]
                        and candle["low"] <= candle["ema50"]
                        and candle["close"] > candle["ema50"]
                        and candle["close"] > candle["open"]
                    ):
                        direction = "BUY"

                    elif (
                        candle["ema50"] < candle["ema200"]
                        and candle["high"] >= candle["ema50"]
                        and candle["close"] < candle["ema50"]
                        and candle["close"] < candle["open"]
                    ):
                        direction = "SELL"

                    if direction is None:
                        i += 1
                        continue

                    entry = float(data.iloc[i + 1]["open"])

                    if direction == "BUY":
                        sl = entry - risk_points
                        tp = entry + risk_points * reward_ratio
                    else:
                        sl = entry + risk_points
                        tp = entry - risk_points * reward_ratio

                    result = "OPEN"
                    exit_index = None

                    for j in range(i + 1, len(data)):
                        bar = data.iloc[j]

                        if direction == "BUY":
                            sl_hit = bar["low"] <= sl
                            tp_hit = bar["high"] >= tp
                        else:
                            sl_hit = bar["high"] >= sl
                            tp_hit = bar["low"] <= tp

                        # Conservative assumption: SL first
                        if sl_hit:
                            result = "LOSS"
                            exit_index = j
                            break
                        elif tp_hit:
                            result = "WIN"
                            exit_index = j
                            break

                    trades.append({
                        "Direction": direction,
                        "Entry": round(entry, 3),
                        "SL": round(sl, 3),
                        "TP": round(tp, 3),
                        "Result": result
                    })

                    if exit_index is None:
                        break

                    i = exit_index + 1

                results = pd.DataFrame(trades)

                if results.empty:
                    st.info("No matching setups found.")
                else:
                    wins = (results["Result"] == "WIN").sum()
                    losses = (results["Result"] == "LOSS").sum()
                    completed = wins + losses

                    win_rate = (
                        wins / completed * 100
                        if completed else 0
                    )

                    st.metric(
                        "Historical Win Rate",
                        f"{win_rate:.2f}%"
                    )
                    st.write("Wins:", int(wins))
                    st.write("Losses:", int(losses))
                    st.write("Completed trades:", int(completed))
                    st.dataframe(results)

    except Exception as e:
        st.error(f"Backtesting error: {e}")

# TUESDAY AI V12 - FILTERED HISTORICAL BACKTEST
# Append this section at the END of your existing app.py (V1-V11).
# Requires streamlit and pandas, already used in V11.
import pandas as pd
import streamlit as st

st.divider()
st.header('🥇 Tuesday AI V12 - Filtered Backtesting')
st.caption('Historical research only. No live feed, trading orders, or guaranteed signals.')

v12_csv = st.file_uploader('Upload XAUUSD M15 OHLC CSV', type=['csv'], key='v12_csv')
with st.expander('V12 backtest settings', expanded=True):
    v12_mode = st.selectbox('Setup', ['EMA Pullback + Breakout', 'EMA Pullback only', 'Breakout only'], key='v12_mode')
    v12_atr_mult = st.number_input('Stop distance (ATR multiple)', 0.5, 5.0, 1.5, 0.1, key='v12_atr')
    v12_rr = st.number_input('Reward / Risk', 0.5, 5.0, 2.0, 0.25, key='v12_rr')
    v12_spread = st.number_input('Assumed spread (XAUUSD price units)', 0.0, 10.0, 0.30, 0.05, key='v12_spread')
    v12_slippage = st.number_input('Assumed entry slippage (price units)', 0.0, 10.0, 0.10, 0.05, key='v12_slip')
    v12_min_atr = st.number_input('Minimum ATR (price units)', 0.0, 100.0, 1.0, 0.5, key='v12_min_atr')
    v12_max_bars = st.number_input('Maximum holding candles', 1, 500, 48, 1, key='v12_hold')
    v12_complete = st.checkbox('Require all 15 M1 candles per M15 bar (when M1_Count available)', value=True, key='v12_complete')

if v12_csv is not None:
    try:
        df12 = pd.read_csv(v12_csv)
        df12.columns = df12.columns.str.strip().str.lower()
        needed = ['open', 'high', 'low', 'close']
        if not all(x in df12.columns for x in needed):
            st.error('CSV must have Open, High, Low, Close columns.')
            st.stop()
        for col in needed:
            df12[col] = pd.to_numeric(df12[col], errors='coerce')
        if 'date' in df12.columns and 'time' in df12.columns:
            df12['timestamp'] = pd.to_datetime(df12['date'].astype(str) + ' ' + df12['time'].astype(str), errors='coerce', format='mixed')
        elif 'datetime' in df12.columns:
            df12['timestamp'] = pd.to_datetime(df12['datetime'], errors='coerce')
        elif 'timestamp' in df12.columns:
            df12['timestamp'] = pd.to_datetime(df12['timestamp'], errors='coerce')
        else:
            st.error('Need Date + Time, Datetime, or Timestamp columns to validate chronological candles.')
            st.stop()
        df12 = df12.dropna(subset=needed + ['timestamp']).sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
        ohlc_valid = (df12['high'] >= df12[needed].max(axis=1)) & (df12['low'] <= df12[needed].min(axis=1))
        if not ohlc_valid.all():
            st.error(f'{int((~ohlc_valid).sum())} invalid OHLC candles. Fix data before testing.')
            st.stop()
        if len(df12) < 250:
            st.error('Need at least 250 valid candles.')
            st.stop()
        df12['ema50'] = df12['close'].ewm(span=50, adjust=False).mean()
        df12['ema200'] = df12['close'].ewm(span=200, adjust=False).mean()
        prior_close = df12['close'].shift(1)
        tr = pd.concat([(df12['high']-df12['low']), (df12['high']-prior_close).abs(), (df12['low']-prior_close).abs()], axis=1).max(axis=1)
        df12['atr'] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        df12['resistance'] = df12['high'].shift(1).rolling(20).max()
        df12['support'] = df12['low'].shift(1).rolling(20).min()
        df12['complete'] = True
        if v12_complete and 'm1_count' in df12.columns:
            df12['complete'] = pd.to_numeric(df12['m1_count'], errors='coerce').eq(15)
        df12['next_contiguous'] = (df12['timestamp'].shift(-1) - df12['timestamp']).eq(pd.Timedelta(minutes=15))
        trades12 = []
        i = 220
        skipped_gap = 0
        while i < len(df12) - 1:
            bar = df12.iloc[i]
            if not bool(bar['complete']) or not bool(bar['next_contiguous']) or not bool(df12.iloc[i+1]['complete']) or pd.isna(bar['atr']) or bar['atr'] < v12_min_atr:
                i += 1
                continue
            bull = bar['close'] > bar['ema50'] > bar['ema200']
            bear = bar['close'] < bar['ema50'] < bar['ema200']
            pull_buy = bull and bar['low'] <= bar['ema50'] and bar['close'] > bar['open']
            pull_sell = bear and bar['high'] >= bar['ema50'] and bar['close'] < bar['open']
            break_buy = bull and bar['close'] > bar['resistance'] and bar['close'] > bar['open']
            break_sell = bear and bar['close'] < bar['support'] and bar['close'] < bar['open']
            if v12_mode == 'EMA Pullback only':
                break_buy = break_sell = False
            elif v12_mode == 'Breakout only':
                pull_buy = pull_sell = False
            direction = 'BUY' if (pull_buy or break_buy) else ('SELL' if (pull_sell or break_sell) else None)
            if direction is None:
                i += 1
                continue
            setup = 'Breakout' if (break_buy or break_sell) else 'Pullback'
            # Adverse fill: buy pays spread/slippage; sell receives less.
            raw_open = float(df12.iloc[i+1]['open'])
            cost = float(v12_spread) + float(v12_slippage)
            entry = raw_open + cost if direction == 'BUY' else raw_open - cost
            dist = float(bar['atr']) * float(v12_atr_mult)
            sl = entry - dist if direction == 'BUY' else entry + dist
            tp = entry + dist * v12_rr if direction == 'BUY' else entry - dist * v12_rr
            outcome, exit_i, exit_price = 'OPEN', None, None
            for j in range(i+1, min(len(df12), i+1+int(v12_max_bars))):
                b = df12.iloc[j]
                if not bool(b['complete']):
                    outcome, exit_i, exit_price = 'DATA_GAP', j, float(b['open'])
                    break
                if j > i+1 and df12.iloc[j]['timestamp'] - df12.iloc[j-1]['timestamp'] != pd.Timedelta(minutes=15):
                    outcome, exit_i, exit_price = 'DATA_GAP', j, float(b['open'])
                    break
                stop_hit = b['low'] <= sl if direction == 'BUY' else b['high'] >= sl
                target_hit = b['high'] >= tp if direction == 'BUY' else b['low'] <= tp
                if stop_hit:  # Includes same-bar TP + SL: assume SL first.
                    outcome, exit_i, exit_price = 'LOSS', j, float(sl)
                    break
                if target_hit:
                    outcome, exit_i, exit_price = 'WIN', j, float(tp)
                    break
            if exit_i is None:
                exit_i = min(len(df12)-1, i+int(v12_max_bars))
                outcome = 'TIME_EXIT' if exit_i < len(df12)-1 else 'END_OF_DATA'
                exit_price = float(df12.iloc[exit_i]['close'])
            r = ((exit_price-entry) if direction == 'BUY' else (entry-exit_price)) / dist
            trades12.append({'Signal time': bar['timestamp'], 'Entry time': df12.iloc[i+1]['timestamp'], 'Exit time': df12.iloc[exit_i]['timestamp'], 'Setup': setup, 'Side': direction, 'Entry': round(entry,3), 'SL': round(sl,3), 'TP': round(tp,3), 'Exit': round(exit_price,3), 'Outcome': outcome, 'R': round(r,4)})
            i = exit_i + 1  # One position at a time.
        results12 = pd.DataFrame(trades12)
        if results12.empty:
            st.warning('No trades matched these settings.')
        else:
            # Exclude data gaps and unfinished trades from performance summary.
            settled = results12[results12['Outcome'].isin(['WIN', 'LOSS', 'TIME_EXIT'])].copy()
            wins = int((settled['Outcome']=='WIN').sum())
            losses = int((settled['Outcome']=='LOSS').sum())
            decisive = wins + losses
            win_rate = 100*wins/decisive if decisive else 0.0
            net_r = float(settled['R'].sum())
            positive = float(settled.loc[settled['R']>0,'R'].sum())
            negative = -float(settled.loc[settled['R']<0,'R'].sum())
            pf = positive/negative if negative else float('inf') if positive else 0.0
            equity = settled['R'].cumsum()
            dd = (equity - equity.cummax().clip(lower=0)).min() if len(equity) else 0
            c1,c2,c3,c4 = st.columns(4)
            c1.metric('Win rate (TP/SL only)', f'{win_rate:.2f}%')
            c2.metric('Wins / Losses', f'{wins} / {losses}')
            c3.metric('Net R (settled)', f'{net_r:.2f}R')
            c4.metric('Profit factor', f'{pf:.2f}')
            st.write(f'Max drawdown: {float(dd):.2f}R | Trades: {len(results12)} | Settled: {len(settled)} | Data gaps: {int((results12.Outcome=="DATA_GAP").sum())} | Unfinished: {int((results12.Outcome=="END_OF_DATA").sum())}')
            st.caption('TIME_EXIT trades count in net R / profit factor, but not TP/SL win rate. DATA_GAP and END_OF_DATA excluded. Costs are simplified; no commissions, financing, news or execution modeling.')
            st.line_chart(equity.reset_index(drop=True), y_label='Cumulative R')
            st.dataframe(results12, use_container_width=True, hide_index=True)
            st.download_button('Download V12 trade log', results12.to_csv(index=False).encode(), 'Tuesday_AI_V12_trades.csv', 'text/csv', key='v12_download')
    except Exception as exc:
        st.error(f'V12 backtest error: {exc}')
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

st.set_page_config(page_title='Tuesday AI V13', page_icon='🥇', layout='wide')
st.title('🥇 Tuesday AI V13 — Gold Strategy Backtester')
st.caption('Research tool only. No live feed, no guaranteed signals. Uses OHLC candle CSVs.')

with st.sidebar:
    st.header('Strategy settings')
    timeframe = st.selectbox('Candle timeframe', ['M15', 'M1'])
    rr = st.number_input('Take-profit / risk (R)', min_value=0.5, max_value=10.0, value=2.0, step=0.25)
    atr_mult = st.number_input('Stop distance (ATR multiple)', min_value=0.25, max_value=10.0, value=1.5, step=0.25)
    breakout_n = st.number_input('Breakout lookback candles', min_value=5, max_value=200, value=20)
    max_hold = st.number_input('Maximum holding candles', min_value=1, max_value=1000, value=48)
    st.caption('Signals form on completed candles; entries fill at next candle open.')

uploaded = st.file_uploader('Upload historical XAUUSD OHLC CSV (not a trade-log CSV)', type=['csv'])
st.info('CSV needs time, open, high, low, close columns. Optional volume. MT5 exports with <DATE>, <TIME>, <OPEN>, etc. are supported.')

def load_ohlc(file):
    raw = pd.read_csv(file, sep=None, engine='python')
    raw.columns = [str(c).strip().lower().replace('<','').replace('>','') for c in raw.columns]
    if 'date' in raw and 'time' in raw:
        raw['datetime'] = pd.to_datetime(raw['date'].astype(str) + ' ' + raw['time'].astype(str), errors='coerce')
    elif 'datetime' in raw:
        raw['datetime'] = pd.to_datetime(raw['datetime'], errors='coerce')
    elif 'timestamp' in raw:
        raw['datetime'] = pd.to_datetime(raw['timestamp'], errors='coerce')
    elif 'time' in raw:
        raw['datetime'] = pd.to_datetime(raw['time'], errors='coerce')
    else:
        raise ValueError('Time column missing. Use datetime, timestamp, time, or date + time.')
    for col in ['open','high','low','close']:
        if col not in raw: raise ValueError(f'Missing {col} column: upload candle data, not a trade log.')
        raw[col] = pd.to_numeric(raw[col], errors='coerce')
    df = raw[['datetime','open','high','low','close']].dropna().sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
    df = df[(df.high >= df[['open','close','low']].max(axis=1)) & (df.low <= df[['open','close','high']].min(axis=1))].reset_index(drop=True)
    if len(df) < 230: raise ValueError('At least 230 valid historical candles needed.')
    return df

def prepare(df):
    df = df.copy()
    df['ema50'] = df.close.ewm(span=50, adjust=False).mean()
    df['ema200'] = df.close.ewm(span=200, adjust=False).mean()
    prev_close = df.close.shift(1)
    tr = pd.concat([df.high-df.low, (df.high-prev_close).abs(), (df.low-prev_close).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()
    df['prev_high'] = df.high.shift(1).rolling(breakout_n).max()
    df['prev_low'] = df.low.shift(1).rolling(breakout_n).min()
    return df

def backtest(df):
    trades = []
    expected = pd.Timedelta(minutes=15 if timeframe == 'M15' else 1)
    i = max(210, breakout_n+2)
    while i < len(df)-1:
        bar = df.iloc[i]
        prev = df.iloc[i-1]
        bullish = bar.ema50 > bar.ema200 and bar.close > bar.ema50
        bearish = bar.ema50 < bar.ema200 and bar.close < bar.ema50
        setup, side = None, None
        if bullish and bar.close > bar.prev_high and prev.close <= prev.prev_high:
            setup, side = 'Breakout', 'BUY'
        elif bearish and bar.close < bar.prev_low and prev.close >= prev.prev_low:
            setup, side = 'Breakout', 'SELL'
        elif bullish and bar.low <= bar.ema50 and bar.close > bar.ema50 and bar.close > bar.open:
            setup, side = 'Pullback', 'BUY'
        elif bearish and bar.high >= bar.ema50 and bar.close < bar.ema50 and bar.close < bar.open:
            setup, side = 'Pullback', 'SELL'
        if not setup or not np.isfinite(bar.atr) or bar.atr <= 0:
            i += 1
            continue
        entry_i = i+1
        entry_bar = df.iloc[entry_i]
        if entry_bar.datetime - bar.datetime != expected:
            i += 1
            continue
        entry = float(entry_bar.open)
        risk = float(bar.atr * atr_mult)
        sl = entry-risk if side == 'BUY' else entry+risk
        tp = entry+rr*risk if side == 'BUY' else entry-rr*risk
        outcome, exit_price, exit_i = 'DATA_GAP', np.nan, entry_i
        last = min(len(df)-1, entry_i+int(max_hold)-1)
        for j in range(entry_i, last+1):
            b = df.iloc[j]
            if j > entry_i and b.datetime-df.iloc[j-1].datetime != expected:
                outcome, exit_i = 'DATA_GAP', j
                break
            hit_sl = b.low <= sl if side == 'BUY' else b.high >= sl
            hit_tp = b.high >= tp if side == 'BUY' else b.low <= tp
            if hit_sl and hit_tp:
                outcome, exit_price, exit_i = 'AMBIGUOUS_SL_FIRST', sl, j
                break
            if hit_sl:
                outcome, exit_price, exit_i = 'LOSS', sl, j
                break
            if hit_tp:
                outcome, exit_price, exit_i = 'WIN', tp, j
                break
        else:
            exit_i = last
            if last == len(df)-1 and last < entry_i+int(max_hold)-1:
                outcome = 'INCOMPLETE'
            else:
                outcome = 'TIME_EXIT'
                exit_price = float(df.iloc[last].close)
        r = ((exit_price-entry)/risk if side == 'BUY' else (entry-exit_price)/risk) if np.isfinite(exit_price) else np.nan
        trades.append({'Signal time':bar.datetime, 'Entry time':entry_bar.datetime, 'Exit time':df.iloc[exit_i].datetime,
                       'Setup':setup,'Side':side,'Entry':entry,'SL':sl,'TP':tp,'Exit':exit_price,'Outcome':outcome,'R':r})
        i = max(i+1, exit_i+1)  # One position at a time; no overlapping trades.
    return pd.DataFrame(trades)

if uploaded:
    try:
        candles = prepare(load_ohlc(uploaded))
        st.write(f'Loaded **{len(candles):,}** candles, from **{candles.datetime.iloc[0]}** to **{candles.datetime.iloc[-1]}**.')
        trades = backtest(candles)
        if trades.empty:
            st.warning('No trades matched these settings. Try more candles or review the setup parameters.')
        else:
            resolved = trades[trades.R.notna()].copy()
            gaps = int((trades.Outcome == 'DATA_GAP').sum())
            wins = int((resolved.R > 0).sum())
            winrate = 100*wins/len(resolved) if len(resolved) else 0
            net_r = float(resolved.R.sum())
            gross_win = float(resolved.loc[resolved.R>0,'R'].sum())
            gross_loss = abs(float(resolved.loc[resolved.R<0,'R'].sum()))
            profit_factor = gross_win/gross_loss if gross_loss else np.nan
            equity = resolved.R.cumsum()
            max_dd = float((equity-equity.cummax().clip(lower=0)).min()) if len(equity) else 0
            a,b,c,d,e = st.columns(5)
            a.metric('Total trades',len(trades))
            b.metric('Resolved win rate',f'{winrate:.1f}%')
            c.metric('Net R',f'{net_r:+.2f}R')
            d.metric('Profit factor',f'{profit_factor:.2f}' if np.isfinite(profit_factor) else 'N/A')
            e.metric('Max drawdown',f'{max_dd:.2f}R')
            st.warning(f'{gaps} data-gap trades excluded from performance calculations. Ambiguous same-candle SL/TP is conservatively scored SL first. Fees, spreads and slippage are NOT included.')
            if len(resolved):
                st.subheader('Cumulative R')
                st.line_chart(equity.reset_index(drop=True))
                st.subheader('Setup performance')
                st.dataframe(resolved.groupby(['Setup','Side']).agg(Trades=('R','size'),Net_R=('R','sum'),Average_R=('R','mean')).reset_index(),use_container_width=True)
            st.subheader('All trades')
            st.dataframe(trades,use_container_width=True)
            st.download_button('Download V13 trade results CSV',trades.to_csv(index=False).encode('utf-8'),file_name='Tuesday_AI_V13_trades.csv',mime='text/csv')
    except Exception as exc:
        st.error(f'Could not process CSV: {exc}')
else:
    st.caption('Upload real historical candle data to run V13. A previous V12 trades CSV cannot be used to reconstruct OHLC candles.')

