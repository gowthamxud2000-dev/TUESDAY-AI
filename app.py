
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
