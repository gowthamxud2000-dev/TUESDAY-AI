
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
  
