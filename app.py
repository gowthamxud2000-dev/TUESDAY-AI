
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
