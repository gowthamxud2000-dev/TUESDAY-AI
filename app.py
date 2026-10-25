
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

