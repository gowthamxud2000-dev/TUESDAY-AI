
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Tuesday AI V14",
    page_icon="🥇",
    layout="wide"
)

st.title("🥇 Tuesday AI V14")
st.subheader("XAUUSD Gold Backtest Analyzer")

st.warning("Research only. No guaranteed live profits.")

st.sidebar.header("Trading Costs")

spread = st.sidebar.number_input(
    "Round-trip spread (price units)",
    min_value=0.0,
    value=0.30,
    step=0.05
)

slippage = st.sidebar.number_input(
    "Round-trip slippage (price units)",
    min_value=0.0,
    value=0.20,
    step=0.05
)

commission = st.sidebar.number_input(
    "Commission per trade (R)",
    min_value=0.0,
    value=0.0,
    step=0.01
)

file = st.file_uploader(
    "Upload V13 Trades CSV",
    type=["csv"]
)

if file is not None:

    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"CSV error: {e}")
        st.stop()

    required = [
        "Setup", "Side", "Entry",
        "SL", "Outcome", "R"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    for col in ["Entry", "SL", "R"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Outcome"] = (
        df["Outcome"].astype(str).str.strip().str.upper()
    )

    df["Risk"] = abs(df["Entry"] - df["SL"])

    df["Resolved"] = df["Outcome"].isin(
        ["WIN", "LOSS"]
    )

    df["ValidRisk"] = (
        df["Risk"].gt(0) &
        np.isfinite(df["Risk"])
    )

    df["Cost_R"] = np.where(
        df["ValidRisk"],
        (spread + slippage) / df["Risk"] + commission,
        np.nan
    )

    df["Net_R"] = np.where(
        df["Resolved"] & df["ValidRisk"],
        df["R"] - df["Cost_R"],
        np.nan
    )

    valid = df[
        df["Resolved"] &
        df["ValidRisk"] &
        df["R"].notna()
    ].copy()

    wins = (valid["Outcome"] == "WIN").sum()
    losses = (valid["Outcome"] == "LOSS").sum()
    total = len(valid)

    winrate = 100 * wins / total if total else 0
    net_r = valid["Net_R"].sum()

    st.header("📊 V14 Performance")

    a, b, c, d = st.columns(4)

    a.metric("Total Trades", len(df))
    b.metric("Wins", int(wins))
    c.metric("Losses", int(losses))
    d.metric("Win Rate", f"{winrate:.1f}%")

    st.metric(
        "Net Result After Costs",
        f"{net_r:+.2f}R"
    )

    unresolved = len(df) - total

    st.info(
        f"Unresolved or invalid trades: {unresolved}"
    )

    st.header("📋 Outcome Audit")

    st.dataframe(
        df["Outcome"].value_counts().reset_index(),
        use_container_width=True
    )

    if total:

        st.header("🎯 Setup Performance")

        setup = valid.groupby("Setup").agg(
            Trades=("R", "size"),
            Gross_R=("R", "sum"),
            Net_R=("Net_R", "sum"),
            Average_R=("Net_R", "mean")
        )

        st.dataframe(
            setup,
            use_container_width=True
        )

        st.header("📈 BUY vs SELL")

        side = valid.groupby("Side").agg(
            Trades=("R", "size"),
            Net_R=("Net_R", "sum")
        )

        st.dataframe(
            side,
            use_container_width=True
        )

        st.header("📉 Equity Curve")

        valid["Cumulative_R"] = (
            valid["Net_R"].cumsum()
        )

        st.line_chart(
            valid["Cumulative_R"].reset_index(drop=True)
        )

        peak = (
            valid["Cumulative_R"]
            .cummax()
            .clip(lower=0)
        )

        drawdown = valid["Cumulative_R"] - peak

        st.metric(
            "Maximum Drawdown",
            f"{drawdown.min():.2f}R"
        )

    st.header("📥 Download Results")

    st.download_button(
        "Download V14 Audit CSV",
        data=df.to_csv(index=False),
        file_name="Tuesday_AI_V14_audit.csv",
        mime="text/csv"
    )

    st.success("Tuesday AI V14 analysis complete!")

else:
    st.info("Upload V13 trades CSV to begin.")
