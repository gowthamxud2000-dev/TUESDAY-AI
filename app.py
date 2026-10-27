
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Tuesday AI V23", page_icon="🥇", layout="wide")
st.title("🥇 Tuesday AI V23")
st.caption("Risk Management & Out-of-Sample Validation")

st.warning("Research only. This replays V22 trade outcomes, not a fresh market backtest. No live signals or profit guarantee.")

uploaded = st.file_uploader("Upload V22 audited trades CSV", type=["csv"])

st.sidebar.header("⚙️ Risk Settings")
capital = st.sidebar.number_input("Starting capital ($)", min_value=100.0, value=1000.0, step=100.0)
risk = st.sidebar.slider("Risk per trade (%)", 0.1, 2.0, 0.5, 0.1)
baseline = st.sidebar.number_input("Assumed V22 baseline risk (%)", min_value=0.1, value=1.0, step=0.1)
daily_limit = st.sidebar.slider("Daily loss limit (%)", 1.0, 10.0, 3.0, 0.5)
max_streak = st.sidebar.slider("Maximum consecutive losses", 1, 10, 3)
max_dd = st.sidebar.slider("Max drawdown lock (%)", 5.0, 40.0, 15.0)
extra_cost = st.sidebar.number_input("Additional cost per executed trade ($)", min_value=0.0, value=0.0, step=0.1)
split_pct = st.sidebar.slider("In-sample portion (%)", 50, 85, 70, 5)

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        required = {"entry_time", "status", "pnl_usd", "equity"}
        if not required.issubset(df.columns):
            st.error("Missing columns: " + ", ".join(sorted(required - set(df.columns))))
            st.stop()

        df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
        for col in ["pnl_usd", "equity"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["entry_time"]).sort_values("entry_time", kind="stable").reset_index(drop=True)
        if df.empty:
            st.error("No valid timestamp rows.")
            st.stop()

        # V22 realized return relative to prior V22 equity.
        # This is an approximation, NOT a verified R-multiple.
        previous_equity = df["equity"] - df["pnl_usd"].fillna(0)
        df["base_return"] = np.where(
            (df["status"] == "Executed") & (previous_equity > 0),
            df["pnl_usd"] / previous_equity,
            0.0
        )

        def simulate(data):
            equity = float(capital)
            peak = equity
            streak = 0
            locked = False
            current_day = None
            day_start = equity
            rows = []

            for _, trade in data.iterrows():
                day = trade["entry_time"].date()
                if day != current_day:
                    current_day = day
                    day_start = equity
                    streak = 0

                original_status = str(trade["status"])
                pnl = 0.0
                status = "V22 skipped"

                if original_status == "Executed":
                    if locked:
                        status = "Drawdown lock"
                    elif (day_start - equity) / max(day_start, 0.01) * 100 >= daily_limit:
                        status = "Daily loss lock"
                    elif streak >= max_streak:
                        status = "Streak lock"
                    else:
                        # Risk-scaled replay of V22 percentage returns.
                        pnl = equity * float(trade["base_return"]) * risk / baseline - extra_cost
                        pnl = max(pnl, -equity)
                        equity += pnl
                        status = "Executed"
                        streak = streak + 1 if pnl < 0 else 0

                peak = max(peak, equity)
                drawdown = (peak - equity) / peak * 100
                if drawdown >= max_dd:
                    locked = True

                rows.append({
                    "entry_time": trade["entry_time"],
                    "side": trade.get("side", ""),
                    "v22_status": original_status,
                    "v23_status": status,
                    "v23_pnl": round(pnl, 4),
                    "v23_equity": round(equity, 4),
                    "v23_drawdown_pct": round(drawdown, 4)
                })
            return pd.DataFrame(rows)

        result = simulate(df)
        executed = result[result["v23_status"] == "Executed"]
        wins = int((executed["v23_pnl"] > 0).sum())
        losses = int((executed["v23_pnl"] < 0).sum())
        total = len(executed)
        final_equity = float(result["v23_equity"].iloc[-1])
        profit = final_equity - capital
        winrate = wins / total * 100 if total else 0
        worst_dd = float(result["v23_drawdown_pct"].max())

        st.subheader("📊 V23 Risk Replay")
        a, b, c = st.columns(3)
        a.metric("Final Equity", f"${final_equity:,.2f}")
        b.metric("Net P/L", f"${profit:,.2f}")
        c.metric("Max DD", f"{worst_dd:.2f}%")
        a, b, c = st.columns(3)
        a.metric("Executed", total)
        b.metric("Win Rate", f"{winrate:.1f}%")
        c.metric("Skipped", len(result) - total)

        st.subheader("📈 Equity Curve")
        st.line_chart(result.set_index("entry_time")["v23_equity"])

        st.subheader("🧪 Chronological 70/30-style Split")
        split = max(1, min(len(df)-1, int(len(df)*split_pct/100)))
        cut_time = df.iloc[split]["entry_time"]
        st.caption(f"Split boundary: {cut_time}. Settings are shared across both sets; tune only on training data.")

        def period_stats(part):
            trades = part[part["v23_status"] == "Executed"]
            return {
                "Trades": len(trades),
                "Net P/L ($)": round(float(trades["v23_pnl"].sum()), 2),
                "Win rate (%)": round(100 * (trades["v23_pnl"] > 0).mean(), 2) if len(trades) else 0
            }

        comparison = pd.DataFrame({
            "In-sample": period_stats(result.iloc[:split]),
            "Out-of-sample": period_stats(result.iloc[split:])
        }).T
        st.dataframe(comparison, use_container_width=True)

        st.info("Out-of-sample is a chronological holdout of already observed V22 trades. It is NOT independent strategy validation. Tune settings on training only, freeze them, then evaluate holdout once.")

        st.subheader("📋 Audited Trades")
        st.dataframe(result, use_container_width=True)

        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Tuesday_AI_V23_audited_trades.csv",
            data=csv,
            file_name="Tuesday_AI_V23_audited_trades.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Could not process CSV: {e}")
else:
    st.info("Upload Tuesday_AI_V22_audited_trades.csv to begin.")
