import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Tuesday AI V24", page_icon="🥇", layout="wide")
st.title("🥇 Tuesday AI V24 — Risk Audit")
st.caption("Historical P&L replay, not live signals or a predictive backtest.")

uploaded = st.file_uploader("Upload V23 audited trades CSV", type="csv")
with st.sidebar:
    st.header("Risk & cost assumptions")
    initial = st.number_input("Starting equity ($)", min_value=1.0, value=1000.0)
    risk_scale = st.slider("Position-size multiplier", 0.1, 1.0, 0.5, 0.1)
    cost = st.number_input("Additional cost per executed trade ($)", min_value=0.0, value=0.50, step=0.10)
    daily_limit = st.slider("Daily loss lock (%)", 0.5, 10.0, 2.0, 0.5)
    max_losses = st.slider("Max consecutive losses per day", 1, 10, 3)
    split = st.slider("Training fraction", 0.5, 0.9, 0.7, 0.05)

if uploaded is None:
    st.info("Upload the V23 audited trades CSV.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
    required = {"entry_time", "v23_status", "v23_pnl"}
    missing = required - set(df.columns)
    if missing:
        st.error("Missing columns: " + ", ".join(sorted(missing)))
        st.stop()
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["v23_pnl"] = pd.to_numeric(df["v23_pnl"], errors="coerce")
    if df[["entry_time", "v23_pnl"]].isna().any().any():
        st.error("Invalid dates or P&L values.")
        st.stop()
    df = df.sort_values("entry_time", kind="stable").reset_index(drop=True)
except Exception as e:
    st.error(f"Cannot read CSV: {e}")
    st.stop()

cut = max(1, min(len(df)-1, int(len(df)*split)))
df["sample"] = np.where(df.index < cut, "Training", "Out-of-sample")

def replay(part, start_equity):
    equity = float(start_equity)
    peak = equity
    day = None
    day_start = equity
    losing_streak = 0
    locked = False
    rows = []
    for _, r in part.iterrows():
        date = r["entry_time"].date()
        if date != day:
            day, day_start, losing_streak, locked = date, equity, 0, False
        original_executed = str(r["v23_status"]).strip().lower() == "executed"
        status = "Original skipped"
        net = 0.0
        if original_executed:
            if locked:
                status = "Risk skipped"
            else:
                net = float(r["v23_pnl"]) * risk_scale - cost * risk_scale
                equity += net
                status = "Executed"
                losing_streak = losing_streak + 1 if net < 0 else 0
                if equity <= day_start * (1 - daily_limit / 100) or losing_streak >= max_losses:
                    locked = True
        peak = max(peak, equity)
        row = r.to_dict()
        row.update(v24_status=status, v24_net_pnl=round(net, 4),
                   v24_equity=round(equity, 4),
                   v24_drawdown_pct=round(100*(peak-equity)/peak, 4))
        rows.append(row)
    return pd.DataFrame(rows), equity

train, after_train = replay(df.iloc[:cut], initial)
test, final_equity = replay(df.iloc[cut:], after_train)
result = pd.concat([train, test], ignore_index=True)

def stats(part, start):
    trades = part.loc[part["v24_status"] == "Executed", "v24_net_pnl"]
    gains = trades[trades > 0].sum()
    losses = -trades[trades < 0].sum()
    return {
        "Executed": len(trades),
        "Skipped": int((part["v24_status"] != "Executed").sum()),
        "Net P&L ($)": round(trades.sum(), 2),
        "Return (%)": round(100*trades.sum()/start, 2),
        "Win rate (%)": round(100*(trades > 0).mean(), 2) if len(trades) else 0,
        "Profit factor": round(gains/losses, 3) if losses > 0 else ("∞" if gains > 0 else "N/A"),
        "Max drawdown (%)": round(part["v24_drawdown_pct"].max(), 2)
    }

st.subheader("Chronological validation")
st.warning("Choose risk parameters using training data only. Freeze them before interpreting holdout results. Holdout performance is not guaranteed.")
st.write(f"Training: {len(train)} rows | Holdout: {len(test)} rows")
st.dataframe(pd.DataFrame({"Training": stats(train, initial), "Out-of-sample": stats(test, after_train)}), use_container_width=True)
a, b, c = st.columns(3)
a.metric("Final equity", f"${final_equity:,.2f}")
b.metric("Net return", f"{(final_equity/initial-1)*100:.2f}%")
c.metric("Risk-skipped trades", int((result["v24_status"]=="Risk skipped").sum()))
st.subheader("Equity curve")
st.line_chart(pd.concat([pd.DataFrame({"v24_equity":[initial]}), result[["v24_equity"]]], ignore_index=True))
st.subheader("Audited trade log")
st.dataframe(result, use_container_width=True)
st.download_button("⬇️ Download V24 audited CSV", result.to_csv(index=False).encode("utf-8"), "Tuesday_AI_V24_audited_trades.csv", "text/csv")
st.caption("Limitations: CSV lacks prices, lot sizes, exit times and broker fee details. Costs are assumed. No new strategy validation is performed.")
