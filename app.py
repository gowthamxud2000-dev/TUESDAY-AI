import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Tuesday AI V17", page_icon="🥇", layout="wide")
st.title("🥇 Tuesday AI V17 — Backtest Audit")
st.caption("Trade-log analysis, chronological out-of-sample split, monthly results and risk simulation. No live signals.")

uploaded = st.file_uploader("Upload Tuesday AI trades CSV (V16 or similar)", type="csv")
starting_balance = st.number_input("Starting balance (USD)", min_value=1.0, value=10000.0, step=1000.0)
train_pct = st.slider("In-sample percentage (chronological)", 50, 90, 70, 5)
risk_pct = st.slider("Risk per trade for separate simulation (%)", 0.1, 3.0, 1.0, 0.1)
max_dd_limit = st.slider("Maximum acceptable drawdown (%)", 1.0, 30.0, 10.0, 0.5)

def stats(frame, initial):
    if frame.empty:
        return {"Trades": 0, "Win rate": 0.0, "Net P&L": 0.0, "Profit factor": None, "Max drawdown": 0.0, "Final balance": initial}
    pnl = frame["pnl_usd"].astype(float)
    eq = initial + pnl.cumsum()
    peak = pd.concat([pd.Series([initial]), eq], ignore_index=True).cummax().iloc[1:].to_numpy()
    dd = (peak - eq.to_numpy()) / peak * 100
    gross_win = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else None)
    return {"Trades": len(frame), "Win rate": 100 * (pnl > 0).mean(),
            "Net P&L": pnl.sum(), "Profit factor": pf,
            "Max drawdown": float(dd.max()), "Final balance": float(eq.iloc[-1])}

if uploaded:
    try:
        data = pd.read_csv(uploaded)
        required = {"entry_time", "exit_time", "pnl_usd", "net_R"}
        missing = required - set(data.columns)
        if missing:
            st.error("Missing columns: " + ", ".join(sorted(missing)))
            st.stop()
        for c in ["entry_time", "exit_time"]:
            data[c] = pd.to_datetime(data[c], errors="coerce")
        for c in ["pnl_usd", "net_R"]:
            data[c] = pd.to_numeric(data[c], errors="coerce")
        invalid = data[list(required)].isna().any(axis=1)
        if invalid.any():
            st.warning(f"Excluded {invalid.sum()} rows with missing/invalid required values.")
            data = data.loc[~invalid].copy()
        if data.empty:
            st.error("No valid trades found.")
            st.stop()
        data = data.sort_values(["entry_time", "exit_time"], kind="stable").reset_index(drop=True)
        if (data["exit_time"] < data["entry_time"]).any():
            st.warning("Some trades exit before entry; check the source CSV.")
        if "equity_usd" in data.columns:
            recorded = pd.to_numeric(data["equity_usd"], errors="coerce")
            calculated = starting_balance + data["pnl_usd"].cumsum()
            if (recorded - calculated).abs().max() > 1:
                st.info("Recorded equity differs from the chosen starting balance or trade P&L. Metrics below use pnl_usd.")
        st.subheader("📊 Full sample")
        full = stats(data, starting_balance)
        a,b,c,d = st.columns(4)
        a.metric("Trades", full["Trades"])
        b.metric("Win rate", f'{full["Win rate"]:.2f}%')
        c.metric("Net P&L", f'${full["Net P&L"]:,.2f}')
        d.metric("Profit factor", "N/A" if full["Profit factor"] is None else f'{full["Profit factor"]:.2f}')
        st.metric("Maximum drawdown", f'{full["Max drawdown"]:.2f}%')
        equity = starting_balance + data["pnl_usd"].cumsum()
        st.line_chart(pd.DataFrame({"Equity USD": equity.to_numpy()}, index=data["exit_time"]))

        st.subheader("🧪 Chronological in-sample vs out-of-sample")
        cut = max(1, min(len(data)-1, int(len(data)*train_pct/100)))
        if len(data) < 2:
            st.warning("At least two trades are needed for a split.")
        else:
            train = data.iloc[:cut]
            test = data.iloc[cut:]
            rows = []
            for label, frame in [("In-sample", train), ("Out-of-sample", test)]:
                s = stats(frame, starting_balance)
                rows.append({"Period": label, **s})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("This is a chronological holdout of existing trades, NOT an independent strategy backtest. No parameters are retrained.")

        st.subheader("📅 Monthly results")
        data["month"] = data["exit_time"].dt.to_period("M").astype(str)
        monthly = data.groupby("month", as_index=False).agg(
            trades=("pnl_usd","size"), pnl_usd=("pnl_usd","sum"),
            wins=("pnl_usd",lambda x: int((x>0).sum())))
        monthly["win_rate_pct"] = 100*monthly["wins"]/monthly["trades"]
        st.dataframe(monthly, hide_index=True, use_container_width=True)

        st.subheader("🛡️ Separate fixed-fraction risk simulation")
        st.caption("Uses net_R from the log, assumes risk is a fixed % of current equity per trade; not broker-executable performance.")
        sim_equity = [starting_balance]
        for r in data["net_R"]:
            sim_equity.append(sim_equity[-1] * (1 + (risk_pct / 100)*float(r)))
        sim = pd.Series(sim_equity)
        sim_dd = ((sim.cummax()-sim)/sim.cummax()*100).max()
        x,y = st.columns(2)
        x.metric("Simulated final balance", f"${sim.iloc[-1]:,.2f}")
        y.metric("Simulated max drawdown", f"{sim_dd:.2f}%")
        st.line_chart(pd.DataFrame({"Simulated equity": sim.iloc[1:].to_numpy()}, index=data["exit_time"]))
        if sim_dd > max_dd_limit:
            st.warning("Risk simulation exceeds your drawdown threshold.")
        else:
            st.info("Risk simulation is within the selected drawdown threshold for this sample only.")

        st.subheader("🔎 Validation checks")
        st.write(f"Date range: {data['entry_time'].min()} to {data['exit_time'].max()}")
        st.write(f"Duplicate rows: {data.duplicated(subset=['entry_time','exit_time','pnl_usd']).sum()}")
        st.write(f"Trades with net_R < -1.1: {(data['net_R'] < -1.1).sum()}")
        st.warning("A profitable historical sample does not validate live execution. Verify data provenance, spread, slippage, commissions, no lookahead bias, and test unseen months before risking money.")
        st.download_button("⬇️ Download monthly audit CSV", monthly.to_csv(index=False).encode(), file_name="Tuesday_AI_V17_monthly_audit.csv", mime="text/csv")
        st.download_button("⬇️ Download cleaned trades CSV", data.to_csv(index=False).encode(), file_name="Tuesday_AI_V17_clean_trades.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Could not process CSV: {e}")
else:
    st.info("Upload your V16 trades CSV to begin.")
