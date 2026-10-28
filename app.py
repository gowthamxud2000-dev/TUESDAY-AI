
# TUESDAY AI V41 - NO DOWNLOAD DEMO MODE
# Paste at the bottom of existing app.py

st.divider()
st.header("5. No-download Demo Test")
st.caption("Synthetic XAUUSD-style data only.")

def generate_demo_m1(seed=41, days=16):
    rng = np.random.default_rng(seed)
    n = days * 24 * 60

    times = pd.date_range(
        "2026-09-01", periods=n, freq="min"
    )

    drift = np.where(
        np.arange(n) < n // 2, 0.006, -0.005
    )

    shocks = rng.normal(0, 0.32, n) + drift
    close = 3400 + np.cumsum(shocks)
    open_ = np.r_[3400.0, close[:-1]]

    high = (
        np.maximum(open_, close)
        + rng.uniform(0.02, 0.25, n)
    )

    low = (
        np.minimum(open_, close)
        - rng.uniform(0.02, 0.25, n)
    )

    return pd.DataFrame({
        "time": times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close
    })


seed = st.number_input(
    "Demo random seed",
    min_value=1,
    max_value=99999,
    value=41,
    key="demo_seed"
)

if st.button(
    "Generate Demo Data & Run Backtest",
    type="primary",
    key="demo_run"
):
    with st.spinner("Running demo backtest..."):
        demo = generate_demo_m1(int(seed))

        midpoint = demo.time.iloc[
            len(demo) // 2
        ]

        first, _ = backtest(
            demo.iloc[:len(demo)//2].copy(),
            balance
        )

        second, signals = backtest(
            demo,
            balance,
            start_time=midpoint
        )

    st.success(
        f"{len(demo):,} synthetic M1 candles generated!"
    )

    st.write(
        "Demo coverage:",
        demo.time.min(),
        "to",
        demo.time.max()
    )

    show_report(first, "Demo First Half")
    show_report(second, "Demo Second Half")

    st.write(
        "Latest synthetic signal:",
        signals.iloc[-1].signal
    )

    st.warning(
        "Software test only. Not real historical "
        "validation or evidence of future profit."
    )
