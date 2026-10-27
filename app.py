import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title='Tuesday AI V21', page_icon='🥇', layout='wide')
st.title('🥇 Tuesday AI V21 — Backtest Risk Lab')
st.caption('Trade-ledger analysis only. Not a live signal engine or independent price-data backtest.')
file = st.file_uploader('Upload V20 audited trades CSV', type=['csv'])
if file is None:
    st.info('Upload Tuesday_AI_V20_audited_trades.csv to begin.')
    st.stop()
try:
    df = pd.read_csv(file)
except Exception as e:
    st.error(f'Cannot read CSV: {e}')
    st.stop()
required = {'entry_time', 'net_usd', 'gross_usd'}
missing = required - set(df.columns)
if missing:
    st.error('Missing columns: ' + ', '.join(sorted(missing)))
    st.stop()
df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce')
for col in ['net_usd', 'gross_usd']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['entry_time', 'net_usd', 'gross_usd']).sort_values('entry_time').reset_index(drop=True)
if df.empty:
    st.error('No valid trades found.')
    st.stop()
st.sidebar.header('Risk assumptions')
starting_equity = st.sidebar.number_input('Starting equity (USD)', min_value=1.0, value=1000.0, step=100.0)
extra_slippage = st.sidebar.number_input('Extra round-trip slippage (XAUUSD price units)', min_value=0.0, value=0.0, step=0.1)
extra_commission = st.sidebar.number_input('Extra commission per trade (USD)', min_value=0.0, value=0.0, step=0.5)
extra_spread = st.sidebar.number_input('Extra round-trip spread cost (XAUUSD price units)', min_value=0.0, value=0.0, step=0.1)
st.sidebar.caption('Extra costs are ADDED to costs already reflected in V20 net_usd. No double-counting the V20 baseline.')
# Infer dollars per XAUUSD price unit from ledger, if present; otherwise require manual setting.
if {'net_price', 'net_usd'}.issubset(df.columns):
    price = pd.to_numeric(df['net_price'], errors='coerce')
    ratio = (df['net_usd'] / price.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    inferred = float(ratio.median()) if len(ratio) else 1.0
else:
    inferred = 1.0
usd_per_price_unit = st.sidebar.number_input('USD per 1.0 XAUUSD price unit (per trade)', min_value=0.0001, value=max(0.0001, round(inferred, 4)), step=0.1, format='%.4f')
extra_cost = (extra_slippage + extra_spread) * usd_per_price_unit + extra_commission
df['stress_net_usd'] = df['net_usd'] - extra_cost
df['equity'] = starting_equity + df['stress_net_usd'].cumsum()
peak = pd.concat([pd.Series([starting_equity]), df['equity']], ignore_index=True).cummax().iloc[1:].reset_index(drop=True)
df['drawdown_usd'] = df['equity'] - peak
df['drawdown_pct'] = 100 * df['drawdown_usd'] / peak
wins = df.loc[df['stress_net_usd'] > 0, 'stress_net_usd'].sum()
losses = -df.loc[df['stress_net_usd'] < 0, 'stress_net_usd'].sum()
pf = wins / losses if losses > 0 else float('inf') if wins > 0 else 0.0
cols = st.columns(4)
cols[0].metric('Trades', len(df))
cols[1].metric('Stressed net P&L', f"${df['stress_net_usd'].sum():,.2f}")
cols[2].metric('Win rate', f"{(df['stress_net_usd'] > 0).mean():.1%}")
cols[3].metric('Profit factor', f'{pf:.2f}')
a, b = st.columns(2)
a.metric('Max drawdown (USD)', f"${df['drawdown_usd'].min():,.2f}")
b.metric('Max drawdown (%)', f"{df['drawdown_pct'].min():.2f}%")
st.subheader('Equity curve')
st.line_chart(df.set_index('entry_time')['equity'])
st.subheader('Chronological holdout (descriptive, not an independent validation)')
st.caption('The last 30% is held out chronologically for reporting. Because this is an already-generated trade ledger, it does NOT prove the strategy was developed without seeing holdout data. True out-of-sample validation requires freezing rules and rerunning on unseen price candles.')
split = max(1, min(len(df) - 1, int(len(df) * 0.7))) if len(df) > 1 else len(df)
rows = []
for label, subset in [('First 70%', df.iloc[:split]), ('Last 30%', df.iloc[split:])]:
    rows.append({'Period': label, 'Trades': len(subset), 'Net USD': round(subset['stress_net_usd'].sum(), 2), 'Win rate %': round(100 * (subset['stress_net_usd'] > 0).mean(), 2) if len(subset) else None})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.subheader('Cost sensitivity')
sensitivity = []
for add in [0, 0.5, 1, 2, 3, 5]:
    pnl = df['net_usd'] - extra_cost - add
    sensitivity.append({'Additional USD / trade': add, 'Net P&L USD': round(pnl.sum(), 2), 'Win rate %': round(100 * (pnl > 0).mean(), 2)})
st.dataframe(pd.DataFrame(sensitivity), hide_index=True, use_container_width=True)
st.download_button('Download V21 stress-tested trades CSV', df.to_csv(index=False).encode(), 'Tuesday_AI_V21_stress_trades.csv', 'text/csv')
st.warning('Backtests can overstate performance: candle ordering, intrabar TP/SL sequencing, data quality, execution, and unseen-market testing are not independently verified here. Do not treat these results as a live trading recommendation.')
