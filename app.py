import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title='Tuesday AI V15', page_icon='🥇', layout='wide')
st.title('🥇 Tuesday AI V15 — Backtest Audit')
st.caption('XAUUSD | Research dashboard — not live trade signals')

uploaded = st.file_uploader('Upload Tuesday AI V14 audit CSV', type=['csv'])
if uploaded is None:
    st.info('Upload Tuesday_AI_V14_audit.csv to begin.')
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f'Could not read CSV: {e}')
    st.stop()

required = {'Outcome', 'Setup', 'Side', 'R', 'Net_R'}
missing = required - set(df.columns)
if missing:
    st.error('Missing columns: ' + ', '.join(sorted(missing)))
    st.stop()

for c in ['R', 'Net_R', 'Risk', 'Cost_R']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
for c in ['Signal time', 'Entry time', 'Exit time']:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors='coerce')

outcomes = df['Outcome'].fillna('MISSING').astype(str).str.upper().str.strip()
resolved_flag = df['Resolved'].astype(str).str.lower().eq('true') if 'Resolved' in df else outcomes.isin(['WIN', 'LOSS'])
valid_flag = df['ValidRisk'].astype(str).str.lower().eq('true') if 'ValidRisk' in df else pd.Series(True, index=df.index)
valid = resolved_flag & valid_flag & outcomes.isin(['WIN', 'LOSS']) & df['Net_R'].notna()
trades = df.loc[valid].copy()
wins = (outcomes.loc[valid] == 'WIN').sum()
losses = (outcomes.loc[valid] == 'LOSS').sum()

st.subheader('📊 Verified outcomes')
a, b, c, d = st.columns(4)
a.metric('All signals', len(df))
b.metric('Resolved valid trades', len(trades))
c.metric('Win rate', f'{wins / len(trades) * 100:.1f}%' if len(trades) else 'N/A')
d.metric('Net R (resolved)', f'{trades.Net_R.sum():+.2f}R' if len(trades) else 'N/A')
st.caption(f'Wins: {wins} | Losses: {losses} | Excluded/unresolved: {len(df) - len(trades)}')

st.subheader('🔎 Excluded / unresolved signals')
excluded = df.loc[~valid].copy()
st.write(excluded['Outcome'].fillna('MISSING').value_counts().rename_axis('Outcome').reset_index(name='Count'))
if len(excluded):
    st.dataframe(excluded, use_container_width=True)
    st.download_button('Download excluded signals', excluded.to_csv(index=False).encode(), 'Tuesday_AI_V15_excluded.csv', 'text/csv')

st.subheader('📈 Performance by setup')
if len(trades):
    summary = trades.groupby('Setup', dropna=False).agg(Trades=('Net_R', 'size'), Net_R=('Net_R', 'sum'), Avg_R=('Net_R', 'mean'))
    summary['Win_rate_%'] = trades.assign(IsWin=outcomes.loc[valid].eq('WIN').astype(int)).groupby('Setup', dropna=False)['IsWin'].mean() * 100
    st.dataframe(summary.round(2), use_container_width=True)
    st.bar_chart(summary['Net_R'])

    st.subheader('📉 Equity curve & drawdown (in R)')
    order_col = 'Exit time' if 'Exit time' in trades.columns else ('Entry time' if 'Entry time' in trades.columns else None)
    if order_col:
        trades = trades.sort_values(order_col, kind='stable', na_position='last')
    trades['Cumulative_R'] = trades['Net_R'].cumsum()
    trades['Peak_R'] = trades['Cumulative_R'].cummax().clip(lower=0)
    trades['Drawdown_R'] = trades['Cumulative_R'] - trades['Peak_R']
    st.line_chart(trades[['Cumulative_R']].reset_index(drop=True))
    st.metric('Maximum drawdown', f'{trades.Drawdown_R.min():.2f}R')
    st.metric('Average net R / trade', f'{trades.Net_R.mean():+.3f}R')
    gross_profit = trades.loc[trades.Net_R > 0, 'Net_R'].sum()
    gross_loss = -trades.loc[trades.Net_R < 0, 'Net_R'].sum()
    st.metric('Profit factor', f'{gross_profit / gross_loss:.2f}' if gross_loss else 'N/A')
    st.subheader('📋 Audited resolved trades')
    st.dataframe(trades, use_container_width=True)
    st.download_button('Download V15 audited trades', trades.to_csv(index=False).encode(), 'Tuesday_AI_V15_audited.csv', 'text/csv')
else:
    st.warning('No valid resolved trades found. Check Outcome, Resolved, ValidRisk and Net_R columns.')

st.warning('Backtests can overstate results. DATA_GAP and ambiguous outcomes are excluded, not assumed to be wins. Confirm data quality, spread, slippage and out-of-sample results before risking real money.')
