import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title='Tuesday AI V22', page_icon='🥇', layout='wide')
st.title('🥇 Tuesday AI V22 — Risk-Control Stress Lab')
st.caption('Research simulator only. No live orders. V21 trade outcomes are historical and do not prove future performance.')
file = st.file_uploader('Upload V21 stress trades CSV', type=['csv'])
with st.sidebar:
    st.header('Simulation settings')
    starting = st.number_input('Starting balance ($)', min_value=100.0, value=1000.0, step=100.0)
    risk = st.slider('Risk per trade (%)', 0.25, 2.0, 1.0, 0.25)
    daily_limit = st.slider('Daily loss limit (%)', 1.0, 10.0, 3.0, 0.5)
    max_dd = st.slider('Maximum drawdown stop (%)', 3.0, 30.0, 15.0, 1.0)
    extra_cost_r = st.number_input('Extra execution cost (R per trade)', 0.0, 1.0, 0.05, 0.01, help='Added spread/slippage/commission stress beyond V21 costs, in units of initial stop risk. Not broker-verified.')
    loss_streak_limit = st.slider('Pause after consecutive losses', 2, 10, 4)

if file is None:
    st.info('Upload Tuesday_AI_V21_stress_trades.csv to run the V22 simulation.')
    st.stop()
try:
    df = pd.read_csv(file)
except Exception as exc:
    st.error(f'Could not read CSV: {exc}')
    st.stop()
required = {'entry_time', 'exit_time', 'side', 'entry', 'sl', 'exit', 'net_r'}
missing = required - set(df.columns)
if missing:
    st.error('Missing required columns: ' + ', '.join(sorted(missing)))
    st.stop()
if df.empty:
    st.error('CSV contains no trades.')
    st.stop()
for c in ['entry', 'sl', 'exit', 'net_r']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
for c in ['entry_time', 'exit_time']:
    df[c] = pd.to_datetime(df[c], errors='coerce')
invalid = df[list(required - {'side'})].isna().any(axis=1) | ((df['entry'] - df['sl']).abs() <= 0)
if invalid.any():
    st.warning(f'{invalid.sum()} invalid rows excluded.')
    df = df.loc[~invalid].copy()
if df.empty:
    st.stop()
df = df.sort_values(['entry_time', 'exit_time']).reset_index(drop=True)
if (df['exit_time'] < df['entry_time']).any():
    st.error('Exit earlier than entry detected. Fix the source CSV.')
    st.stop()
# Conservative sequential simulation: never overlap positions, no look-ahead on open trades.
balance = starting
peak = starting
day = None
day_start = starting
daily_block = False
streak = 0
halted = False
last_exit = pd.Timestamp.min
records = []
for _, trade in df.iterrows():
    entry_t = trade['entry_time']
    if entry_t.date() != day:
        day = entry_t.date()
        day_start = balance
        daily_block = False
        streak = 0
    reason = ''
    if halted:
        reason = 'Max drawdown halt'
    elif entry_t < last_exit:
        reason = 'Overlapping trade skipped'
    elif daily_block:
        reason = 'Daily loss lock'
    elif streak >= loss_streak_limit:
        reason = 'Loss streak lock'
    if reason:
        records.append({'entry_time': entry_t, 'side': trade['side'], 'status': reason, 'net_r': np.nan, 'pnl_usd': 0.0, 'equity': balance, 'drawdown_pct': (peak-balance)/peak*100})
        continue
    # net_r from V21 includes its original simulated costs; subtract incremental stress cost.
    realized_r = float(trade['net_r']) - extra_cost_r
    position_risk = balance * risk / 100
    pnl = position_risk * realized_r
    balance += pnl
    peak = max(peak, balance)
    dd = (peak-balance)/peak*100
    last_exit = trade['exit_time']
    streak = streak+1 if pnl < 0 else 0
    if balance <= day_start * (1-daily_limit/100):
        daily_block = True
    if dd >= max_dd or balance <= 0:
        halted = True
    records.append({'entry_time': entry_t, 'side': trade['side'], 'status': 'Executed', 'net_r': realized_r, 'pnl_usd': pnl, 'equity': balance, 'drawdown_pct': dd})
result = pd.DataFrame(records)
executed = result[result['status']=='Executed']
wins = int((executed['pnl_usd']>0).sum())
losses = int((executed['pnl_usd']<0).sum())
max_observed_dd = result['drawdown_pct'].max()
profit = balance - starting
c1,c2,c3,c4 = st.columns(4)
c1.metric('Final balance', f'${balance:,.2f}')
c2.metric('Net P&L', f'${profit:+,.2f}')
c3.metric('Win rate', f'{wins/len(executed)*100:.1f}%' if len(executed) else 'N/A')
c4.metric('Max drawdown', f'{max_observed_dd:.2f}%')
st.write(f'Executed: **{len(executed)}** | Skipped: **{len(result)-len(executed)}** | Losses: **{losses}**')
if halted:
    st.warning('Maximum drawdown threshold was reached: later trades were halted. A loss may overshoot the limit.')
st.subheader('Equity curve')
st.line_chart(pd.concat([pd.Series([starting]),result['equity']], ignore_index=True))
st.subheader('Trade audit')
st.dataframe(result, use_container_width=True)
st.download_button('Download V22 audited trades CSV', result.to_csv(index=False).encode(), 'Tuesday_AI_V22_audited_trades.csv', 'text/csv')
st.info('Limitations: V21 net_r already contains modeled costs. The extra cost is an additional R-based stress assumption, not independent bid/ask fill simulation. Intrabar stop/target ordering, gaps, variable spread, partial fills, broker lot rules and actual order latency are not verified. Day boundaries follow the CSV timestamps. Daily and drawdown locks act after a trade closes; they cannot cap a single trade loss.')
