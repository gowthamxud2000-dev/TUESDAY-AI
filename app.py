import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title='Tuesday AI V20', page_icon='🥇', layout='wide')
st.title('🥇 Tuesday AI V20 — Cost-Aware Backtest Audit')
st.caption('Historical trade-log audit only. Not live signals or proof of executable returns.')
file = st.file_uploader('Upload Tuesday_AI_V19_trades.csv', type=['csv'])
st.sidebar.header('Trading cost assumptions')
spread = st.sidebar.number_input('Round-trip spread cost (XAUUSD price units)', min_value=0.0, value=0.30, step=0.05, format='%.2f', help='Example: 0.30 means $0.30 per ounce in price movement; do not double-count if spread is already in fills.')
slippage = st.sidebar.number_input('Slippage per side (price units)', min_value=0.0, value=0.10, step=0.05, format='%.2f')
commission = st.sidebar.number_input('Round-trip commission ($ per lot)', min_value=0.0, value=7.0, step=0.5)
lots = st.sidebar.number_input('Lots per trade', min_value=0.01, value=0.02, step=0.01)
contract = st.sidebar.number_input('Ounces per lot (check broker)', min_value=1.0, value=100.0, step=1.0)
st.sidebar.info('Defaults are examples, NOT verified broker charges. Adjust to your broker and contract.')
if file is None:
    st.info('Upload the V19 trades CSV to start. No data is stored by this app.')
    st.stop()
try:
    df = pd.read_csv(file)
except Exception as e:
    st.error(f'Cannot read CSV: {e}')
    st.stop()
required = ['signal_time','entry_time','exit_time','side','entry','sl','tp','exit','outcome','r_multiple']
missing = [c for c in required if c not in df.columns]
if missing:
    st.error('Missing columns: ' + ', '.join(missing))
    st.stop()
df = df.copy()
for c in ['entry','sl','tp','exit','r_multiple']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
for c in ['signal_time','entry_time','exit_time']:
    df[c] = pd.to_datetime(df[c], errors='coerce')
df['side'] = df['side'].astype(str).str.upper().str.strip()
df['risk_price'] = (df['entry']-df['sl']).abs()
df['direction'] = df['side'].map({'BUY':1,'SELL':-1})
df['gross_price'] = (df['exit']-df['entry'])*df['direction']
df['recomputed_gross_r'] = df['gross_price']/df['risk_price'].replace(0,np.nan)
df['cost_price'] = spread + 2*slippage + commission / (lots*contract)
df['net_price'] = df['gross_price']-df['cost_price']
df['net_r'] = df['net_price']/df['risk_price'].replace(0,np.nan)
df['net_usd'] = df['net_price']*lots*contract
df['gross_usd'] = df['gross_price']*lots*contract
issues = pd.Series('', index=df.index, dtype='object')
def flag(mask, msg):
    global issues
    issues = issues + np.where(mask.fillna(True), msg+'; ', '')
flag(df[required].isna().any(axis=1), 'Missing/invalid data')
flag(~df['side'].isin(['BUY','SELL']), 'Invalid side')
flag(df['risk_price'].le(0), 'Zero stop distance')
flag((df['side'].eq('BUY') & ((df['sl']>=df['entry']) | (df['tp']<=df['entry']))) | (df['side'].eq('SELL') & ((df['sl']<=df['entry']) | (df['tp']>=df['entry']))), 'Invalid SL/TP direction')
flag((df['entry_time'] < df['signal_time']) | (df['exit_time'] < df['entry_time']), 'Time ordering')
flag((df['recomputed_gross_r']-df['r_multiple']).abs()>0.03, 'R mismatch')
flag(df.duplicated(subset=['entry_time','side','entry','exit'], keep=False), 'Possible duplicate')
df['audit_issues'] = issues.str.rstrip('; ')
valid = df[df['audit_issues'].eq('')].sort_values('exit_time').copy()
st.subheader('Data validation')
a,b,c=st.columns(3)
a.metric('Uploaded rows',len(df));b.metric('Valid rows',len(valid));c.metric('Flagged rows',len(df)-len(valid))
if len(valid)==0:
    st.error('No valid rows. Review audit issues below.')
else:
    valid['cumulative_net_usd'] = valid['net_usd'].cumsum()
    valid['drawdown_usd'] = valid['cumulative_net_usd']-valid['cumulative_net_usd'].cummax().clip(lower=0)
    wins=valid.loc[valid.net_usd>0,'net_usd'].sum()
    losses=-valid.loc[valid.net_usd<0,'net_usd'].sum()
    pf= wins/losses if losses>0 else float('inf')
    st.subheader('Net results (valid trades only)')
    a,b,c,d=st.columns(4)
    a.metric('Net P/L',f'${valid.net_usd.sum():,.2f}')
    b.metric('Net win rate',f'{(valid.net_usd>0).mean()*100:.1f}%')
    c.metric('Profit factor',f'{pf:.2f}')
    d.metric('Max closed-trade DD',f'${-valid.drawdown_usd.min():,.2f}')
    st.metric('Total net R',f'{valid.net_r.sum():.2f}R')
    st.line_chart(valid.set_index('exit_time')['cumulative_net_usd'])
    st.caption('Closed-trade drawdown excludes unrealized intratrade drawdown. Costs are modeled, not measured fills.')
st.subheader('Full audit')
st.dataframe(df, use_container_width=True)
st.download_button('⬇️ Download V20 audited trades CSV',df.to_csv(index=False).encode('utf-8'),'Tuesday_AI_V20_audited_trades.csv','text/csv')
st.warning('This trade CSV cannot prove whether historical candle highs/lows really hit TP or SL, whether both were touched in the same bar, or whether entries were fillable. For those checks upload original M1 OHLC data in a future candle-level audit. Broker time zone and contract specifications must also be verified.')
