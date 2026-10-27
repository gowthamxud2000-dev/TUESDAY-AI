import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title='Tuesday AI V18', page_icon='🥇', layout='wide')
st.title('🥇 Tuesday AI V18 — XAUUSD Backtest Audit')
st.caption('Historical CSV analysis only. No live prices or verified trading signals.')

uploaded = st.file_uploader('Upload XAUUSD OHLC candles or completed trade history CSV', type=['csv'])
if uploaded is None:
    st.info('Upload a CSV to begin. OHLC columns: time, open, high, low, close. Trade-history columns: pnl (or profit).')
    st.stop()
try:
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f'Could not read CSV: {exc}')
    st.stop()
if df.empty:
    st.error('CSV has no rows.')
    st.stop()
df.columns = [str(c).strip().lower().replace(' ', '_').strip('<>') for c in df.columns]
st.write('Columns found:', ', '.join(df.columns))

pnl_col = next((c for c in ['pnl', 'net_pnl', 'profit', 'net_profit', 'profit_loss', 'pl'] if c in df.columns), None)
if pnl_col is not None:
    st.subheader('📊 Recorded trades audit')
    trades = df.copy()
    trades['net_result'] = pd.to_numeric(trades[pnl_col], errors='coerce')
    if 'commission' in trades.columns and pnl_col in ['profit', 'pl', 'profit_loss']:
        trades['net_result'] += pd.to_numeric(trades['commission'], errors='coerce').fillna(0)
    if 'swap' in trades.columns and pnl_col in ['profit', 'pl', 'profit_loss']:
        trades['net_result'] += pd.to_numeric(trades['swap'], errors='coerce').fillna(0)
    trades = trades.dropna(subset=['net_result']).copy()
    if trades.empty:
        st.error('No valid numeric trade P&L values found.')
        st.stop()
    date_col = next((c for c in ['exit_time', 'close_time', 'date', 'time', 'timestamp'] if c in trades.columns), None)
    if date_col:
        trades['_date'] = pd.to_datetime(trades[date_col], errors='coerce')
        if trades['_date'].notna().any():
            trades = trades.sort_values('_date', kind='stable')
    if 'trade_id' in trades.columns:
        st.caption('Each row is assumed to represent one closed trade; check for duplicate trade IDs.')
    equity_start = st.number_input('Starting account balance (account currency)', min_value=0.01, value=10000.0, step=100.0)
    pnl = trades['net_result']
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win = wins.sum()
    gross_loss = -losses.sum()
    factor = gross_win / gross_loss if gross_loss > 0 else np.nan
    equity = equity_start + pnl.cumsum()
    equity_curve = pd.concat([pd.Series([equity_start]), equity.reset_index(drop=True)], ignore_index=True)
    drawdown = equity_curve - equity_curve.cummax()
    max_dd = -drawdown.min()
    peak = equity_curve.cummax().replace(0, np.nan)
    max_dd_pct = -(drawdown / peak).min() * 100
    a,b,c,d = st.columns(4)
    a.metric('Trades', len(trades))
    b.metric('Win rate', f'{len(wins)/len(trades)*100:.2f}%')
    c.metric('Net P&L', f'{pnl.sum():,.2f}')
    d.metric('Profit factor', f'{factor:.2f}' if np.isfinite(factor) else 'N/A')
    e,f,g,h = st.columns(4)
    e.metric('Max closed-trade drawdown', f'{max_dd:,.2f}')
    f.metric('Max drawdown %', f'{max_dd_pct:.2f}%')
    g.metric('Avg trade', f'{pnl.mean():,.2f}')
    h.metric('Wins / Losses / Flat', f'{len(wins)} / {len(losses)} / {(pnl==0).sum()}')
    st.line_chart(pd.DataFrame({'Closed-trade equity': equity_curve}))
    if '_date' in trades and trades['_date'].notna().any():
        monthly = trades.dropna(subset=['_date']).assign(month=lambda x: x['_date'].dt.to_period('M').astype(str)).groupby('month', as_index=False).agg(trades=('net_result','size'), net_pnl=('net_result','sum'))
        st.subheader('Monthly performance')
        st.dataframe(monthly, use_container_width=True)
    st.subheader('Trades')
    st.dataframe(trades.drop(columns=['_date'], errors='ignore'), use_container_width=True)
    st.download_button('Download audited trades CSV', trades.drop(columns=['_date'], errors='ignore').to_csv(index=False).encode(), 'Tuesday_AI_V18_audited_trades.csv', 'text/csv')
    st.warning('Drawdown uses closed-trade P&L only, not intratrade equity. Results depend on the CSV accounting for spread, fees, slippage and complete trade history.')
elif all(c in df.columns for c in ['open','high','low','close']):
    st.subheader('📈 Historical candle data')
    time_col = next((c for c in ['time','datetime','timestamp','date'] if c in df.columns), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col]).sort_values(time_col).drop_duplicates(subset=[time_col])
    for c in ['open','high','low','close']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['open','high','low','close']).copy()
    df = df[(df['high'] >= df[['open','close','low']].max(axis=1)) & (df['low'] <= df[['open','close','high']].min(axis=1))]
    if len(df) < 220:
        st.warning('Need at least 220 valid candles for EMA 50/200 crossover evaluation.')
        st.stop()
    st.caption('Illustrative EMA 50/200 crossover test, not an optimized strategy. Signals use prior-bar EMAs; entries at next bar open.')
    spread = st.number_input('Round-trip cost per 1 oz in price units (spread + fees + slippage estimate)', min_value=0.0, value=0.5, step=0.1)
    ounces = st.number_input('Ounces per trade (broker contract size varies)', min_value=0.01, value=1.0, step=1.0)
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    position = 0
    entry = None
    entry_date = None
    rows = []
    for i in range(201, len(df)):
        prev = df.iloc[i-1]
        before = df.iloc[i-2]
        desired = 1 if prev.ema50 > prev.ema200 else -1
        prior_desired = 1 if before.ema50 > before.ema200 else -1
        if desired != prior_desired:
            price = float(df.iloc[i]['open'])
            dt = df.iloc[i][time_col] if time_col else i
            if position:
                rows.append({'entry_time': entry_date, 'exit_time': dt, 'side': 'BUY' if position == 1 else 'SELL', 'entry': entry, 'exit': price, 'pnl': ((price-entry)*position-spread)*ounces})
            position, entry, entry_date = desired, price, dt
    if position:
        st.info('Last open position excluded; no realized P&L until closed.')
    if not rows:
        st.warning('No completed crossover trades found. Upload more historical data.')
    else:
        result = pd.DataFrame(rows)
        st.metric('Completed trades', len(result))
        st.metric('Net realized P&L', f'{result.pnl.sum():,.2f}')
        st.metric('Win rate', f'{(result.pnl>0).mean()*100:.2f}%')
        st.dataframe(result, use_container_width=True)
        st.download_button('Download V18 crossover trades CSV', result.to_csv(index=False).encode(), 'Tuesday_AI_V18_crossover_trades.csv', 'text/csv')
        st.warning('Illustrative simplified backtest: no stop-loss, take-profit, intrabar execution, funding, liquidity or position sizing model. Not suitable for live-trading decisions.')
else:
    st.warning('This looks like a summary/audit CSV, not individual trades or OHLC candles. For full metrics, upload trade-level CSV with a P&L column or OHLC CSV with open/high/low/close.')
    st.dataframe(df.head(30), use_container_width=True)
