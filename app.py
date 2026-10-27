import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title='Tuesday AI V25', page_icon='🥇', layout='wide')
st.title('🥇 Tuesday AI V25 — Research Backtester')
st.warning('Research and demo use only. No live feed, broker connection, or guaranteed signals. Funded-account rules must be checked separately.')

with st.sidebar:
    st.header('Risk & execution assumptions')
    initial = st.number_input('Starting equity ($)', min_value=100.0, value=1000.0, step=100.0)
    risk_pct = st.slider('Risk per trade (%)', 0.1, 1.0, 0.25, 0.05)
    daily_limit = st.slider('Daily loss limit (%)', 0.5, 5.0, 2.0, 0.25)
    spread = st.number_input('Round-trip spread cost (price units)', min_value=0.0, value=0.25, step=0.05)
    slip = st.number_input('Slippage per side (price units)', min_value=0.0, value=0.10, step=0.05)
    commission = st.number_input('Round-trip commission ($ per 1 oz)', min_value=0.0, value=0.02, step=0.01)
    train_frac = st.slider('Training fraction (%)', 50, 85, 70, 5)

st.subheader('1. Review V24 audit')
audit = st.file_uploader('Optional: upload Tuesday_AI_V24_audited_trades.csv', type='csv', key='audit')
if audit:
    try:
        a = pd.read_csv(audit)
        if {'v24_status','v24_net_pnl','sample'}.issubset(a.columns):
            executed = a[a['v24_status'].astype(str).str.lower() == 'executed'].copy()
            executed['v24_net_pnl'] = pd.to_numeric(executed['v24_net_pnl'], errors='coerce')
            executed = executed.dropna(subset=['v24_net_pnl'])
            st.metric('Executed V24 trades', len(executed))
            st.metric('V24 net P&L', f"${executed['v24_net_pnl'].sum():,.2f}")
            st.dataframe(executed.groupby('sample')['v24_net_pnl'].agg(['count','sum','mean']), use_container_width=True)
        else:
            st.info('This CSV is not a recognized V24 audit. Upload the original V24 audit for comparison.')
    except Exception as exc:
        st.error(f'Could not read audit: {exc}')

st.subheader('2. Historical XAUUSD OHLC backtest')
st.caption('Upload genuine chronological M15 OHLC candles. M1 files may be resampled to M15 below. Never use a trade-report CSV as candle data.')
price_file = st.file_uploader('Upload XAUUSD M1/M15 CSV or tab-separated TXT', type=['csv','txt'], key='ohlc')

def load_ohlc(file):
    file.seek(0)
    try:
        d = pd.read_csv(file, sep=None, engine='python')
    except Exception:
        file.seek(0)
        d = pd.read_csv(file, sep=r'[,;\t ]+', engine='python', header=None)
    d.columns = [str(x).strip().lower().replace('<','').replace('>','') for x in d.columns]
    if not {'open','high','low','close'}.issubset(d.columns):
        file.seek(0)
        d = pd.read_csv(file, sep=r'[,;\t ]+', engine='python', header=None)
        if d.shape[1] >= 6:
            d = d.iloc[:, :6]
            d.columns = ['date','time','open','high','low','close']
        elif d.shape[1] == 5:
            d.columns = ['datetime','open','high','low','close']
        else:
            raise ValueError('Need date/time and open, high, low, close columns.')
    if 'datetime' in d:
        dates = pd.to_datetime(d['datetime'], errors='coerce')
    elif 'date' in d and 'time' in d:
        dates = pd.to_datetime(d['date'].astype(str) + ' ' + d['time'].astype(str), errors='coerce')
    elif 'date' in d:
        dates = pd.to_datetime(d['date'], errors='coerce')
    else:
        raise ValueError('Missing datetime/date column.')
    d.index = dates
    d = d.loc[d.index.notna(), ['open','high','low','close']].copy()
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna().sort_index()
    d = d[~d.index.duplicated(keep='first')]
    d = d[(d.high >= d[['open','close','low']].max(axis=1)) & (d.low <= d[['open','close']].min(axis=1))]
    return d

def backtest(raw, equity, risk, day_loss, cost_spread, cost_slip, cost_commission, frac):
    # 15-minute candle bars; decisions use only previously completed bars.
    o = raw.resample('15min', label='left', closed='left').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    o['ema50'] = o.close.ewm(span=50, adjust=False).mean()
    o['ema200'] = o.close.ewm(span=200, adjust=False).mean()
    prev_close = o.close.shift(1)
    tr = pd.concat([o.high-o.low, (o.high-prev_close).abs(), (o.low-prev_close).abs()], axis=1).max(axis=1)
    o['atr'] = tr.rolling(14, min_periods=14).mean()
    o['break_high'] = o.high.shift(1).rolling(20).max()
    o['break_low'] = o.low.shift(1).rolling(20).min()
    o = o.dropna().copy()
    if len(o) < 300:
        raise ValueError('Need at least 300 usable M15 candles after indicator warm-up.')
    split = int(len(o)*frac/100)
    if split < 100 or len(o)-split < 50:
        raise ValueError('Not enough candles for train/test split.')
    rows = []
    cash = equity
    peak = equity
    day_start = equity
    current_day = None
    pending = None
    position = None
    for i in range(1, len(o)):
        now = o.iloc[i]
        previous = o.iloc[i-1]
        ts = o.index[i]
        if current_day != ts.date():
            current_day = ts.date()
            day_start = cash
        # Exit open positions first; if both stop and target touched, assume stop first.
        if position is not None:
            side = position['side']
            stop_hit = now.low <= position['stop'] if side == 1 else now.high >= position['stop']
            target_hit = now.high >= position['target'] if side == 1 else now.low <= position['target']
            if stop_hit or target_hit:
                exit_raw = position['stop'] if stop_hit else position['target']
                # Gaps can produce a worse stop fill.
                if stop_hit:
                    exit_raw = min(exit_raw, now.open) if side == 1 else max(exit_raw, now.open)
                exit_fill = exit_raw - side*(cost_spread/4 + cost_slip)
                gross = side*(exit_fill-position['entry'])*position['qty']
                fees = cost_commission*position['qty']
                pnl = gross-fees
                cash += pnl
                peak = max(peak, cash)
                rows.append({'entry_time':position['time'], 'exit_time':ts, 'sample':position['sample'],
                             'side':'BUY' if side==1 else 'SELL', 'entry':position['entry'],
                             'exit':exit_fill, 'stop':position['stop'], 'target':position['target'],
                             'qty_oz':position['qty'], 'net_pnl':pnl, 'equity':cash,
                             'drawdown_pct':100*(peak-cash)/peak, 'reason':'Stop' if stop_hit else 'Target'})
                position = None
        # Pending signal was generated at previous close, enters current open.
        if position is None and pending is not None:
            if cash > 0 and cash > day_start*(1-day_loss/100):
                side, atr, sample = pending
                entry = now.open + side*(cost_spread/4 + cost_slip)
                stop_distance = max(1.5*atr, 0.01)
                qty = max(0.0, cash*risk/100/(stop_distance+cost_spread/2+2*cost_slip+cost_commission))
                if qty > 0:
                    position = {'time':ts, 'side':side, 'entry':entry, 'stop':entry-side*stop_distance,
                                'target':entry+side*stop_distance*1.8, 'qty':qty, 'sample':sample}
            pending = None
        if position is None and pending is None and cash > day_start*(1-day_loss/100):
            # Signals are formed from current completed candle; filled NEXT bar only.
            if now.close > now.break_high and now.ema50 > now.ema200:
                pending = (1, now.atr, 'Training' if i < split else 'Out-of-sample')
            elif now.close < now.break_low and now.ema50 < now.ema200:
                pending = (-1, now.atr, 'Training' if i < split else 'Out-of-sample')
    # Open position excluded from realized P&L and shown explicitly.
    return pd.DataFrame(rows), o, position is not None

if price_file:
    try:
        raw = load_ohlc(price_file)
        st.write(f'Valid input candles: {len(raw):,} | {raw.index.min()} to {raw.index.max()}')
        if st.button('Run V25 historical backtest', type='primary'):
            trades, candles, open_position = backtest(raw, initial, risk_pct, daily_limit, spread, slip, commission, train_frac)
            if trades.empty:
                st.warning('No closed trades under these settings. Try a longer historical dataset.')
            else:
                st.subheader('Backtest results')
                st.metric('Closed trades', len(trades))
                st.metric('Net P&L', f"${trades.net_pnl.sum():,.2f}")
                st.metric('Win rate', f"{100*(trades.net_pnl>0).mean():.1f}%")
                st.metric('Max realized-equity drawdown', f"{trades.drawdown_pct.max():.2f}%")
                st.dataframe(trades.groupby('sample').net_pnl.agg(['count','sum','mean']), use_container_width=True)
                st.line_chart(trades.set_index('exit_time')['equity'])
                st.download_button('Download V25 trades CSV', trades.to_csv(index=False).encode(), 'Tuesday_AI_V25_trades.csv', 'text/csv')
                st.dataframe(trades.tail(20), use_container_width=True)
                if open_position:
                    st.info('One position remained open at the end of the dataset; unrealized P&L is not included.')
                st.caption('Caveats: illustrative 1-oz price-unit model, fixed user-supplied costs, no swap, liquidity or margin modeling; intrabar path unknown. Daily limit checks realized P&L only, not floating losses. Split is chronological but parameters are not optimized or walk-forward validated. Do not treat as broker-executable results.')
    except Exception as exc:
        st.error(f'Input/backtest error: {exc}')
