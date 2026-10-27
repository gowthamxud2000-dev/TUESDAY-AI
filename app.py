import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title='Tuesday AI V26 Fibonacci', page_icon='🥇', layout='wide')
st.title('🥇 Tuesday AI V26 — Fibonacci Backtest')
st.caption('Research only • XAUUSD M15 • No live execution • Confirmed pivots only')

with st.sidebar:
    capital = st.number_input('Starting balance ($)', min_value=100.0, value=1000.0, step=100.0)
    risk_pct = st.slider('Risk per trade (%)', 0.1, 1.0, 0.5, 0.1)
    spread = st.number_input('Spread in gold price units ($/oz)', min_value=0.0, value=0.30, step=0.05)
    slippage = st.number_input('Slippage per side ($/oz)', min_value=0.0, value=0.10, step=0.05)
    commission = st.number_input('Commission round-trip ($ per oz)', min_value=0.0, value=0.10, step=0.05)
    pivot = st.slider('Pivot confirmation bars', 2, 12, 5)
    min_stop = st.number_input('Minimum stop distance ($/oz)', min_value=0.1, value=1.0)
    st.caption('Costs are assumptions. Set to your broker’s actual contract and execution terms.')

uploaded = st.file_uploader('Upload historical XAUUSD OHLC M1/M15 CSV or tab-separated TXT', type=['csv','txt'])
st.info('V25 trades CSV alone cannot reconstruct a Fibonacci strategy. Upload original OHLC candles (e.g. DAT_ASCII_XAUUSD_M1_202608.csv).')

def load_candles(file):
    raw = file.getvalue()
    frame = None
    for sep in [None, '\t', ',', ';']:
        try:
            temp = pd.read_csv(BytesIO(raw), sep=sep, engine='python')
            if temp.shape[1] >= 5:
                frame = temp
                break
        except Exception:
            pass
    if frame is None:
        raise ValueError('Could not parse OHLC data')
    frame.columns = [str(c).strip().lower().replace('<','').replace('>','') for c in frame.columns]
    aliases = {'open':'open','high':'high','low':'low','close':'close', 'date':'date','time':'time','datetime':'datetime','timestamp':'datetime'}
    frame = frame.rename(columns={c: aliases.get(c,c) for c in frame.columns})
    if 'datetime' in frame:
        dates = pd.to_datetime(frame['datetime'], errors='coerce')
    elif 'date' in frame and 'time' in frame:
        dates = pd.to_datetime(frame['date'].astype(str)+' '+frame['time'].astype(str), errors='coerce')
    elif 'date' in frame:
        dates = pd.to_datetime(frame['date'], errors='coerce')
    else:
        raise ValueError('Need datetime or date/time columns')
    frame.index = dates
    frame = frame.loc[~frame.index.isna()]
    for col in ['open','high','low','close']:
        if col not in frame: raise ValueError('Missing '+col+' column')
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    frame = frame[['open','high','low','close']].dropna().sort_index()
    frame = frame[~frame.index.duplicated(keep='last')]
    if frame.empty: raise ValueError('No valid rows')
    # 15-minute OHLC bars; right label means all bar inputs are available at that timestamp
    frame = frame.resample('15min', label='right', closed='left').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    frame = frame[(frame.high >= frame[['open','close','low']].max(axis=1)) & (frame.low <= frame[['open','close','high']].min(axis=1))]
    return frame

def run_strategy(df, zone_low, zone_high):
    d = df.copy()
    d['ema50'] = d.close.ewm(span=50, adjust=False, min_periods=50).mean()
    d['ema200'] = d.close.ewm(span=200, adjust=False, min_periods=200).mean()
    # A pivot at bar j is not known until bar j+pivot has CLOSED.
    confirmed = []
    trades = []
    equity = capital
    peak = capital
    position = None
    pending = None
    for i in range(len(d)):
        row = d.iloc[i]
        if i >= 2*pivot:
            j = i-pivot
            window = d.iloc[j-pivot:j+pivot+1]
            if d.high.iloc[j] == window.high.max() and (window.high == d.high.iloc[j]).sum() == 1:
                confirmed.append(('H', float(d.high.iloc[j]), j))
            if d.low.iloc[j] == window.low.min() and (window.low == d.low.iloc[j]).sum() == 1:
                confirmed.append(('L', float(d.low.iloc[j]), j))
        if position is not None:
            # Conservative same-bar handling: SL first if SL and TP both touched.
            hit_stop = row.low <= position['stop'] if position['side']=='BUY' else row.high >= position['stop']
            hit_target = row.high >= position['target'] if position['side']=='BUY' else row.low <= position['target']
            if hit_stop or hit_target:
                intended = position['stop'] if hit_stop else position['target']
                # Gaps can worsen stops; no favorable fill beyond target assumed.
                if hit_stop:
                    intended = min(intended, row.open) if position['side']=='BUY' else max(intended, row.open)
                exit_fill = intended - (spread/2+slippage) if position['side']=='BUY' else intended+(spread/2+slippage)
                pnl = ((exit_fill-position['entry']) if position['side']=='BUY' else (position['entry']-exit_fill))*position['qty'] - commission*position['qty']
                equity += pnl
                peak = max(peak, equity)
                trades.append({**position, 'exit_time':d.index[i], 'exit':exit_fill, 'net_pnl':pnl,'equity':equity,'drawdown_pct':(peak-equity)/peak*100,'reason':'Stop' if hit_stop else 'Target'})
                position = None
        # Signal from completed bar i-1, entry on NEXT bar open i.
        if pending is not None and position is None and equity > 0:
            side, stop, target = pending
            entry = float(row.open) + (spread/2+slippage)*(1 if side=='BUY' else -1)
            dist = entry-stop if side=='BUY' else stop-entry
            if dist >= min_stop and ((target > entry) if side=='BUY' else (target < entry)):
                qty = equity*risk_pct/100 / dist
                position = {'entry_time':d.index[i], 'side':side,'entry':entry,'stop':stop,'target':target,'qty':qty}
            pending = None
            # Entry-bar exits are processed conservatively, too.
            if position is not None:
                hit_stop = row.low <= position['stop'] if side=='BUY' else row.high >= position['stop']
                hit_target = row.high >= position['target'] if side=='BUY' else row.low <= position['target']
                if hit_stop or hit_target:
                    intended = position['stop'] if hit_stop else position['target']
                    if hit_stop: intended = min(intended,row.open) if side=='BUY' else max(intended,row.open)
                    fill = intended - (spread/2+slippage) if side=='BUY' else intended+(spread/2+slippage)
                    pnl = ((fill-entry) if side=='BUY' else (entry-fill))*qty - commission*qty
                    equity += pnl
                    peak = max(peak,equity)
                    trades.append({**position,'exit_time':d.index[i],'exit':fill,'net_pnl':pnl,'equity':equity,'drawdown_pct':(peak-equity)/peak*100,'reason':'Stop' if hit_stop else 'Target'})
                    position=None
        if position is not None or pending is not None or i < 201: continue
        prev = d.iloc[i-1]
        if not np.isfinite(prev.ema200): continue
        # Only pivots confirmed by current closed candle are eligible.
        highs = [(v,j) for typ,v,j in confirmed if typ=='H']
        lows = [(v,j) for typ,v,j in confirmed if typ=='L']
        if not highs or not lows: continue
        h,hj = highs[-1]; l,lj = lows[-1]
        if hj <= lj and prev.ema50 > prev.ema200 and prev.close > prev.ema200 and h>l:
            # Last confirmed low precedes high: up impulse, retrace downward.
            a = h-(h-l)*zone_high; b = h-(h-l)*zone_low
            if row.low <= b and row.high >= a and row.close > row.open and row.close > a:
                stop = l - max(0.2,0.05*(h-l))
                entry_proxy = row.close+spread/2+slippage
                if entry_proxy>stop+min_stop: pending=('BUY',stop,entry_proxy+2*(entry_proxy-stop))
        elif lj <= hj and prev.ema50 < prev.ema200 and prev.close < prev.ema200 and h>l:
            a = l+(h-l)*zone_low; b = l+(h-l)*zone_high
            if row.high >= a and row.low <= b and row.close < row.open and row.close < b:
                stop = h + max(0.2,0.05*(h-l))
                entry_proxy = row.close-spread/2-slippage
                if stop>entry_proxy+min_stop: pending=('SELL',stop,entry_proxy-2*(stop-entry_proxy))
    return pd.DataFrame(trades)

def metrics(t):
    if t.empty: return {'Trades':0,'Win rate':0,'Net P&L':0,'Profit factor':0,'Max drawdown':0}
    wins = t.loc[t.net_pnl>0,'net_pnl'].sum()
    losses = -t.loc[t.net_pnl<0,'net_pnl'].sum()
    return {'Trades':len(t),'Win rate':round((t.net_pnl>0).mean()*100,2),'Net P&L':round(t.net_pnl.sum(),2),'Profit factor':round(wins/losses,2) if losses>0 else float('inf'),'Max drawdown':round(t.drawdown_pct.max(),2)}

if uploaded:
    try:
        df = load_candles(uploaded)
        st.write(f'Loaded {len(df):,} M15 bars: {df.index.min()} to {df.index.max()}')
        if len(df)<400: st.warning('Very limited history: need substantially more data for meaningful validation.')
        if st.button('Run Golden Zone vs Deep Retracement backtest', type='primary'):
            split = int(len(df)*0.7)
            # Re-run whole series once per candidate, then segment by entry time.
            # OOS is strictly after split, and no strategy parameters are fitted on OOS.
            for name,z1,z2 in [('Golden Zone',0.5,0.618),('Deep Retracement',0.618,0.786)]:
                trades=run_strategy(df,z1,z2)
                train=trades[trades.entry_time < df.index[split]] if not trades.empty else trades
                test=trades[trades.entry_time >= df.index[split]] if not trades.empty else trades
                st.subheader(name)
                st.dataframe(pd.DataFrame([{'Sample':'Training',**metrics(train)},{'Sample':'Out of sample',**metrics(test)}]),hide_index=True)
                if not trades.empty:
                    st.download_button(f'Download {name} trades CSV',trades.to_csv(index=False),file_name=f'Tuesday_AI_V26_{name.replace(" ","_")}_trades.csv',mime='text/csv')
            st.warning('Experimental backtest. Same-bar fills, liquidity, execution and data quality may differ from real trading. Do not use as funded-account proof.')
    except Exception as exc:
        st.error(f'Input issue: {exc}')
