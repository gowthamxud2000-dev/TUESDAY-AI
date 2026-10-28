
import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Tuesday AI V30', page_icon='🥇', layout='wide')
st.title('🥇 Tuesday AI V30')
st.caption('XAUUSD M15 | Manual signals + historical backtest | No auto orders')
st.warning('Experimental research tool. Historical performance does not guarantee future profits.')

@st.cache_data(show_spinner=False)
def load_ohlc(raw, filename):
    if filename.lower().endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(('.csv', '.txt')) and not n.startswith('__MACOSX')]
            if not names:
                raise ValueError('ZIP contains no CSV/TXT data')
            raw = archive.read(names[0])
    text = raw.decode('utf-8-sig', errors='replace')
    first = next((line for line in text.splitlines() if line.strip()), '')
    if not first:
        raise ValueError('Empty file')
    sep = ';' if ';' in first else ('\t' if '\t' in first else ',')
    has_header = any(word in first.lower() for word in ['datetime', 'date,', 'date;', '<date>', 'open,', 'open;'])
    raw_df = pd.read_csv(io.StringIO(text), sep=sep, header=0 if has_header else None, low_memory=False)
    if has_header:
        raw_df.columns = [str(c).strip().lower().replace('<', '').replace('>', '') for c in raw_df.columns]
        if {'date','time','open','high','low','close'}.issubset(raw_df.columns):
            dates = raw_df['date'].astype(str) + ' ' + raw_df['time'].astype(str)
        elif 'datetime' in raw_df.columns:
            dates = raw_df['datetime'].astype(str)
        elif 'date' in raw_df.columns:
            dates = raw_df['date'].astype(str)
        else:
            raise ValueError('Missing date/time column')
        values = raw_df[['open','high','low','close']].copy()
    else:
        if raw_df.shape[1] >= 7:
            dates = raw_df.iloc[:,0].astype(str) + ' ' + raw_df.iloc[:,1].astype(str)
            values = raw_df.iloc[:,2:6].copy()
        elif raw_df.shape[1] >= 5:
            dates = raw_df.iloc[:,0].astype(str)
            values = raw_df.iloc[:,1:5].copy()
        else:
            raise ValueError('Expected date and OHLC columns')
        values.columns = ['open','high','low','close']
    dt = pd.to_datetime(dates, format='mixed', errors='coerce')
    if dt.isna().all():
        dt = pd.to_datetime(dates, dayfirst=True, errors='coerce')
    df = pd.DataFrame({c: pd.to_numeric(values[c], errors='coerce').to_numpy() for c in ['open','high','low','close']}, index=pd.DatetimeIndex(dt))
    df = df.loc[~df.index.isna()].dropna().sort_index()
    df = df[~df.index.duplicated(keep='last')]
    df = df[(df.low > 0) & (df.high >= df[['open','close','low']].max(axis=1)) & (df.low <= df[['open','close','high']].min(axis=1))]
    if df.empty:
        raise ValueError('No valid OHLC rows')
    return df

def to_m15(df, source):
    if source == 'M15':
        return df
    count = df['close'].resample('15min').count()
    bars = df.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last'})
    return bars.loc[count == 15].dropna()

def indicators(df):
    d = df.copy()
    d['ema50'] = d.close.ewm(span=50, adjust=False).mean()
    d['ema200'] = d.close.ewm(span=200, adjust=False).mean()
    prev = d.close.shift(1)
    tr = pd.concat([(d.high-d.low), (d.high-prev).abs(), (d.low-prev).abs()],axis=1).max(axis=1)
    d['atr'] = tr.ewm(alpha=1/14,adjust=False).mean()
    return d

def signal_at(d, i, lookback, rr, buffer):
    if i < max(250,lookback+2):
        return None
    current = d.iloc[i]
    prior = d.iloc[i-1]
    window = d.iloc[i-lookback:i]
    hi, lo = float(window.high.max()), float(window.low.min())
    rng = hi-lo
    atr = float(current.atr)
    if rng <= 0 or atr <= 0 or not np.isfinite(atr):
        return None
    buy_lo, buy_hi = hi-0.618*rng, hi-0.5*rng
    sell_lo, sell_hi = lo+0.5*rng, lo+0.618*rng
    up = current.ema50 > current.ema200 and current.close > current.ema200
    down = current.ema50 < current.ema200 and current.close < current.ema200
    buy = up and current.low <= buy_hi and current.high >= buy_lo and current.close > current.open and current.close > prior.high
    sell = down and current.high >= sell_lo and current.low <= sell_hi and current.close < current.open and current.close < prior.low
    if buy:
        sl = min(float(current.low),buy_lo)-buffer*atr
        if sl < current.close:
            return {'side':'BUY','sl':sl,'signal_close':float(current.close),'signal_time':d.index[i]}
    if sell:
        sl = max(float(current.high),sell_hi)+buffer*atr
        if sl > current.close:
            return {'side':'SELL','sl':sl,'signal_close':float(current.close),'signal_time':d.index[i]}
    return None

def backtest(d, lookback, rr, buffer, spread, slippage, fee, max_hold, dollars_per_unit):
    records=[]
    i=max(250,lookback+2)
    while i < len(d)-1:
        s=signal_at(d,i,lookback,rr,buffer)
        if s is None:
            i+=1
            continue
        side=s['side']
        direction=1 if side=='BUY' else -1
        next_bar=d.iloc[i+1]
        entry=float(next_bar.open)+direction*(spread/2+slippage)
        sl=s['sl']
        risk=(entry-sl)*direction
        if risk<=0:
            i+=1
            continue
        tp=entry+direction*rr*risk
        exit_price=None
        reason='TIME'
        last_j=min(len(d)-1,i+max_hold)
        exit_j=last_j
        for j in range(i+1,last_j+1):
            b=d.iloc[j]
            if side=='BUY':
                hit_sl=b.low<=sl
                hit_tp=b.high>=tp
            else:
                hit_sl=b.high>=sl
                hit_tp=b.low<=tp
            if hit_sl:
                exit_price=sl-direction*(spread/2+slippage)
                reason='SL (conservative if both hit)'
            elif hit_tp:
                exit_price=tp-direction*(spread/2+slippage)
                reason='TP'
            if exit_price is not None:
                exit_j=j
                break
        if exit_price is None:
            exit_price=float(d.iloc[exit_j].close)-direction*(spread/2+slippage)
        pnl=direction*(exit_price-entry)*dollars_per_unit-fee
        records.append({'signal_time':s['signal_time'],'entry_time':d.index[i+1], 'exit_time':d.index[exit_j], 'side':side, 'entry':entry, 'sl':sl, 'tp':tp, 'exit':exit_price, 'reason':reason, 'pnl_usd':pnl})
        i=exit_j+1
    return pd.DataFrame(records)

st.sidebar.header('Settings')
source=st.sidebar.selectbox('Input timeframe',['M15','M1'])
lookback=st.sidebar.number_input('Swing lookback',20,200,60)
rr=st.sidebar.slider('Reward / Risk',1.0,4.0,2.0,0.25)
buffer=st.sidebar.slider('SL ATR buffer',0.0,2.0,0.3,0.1)
spread=st.sidebar.number_input('Round-trip spread in XAUUSD price units',0.0,10.0,0.30,0.05)
slippage=st.sidebar.number_input('Slippage per fill in price units',0.0,10.0,0.10,0.05)
fee=st.sidebar.number_input('Commission per completed trade ($)',0.0,100.0,0.0,0.5)
dollars_per_unit=st.sidebar.number_input('USD P/L per $1 gold move (position exposure)',0.01,10000.0,1.0,0.01)
max_hold=st.sidebar.number_input('Max holding candles',1,500,32)
balance=st.sidebar.number_input('Starting balance ($)',10.0,1000000.0,2500.0,10.0)
st.sidebar.caption('Adjust cost and exposure assumptions to your broker. No position sizing automation.')
files=st.file_uploader('Upload XAUUSD historical CSV/TXT/ZIP (one or more months)',type=['csv','txt','zip'],accept_multiple_files=True)
if files:
    try:
        frames=[load_ohlc(f.getvalue(),f.name) for f in files]
        raw=pd.concat(frames).sort_index()
        raw=raw[~raw.index.duplicated(keep='last')]
        d=to_m15(raw,source)
        # Conservatively exclude last bar, which could be incomplete.
        d=d.iloc[:-1]
        if len(d)<260:
            st.error('Need at least 260 completed M15 candles')
            st.stop()
        d=indicators(d)
        st.success(f'{len(d):,} completed M15 candles loaded')
        st.caption(f'Data range: {d.index.min()} to {d.index.max()} (timestamps as supplied)')
        current=signal_at(d,len(d)-1,int(lookback),float(rr),float(buffer))
        st.subheader('Latest historical signal')
        if current:
            st.success(f"{current['side']} — confirmation at {current['signal_time']}") if current['side']=='BUY' else st.error(f"{current['side']} — confirmation at {current['signal_time']}")
            st.write('Reference signal close:',round(current['signal_close'],2),'| Proposed SL:',round(current['sl'],2))
            st.caption('Actual entry would require the next completed candle/open; no future fill assumed here.')
        else:
            st.warning('WAIT — no fully confirmed setup')
        st.subheader('Historical backtest')
        if st.button('Run V30 Backtest',type='primary'):
            trades=backtest(d,int(lookback),float(rr),float(buffer),float(spread),float(slippage),float(fee),int(max_hold),float(dollars_per_unit))
            if trades.empty:
                st.info('No completed trades with these settings.')
            else:
                pnl=trades.pnl_usd
                equity=balance+pnl.cumsum()
                peaks=pd.concat([pd.Series([balance]),equity.reset_index(drop=True)]).cummax().iloc[1:].to_numpy()
                drawdown=(equity.to_numpy()-peaks)
                gains=pnl[pnl>0].sum()
                losses=-pnl[pnl<0].sum()
                pf=gains/losses if losses>0 else float('inf') if gains>0 else 0
                a,b,c,dcol=st.columns(4)
                a.metric('Trades',len(trades))
                b.metric('Win rate',f'{(pnl>0).mean()*100:.1f}%')
                c.metric('Net P/L',f'${pnl.sum():,.2f}')
                dcol.metric('Profit factor',f'{pf:.2f}')
                st.metric('Max dollar drawdown',f'${abs(min(0,float(drawdown.min()))):,.2f}')
                st.metric('Ending balance',f'${balance+pnl.sum():,.2f}')
                st.line_chart(pd.DataFrame({'Equity':equity.to_numpy()},index=trades.exit_time))
                st.dataframe(trades,use_container_width=True)
                st.download_button('Download V30 trades CSV',trades.to_csv(index=False).encode('utf-8'),'Tuesday_AI_V30_trades.csv','text/csv')
                st.warning('Backtest assumptions: one position at a time, no compounding, no margin model. Both SL and TP in one candle count as SL. Intrabar gaps and execution may differ. Costs are estimates; no independent out-of-sample validation yet.')
    except Exception as exc:
        st.error(f'Data processing error: {exc}')
else:
    st.info('Upload historical data to analyze signals and run the backtest.')
