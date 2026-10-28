import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Tuesday AI V40', layout='wide')
st.title('Tuesday AI V40 | XAUUSD M1 + M5')
st.caption('Demo research only. No live data, broker connection, or automatic orders. CSV timestamps determine sessions.')

with st.sidebar:
    st.header('Risk & execution')
    balance = st.number_input('Starting balance ($)', min_value=1.0, value=2500.0)
    daily_limit = st.number_input('Daily loss lock ($)', min_value=1.0, value=40.0)
    risk_pct = st.number_input('Risk per trade (%)', min_value=0.05, max_value=5.0, value=0.25, step=0.05)
    lot_min = st.number_input('Minimum lot', min_value=0.01, value=0.01, step=0.01)
    lot_step = st.number_input('Lot step', min_value=0.01, value=0.01, step=0.01)
    max_lot = st.number_input('Maximum lot', min_value=0.01, value=1.0, step=0.01)
    contract = st.number_input('Gold ounces per 1 lot (verify broker)', min_value=1.0, value=100.0)
    spread = st.number_input('Spread (price units, full round trip)', min_value=0.0, value=0.30, step=0.05)
    slip = st.number_input('Slippage per side (price units)', min_value=0.0, value=0.10, step=0.05)
    commission = st.number_input('Round-trip commission ($/lot)', min_value=0.0, value=7.0)
    rr = st.number_input('Reward/risk target', min_value=1.0, max_value=5.0, value=2.0, step=0.25)
    st.caption('Costs are illustrative. Confirm broker bid/ask, contract, lot and commissions.')

files = st.file_uploader('Upload HISTDATA XAUUSD M1 ZIP/CSV/TXT files', type=['zip','csv','txt'], accept_multiple_files=True)

def read_one(data):
    for sep in [';', ',', '\t']:
        try:
            d = pd.read_csv(io.BytesIO(data), sep=sep, header=None, engine='python', on_bad_lines='skip')
            if d.shape[1] >= 5:
                return d
        except Exception:
            pass
    raise ValueError('Unsupported CSV layout')

def parse_data(uploaded):
    parts=[]
    for f in uploaded:
        raw=f.getvalue()
        if f.name.lower().endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for n in z.namelist():
                    if n.lower().endswith(('.csv','.txt')) and not n.startswith('__MACOSX'):
                        parts.append(read_one(z.read(n)))
        else:
            parts.append(read_one(raw))
    if not parts:
        raise ValueError('No CSV/TXT data found')
    frames=[]
    for d in parts:
        d=d.dropna(how='all').copy()
        if d.shape[1] >= 7:
            dates=d.iloc[:,0].astype(str).str.strip()
            times=d.iloc[:,1].astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.zfill(6)
            dt=pd.to_datetime(dates+' '+times, format='%Y%m%d %H%M%S', errors='coerce')
            if dt.notna().mean() < 0.5:
                dt=pd.to_datetime(dates+' '+d.iloc[:,1].astype(str), errors='coerce')
            vals=d.iloc[:,2:6]
        else:
            dt=pd.to_datetime(d.iloc[:,0].astype(str).str.strip(), errors='coerce')
            vals=d.iloc[:,1:5]
        temp=pd.DataFrame(vals.to_numpy(),columns=['open','high','low','close'])
        temp['time']=dt.to_numpy()
        for c in ['open','high','low','close']:
            temp[c]=pd.to_numeric(temp[c],errors='coerce')
        frames.append(temp)
    out=pd.concat(frames,ignore_index=True).dropna().drop_duplicates('time').sort_values('time').set_index('time')
    out=out[(out.high>=out[['open','close','low']].max(axis=1)) & (out.low<=out[['open','close','high']].min(axis=1))]
    return out

def ema(s,n):
    return s.ewm(span=n,adjust=False,min_periods=n).mean()

def signals(m1):
    x=m1.copy()
    x['e20']=ema(x.close,20)
    x['e50']=ema(x.close,50)
    tr=pd.concat([x.high-x.low,(x.high-x.close.shift()).abs(),(x.low-x.close.shift()).abs()],axis=1).max(axis=1)
    x['atr']=tr.rolling(14,min_periods=14).mean()
    # M1 timestamps represent bar OPEN. Shift M5 availability forward to its completion.
    m5=x[['open','high','low','close']].resample('5min',label='right',closed='left').agg({'open':'first','high':'max','low':'min','close':'last'})
    counts=x.close.resample('5min',label='right',closed='left').count()
    m5=m5[counts==5].dropna()
    m5['fast']=ema(m5.close,20)
    m5['slow']=ema(m5.close,50)
    # For an M1 bar opening at T, only M5 bars ending at or before T are known.
    aligned=m5[['close','fast','slow']].reindex(x.index,method='ffill')
    x[['m5close','m5fast','m5slow']]=aligned
    prevhigh=x.high.shift(1).rolling(5,min_periods=5).max()
    prevlow=x.low.shift(1).rolling(5,min_periods=5).min()
    x['buy']=(x.m5fast>x.m5slow)&(x.m5close>x.m5fast)&(x.e20>x.e50)&(x.close>prevhigh)&(x.close>x.e20)
    x['sell']=(x.m5fast<x.m5slow)&(x.m5close<x.m5fast)&(x.e20<x.e50)&(x.close<prevlow)&(x.close<x.e20)
    return x

def backtest(x):
    trades=[]; equity=float(balance); pos=None; locked=set(); day_pnl={}; skipped=0
    costs=(spread+2*slip)*contract+commission
    for i in range(1,len(x)):
        bar=x.iloc[i]; prev=x.iloc[i-1]; t=x.index[i]; day=t.date()
        # Only next consecutive M1 bar is valid for entry/exit simulation.
        if t-x.index[i-1]!=pd.Timedelta(minutes=1):
            if pos is not None:
                # Conservative gap exit at next observed open, accounting for costs.
                direction=1 if pos['signal']=='BUY' else -1
                pnl=direction*(float(bar.open)-pos['entry_raw'])*pos['lot']*contract-costs*pos['lot']
                equity+=pnl; day_pnl[day]=day_pnl.get(day,0)+pnl
                trades.append({**pos,'exit_time':t,'exit':float(bar.open),'pnl':round(pnl,2),'equity':round(equity,2),'reason':'DATA_GAP'})
                pos=None
            skipped+=1
            continue
        if pos is not None:
            direction=1 if pos['signal']=='BUY' else -1
            stop=pos['sl']; target=pos['tp']
            if direction==1:
                px=(min(float(bar.open),stop) if bar.open<=stop else stop if bar.low<=stop else target if bar.high>=target else None)
            else:
                px=(max(float(bar.open),stop) if bar.open>=stop else stop if bar.high>=stop else target if bar.low<=target else None)
            # Floating mark-to-market lock, approximated from candle extremes.
            adverse=float(bar.low) if direction==1 else float(bar.high)
            floating=direction*(adverse-pos['entry_raw'])*pos['lot']*contract-costs*pos['lot']
            if px is None and day_pnl.get(day,0)+floating<=-daily_limit:
                # Exit at adverse candle extreme: pessimistic candle-only proxy.
                px=adverse
                reason='DAILY_LOCK_INTRABAR'
            else:
                reason='STOP_OR_TARGET'
            if px is not None:
                pnl=direction*(px-pos['entry_raw'])*pos['lot']*contract-costs*pos['lot']
                equity+=pnl; day_pnl[day]=day_pnl.get(day,0)+pnl
                trades.append({**pos,'exit_time':t,'exit':round(px,3),'pnl':round(pnl,2),'equity':round(equity,2),'reason':reason})
                pos=None
        if day_pnl.get(day,0)<=-daily_limit:
            locked.add(day)
        if pos is None and day not in locked and pd.notna(prev.atr) and pd.notna(prev.m5slow) and equity>0:
            side='BUY' if prev.buy else 'SELL' if prev.sell else None
            if side:
                entry=float(bar.open)
                distance=max(float(prev.atr)*1.5,spread+2*slip+0.01)
                stop=entry-distance if side=='BUY' else entry+distance
                target=entry+rr*distance if side=='BUY' else entry-rr*distance
                remaining=max(0,daily_limit+min(0,day_pnl.get(day,0)))
                budget=min(equity*risk_pct/100,remaining)
                perlot=distance*contract+costs
                lot=np.floor((budget/perlot+1e-12)/lot_step)*lot_step
                lot=round(min(float(lot),max_lot),8)
                if lot>=lot_min and budget>0:
                    pos={'entry_time':t,'signal':side,'entry_raw':entry,'entry':round(entry,3),'sl':round(stop,3),'tp':round(target,3),'lot':lot,'risk_budget':round(budget,2)}
    return pd.DataFrame(trades),equity,pos,locked,skipped

if files:
    try:
        m1=parse_data(files)
        if len(m1)<500:
            st.error('Need at least 500 M1 candles.'); st.stop()
        x=signals(m1)
        st.success(f'Loaded {len(m1):,} M1 bars: {m1.index.min()} to {m1.index.max()}')
        st.warning('Assumes CSV timestamps are M1 bar OPEN times. Candle-level backtests cannot guarantee fill order, execution, or a hard $40 cap. Verify broker-specific execution and timezone.')
        latest=x.iloc[-1]
        state='BUY' if latest.buy else 'SELL' if latest.sell else 'WAIT'
        st.metric('Last historical M1 candle signal (NOT live)',state)
        trades,final_equity,open_pos,locked,skipped=backtest(x)
        if trades.empty:
            st.warning('No closed trades. Check lot minimum, risk budget, and data.')
        else:
            a,b,c,d=st.columns(4)
            a.metric('Closed trades',len(trades))
            b.metric('Win rate',f'{100*(trades.pnl>0).mean():.1f}%')
            c.metric('Net P&L',f'${trades.pnl.sum():,.2f}')
            d.metric('Realized balance',f'${final_equity:,.2f}')
            curve=pd.concat([pd.Series([balance],index=[m1.index[0]]),pd.Series(trades.equity.to_numpy(),index=pd.to_datetime(trades.exit_time))]).sort_index()
            st.metric('Closed-equity max drawdown',f'${(curve.cummax()-curve).max():,.2f}')
            st.line_chart(curve)
            trades['month']=pd.to_datetime(trades.entry_time).dt.to_period('M').astype(str)
            st.subheader('Month-by-month evaluation (not proof of independent validation)')
            st.dataframe(trades.groupby('month').agg(trades=('pnl','size'),net_pnl=('pnl','sum'),win_rate=('pnl',lambda s:round(100*(s>0).mean(),1))))
            st.caption('Freeze rules before testing genuinely unseen months. No optimization is performed here.')
            st.dataframe(trades,hide_index=True)
            st.download_button('Download trades CSV',trades.to_csv(index=False).encode(),'Tuesday_AI_V40_audit_trades.csv','text/csv')
        st.caption(f'Locked days: {len(locked)} | Data gaps: {skipped}. Daily reset follows CSV calendar dates.')
        if open_pos:
            st.warning('Position open at end of file; unrealized P&L excluded.')
    except Exception as e:
        st.error(f'Input/data error: {e}')
else:
    st.info('Upload September and October 2026 HISTDATA XAUUSD M1 ZIPs to begin.')
