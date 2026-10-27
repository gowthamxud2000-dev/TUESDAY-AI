import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Tuesday AI V16', page_icon='🥇', layout='wide')
st.title('🥇 Tuesday AI V16 — XAUUSD Historical Backtester')
st.caption('Research tool only. Uses completed-bar signals and next-bar entry; not live trading advice.')

with st.sidebar:
    st.header('Strategy settings')
    fast = st.number_input('Fast EMA', 2, 500, 50)
    slow = st.number_input('Slow EMA', 3, 1000, 200)
    lookback = st.number_input('Breakout lookback (bars)', 2, 100, 20)
    atr_n = st.number_input('ATR period', 2, 100, 14)
    atr_mult = st.number_input('Stop distance (ATR)', 0.1, 10.0, 1.5, 0.1)
    rr = st.number_input('Take profit / stop ratio', 0.2, 10.0, 2.0, 0.1)
    spread = st.number_input('Round-trip spread cost in price units ($/oz)', 0.0, 20.0, 0.30, 0.01)
    slip = st.number_input('Slippage per side in price units ($/oz)', 0.0, 20.0, 0.05, 0.01)
    commission = st.number_input('Round-trip commission in R', 0.0, 1.0, 0.0, 0.01)
    capital = st.number_input('Starting equity (USD)', 100.0, 10000000.0, 10000.0, 100.0)
    risk_pct = st.number_input('Risk per trade (%)', 0.1, 10.0, 1.0, 0.1)
    conservative = st.checkbox('If SL and TP hit in same bar, assume SL first', value=True, disabled=True)

st.info('CSV must contain OHLC columns (Open, High, Low, Close) and preferably Date/Time. MT5 tab-separated exports are supported. All costs are estimates; spread is charged as a round-trip price-unit deduction.')
file = st.file_uploader('Upload real XAUUSD CSV', type=['csv','txt'])


def load_prices(upload):
    raw = upload.getvalue()
    df = None
    for sep in [None, '\t', ',', ';']:
        try:
            test = pd.read_csv(io.BytesIO(raw), sep=sep, engine='python', encoding='utf-8-sig')
            if len(test.columns) >= 4:
                df = test
                break
        except Exception:
            continue
    if df is None:
        raise ValueError('CSV could not be read.')
    df.columns = [str(c).strip().lower().strip('<> ').replace(' ', '_') for c in df.columns]
    mapping = {}
    for key in ['open','high','low','close']:
        options = [c for c in df.columns if c == key or c.endswith('_'+key)]
        if not options:
            raise ValueError(f'Missing {key} column. Found: {list(df.columns)}')
        mapping[options[0]] = key
    df = df.rename(columns=mapping)
    if 'date' in df and 'time' in df:
        df['timestamp'] = pd.to_datetime(df['date'].astype(str)+' '+df['time'].astype(str), errors='coerce')
    else:
        choices = [c for c in ['timestamp','datetime','date','time'] if c in df]
        if choices:
            df['timestamp'] = pd.to_datetime(df[choices[0]], errors='coerce')
        else:
            df['timestamp'] = pd.RangeIndex(len(df))
            st.warning('No timestamp column found; CSV row order is assumed chronological.')
    for c in ['open','high','low','close']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['open','high','low','close','timestamp']).copy()
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    df = df[(df['high'] >= df[['open','close','low']].max(axis=1)) & (df['low'] <= df[['open','close','high']].min(axis=1))].reset_index(drop=True)
    if len(df) < max(slow, lookback, atr_n)+5:
        raise ValueError('Not enough valid candles for selected indicators.')
    return df


def backtest(df):
    d = df.copy()
    d['ema_fast'] = d.close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    d['ema_slow'] = d.close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    prev_close = d.close.shift(1)
    tr = pd.concat([d.high-d.low, (d.high-prev_close).abs(), (d.low-prev_close).abs()], axis=1).max(axis=1)
    d['atr'] = tr.ewm(alpha=1/atr_n, adjust=False, min_periods=atr_n).mean()
    d['prior_high'] = d.high.shift(1).rolling(lookback).max()
    d['prior_low'] = d.low.shift(1).rolling(lookback).min()
    d['buy'] = (d.ema_fast > d.ema_slow) & (d.close > d.prior_high) & (d.close > d.open)
    d['sell'] = (d.ema_fast < d.ema_slow) & (d.close < d.prior_low) & (d.close < d.open)
    trades = []
    equity = float(capital)
    i = max(slow, lookback, atr_n)
    while i < len(d)-1:
        signal = d.iloc[i]
        side = 1 if signal['buy'] else (-1 if signal['sell'] else 0)
        if side == 0 or not np.isfinite(signal.atr) or signal.atr <= 0:
            i += 1
            continue
        entry_idx = i+1
        entry = float(d.iloc[entry_idx].open)
        distance = float(signal.atr*atr_mult)
        stop = entry-side*distance
        target = entry+side*distance*rr
        exit_idx = len(d)-1
        exit_price = float(d.iloc[exit_idx].close)
        reason = 'End of data'
        for j in range(entry_idx, len(d)):
            bar = d.iloc[j]
            # Opening gaps are filled at the opening price, not the intended stop/target.
            if (side == 1 and bar.open <= stop) or (side == -1 and bar.open >= stop):
                exit_idx, exit_price, reason = j, float(bar.open), 'Stop gap'
                break
            if (side == 1 and bar.open >= target) or (side == -1 and bar.open <= target):
                exit_idx, exit_price, reason = j, float(bar.open), 'Target gap'
                break
            hit_stop = bar.low <= stop if side == 1 else bar.high >= stop
            hit_target = bar.high >= target if side == 1 else bar.low <= target
            if hit_stop:
                exit_idx, exit_price, reason = j, float(stop), 'Stop (conservative if both)'
                break
            if hit_target:
                exit_idx, exit_price, reason = j, float(target), 'Target'
                break
        gross_price = side*(exit_price-entry)
        net_r = (gross_price-spread-2*slip)/distance-commission
        before = equity
        pnl = before*risk_pct/100*net_r
        equity += pnl
        trades.append({'signal_time':signal.timestamp, 'entry_time':d.iloc[entry_idx].timestamp,
                       'exit_time':d.iloc[exit_idx].timestamp,'side':'BUY' if side==1 else 'SELL',
                       'entry':entry,'stop':stop,'target':target,'exit':exit_price,'exit_reason':reason,
                       'gross_R':gross_price/distance,'net_R':net_r,'pnl_usd':pnl,'equity_usd':equity})
        if equity <= 0:
            break
        i = exit_idx+1  # one position at a time; no same-bar re-entry
    return d, pd.DataFrame(trades)

if file:
    try:
        prices = load_prices(file)
        if fast >= slow:
            st.error('Fast EMA must be lower than Slow EMA.')
            st.stop()
        candles, trades = backtest(prices)
        st.success(f'Loaded {len(prices):,} valid candles. Period: {prices.timestamp.iloc[0]} to {prices.timestamp.iloc[-1]}')
        if trades.empty:
            st.warning('No trades found. Try more historical data or review parameters.')
        else:
            r = trades.net_R
            wins = r[r>0]
            losses = r[r<0]
            pf = wins.sum()/abs(losses.sum()) if len(losses) else float('inf')
            eq = pd.concat([pd.Series([capital]), trades.equity_usd], ignore_index=True)
            dd = (eq/eq.cummax()-1)*100
            metrics = [len(trades), f'{(r>0).mean()*100:.1f}%', f'{r.sum():+.2f}R',
                       f'{pf:.2f}' if np.isfinite(pf) else '∞', f'{r.mean():+.3f}R',
                       f'{dd.min():.2f}%']
            labels = ['Trades','Win rate','Sum of net R','Profit factor','Expectancy / trade','Max equity drawdown']
            for col, label, value in zip(st.columns(6), labels, metrics):
                col.metric(label, value)
            st.metric('Final simulated equity', f'${eq.iloc[-1]:,.2f}')
            st.subheader('Equity curve (USD)')
            st.line_chart(eq.rename('Equity'))
            st.subheader('Trade history')
            st.dataframe(trades, use_container_width=True)
            st.download_button('Download V16 trades CSV', trades.to_csv(index=False).encode(),
                               'Tuesday_AI_V16_trades.csv','text/csv')
            st.caption('One trade at a time. Signal on completed candle; entry at next open. No position sizing by contract/lot; USD equity is a risk-per-trade simulation. Net R deducts assumed spread/slippage and commission. No swap, funding, liquidity constraints or intrabar tick sequencing.')
    except Exception as e:
        st.error(f'Unable to run backtest: {e}')
