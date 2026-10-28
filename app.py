
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title='Tuesday AI V38', page_icon='📈', layout='wide')
st.title('Tuesday AI V38 | M1 + M5 Gold Scalping')
st.caption('Historical research only • Manual trading • No live feed')

with st.sidebar:
    balance = st.number_input('Starting balance ($)', min_value=1.0, value=100.0)
    risk_pct = st.number_input('Risk per trade (%)', min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    daily_cap = st.number_input('Daily loss cap ($)', min_value=0.1, value=3.0)
    min_lot = st.number_input('Minimum lot', min_value=0.001, value=0.01, format='%.3f')
    contract = st.number_input('Ounces per lot (verify with broker)', min_value=1.0, value=100.0)
    spread = st.number_input('Assumed spread (XAUUSD price)', min_value=0.0, value=0.30)
    slippage = st.number_input('Slippage per side (price)', min_value=0.0, value=0.10)
    commission = st.number_input('Round-trip commission per lot ($)', min_value=0.0, value=7.0)
    rr = st.number_input('Reward / risk', min_value=1.0, value=1.5)
    max_trades = st.number_input('Max trades per day', min_value=1, value=3)
    stop_buffer = st.number_input('Stop buffer (price)', min_value=0.01, value=0.10)


def load_csv(upload):
    raw = upload.getvalue()
    sample = raw[:2000].decode('utf-8-sig', errors='replace')
    first = sample.splitlines()[0]
    sep = ';' if first.count(';') > first.count(',') else (',' if ',' in first else '\t')
    header = 0 if any(x in first.lower() for x in ('open', 'close', 'date', 'time', 'timestamp')) else None
    frame = pd.read_csv(BytesIO(raw), sep=sep, header=header, encoding='utf-8-sig')
    if header is None:
        if frame.shape[1] >= 6 and str(frame.iloc[0, 1]).count(':') >= 1:
            frame = frame.iloc[:, :6].copy()
            frame.columns = ['date', 'time', 'open', 'high', 'low', 'close']
        elif frame.shape[1] >= 5:
            frame = frame.iloc[:, :5].copy()
            frame.columns = ['datetime', 'open', 'high', 'low', 'close']
        else:
            raise ValueError('Expected at least 5 columns.')
    frame.columns = [str(c).strip().lower().replace('<', '').replace('>', '') for c in frame.columns]
    if 'datetime' in frame:
        raw_time = frame['datetime'].astype(str)
    elif 'timestamp' in frame:
        raw_time = frame['timestamp'].astype(str)
    elif 'date' in frame and 'time' in frame:
        raw_time = frame['date'].astype(str) + ' ' + frame['time'].astype(str)
    elif 'date' in frame:
        raw_time = frame['date'].astype(str)
    else:
        raise ValueError('Missing date/time column.')
    raw_time = raw_time.str.strip()
    if raw_time.str.match(r'^\d{8} \d{6}$').mean() > .9:
        dates = pd.to_datetime(raw_time, format='%Y%m%d %H%M%S', errors='coerce')
    elif raw_time.str.match(r'^\d{8} \d{4}$').mean() > .9:
        dates = pd.to_datetime(raw_time, format='%Y%m%d %H%M', errors='coerce')
    else:
        dates = pd.to_datetime(raw_time, errors='coerce')
    frame['dt'] = dates
    for c in ['open', 'high', 'low', 'close']:
        if c not in frame:
            raise ValueError('Missing OHLC column: ' + c)
        frame[c] = pd.to_numeric(frame[c], errors='coerce')
    invalid = frame['dt'].isna().sum()
    frame = frame.dropna(subset=['dt', 'open', 'high', 'low', 'close'])
    duplicates = frame.duplicated('dt').sum()
    frame = frame.drop_duplicates('dt').sort_values('dt').set_index('dt')
    frame = frame[['open', 'high', 'low', 'close']]
    frame = frame[(frame.high >= frame[['open', 'close', 'low']].max(axis=1)) &
                  (frame.low <= frame[['open', 'close', 'high']].min(axis=1))]
    if len(frame) < 500:
        raise ValueError('At least 500 valid M1 candles required.')
    diffs = frame.index.to_series().diff().dropna()
    minute_share = float((diffs == pd.Timedelta(minutes=1)).mean())
    return frame, invalid, duplicates, minute_share


def candles(frame, rule):
    return frame.resample(rule, label='right', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()


def build_signals(m1, timeframe):
    base = candles(m1, timeframe)
    higher = candles(m1, '5min' if timeframe == '1min' else '15min')
    higher['ema50'] = higher.close.ewm(span=50, adjust=False).mean()
    higher['ema200'] = higher.close.ewm(span=200, adjust=False).mean()
    higher['trend'] = np.where(higher.ema50 > higher.ema200, 1, -1)
    trend = higher.trend.reindex(base.index, method='ffill').fillna(0)
    ph = base.high.shift(1).rolling(10).max()
    pl = base.low.shift(1).rolling(10).min()
    buy = (trend == 1) & (base.close > ph) & (base.close > base.open)
    sell = (trend == -1) & (base.close < pl) & (base.close < base.open)
    base['signal'] = np.select([buy, sell], [1, -1], default=0)
    return base


def run_backtest(base, timeframe):
    trades, rejects = [], []
    equity = balance
    day, day_start, count = None, balance, 0
    i = 220
    while i < len(base) - 1:
        ts = base.index[i]
        if day != ts.date():
            day, day_start, count = ts.date(), equity, 0
        row = base.iloc[i]
        side = int(row.signal)
        if side == 0:
            i += 1
            continue
        if count >= max_trades or (day_start - equity) >= daily_cap:
            rejects.append({'time': ts, 'reason': 'Daily limit'})
            i += 1
            continue
        next_bar = base.iloc[i + 1]
        entry = float(next_bar.open) + side * (spread / 2 + slippage)
        stop = float(row.low - stop_buffer if side == 1 else row.high + stop_buffer)
        stop_fill = stop - side * (spread / 2 + slippage)
        distance = side * (entry - stop_fill)
        if distance <= 0:
            rejects.append({'time': ts, 'reason': 'Invalid stop'})
            i += 1
            continue
        estimated_risk = distance * contract * min_lot + commission * min_lot
        remaining = daily_cap - (day_start - equity)
        if estimated_risk > min(equity * risk_pct / 100, remaining):
            rejects.append({'time': ts, 'reason': 'Minimum lot exceeds risk limit'})
            i += 1
            continue
        target = entry + side * rr * distance
        exit_i, exit_fill, reason = len(base) - 1, None, 'END'
        for j in range(i + 1, len(base)):
            b = base.iloc[j]
            hit_sl = (b.low <= stop) if side == 1 else (b.high >= stop)
            hit_tp = (b.high >= target + spread / 2) if side == 1 else (b.low <= target - spread / 2)
            if hit_sl or hit_tp:
                exit_i, reason = j, ('SL' if hit_sl else 'TP')
                exit_fill = stop_fill if hit_sl else target - side * (spread / 2 + slippage)
                break
        if exit_fill is None:
            exit_fill = float(base.iloc[-1].close) - side * (spread / 2 + slippage)
        pnl = side * (exit_fill - entry) * contract * min_lot - commission * min_lot
        equity += pnl
        count += 1
        trades.append({'entry_time': ts, 'exit_time': base.index[exit_i],
                       'timeframe': timeframe, 'side': 'BUY' if side == 1 else 'SELL',
                       'entry': round(entry, 3), 'stop': round(stop, 3),
                       'target': round(target, 3), 'exit': round(exit_fill, 3),
                       'reason': reason, 'pnl': round(pnl, 2), 'equity': round(equity, 2)})
        i = exit_i + 1
    return pd.DataFrame(trades), pd.DataFrame(rejects, columns=['time', 'reason'])


upload = st.file_uploader('Upload original XAUUSD M1 CSV', type=['csv', 'txt'])
if upload:
    try:
        m1, invalid, duplicates, minute_share = load_csv(upload)
        st.write(f'M1 candles: {len(m1):,} | Invalid dates: {invalid} | Duplicate times: {duplicates}')
        st.write(f'Exact one-minute intervals: {minute_share:.1%}')
        if minute_share < 0.90:
            st.error('Data does not look like continuous M1 candles. Check timestamps and upload original M1 data. Backtest blocked.')
            st.stop()
        if minute_share < 0.98:
            st.warning('Some M1 gaps detected. Review session/weekend gaps before trusting results.')
        if (m1.index[-1] - m1.index[0]).days < 3:
            st.warning('Very short dataset; results are not statistically reliable.')
        tabs = st.tabs(['M1 Scalping', 'M5 Scalping', 'Compare'])
        results = {}
        for tab, tf, label in zip(tabs[:2], ['1min', '5min'], ['M1', 'M5']):
            with tab:
                base = build_signals(m1, tf)
                trades, rejected = run_backtest(base, label)
                results[label] = (trades, rejected)
                latest = int(base.iloc[-1].signal)
                st.subheader({1: '🟢 BUY setup', -1: '🔴 SELL setup', 0: '🟡 WAIT'}[latest])
                st.caption('Last completed historical candle, not live execution advice.')
                if trades.empty:
                    st.warning('No trades passed risk rules. WAIT.')
                else:
                    wins = (trades.pnl > 0).mean() * 100
                    eq = pd.concat([pd.Series([balance]), trades.equity.reset_index(drop=True)], ignore_index=True)
                    dd = (eq - eq.cummax()).min()
                    a, b, c, d = st.columns(4)
                    a.metric('Trades', len(trades))
                    b.metric('Win rate', f'{wins:.1f}%')
                    c.metric('Net P/L', f'${trades.pnl.sum():.2f}')
                    d.metric('Max drawdown ($)', f'{dd:.2f}')
                    st.line_chart(trades.set_index('exit_time').equity)
                    st.dataframe(trades, use_container_width=True)
                    st.download_button(f'Download {label} trades', trades.to_csv(index=False), f'V38_{label}_trades.csv', 'text/csv')
                st.write('Rejected setups:', len(rejected))
                if not rejected.empty:
                    st.dataframe(rejected.reason.value_counts())
                    st.download_button(f'Download {label} rejects', rejected.to_csv(index=False), f'V38_{label}_rejected.csv', 'text/csv')
        with tabs[2]:
            summary = []
            for label, (trades, rejected) in results.items():
                summary.append({'Timeframe': label, 'Trades': len(trades),
                                'Rejected': len(rejected),
                                'Net P/L ($)': round(trades.pnl.sum(), 2) if not trades.empty else 0.0,
                                'Win rate (%)': round((trades.pnl > 0).mean() * 100, 1) if not trades.empty else None})
            st.dataframe(pd.DataFrame(summary), use_container_width=True)
            st.caption('Same historical sample is used for comparison; it is not independent validation.')
        st.warning('Experimental OHLC backtest. Intrabar execution, real spreads, rollover, news and slippage can differ. No live trading until independent and demo validation.')
    except Exception as exc:
        st.error(f'CSV processing error: {exc}')
else:
    st.info('Upload genuine XAUUSD M1 data. M5 and M15 are generated automatically.')
