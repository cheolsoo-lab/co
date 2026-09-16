import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import ta
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

st.set_page_config(page_title="🔥 크립토 종합 AI 추천 & 차트 백테스트 대시보드", layout="wide")

TOP_MAJORS = {'BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'ADA', 'AVAX', 'DOT', 'LINK', 'SUI', 'APT', 'BCH', 'NEAR'}

SECTORS = {
    'AI / Big Data': {'NEAR', 'TAO', 'RENDER', 'RNDR', 'FET', 'AGIX', 'OCEAN', 'GRT', 'VIRTUAL', 'AKT', 'THETA'},
    'DeFi': {'UNI', 'AAVE', 'CRV', 'MKR', 'SNX', 'COMP', 'LDO', 'PENDLE', 'RUNE', 'INJ'},
    'Meme': {'DOGE', 'SHIB', 'PEPE', 'BONK', 'WIF', 'FLOKI', '1000SATS', 'BOME'},
    'Layer 1 & 2': {'ADA', 'AVAX', 'DOT', 'LINK', 'BNB', 'BCH', 'SUI', 'APT', 'SEI', 'MATIC', 'OP', 'ARB'},
    'GameFi & Metaverse': {'GALA', 'SAND', 'MANA', 'IMX', 'AXS', 'BEAM'}
}

def get_crypto_logo_url(symbol_base):
    base_lower = symbol_base.lower()
    return f"https://raw.githubusercontent.com/spothit/cryptocurrency-icons/master/128/color/{base_lower}.png"

def get_sector_label(base_asset):
    if base_asset in TOP_MAJORS:
        return '👑 Key Major'
    for sector, coins in SECTORS.items():
        if base_asset in coins:
            return sector
    return ' 기타 알트'

# ==========================================
# 1. 시세 데이터 및 거래소 API 로드 (Fallback)
# ==========================================
@st.cache_data(ttl=30)
def load_market_data():
    exchanges_to_try = [
        ('MEXC', getattr(ccxt, 'mexc', None)),
        ('Gate.io', getattr(ccxt, 'gate', None)),
        ('Bybit', getattr(ccxt, 'bybit', None))
    ]
    
    tickers, exchange = None, None
    for ex_name, ex_class in exchanges_to_try:
        if ex_class is None:
            continue
        try:
            ex_instance = ex_class({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
            tickers = ex_instance.fetch_tickers()
            if tickers:
                exchange = ex_instance
                break
        except Exception:
            continue
            
    if not tickers or not exchange:
        st.error("모든 거래소 API 접근이 일시적으로 제한되었습니다. 잠시 후 다시 시도해 주세요.")
        return pd.DataFrame(), None

    market_data = []
    for symbol, data in tickers.items():
        if not symbol.endswith('/USDT'):
            continue
        
        base_asset = symbol.split('/')[0]
        quote_vol = data.get('quoteVolume', 0) or 0
        if quote_vol < 1000000:
            continue
            
        change_pct = data.get('percentage', 0) or 0
        high_24h = data.get('high')
        low_24h = data.get('low')
        last_price = data.get('last')
        
        if high_24h and low_24h and last_price and high_24h > 0 and low_24h > 0:
            volatility_pct = ((high_24h - low_24h) / low_24h) * 100
            market_data.append({
                'symbol': symbol,
                'base': base_asset,
                'change_pct': change_pct,
                'high_24h': high_24h,
                'low_24h': low_24h,
                'last_price': last_price,
                'volatility_pct': volatility_pct,
                'sector': get_sector_label(base_asset),
                'logo_url': get_crypto_logo_url(base_asset)
            })

    df = pd.DataFrame(market_data)
    if not df.empty:
        df['drawdown_pct'] = ((df['last_price'] - df['high_24h']) / df['high_24h']) * 100
    return df, exchange

@st.cache_data(ttl=60)
def fetch_ohlcv_data(_exchange, symbol, timeframe='1d', limit=150):
    exchanges_to_try = []
    if _exchange is not None:
        exchanges_to_try.append(_exchange)
        
    for ex_name, ex_class in [
        ('MEXC', getattr(ccxt, 'mexc', None)),
        ('Gate.io', getattr(ccxt, 'gate', None)),
        ('Bybit', getattr(ccxt, 'bybit', None))
    ]:
        if ex_class is not None:
            try:
                ex_inst = ex_class({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
                if _exchange is None or ex_inst.id != _exchange.id:
                    exchanges_to_try.append(ex_inst)
            except Exception:
                continue

    for ex in exchanges_to_try:
        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if ohlcv and len(ohlcv) >= 30:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
        except Exception:
            continue

    return pd.DataFrame()

# ==========================================
# 2. 거래량 오더블록(Volume OB) & RSI 분석
# ==========================================
@st.cache_data(ttl=60)
def analyze_volume_ob_and_rsi(symbol, _exchange):
    try:
        df = fetch_ohlcv_data(_exchange, symbol, timeframe='1d', limit=150)
        if df.empty or len(df) < 60:
            return None
        
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        df['rsi_sma50'] = ta.trend.sma_indicator(df['rsi'], window=50)
        df['rsi_sma200'] = ta.trend.sma_indicator(df['rsi'], window=200)
        df['vol_ma20'] = ta.trend.sma_indicator(df['Volume'], window=20)
        
        df['vol_spike'] = df['Volume'] > (df['vol_ma20'] * 1.8)
        
        latest_bull_ob = "없음"
        latest_bear_ob = "없음"
        bull_ob_low, bull_ob_high = 0.0, 0.0
        bear_ob_low, bear_ob_high = 0.0, 0.0
        ob_status = "일반"
        
        recent_df = df.iloc[-30:]
        current_price = df.iloc[-1]['Close']
        
        for i in range(len(recent_df)-1, 1, -1):
            row = recent_df.iloc[i]
            prev_row = recent_df.iloc[i-1]
            
            if row['vol_spike'] and row['Close'] > row['Open']:
                bull_ob_low = float(min(prev_row['Low'], row['Low']))
                bull_ob_high = float(row['High'])
                latest_bull_ob = f"${bull_ob_low:,.2f} ~ ${bull_ob_high:,.2f}"
                if bull_ob_low <= current_price <= bull_ob_high:
                    ob_status = "🎯 매수 지지대 재진입 (OB Retest)"
                break

            elif row['vol_spike'] and row['Close'] < row['Open']:
                bear_ob_low = float(row['Low'])
                bear_ob_high = float(max(prev_row['High'], row['High']))
                latest_bear_ob = f"${bear_ob_low:,.2f} ~ ${bear_ob_high:,.2f}"
                if bear_ob_low <= current_price <= bear_ob_high:
                    ob_status = "⚠️ 매도 저항대 진입 (Bear OB)"
                break

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        rsi_val = latest['rsi'] if not pd.isna(latest['rsi']) else 50.0
        sma50 = latest['rsi_sma50'] if not pd.isna(latest['rsi_sma50']) else 50.0
        sma200 = latest['rsi_sma200'] if not pd.isna(latest['rsi_sma200']) else 50.0
        prev_sma50 = prev['rsi_sma50'] if not pd.isna(prev['rsi_sma50']) else 50.0
        prev_sma200 = prev['rsi_sma200'] if not pd.isna(prev['rsi_sma200']) else 50.0
        
        if prev_sma50 <= prev_sma200 and sma50 > sma200:
            cross_status = "🚀 골든크로스"
        elif prev_sma50 >= prev_sma200 and sma50 < sma200:
            cross_status = "📉 데드크로스"
        elif sma50 > sma200:
            cross_status = "🟢 강세 추세"
        else:
            cross_status = "🔴 약세 추세"
            
        rsi_gap = abs(sma50 - sma200)
        is_squeezed = "⚡ 수렴" if rsi_gap <= 2.5 else "일반"
        
        return {
            'rsi': rsi_val,
            'rsi_sma50': sma50,
            'rsi_sma200': sma200,
            'rsi_gap': rsi_gap,
            'cross_status': cross_status,
            'is_squeezed': is_squeezed,
            'bull_ob': latest_bull_ob,
            'bear_ob': latest_bear_ob,
            'bull_ob_low': bull_ob_low,
            'bull_ob_high': bull_ob_high,
            'bear_ob_low': bear_ob_low,
            'bear_ob_high': bear_ob_high,
            'ob_status': ob_status
        }
    except Exception:
        return None

# ==========================================
# 3. 백테스팅 및 차트 패턴 감지 엔진
# ==========================================
def calculate_indicators_bt(df, sma_short_p=20, sma_long_p=60, rsi_p=14):
    df_calc = df.copy()
    df_calc[f'SMA_{sma_short_p}'] = df_calc['Close'].rolling(window=sma_short_p).mean()
    df_calc[f'SMA_{sma_long_p}'] = df_calc['Close'].rolling(window=sma_long_p).mean()
    
    delta = df_calc['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_p, adjust=False, min_periods=rsi_p).mean()
    avg_loss = loss.ewm(alpha=1/rsi_p, adjust=False, min_periods=rsi_p).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df_calc['RSI'] = 100 - (100 / (1 + rs))
    return df_calc

def detect_signals(df, tolerance_pct=0.02, pole_gain_pct=0.06):
    signals = []
    lows, highs, closes, volumes = df['Low'].values, df['High'].values, df['Close'].values, df['Volume'].values
    
    prominence = np.mean(lows) * 0.015
    peaks_idx, _ = find_peaks(-lows, distance=5, prominence=prominence)
    
    for i in range(len(peaks_idx) - 1):
        idx1, idx2 = peaks_idx[i], peaks_idx[i+1]
        trough1, trough2 = lows[idx1], lows[idx2]
        if 5 <= (idx2 - idx1) <= 35:
            if abs(trough1 - trough2) / min(trough1, trough2) <= tolerance_pct:
                neckline = highs[idx1:idx2+1].max()
                post_df = df.iloc[idx2:]
                breakout = post_df[post_df['Close'] > neckline]
                if not breakout.empty:
                    b_idx = df.index.get_loc(breakout.index[0])
                    signals.append({'breakout_idx': b_idx, 'pattern_type': 'Double Bottom'})

    pole_win = 5
    for i in range(pole_win, len(df) - 15):
        pole_start, pole_end = i - pole_win, i
        gain = (closes[pole_end] - lows[pole_start]) / lows[pole_start]
        if gain >= pole_gain_pct:
            flag_df = df.iloc[pole_end+1 : pole_end+12]
            if not flag_df.empty:
                if flag_df['Volume'].mean() < volumes[pole_start:pole_end].mean() * 1.1:
                    flag_high = flag_df['High'].max()
                    breakout = flag_df[flag_df['Close'] > flag_high * 0.99]
                    if not breakout.empty:
                        b_idx = df.index.get_loc(breakout.index[0])
                        signals.append({'breakout_idx': b_idx, 'pattern_type': 'Bull Flag'})
                        
    seen = set()
    unique_signals = []
    for s in sorted(signals, key=lambda x: x['breakout_idx']):
        if s['breakout_idx'] not in seen:
            seen.add(s['breakout_idx'])
            unique_signals.append(s)
            
    return unique_signals

def run_backtest_engine(df, signals, tp_ratio=0.05, sl_ratio=0.03, max_holding_bars=15, position_pct=0.20, max_positions=4, initial_capital=10000.0):
    cash = initial_capital
    active_positions = []
    closed_trades = []
    equity_curve = []
    
    signal_map = {s['breakout_idx']: s['pattern_type'] for s in signals}
    
    for i in range(len(df)):
        c_date = df.index[i]
        c_close, c_high, c_low = df['Close'].iloc[i], df['High'].iloc[i], df['Low'].iloc[i]
        
        remaining = []
        for pos in active_positions:
            entry_p = pos['entry_price']
            holding = i - pos['entry_idx']
            tp_p, sl_p = entry_p * (1 + tp_ratio), entry_p * (1 - sl_ratio)
            
            exit_reason, exit_p = None, c_close
            if c_high >= tp_p:
                exit_reason, exit_p = 'Take Profit', tp_p
            elif c_low <= sl_p:
                exit_reason, exit_p = 'Stop Loss', sl_p
            elif holding >= max_holding_bars:
                exit_reason, exit_p = 'Time Out', c_close
                
            if exit_reason:
                cash += pos['shares'] * exit_p
                profit = (exit_p - entry_p) * pos['shares']
                pnl = ((exit_p - entry_p) / entry_p) * 100
                closed_trades.append({
                    'pattern': pos['pattern'],
                    'entry_date': df.index[pos['entry_idx']],
                    'entry_price': entry_p,
                    'exit_date': c_date,
                    'exit_price': exit_p,
                    'return_pct': round(pnl, 2),
                    'profit': round(profit, 2),
                    'exit_reason': exit_reason
                })
            else:
                remaining.append(pos)
        active_positions = remaining
        
        if i in signal_map and len(active_positions) < max_positions and cash > 0:
            unrealized = sum(p['shares'] * c_close for p in active_positions)
            total_eq = cash + unrealized
            alloc_cash = min(cash, total_eq * position_pct)
            
            if alloc_cash >= 10.0:
                shares = alloc_cash / c_close
                cash -= alloc_cash
                active_positions.append({
                    'pattern': signal_map[i],
                    'entry_idx': i,
                    'entry_price': c_close,
                    'shares': shares
                })
                
        unrealized = sum(p['shares'] * c_close for p in active_positions)
        equity_curve.append({
            'Date': c_date,
            'Total_Equity': cash + unrealized,
            'Active_Pos': len(active_positions)
        })
        
    last_c = df['Close'].iloc[-1]
    for pos in active_positions:
        cash += pos['shares'] * last_c
        profit = (last_c - pos['entry_price']) * pos['shares']
        pnl = ((last_c - pos['entry_price']) / pos['entry_price']) * 100
        closed_trades.append({
            'pattern': pos['pattern'],
            'entry_date': df.index[pos['entry_idx']],
            'entry_price': pos['entry_price'],
            'exit_date': df.index[-1],
            'exit_price': last_c,
            'return_pct': round(pnl, 2),
            'profit': round(profit, 2),
            'exit_reason': 'Unclosed'
        })
        
    trades_df = pd.DataFrame(closed_trades)
    equity_df = pd.DataFrame(equity_curve).set_index('Date')
    
    if not trades_df.empty:
        win_t = trades_df[trades_df['return_pct'] > 0]
        win_rate = (len(win_t) / len(trades_df)) * 100
        final_eq = equity_df['Total_Equity'].iloc[-1]
        tot_ret = ((final_eq - initial_capital) / initial_capital) * 100
        peak = equity_df['Total_Equity'].cummax()
        mdd = abs(((equity_df['Total_Equity'] - peak) / peak).min()) * 100
        metrics = {
            'total_trades': len(trades_df),
            'win_rate': round(win_rate, 2),
            'total_return': round(tot_ret, 2),
            'mdd': round(mdd, 2),
            'final_capital': round(final_eq, 2)
        }
    else:
        metrics = {'total_trades': 0, 'win_rate': 0.0, 'total_return': 0.0, 'mdd': 0.0, 'final_capital': round(initial_capital, 2)}
        
    return trades_df, equity_df, metrics

def perform_grid_search(df_raw, sma_range, tp_range, sl_range, rsi_period=14, max_bars=15):
    results = []
    total_iterations = len(sma_range) * len(tp_range) * len(sl_range)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    count = 0
    for sma_s in sma_range:
        df_calc = calculate_indicators_bt(df_raw, sma_s, sma_s * 3, rsi_period)
        signals = detect_signals(df_calc)
        
        for tp in tp_range:
            for sl in sl_range:
                count += 1
                _, _, metrics = run_backtest_engine(df_calc, signals, tp, sl, max_holding_bars=max_bars)
                results.append({
                    '단기 SMA': sma_s,
                    '장기 SMA': sma_s * 3,
                    '익절률 (TP %)': round(tp * 100, 1),
                    '손절률 (SL %)': round(sl * 100, 1),
                    '손익비 (R:R)': round(tp / sl, 2),
                    '총 거래 수': metrics['total_trades'],
                    '승률 (%)': metrics['win_rate'],
                    '누적 수익률 (%)': metrics['total_return'],
                    'MDD (%)': metrics['mdd'],
                    '최종 자산 ($)': metrics['final_capital']
                })
                progress_bar.progress(count / total_iterations)
                status_text.text(f"🔍 파라미터 Grid 탐색 진행 중... ({count}/{total_iterations})")
                
    progress_bar.empty()
    status_text.empty()
    res_df = pd.DataFrame(results)
    return res_df.sort_values(by='누적 수익률 (%)', ascending=False).reset_index(drop=True)

# ==========================================
# 4. 대시보드 메인 UI
# ==========================================
st.title("🔥 크립토 종합 AI 추천 & 차트 백테스트 대시보드")
st.caption("실시간 오더블록 포착 + RSI 수렴/크로스 + 리스크 레버리지 산출 + 차트 패턴 백테스트")

df_all, exchange = load_market_data()

if not df_all.empty and exchange is not None:
    top_30 = df_all.sort_values(by='change_pct', ascending=False).head(30).copy()
    
    with st.spinner("거래량 기반 오더블록(Volume OB) 및 RSI 장기 지표 분석 중..."):
        analysis_results = []
        for sym in top_30['symbol']:
            res = analyze_volume_ob_and_rsi(sym, exchange)
            if res:
                analysis_results.append(res)
            else:
                analysis_results.append({
                    'rsi': 50.0, 'rsi_sma50': 50.0, 'rsi_sma200': 50.0,
                    'rsi_gap': 0.0, 'cross_status': 'N/A', 'is_squeezed': 'N/A',
                    'bull_ob': '없음', 'bear_ob': '없음',
                    'bull_ob_low': 0.0, 'bull_ob_high': 0.0,
                    'bear_ob_low': 0.0, 'bear_ob_high': 0.0,
                    'ob_status': '일반'
                })
        
        df_analysis = pd.DataFrame(analysis_results)
        top_30 = pd.concat([top_30.reset_index(drop=True), df_analysis], axis=1)

    top_30_sorted = top_30.sort_values(by='drawdown_pct', ascending=True).reset_index(drop=True)

    # 탭 구성 (복원 완료 + 새로 추가된 기능 통합)
    tab_major, tab_signal, tab_ob, tab_calc, tab_all, tab_rsi, tab_chart, tab_grid = st.tabs([
        "👑 메이저 AI 추천",
        "🤖 전체 AI 추천 포지션",
        "🧱 거래량 오더블록 (Volume OB)",
        "🧮 레버리지 & 손익비 계산기",
        "🔥 Top 30 종합 시세",
        "🎯 RSI 수렴/크로스",
        "📈 차트 & 백테스팅",
        "🔍 Grid Search 최적화"
    ])

    # 추천 종목 추출 공통 함수
    def get_trade_signals(df_input):
        longs, shorts = [], []
        for _, row in df_input.iterrows():
            price = row['last_price']
            is_bull_trend = row['cross_status'] in ["🚀 골든크로스", "🟢 강세 추세"]
            is_bear_trend = row['cross_status'] in ["📉 데드크로스", "🔴 약세 추세"]
            is_bull_ob = "매수 지지대" in row['ob_status']
            is_bear_ob = "매도 저항대" in row['ob_status']
            
            if (is_bull_trend and not is_bear_ob) or is_bull_ob:
                sl = row['bull_ob_low'] * 0.985 if row['bull_ob_low'] > 0 else price * 0.96
                tp = price + (price - sl) * 1.8
                longs.append({
                    'symbol': row['symbol'], 'price': price, 'ob_status': row['ob_status'],
                    'cross_status': row['cross_status'], 'rsi': row['rsi'],
                    'bull_ob': row['bull_ob'], 'sl': sl, 'tp': tp, 'rr': 1.8
                })
            elif (is_bear_trend and not is_bull_ob) or is_bear_ob:
                sl = row['bear_ob_high'] * 1.015 if row['bear_ob_high'] > 0 else price * 1.04
                tp = price - (sl - price) * 1.8
                shorts.append({
                    'symbol': row['symbol'], 'price': price, 'ob_status': row['ob_status'],
                    'cross_status': row['cross_status'], 'rsi': row['rsi'],
                    'bear_ob': row['bear_ob'], 'sl': sl, 'tp': tp, 'rr': 1.8
                })
        return longs, shorts

    # 1. 👑 메이저 AI 추천
    with tab_major:
        st.subheader("👑 비트코인 및 주요 메이저 코인 전용 추천")
        st.caption("비트코인(BTC), 이더리움(ETH), 솔라나(SOL), 리플(XRP) 등 주요 주도 자산의 추천 포지션입니다.")
        
        major_df = top_30_sorted[top_30_sorted['base'].isin(TOP_MAJORS)].reset_index(drop=True)
        if not major_df.empty:
            m_longs, m_shorts = get_trade_signals(major_df)
            col_ml, col_ms = st.columns(2)
            with col_ml:
                st.markdown("### 🚀 메이저 LONG (매수)")
                if m_longs:
                    for item in m_longs:
                        with st.expander(f"🟢 **{item['symbol']}** (현재가: ${item['price']:,.2f})", expanded=True):
                            st.markdown(f"""
                            * **기술적 근거:** {item['cross_status']} | {item['ob_status']}
                            * **RSI(14):** {item['rsi']:.1f}
                            * **상승 오더블록:** {item['bull_ob']}
                            * **추천 진입가:** ${item['price']:,.2f}
                            * **목표가 (TP):** `${item['tp']:,.2f}` | **손절가 (SL):** `${item['sl']:,.2f}` (손익비 1:{item['rr']:.1f})
                            """)
                else:
                    st.info("현재 매수 조건에 부합하는 메이저 코인이 없습니다.")
            with col_ms:
                st.markdown("### 📉 메이저 SHORT (매도)")
                if m_shorts:
                    for item in m_shorts:
                        with st.expander(f"🔴 **{item['symbol']}** (현재가: ${item['price']:,.2f})", expanded=True):
                            st.markdown(f"""
                            * **기술적 근거:** {item['cross_status']} | {item['ob_status']}
                            * **RSI(14):** {item['rsi']:.1f}
                            * **하락 오더블록:** {item['bear_ob']}
                            * **추천 진입가:** ${item['price']:,.2f}
                            * **목표가 (TP):** `${item['tp']:,.2f}` | **손절가 (SL):** `${item['sl']:,.2f}` (손익비 1:{item['rr']:.1f})
                            """)
                else:
                    st.info("현재 매도 조건에 부합하는 메이저 코인이 없습니다.")

    # 2. 🤖 전체 AI 추천 포지션
    with tab_signal:
        st.subheader("💡 Top 30 거래량 OB + RSI 종합 추천 종목")
        all_longs, all_shorts = get_trade_signals(top_30_sorted)
        col_l, col_s = st.columns(2)
        with col_l:
            st.markdown("### 🚀 LONG (매수) 추천 종목")
            if all_longs:
                for item in all_longs[:4]:
                    with st.expander(f"🟢 **{item['symbol']}** (현재가: ${item['price']:,.4f})", expanded=True):
                        st.markdown(f"""
                        * **기술적 근거:** {item['cross_status']} | {item['ob_status']}
                        * **RSI(14):** {item['rsi']:.1f} | **상승 오더블록:** {item['bull_ob']}
                        * **목표가 (TP):** `${item['tp']:,.4f}` | **손절가 (SL):** `${item['sl']:,.4f}`
                        """)
            else:
                st.info("현재 조건에 부합하는 LONG 종목이 없습니다.")

        with col_s:
            st.markdown("### 📉 SHORT (매도) 추천 종목")
            if all_shorts:
                for item in all_shorts[:4]:
                    with st.expander(f"🔴 **{item['symbol']}** (현재가: ${item['price']:,.4f})", expanded=True):
                        st.markdown(f"""
                        * **기술적 근거:** {item['cross_status']} | {item['ob_status']}
                        * **RSI(14):** {item['rsi']:.1f} | **하락 오더블록:** {item['bear_ob']}
                        * **목표가 (TP):** `${item['tp']:,.4f}` | **손절가 (SL):** `${item['sl']:,.4f}`
                        """)
            else:
                st.info("현재 조건에 부합하는 SHORT 종목이 없습니다.")

    # 3. 🧱 거래량 오더블록
    with tab_ob:
        st.subheader("🧱 거래량 기반 유효 오더블록(Volume Order Block) 매수/매도 구간")
        display_ob_df = pd.DataFrame({
            '종목코드': top_30_sorted['symbol'],
            '현재가': top_30_sorted['last_price'],
            '상승 오더블록 (매수 지지대)': top_30_sorted['bull_ob'],
            '하락 오더블록 (매도 저항대)': top_30_sorted['bear_ob'],
            '오더블록 도달 상태': top_30_sorted['ob_status'],
            '24h 변동성': top_30_sorted['volatility_pct'].map('{:.2f}%'.format),
            '24h 상승률': top_30_sorted['change_pct'].map('{:+.2f}%'.format)
        })
        st.dataframe(display_ob_df, use_container_width=True, height=500, hide_index=True)

    # 4. 🧮 레버리지 & 손익비 계산기
    with tab_calc:
        st.subheader("🧮 리스크 관리 및 적정 레버리지 계산기")
        col_in1, col_in2 = st.columns([1, 1])
        with col_in1:
            total_balance = st.number_input("총 시드 자산 ($)", value=10000.0, step=500.0)
            max_risk_pct = st.slider("1회 매매 최대 감수 리스크 (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
            position_type = st.radio("포지션 방향", ["LONG (매수)", "SHORT (매도)"], horizontal=True)
        with col_in2:
            entry_price = st.number_input("진입 가격 ($)", value=100.0, step=0.1)
            default_sl = entry_price * 0.95 if "LONG" in position_type else entry_price * 1.05
            default_tp = entry_price * 1.10 if "LONG" in position_type else entry_price * 0.90
            stop_loss = st.number_input("손절 가격 (Stop Loss) ($)", value=default_sl, step=0.1)
            take_profit = st.number_input("목표 가격 (Take Profit) ($)", value=default_tp, step=0.1)

        if entry_price > 0 and stop_loss > 0 and take_profit > 0:
            sl_dist = (entry_price - stop_loss)/entry_price * 100 if "LONG" in position_type else (stop_loss - entry_price)/entry_price * 100
            tp_dist = (take_profit - entry_price)/entry_price * 100 if "LONG" in position_type else (entry_price - take_profit)/entry_price * 100
            if sl_dist > 0 and tp_dist > 0:
                rr_ratio = tp_dist / sl_dist
                max_loss = total_balance * (max_risk_pct / 100)
                pos_size = max_loss / (sl_dist / 100)
                rec_lev = pos_size / total_balance
                exp_profit = pos_size * (tp_dist / 100)
                
                res1, res2, res3, res4 = st.columns(4)
                res1.metric("손익비 (R:R Ratio)", f"1 : {rr_ratio:.2f}")
                res2.metric("권장 레버리지", f"{rec_lev:.1f}x")
                res3.metric("최대 예상 손실액", f"-${max_loss:,.2f}")
                res4.metric("목표 예상 수익액", f"+${exp_profit:,.2f}")

    # 5. 🔥 Top 30 종합 시세
    with tab_all:
        st.subheader("🔥 Top 30 종합 시세 및 변동성 모니터링")
        display_df = pd.DataFrame({
            '종목코드': top_30_sorted['symbol'],
            '분야(섹터)': top_30_sorted['sector'],
            '현재가': top_30_sorted['last_price'],
            '24h 변동성': top_30_sorted['volatility_pct'].map('{:.2f}%'.format),
            'RSI(14)': top_30_sorted['rsi'].map('{:.1f}'.format),
            '추세 상태': top_30_sorted['cross_status'],
            '오더블록 상태': top_30_sorted['ob_status'],
            '고점 대비 조정률': top_30_sorted['drawdown_pct'].map('{:.2f}%'.format)
        })
        st.dataframe(display_df, use_container_width=True, height=500, hide_index=True)

    # 6. 🎯 RSI 수렴/크로스
    with tab_rsi:
        st.subheader("🎯 RSI 50-200 수렴 및 크로스 종목")
        squeezed_df = top_30_sorted.sort_values(by='rsi_gap', ascending=True)
        rsi_table = pd.DataFrame({
            '종목코드': squeezed_df['symbol'],
            '분야(섹터)': squeezed_df['sector'],
            'RSI 50-200 이격도': squeezed_df['rsi_gap'].map('{:.2f}'.format),
            '수렴 여부': squeezed_df['is_squeezed'],
            '추세 상태': squeezed_df['cross_status'],
            '상승 오더블록(지지대)': squeezed_df['bull_ob']
        })
        st.dataframe(rsi_table, use_container_width=True, hide_index=True)

    # 7. 📈 차트 & 백테스팅
    selected_symbol = st.sidebar.selectbox("차트 분석 코인 선택", top_30['symbol'].tolist(), index=0)
    sma_short = st.sidebar.slider("단기 SMA", 5, 40, 20)
    sma_long = st.sidebar.slider("장기 SMA", 20, 150, 60)
    rsi_p = st.sidebar.slider("RSI 기간", 5, 30, 14)
    tp_input = st.sidebar.slider("목표 익절률 (%)", 1.0, 15.0, 5.0, 0.5) / 100
    sl_input = st.sidebar.slider("목표 손절률 (%)", 1.0, 10.0, 3.0, 0.5) / 100
    max_bars = st.sidebar.slider("최대 보유 봉 수", 3, 40, 15)

    df_symbol = fetch_ohlcv_data(exchange, selected_symbol)
    
    with tab_chart:
        if df_symbol is not None and not df_symbol.empty:
            df_calc = calculate_indicators_bt(df_symbol, sma_short, sma_long, rsi_p)
            signals = detect_signals(df_calc)
            trades_df, equity_df, metrics = run_backtest_engine(df_calc, signals, tp_input, sl_input, max_bars)
            
            st.subheader(f"📊 {selected_symbol} 백테스트 성과 리포트")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("총 거래 수", f"{metrics['total_trades']}회")
            c2.metric("승률", f"{metrics['win_rate']}%")
            c3.metric("누적 수익률", f"{metrics['total_return']}%")
            c4.metric("MDD", f"{metrics['mdd']}%")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df_calc.index, open=df_calc['Open'], high=df_calc['High'], low=df_calc['Low'], close=df_calc['Close'], name='OHLC'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc[f'SMA_{sma_short}'], mode='lines', name=f'{sma_short} SMA', line=dict(color='#FFD700')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc[f'SMA_{sma_long}'], mode='lines', name=f'{sma_long} SMA', line=dict(color='#00BFFF')), row=1, col=1)
            
            if not trades_df.empty:
                for _, tr in trades_df.iterrows():
                    clr = '#00E676' if tr['return_pct'] > 0 else '#FF5252'
                    fig.add_trace(go.Scatter(x=[tr['entry_date'], tr['exit_date']], y=[tr['entry_price'], tr['exit_price']], mode='lines', line=dict(color=clr, width=1.8, dash='dash'), showlegend=False), row=1, col=1)
                    
            fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc['RSI'], mode='lines', name='RSI', line=dict(color='#AB63FA')), row=2, col=1)
            fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.error("캔들스틱 데이터를 불러올 수 없습니다.")

    # 8. 🔍 Grid Search
    with tab_grid:
        st.subheader("🔍 파라미터 Grid Search 최적화")
        sma_g_min, sma_g_max = st.sidebar.slider("Grid SMA 범위", 5, 30, (10, 25))
        tp_g_min, tp_g_max = st.sidebar.slider("Grid TP 범위 (%)", 2.0, 12.0, (3.0, 8.0))
        sl_g_min, sl_g_max = st.sidebar.slider("Grid SL 범위 (%)", 1.0, 6.0, (2.0, 4.0))
        run_grid_btn = st.sidebar.button("🚀 Grid Search 탐색 시작", type="primary")
        
        if 'grid_res_df' not in st.session_state:
            st.session_state.grid_res_df = None
            
        if run_grid_btn and df_symbol is not None and not df_symbol.empty:
            sma_range = list(range(sma_g_min, sma_g_max + 1, 5))
            tp_range = [round(x/100, 3) for x in np.arange(tp_g_min, tp_g_max + 0.1, 1.5)]
            sl_range = [round(x/100, 3) for x in np.arange(sl_g_min, sl_g_max + 0.1, 1.0)]
            
            with st.spinner("최적 조합 탐색 중..."):
                st.session_state.grid_res_df = perform_grid_search(df_symbol, sma_range, tp_range, sl_range, rsi_period=rsi_p, max_bars=max_bars)
                st.success("탐색 완료!")
                
        if st.session_state.grid_res_df is not None:
            res_df = st.session_state.grid_res_df
            st.markdown("##### 🏆 최적 파라미터 조합 (Top 1)")
            st.dataframe(res_df.head(10), use_container_width=True)
            
            fig_hm = px.imshow(res_df.pivot_table(index='익절률 (TP %)', columns='손절률 (SL %)', values='누적 수익률 (%)', aggfunc='mean'), color_continuous_scale="Viridis", text_auto=True)
            fig_hm.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_hm, use_container_width=True)

    time.sleep(30)
    st.rerun()