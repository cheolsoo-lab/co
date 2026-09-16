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

st.set_page_config(page_title="🔥 크립토 정밀 차트 패턴 & Grid Search 대시보드", layout="wide")

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
# 1. 거래소 데이터 로드 (다중 우회 Fallback)
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
            market_data.append({
                'symbol': symbol,
                'base': base_asset,
                'change_pct': change_pct,
                'high_24h': high_24h,
                'low_24h': low_24h,
                'last_price': last_price,
                'sector': get_sector_label(base_asset),
                'logo_url': get_crypto_logo_url(base_asset)
            })

    df = pd.DataFrame(market_data)
    if not df.empty:
        df['drawdown_pct'] = ((df['last_price'] - df['high_24h']) / df['high_24h']) * 100
    return df, exchange

@st.cache_data(ttl=60)
def fetch_ohlcv_data(_exchange, symbol, timeframe='1d', limit=150):
    try:
        ohlcv = _exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv or len(ohlcv) < 60:
            return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception:
        return None

# ==========================================
# 2. 기술적 지표 및 패턴 감지 엔진
# ==========================================
def calculate_indicators(df, sma_short_p=20, sma_long_p=60, rsi_p=14):
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
    
    # Double Bottom
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

    # Bull Flag
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

# ==========================================
# 3. 백테스트 시뮬레이션 엔진
# ==========================================
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

# ==========================================
# 4. Grid Search 탐색 함수
# ==========================================
def perform_grid_search(df_raw, sma_range, tp_range, sl_range, rsi_period=14, max_bars=15):
    results = []
    total_iterations = len(sma_range) * len(tp_range) * len(sl_range)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    count = 0
    for sma_s in sma_range:
        df_calc = calculate_indicators(df_raw, sma_s, sma_s * 3, rsi_period)
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
# 5. UI 및 대시보드 메인
# ==========================================
st.title("🔥 크립토 정밀 차트 패턴 & Grid Search 대시보드")
st.caption("실시간 오더블록/매물대 분석 + 차트 패턴 자동 포착 + Grid Search 최적 파라미터 추출")

df_all, exchange = load_market_data()

if not df_all.empty and exchange is not None:
    top_30 = df_all.sort_values(by='change_pct', ascending=False).head(30).copy()
    
    # 사이드바 파라미터
    st.sidebar.header("⚙️ 1. 차트 & 백테스트 설정")
    selected_symbol = st.sidebar.selectbox("분석 대상 코인 선택", top_30['symbol'].tolist(), index=0)
    sma_short = st.sidebar.slider("단기 SMA 주기", 5, 40, 20)
    sma_long = st.sidebar.slider("장기 SMA 주기", 20, 150, 60)
    rsi_period = st.sidebar.slider("RSI 계산 기간", 5, 30, 14)
    
    st.sidebar.header("🎯 2. 손익비 & 자금 관리")
    tp_input = st.sidebar.slider("목표 익절률 (TP %)", 1.0, 15.0, 5.0, 0.5) / 100
    sl_input = st.sidebar.slider("목표 손절률 (SL %)", 1.0, 10.0, 3.0, 0.5) / 100
    max_bars = st.sidebar.slider("최대 보유 봉 수", 3, 40, 15)
    
    st.sidebar.header("🔍 3. Grid Search 범위 설정")
    sma_grid_min, sma_grid_max = st.sidebar.slider("SMA 탐색 범위 (일)", 5, 30, (10, 25))
    tp_grid_min, tp_grid_max = st.sidebar.slider("익절 (TP %) 범위", 2.0, 12.0, (3.0, 8.0))
    sl_grid_min, sl_grid_max = st.sidebar.slider("손절 (SL %) 범위", 1.0, 6.0, (2.0, 4.0))
    
    run_grid_btn = st.sidebar.button("🚀 Grid Search 최적화 실행", type="primary")

    df_symbol = fetch_ohlcv_data(exchange, selected_symbol)
    
    if df_symbol is not None and not df_symbol.empty:
        df_calc = calculate_indicators(df_symbol, sma_short, sma_long, rsi_period)
        signals = detect_signals(df_calc)
        trades_df, equity_df, metrics = run_backtest_engine(df_calc, signals, tp_input, sl_input, max_bars)
        
        tab_chart, tab_grid, tab_major = st.tabs([
            "📈 백테스트 시각화 & 차트",
            "🔍 Grid Search 최적화 결과",
            "👑 메이저 코인 시세 목록"
        ])
        
        # TAB 1: 차트 시각화
        with tab_chart:
            st.subheader(f"📊 {selected_symbol} 기술적 분석 및 백테스트 성과")
            
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("총 포착 신호 / 거래 수", f"{len(signals)}개 / {metrics['total_trades']}회")
            k2.metric("승률", f"{metrics['win_rate']} %")
            k3.metric("누적 수익률", f"{metrics['total_return']} %", delta=f"{metrics['total_return']:.1f}%")
            k4.metric("MDD", f"{metrics['mdd']} %")
            k5.metric("최종 자산", f"${metrics['final_capital']:,.2f}")
            
            # Plotly 통합 차트
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3],
                                subplot_titles=(f"<b>{selected_symbol} 가격 & 이동평균선 & 포지션</b>", "<b>RSI (14)</b>"))
            
            fig.add_trace(go.Candlestick(x=df_calc.index, open=df_calc['Open'], high=df_calc['High'], low=df_calc['Low'], close=df_calc['Close'], name='OHLC'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc[f'SMA_{sma_short}'], mode='lines', name=f'{sma_short} SMA', line=dict(color='#FFD700')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc[f'SMA_{sma_long}'], mode='lines', name=f'{sma_long} SMA', line=dict(color='#00BFFF')), row=1, col=1)
            
            if not trades_df.empty:
                for _, tr in trades_df.iterrows():
                    clr = '#00E676' if tr['return_pct'] > 0 else '#FF5252'
                    fig.add_trace(go.Scatter(x=[tr['entry_date'], tr['exit_date']], y=[tr['entry_price'], tr['exit_price']], mode='lines', line=dict(color=clr, width=1.8, dash='dash'), showlegend=False), row=1, col=1)
                    fig.add_trace(go.Scatter(x=[tr['entry_date']], y=[tr['entry_price']], mode='markers', marker=dict(symbol='triangle-up', size=11, color='#00E676'), showlegend=False), row=1, col=1)
                    fig.add_trace(go.Scatter(x=[tr['exit_date']], y=[tr['exit_price']], mode='markers', marker=dict(symbol='triangle-down', size=11, color=clr), showlegend=False), row=1, col=1)
                    
            fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc['RSI'], mode='lines', name='RSI', line=dict(color='#AB63FA')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="#FF5252", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="#00E676", row=2, col=1)
            
            fig.update_layout(template="plotly_dark", height=650, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📋 상세 트레이딩 내역 로그")
            st.dataframe(trades_df, use_container_width=True)

        # TAB 2: Grid Search
        with tab_grid:
            st.subheader("🔍 파라미터 자동 최적화 (Grid Search)")
            st.caption("설정한 범위 내의 이동평균선 및 TP/SL 파라미터를 완전 시뮬레이션하여 최상위 수익률 조합을 추출합니다.")
            
            if 'grid_res_df' not in st.session_state:
                st.session_state.grid_res_df = None
                
            if run_grid_btn:
                sma_range = list(range(sma_grid_min, sma_grid_max + 1, 5))
                tp_range = [round(x/100, 3) for x in np.arange(tp_grid_min, tp_grid_max + 0.1, 1.5)]
                sl_range = [round(x/100, 3) for x in np.arange(sl_grid_min, sl_grid_max + 0.1, 1.0)]
                
                with st.spinner("Grid Search 탐색을 진행하고 있습니다..."):
                    st.session_state.grid_res_df = perform_grid_search(df_symbol, sma_range, tp_range, sl_range, rsi_period=rsi_period, max_bars=max_bars)
                    st.success("Grid Search 최적화 탐색이 완료되었습니다!")
                    
            if st.session_state.grid_res_df is not None:
                res_df = st.session_state.grid_res_df
                best = res_df.iloc[0]
                
                st.markdown("##### 🏆 최적 파라미터 (Top 1 Combination)")
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("최적 단기 SMA", f"{int(best['단기 SMA'])}일")
                b2.metric("최적 익절 / 손절", f"{best['익절률 (TP %)']}% / {best['손절률 (SL %)']}%")
                b3.metric("최고 수익률", f"{best['누적 수익률 (%)']}%")
                b4.metric("승률 / MDD", f"{best['승률 (%)']}% / {best['MDD (%)']}%")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("##### 🔥 TP vs SL 수익률 히트맵")
                    pivot_hm = res_df.pivot_table(index='익절률 (TP %)', columns='손절률 (SL %)', values='누적 수익률 (%)', aggfunc='mean')
                    fig_hm = px.imshow(pivot_hm, color_continuous_scale="Viridis", text_auto=True)
                    fig_hm.update_layout(template="plotly_dark", height=380)
                    st.plotly_chart(fig_hm, use_container_width=True)
                with c2:
                    st.markdown("##### 📊 SMA별 수익률 분포")
                    fig_box = px.box(res_df, x='단기 SMA', y='누적 수익률 (%)', template="plotly_dark")
                    fig_box.update_layout(height=380)
                    st.plotly_chart(fig_box, use_container_width=True)
                    
                st.markdown("##### 📋 상위 10개 최적 조합 순위표")
                st.dataframe(res_df.head(10), use_container_width=True)
            else:
                st.info("사이드바에서 범위를 조절하신 후 **[🚀 Grid Search 최적화 실행]** 버튼을 눌러주세요.")

        # TAB 3: 메이저 코인
        with tab_major:
            st.subheader("👑 메이저 코인 종목 시세 모니터링")
            major_df = top_30[top_30['base'].isin(TOP_MAJORS)]
            st.dataframe(major_df[['symbol', 'sector', 'last_price', 'change_pct', 'drawdown_pct']], use_container_width=True, hide_index=True)
    else:
        st.error("해당 종목의 캔들스틱 데이터를 불러올 수 없습니다.")
        
    time.sleep(30)
    st.rerun()