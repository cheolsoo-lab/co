import streamlit as st
import ccxt
import pandas as pd
import ta
import time

st.set_page_config(page_title="거래량 오더블록 & 리스크/레버리지 계산기", layout="wide")

TOP_MAJORS = {'BTC', 'ETH', 'SOL', 'XRP'}

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

# 캐시 주기를 5분(300초)으로 늘려 Rate Limit 방지
@st.cache_data(ttl=300)
def analyze_volume_ob_and_rsi(symbol):
    try:
        exchange = ccxt.bybit({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=150)
        time.sleep(0.05) # 호출 간 간격 부여
        
        if len(ohlcv) < 60:
            return None
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_sma50'] = ta.trend.sma_indicator(df['rsi'], window=50)
        df['rsi_sma200'] = ta.trend.sma_indicator(df['rsi'], window=200)
        df['vol_ma20'] = ta.trend.sma_indicator(df['volume'], window=20)
        df['vol_spike'] = df['volume'] > (df['vol_ma20'] * 1.8)
        
        latest_bull_ob, latest_bear_ob = "없음", "없음"
        bull_ob_low, bull_ob_high = 0.0, 0.0
        bear_ob_low, bear_ob_high = 0.0, 0.0
        ob_status = "일반"
        
        recent_df = df.iloc[-30:]
        current_price = df.iloc[-1]['close']
        
        for i in range(len(recent_df)-1, 1, -1):
            row = recent_df.iloc[i]
            prev_row = recent_df.iloc[i-1]
            
            if row['vol_spike'] and row['close'] > row['open']:
                bull_ob_low = float(min(prev_row['low'], row['low']))
                bull_ob_high = float(row['high'])
                latest_bull_ob = f"${bull_ob_low:,.2f} ~ ${bull_ob_high:,.2f}"
                if bull_ob_low <= current_price <= bull_ob_high:
                    ob_status = "🎯 매수 지지대 재진입 (OB Retest)"
                break

            elif row['vol_spike'] and row['close'] < row['open']:
                bear_ob_low = float(row['low'])
                bear_ob_high = float(max(prev_row['high'], row['high']))
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

# 마켓 전체 티커 캐시 5분 적용
@st.cache_data(ttl=300)
def load_market_data():
    try:
        exchange = ccxt.bybit({'enableRateLimit': True})
        tickers = exchange.fetch_tickers()
        market_data = []
        
        for symbol, data in tickers.items():
            if not symbol.endswith('/USDT'):
                continue
            
            base_asset = symbol.split('/')[0]
            if data.get('quoteVolume') is None or data['quoteVolume'] < 2000000:
                continue
                
            change_pct = data.get('percentage', 0)
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
        if df.empty:
            return pd.DataFrame()
            
        df['drawdown_pct'] = ((df['last_price'] - df['high_24h']) / df['high_24h']) * 100
        return df
    except Exception as e:
        st.error(f"거래소 데이터 불러오기 일시 제한 중입니다. 잠시 후 [다시 불러오기]를 눌러주세요. ({e})")
        return pd.DataFrame()

# 메인 UI
st.title("🔥 거래량 오더블록 & 리스크/레버리지 계산기")

col_head1, col_head2 = st.columns([4, 1])
with col_head1:
    st.caption("실시간 오더블록 포착 및 포지션별 손익비(R:R)·적정 레버리지 산출 (5분 주기 자동 갱신)")
with col_head2:
    if st.button("🔄 시세 새로고침"):
        st.cache_data.clear()
        st.rerun()

df_all = load_market_data()

if not df_all.empty:
    top_30 = df_all.sort_values(by='change_pct', ascending=False).head(30).copy()
    
    with st.spinner("거래량 기반 오더블록(Volume OB) 분석 중..."):
        analysis_results = []
        for sym in top_30['symbol']:
            res = analyze_volume_ob_and_rsi(sym)
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

    tab_signal, tab_ob, tab_calc, tab_all, tab_rsi = st.tabs([
        "🤖 AI 추천 포지션 (LONG / SHORT)",
        "🧱 거래량 오더블록 (Volume OB)",
        "🧮 레버리지 & 손익비(R:R) 계산기",
        "🔥 Top 30 종합 리스트",
        "🎯 RSI 수렴/크로스"
    ])

    with tab_signal:
        st.subheader("💡 Top 30 + Volume OB + RSI 기반 추천 종목")
        
        long_candidates = []
        short_candidates = []
        
        for _, row in top_30_sorted.iterrows():
            price = row['last_price']
            is_bull_trend = row['cross_status'] in ["🚀 골든크로스", "🟢 강세 추세"]
            is_bear_trend = row['cross_status'] in ["📉 데드크로스", "🔴 약세 추세"]
            is_bull_ob = "매수 지지대" in row['ob_status']
            is_bear_ob = "매도 저항대" in row['ob_status']
            
            if (is_bull_trend and not is_bear_ob) or is_bull_ob:
                sl = row['bull_ob_low'] * 0.985 if row['bull_ob_low'] > 0 else price * 0.96
                tp = price + (price - sl) * 1.8
                long_candidates.append({
                    'symbol': row['symbol'], 'price': price, 'ob_status': row['ob_status'],
                    'cross_status': row['cross_status'], 'rsi': row['rsi'], 'bull_ob': row['bull_ob'],
                    'sl': sl, 'tp': tp, 'rr': 1.8
                })
            elif (is_bear_trend and not is_bull_ob) or is_bear_ob:
                sl = row['bear_ob_high'] * 1.015 if row['bear_ob_high'] > 0 else price * 1.04
                tp = price - (sl - price) * 1.8
                short_candidates.append({
                    'symbol': row['symbol'], 'price': price, 'ob_status': row['ob_status'],
                    'cross_status': row['cross_status'], 'rsi': row['rsi'], 'bear_ob': row['bear_ob'],
                    'sl': sl, 'tp': tp, 'rr': 1.8
                })

        col_l, col_s = st.columns(2)
        with col_l:
            st.markdown("### 🚀 LONG (매수) 추천 종목")
            if long_candidates:
                for item in long_candidates[:3]:
                    with st.expander(f"🟢 **{item['symbol']}** (현재가: ${item['price']:,.4f})", expanded=True):
                        st.markdown(f"""
                        * **기술적 근거:** {item['cross_status']} | {item['ob_status']}
                        * **RSI(14):** {item['rsi']:.1f}
                        * **상승 오더블록:** {item['bull_ob']}
                        * **추천 진입가:** ${item['price']:,.4f} | **목표가 (TP):** `${item['tp']:,.4f}` | **손절가 (SL):** `${item['sl']:,.4f}`
                        """)
            else:
                st.info("현재 조건에 부합하는 LONG 종목이 없습니다.")

        with col_s:
            st.markdown("### 📉 SHORT (매도) 추천 종목")
            if short_candidates:
                for item in short_candidates[:3]:
                    with st.expander(f"🔴 **{item['symbol']}** (현재가: ${item['price']:,.4f})", expanded=True):
                        st.markdown(f"""
                        * **기술적 근거:** {item['cross_status']} | {item['ob_status']}
                        * **RSI(14):** {item['rsi']:.1f}
                        * **하락 오더블록:** {item['bear_ob']}
                        * **추천 진입가:** ${item['price']:,.4f} | **목표가 (TP):** `${item['tp']:,.4f}` | **손절가 (SL):** `${item['sl']:,.4f}`
                        """)
            else:
                st.info("현재 조건에 부합하는 SHORT 종목이 없습니다.")

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
            sl_distance_pct = (entry_price - stop_loss) / entry_price * 100 if "LONG" in position_type else (stop_loss - entry_price) / entry_price * 100
            tp_distance_pct = (take_profit - entry_price) / entry_price * 100 if "LONG" in position_type else (entry_price - take_profit) / entry_price * 100
            
            if sl_distance_pct > 0 and tp_distance_pct > 0:
                rr_ratio = tp_distance_pct / sl_distance_pct
                max_loss_amount = total_balance * (max_risk_pct / 100)
                position_size_usd = max_loss_amount / (sl_distance_pct / 100)
                rec_leverage = position_size_usd / total_balance
                expected_profit_amount = position_size_usd * (tp_distance_pct / 100)
                
                res1, res2, res3, res4 = st.columns(4)
                res1.metric("손익비 (R:R Ratio)", f"1 : {rr_ratio:.2f}")
                res2.metric("권장 레버리지", f"{rec_leverage:.1f}x")
                res3.metric("최대 예상 손실액", f"-${max_loss_amount:,.2f}")
                res4.metric("목표 예상 수익액", f"+${expected_profit_amount:,.2f}")

    with tab_ob:
        st.subheader("🧱 거래량 기반 유효 오더블록(Volume Order Block)")
        display_ob_df = pd.DataFrame({
            '마크': top_30_sorted['logo_url'],
            '종목코드': top_30_sorted['symbol'],
            '현재가': top_30_sorted['last_price'],
            '상승 오더블록': top_30_sorted['bull_ob'],
            '하락 오더블록': top_30_sorted['bear_ob'],
            '오더블록 상태': top_30_sorted['ob_status']
        })
        st.dataframe(display_ob_df, column_config={"마크": st.column_config.ImageColumn("마크", width="small")}, use_container_width=True, hide_index=True)

    with tab_all:
        st.subheader("🔥 Top 30 종합 리스트")
        display_df = pd.DataFrame({
            '마크': top_30_sorted['logo_url'],
            '종목코드': top_30_sorted['symbol'],
            '분야(섹터)': top_30_sorted['sector'],
            'RSI(14)': top_30_sorted['rsi'].map('{:.1f}'.format),
            '추세 상태': top_30_sorted['cross_status'],
            '오더블록 상태': top_30_sorted['ob_status']
        })
        st.dataframe(display_df, column_config={"마크": st.column_config.ImageColumn("마크", width="small")}, use_container_width=True, hide_index=True)

    with tab_rsi:
        st.subheader("🎯 RSI 50-200 수렴 및 크로스 종목")
        rsi_table = pd.DataFrame({
            '종목코드': top_30_sorted['symbol'],
            'RSI 50-200 이격도': top_30_sorted['rsi_gap'].map('{:.2f}'.format),
            '수렴 여부': top_30_sorted['is_squeezed'],
            '추세 상태': top_30_sorted['cross_status']
        })
        st.dataframe(rsi_table, use_container_width=True, hide_index=True)
else:
    st.warning("데이터를 불러오지 못했습니다. 잠시 후 우측 상단 '시세 새로고침' 버튼을 눌러주세요.")