import streamlit as st
import ccxt
import pandas as pd
import ta
import time

st.set_page_config(page_title="거래량 오더블록 & 리스크/레버리지 계산기", layout="wide")

TOP_MAJORS = {'BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'ADA', 'AVAX', 'DOT', 'LINK', 'SUI', 'APT', 'BCH', 'MATIC', 'NEAR'}

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

@st.cache_data(ttl=60)
def analyze_volume_ob_and_rsi(symbol, _exchange):
    try:
        ohlcv = _exchange.fetch_ohlcv(symbol, timeframe='1d', limit=150)
        if not ohlcv or len(ohlcv) < 60:
            return None
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_sma50'] = ta.trend.sma_indicator(df['rsi'], window=50)
        df['rsi_sma200'] = ta.trend.sma_indicator(df['rsi'], window=200)
        df['vol_ma20'] = ta.trend.sma_indicator(df['volume'], window=20)
        
        df['vol_spike'] = df['volume'] > (df['vol_ma20'] * 1.8)
        
        latest_bull_ob = "없음"
        latest_bear_ob = "없음"
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

@st.cache_data(ttl=30)
def load_market_data():
    exchanges_to_try = [
        ('MEXC', getattr(ccxt, 'mexc', None)),
        ('Gate.io', getattr(ccxt, 'gate', None)),
        ('Bybit', getattr(ccxt, 'bybit', None))
    ]
    
    tickers = None
    exchange = None

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
    if df.empty:
        return pd.DataFrame(), exchange
        
    df['drawdown_pct'] = ((df['last_price'] - df['high_24h']) / df['high_24h']) * 100
    return df, exchange

st.title("🔥 거래량 오더블록 & 리스크/레버리지 계산기")
st.caption("실시간 오더블록 포착 및 포지션별 손익비(R:R)·적정 레버리지 산출")

df_all, exchange = load_market_data()

if not df_all.empty and exchange is not None:
    top_30 = df_all.sort_values(by='change_pct', ascending=False).head(30).copy()
    
    with st.spinner("거래량 기반 오더블록(Volume OB) 및 RSI 장기 지표 산출 중..."):
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

    # 탭 순서 변경 및 "👑 메이저 AI 추천" 추가
    tab_major, tab_signal, tab_ob, tab_calc, tab_all, tab_rsi = st.tabs([
        "👑 메이저 AI 추천 (BTC/ETH/SOL 등)",
        "🤖 전체 AI 추천 포지션",
        "🧱 거래량 오더블록 (Volume OB)",
        "🧮 레버리지 & 손익비(R:R) 계산기",
        "🔥 Top 30 종합 리스트",
        "🎯 RSI 수렴/크로스"
    ])

    # 추천 종목 추출 공통 함수
    def get_trade_signals(df_input):
        longs = []
        shorts = []
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

    # 👑 메이저 AI 추천 탭
    with tab_major:
        st.subheader("👑 비트코인 및 주요 메이저 코인 전용 추천")
        st.caption("비트코인(BTC), 이더리움(ETH), 솔라나(SOL), 리플(XRP) 등 시장 주도 메이저 자산의 타점 분석 결과입니다.")
        
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
                            * **목표가 (TP):** `${item['tp']:,.2f}`
                            * **손절가 (SL):** `${item['sl']:,.2f}` (손익비 1 : {item['rr']:.1f})
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
                            * **목표가 (TP):** `${item['tp']:,.2f}`
                            * **손절가 (SL):** `${item['sl']:,.2f}` (손익비 1 : {item['rr']:.1f})
                            """)
                else:
                    st.info("현재 매도 조건에 부합하는 메이저 코인이 없습니다.")
        else:
            st.info("현재 거래량 상위 30개 항목 내에 포함된 메이저 코인이 없습니다.")

    # 🤖 전체 AI 추천 포지션 탭
    with tab_signal:
        st.subheader("💡 Top 30 거래량 OB + RSI 종합 추천 종목")
        st.caption("거래량 오더블록과 RSI 방향성이 일치하는 전체 종목 추천입니다.")
        
        all_longs, all_shorts = get_trade_signals(top_30_sorted)
        col_l, col_s = st.columns(2)
        
        with col_l:
            st.markdown("### 🚀 LONG (매수) 추천 종목")
            if all_longs:
                for item in all_longs[:4]:
                    with st.expander(f"🟢 **{item['symbol']}** (현재가: ${item['price']:,.4f})", expanded=True):
                        st.markdown(f"""
                        * **기술적 근거:** {item['cross_status']} | {item['ob_status']}
                        * **RSI(14):** {item['rsi']:.1f}
                        * **상승 오더블록:** {item['bull_ob']}
                        * **추천 진입가:** ${item['price']:,.4f}
                        * **목표가 (TP):** `${item['tp']:,.4f}`
                        * **손절가 (SL):** `${item['sl']:,.4f}` (손익비 1 : {item['rr']:.1f})
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
                        * **RSI(14):** {item['rsi']:.1f}
                        * **하락 오더블록:** {item['bear_ob']}
                        * **추천 진입가:** ${item['price']:,.4f}
                        * **목표가 (TP):** `${item['tp']:,.4f}`
                        * **손절가 (SL):** `${item['sl']:,.4f}` (손익비 1 : {item['rr']:.1f})
                        """)
            else:
                st.info("현재 조건에 부합하는 SHORT 종목이 없습니다.")

    # 🧮 레버리지 & 손익비 계산기 탭
    with tab_calc:
        st.subheader("🧮 리스크 관리 및 적정 레버리지 계산기")
        st.caption("손절 시 손실 금액을 시드의 일정 비율로 제한하는 적정 레버리지와 손익비를 산출합니다.")
        
        col_in1, col_in2 = st.columns([1, 1])
        
        with col_in1:
            st.markdown("##### 1️⃣ 계좌 및 매매 조건 설정")
            total_balance = st.number_input("총 시드 자산 ($)", value=10000.0, step=500.0)
            max_risk_pct = st.slider("1회 매매 최대 감수 리스크 (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
            position_type = st.radio("포지션 방향", ["LONG (매수)", "SHORT (매도)"], horizontal=True)
            
        with col_in2:
            st.markdown("##### 2️⃣ 가격 타점 입력 ($)")
            entry_price = st.number_input("진입 가격 ($)", value=100.0, step=0.1)
            
            default_sl = entry_price * 0.95 if "LONG" in position_type else entry_price * 1.05
            default_tp = entry_price * 1.10 if "LONG" in position_type else entry_price * 0.90
            
            stop_loss = st.number_input("손절 가격 (Stop Loss) ($)", value=default_sl, step=0.1)
            take_profit = st.number_input("목표 가격 (Take Profit) ($)", value=default_tp, step=0.1)

        st.markdown("---")
        
        if entry_price > 0 and stop_loss > 0 and take_profit > 0:
            if "LONG" in position_type:
                sl_distance_pct = (entry_price - stop_loss) / entry_price * 100
                tp_distance_pct = (take_profit - entry_price) / entry_price * 100
            else:
                sl_distance_pct = (stop_loss - entry_price) / entry_price * 100
                tp_distance_pct = (entry_price - take_profit) / entry_price * 100
                
            if sl_distance_pct <= 0:
                st.error("⚠️ 손절가가 진입가보다 올바르지 않은 위치에 있습니다.")
            elif tp_distance_pct <= 0:
                st.error("⚠️ 목표가가 진입가보다 올바르지 않은 위치에 있습니다.")
            else:
                rr_ratio = tp_distance_pct / sl_distance_pct
                max_loss_amount = total_balance * (max_risk_pct / 100)
                position_size_usd = max_loss_amount / (sl_distance_pct / 100)
                rec_leverage = position_size_usd / total_balance
                expected_profit_amount = position_size_usd * (tp_distance_pct / 100)
                
                st.markdown("##### 📊 리스크 분석 결과")
                res1, res2, res3, res4 = st.columns(4)
                
                res1.metric("손익비 (R:R Ratio)", f"1 : {rr_ratio:.2f}", delta="손익비 양호" if rr_ratio >= 1.5 else "손익비 낮음")
                res2.metric("권장 레버리지", f"{rec_leverage:.1f}x")
                res3.metric("최대 예상 손실액", f"-${max_loss_amount:,.2f}", f"-{max_risk_pct:.1f}% 시드")
                res4.metric("목표 예상 수익액", f"+${expected_profit_amount:,.2f}", f"+{(expected_profit_amount/total_balance)*100:.1f}% 시드")

                st.markdown("---")
                st.info(f"""
                💡 **매매 실행 가이드:**
                * **추천 손익비:** 보통 **1 : 1.5 이상**일 때 진입하는 것이 통계적으로 유효합니다. (현재: **1 : {rr_ratio:.2f}**)
                * **포지션 규모:** 총 **${position_size_usd:,.2f}** 상당의 코인 수량을 체결해야 합니다.
                * **레버리지 활용법:** 시드 전체(${total_balance:,.0f})를 증거금으로 쓸 경우 **{rec_leverage:.1f}x 레버리지**를 적용하면 손절 시 딱 **${max_loss_amount:,.2f} ({max_risk_pct}%)**만 손실 처리됩니다.
                """)

    with tab_ob:
        st.subheader("🧱 거래량 기반 유효 오더블록(Volume Order Block) 매수/매도 구간")
        display_ob_df = pd.DataFrame({
            '종목코드': top_30_sorted['symbol'],
            '현재가': top_30_sorted['last_price'],
            '상승 오더블록 (매수 지지대)': top_30_sorted['bull_ob'],
            '하락 오더블록 (매도 저항대)': top_30_sorted['bear_ob'],
            '오더블록 도달 상태': top_30_sorted['ob_status'],
            '24h 상승률': top_30_sorted['change_pct'].map('{:+.2f}%'.format)
        })
        st.dataframe(
            display_ob_df,
            use_container_width=True,
            height=500,
            hide_index=True
        )

    with tab_all:
        st.subheader("🔥 Top 30 종합 시세 및 분야별 분류")
        display_df = pd.DataFrame({
            '종목코드': top_30_sorted['symbol'],
            '분야(섹터)': top_30_sorted['sector'],
            'RSI(14)': top_30_sorted['rsi'].map('{:.1f}'.format),
            '추세 상태': top_30_sorted['cross_status'],
            '오더블록 상태': top_30_sorted['ob_status'],
            '고점 대비 조정률': top_30_sorted['drawdown_pct'].map('{:.2f}%'.format)
        })
        st.dataframe(
            display_df,
            use_container_width=True,
            height=500,
            hide_index=True
        )

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

    time.sleep(30)
    st.rerun()
else:
    st.warning("거래소 API 데이터를 불러올 수 없어 30초 후 자동으로 재시도합니다.")
    time.sleep(30)
    st.rerun()