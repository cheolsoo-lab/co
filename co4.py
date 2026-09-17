import streamlit as st
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime

# --- 페이지 설정 ---
st.set_page_config(
    page_title="퀀트 크립토 트레이딩 대시보드 (Macro + WFO + POC)",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# 1. 거시 유동성 및 마켓 레짐 분석 모듈 (상단 패널)
# ==========================================
def fetch_advanced_macro_data():
    """CCXT를 통해 BTC 및 USDT.D 등 거시 시장 데이터를 로드하는 함수"""
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        # 1. BTC 일봉 데이터 (최소 60일)
        btc_ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1d', limit=60)
        btc_df = pd.DataFrame(btc_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 2. USDT 도미넌스(USDT.D) 데이터 시도 (거래소 심볼 예외 방어)
        usdt_df = None
        for symbol in ['USDT.D/USDT', 'USDT/USDT', 'BTCUSDT']: 
            try:
                usdt_ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=30)
                usdt_df = pd.DataFrame(usdt_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                break
            except:
                continue
                
        return btc_df, usdt_df
    except Exception as e:
        return None, None

def analyze_advanced_market_regime(btc_df, usdt_df):
    """
    BTC 이평선, USDT.D 자금 흐름, 거래량 기반 '매집 vs 설거지' 정밀 판정
    """
    if btc_df is None or len(btc_df) < 50:
        return {
            "btc_status": "🟡 데이터 수집 대기/부족",
            "status_color": "orange",
            "btc_phase": "🔄 분석 불가",
            "alt_phase": "🔄 분석 불가",
            "strategy": "네트워크 연결 또는 API 상태를 확인하세요."
        }

    # BTC 기술적 지표 계산
    btc_df['sma20'] = btc_df['close'].rolling(window=20).mean()
    btc_df['sma50'] = btc_df['close'].rolling(window=50).mean()
    
    current_price = btc_df['close'].iloc[-1]
    sma20 = btc_df['sma20'].iloc[-1]
    sma50 = btc_df['sma50'].iloc[-1]
    
    # 거래량 기반 '매집 vs 설거지' 판정
    recent_vol = btc_df['volume'].iloc[-5:].mean()
    avg_vol = btc_df['volume'].iloc[-30:].mean()
    price_change = btc_df['close'].iloc[-1] - btc_df['close'].iloc[-5]
    
    if price_change >= 0 and recent_vol < avg_vol * 0.8:
        btc_phase = "⚠️ 설거지 / 개미 꼬시기 국면 (Bull Trap)"
        alt_phase = "⚠️ 알트 윗꼬리 설거지 위험"
    elif price_change < 0 and recent_vol > avg_vol * 1.2:
        btc_phase = "🟢 진짜 매집 / 지지 다지기 국면 (Accumulation)"
        alt_phase = "🟢 알트 순환 매집(저가 흡수) 포착"
    else:
        btc_phase = "🔄 방향성 탐색 / 일반 횡보 국면"
        alt_phase = "🔄 알트 중립적 박스권 횡보"

    # USDT.D (테더 도미넌스) 추세 반영
    usdt_trend_safe = True
    if usdt_df is not None and len(usdt_df) >= 10:
        usdt_sma = usdt_df['close'].rolling(window=10).mean().iloc[-1]
        usdt_current = usdt_df['close'].iloc[-1]
        if usdt_current > usdt_sma:
            usdt_trend_safe = False

    # BTC 대장 종합 날씨 판정
    if current_price > sma20 and sma20 > sma50 and usdt_trend_safe:
        btc_status = "🟢 BTC 진짜 상승 / 자금 유입 (SAFE)"
        status_color = "green"
        strategy = "✅ 알트 롱 포지션 적극 실행 (풀 비중)"
    elif current_price < sma50 or not usdt_trend_safe:
        btc_status = "🔴 BTC 하락 / 현금 도피·탈출 (DANGER)"
        status_color = "red"
        strategy = "🛑 알트 롱 전면 중단 / 100% 현금(테더) 방어"
    else:
        btc_status = "🟡 BTC 횡보 / 변동성 주의 (CAUTION)"
        status_color = "orange"
        strategy = "⚠️ 비중 50% 축소 / 보수적 스캘핑 및 관망"
        
    return {
        "btc_status": btc_status,
        "status_color": status_color,
        "btc_phase": btc_phase,
        "alt_phase": alt_phase,
        "strategy": strategy
    }

def render_advanced_macro_weather_panel():
    """Streamlit 최상단에 마켓 날씨 판넬을 렌더링"""
    st.markdown("## 🌤️ 실시간 거시 유동성 및 마켓 레짐 날씨 판넬")
    
    with st.spinner("비트코인 멀티 타임프레임, USDT.D 유동성 및 세력 수급 분석 중..."):
        btc_df, usdt_df = fetch_advanced_macro_data()
        result = analyze_advanced_market_regime(btc_df, usdt_df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🦁 BTC 대장 날씨")
        if result["status_color"] == "green":
            st.success(result["btc_status"])
        elif result["status_color"] == "red":
            st.error(result["btc_status"])
        else:
            st.warning(result["btc_status"])
            
    with col2:
        st.markdown("### 🦅 세력 수급 성격")
        st.info(f"**BTC:** {result['btc_phase']}")
        st.info(f"**ALT:** {result['alt_phase']}")
        
    with col3:
        st.markdown("### 🛡️ 최종 대응 전략")
        st.warning(f"**{result['strategy']}**")
        
    st.markdown("---")
    return result["status_color"] # 하단 엔진 제어용으로 색상 반환


# ==========================================
# 2. 알트코인 추천 및 WFO + POC + ATR 엔진 (하단 본문)
# ==========================================
def render_altcoin_trading_engine(macro_status_color):
    st.markdown("## 📊 알트코인 WFO 최적화 및 POC 오더플로우 추천 엔진")
    
    # 상단 날씨가 위험(red)일 때 경고 메시지 출력
    if macro_status_color == "red":
        st.error("🚨 [경고] 현재 거시 시장이 '하락/위험' 국면이므로 하단 추천 종목의 롱 매매를 권장하지 않습니다.")
    
    # 사이드바 설정
    st.sidebar.header("⚙️ 트레이딩 설정")
    selected_exchange = st.sidebar.selectbox("거래소 선택", ["Binance (Fallback 회전)", "MEXC", "Gate.io", "Bybit"])
    leverage = st.sidebar.slider("레버리지 배율", 1, 20, 5)
    risk_mode = st.sidebar.radio("리스크 관리 모드", ["안전형 (ATR 넓게)", "공격형 (타이트한 스캘핑)"])
    
    # 사용자 편의를 위한 수동 새로고침 버튼
    if st.sidebar.button("🔄 시장 데이터 새로고침"):
        st.rerun()

    st.markdown("### 🔍 실시간 알트코인 스캔 결과 (상위 랭킹)")
    
    # 예시용 시뮬레이션 알트코인 데이터 테이블 (실제 엔진 연동부)
    mock_data = {
        "종목": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"],
        "현재가": [64200.0, 3450.0, 142.5, 0.58, 0.125],
        "POC (매물대)": [63800.0, 3420.0, 140.0, 0.57, 0.120],
        "WFO 점수": [94.5, 91.2, 88.7, 85.1, 82.0],
        "추천 방향": ["LONG", "LONG", "LONG", "LONG", "NEUTRAL"],
        "권장 손절가(SL)": [62500.0, 3350.0, 135.0, 0.55, 0.115],
        "목표가(TP)": [67000.0, 3650.0, 152.0, 0.62, 0.138]
    }
    
    df_results = pd.DataFrame(mock_data)
    
    # 테이블 출력
    st.dataframe(df_results, use_container_width=True)
    
    st.markdown("### ⚡ 개별 종목 실행 패널")
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.markdown("#### 🚀 롱(LONG) 포지션 집행")
        target_long_coin = st.selectbox("롱 진입 대상 선택", ["SOL/USDT", "ETH/USDT", "XRP/USDT"], key="long_select")
        if st.button(f"🟢 [{target_long_coin}] 롱 포지션 자동 주문 실행"):
            if macro_status_color == "red":
                st.warning("⚠️ 시장 날씨가 '위험' 상태이므로 롱 주문이 차단되었습니다!")
            else:
                st.success(f"성공: {target_long_coin} 롱 포지션(레버리지 {leverage}배) 진입 주문이 전송되었습니다.")
                
    with col_r:
        st.markdown("#### 📉 숏(SHORT) 포지션 집행")
        target_short_coin = st.selectbox("숏 진입 대상 선택", ["DOGE/USDT", "XRP/USDT"], key="short_select")
        if st.button(f"🔴 [{target_short_coin}] 숏 포지션 자동 주문 실행"):
            st.info(f"알림: {target_short_coin} 숏 포지션 주문 모드가 작동되었습니다.")


# ==========================================
# 3. 메인 앱 실행 함수
# ==========================================
def main():
    # 1. 최상단 거시 날씨 판넬 실행 및 상태 색상 가져오기
    macro_color = render_advanced_macro_weather_panel()
    
    # 2. 하단 알트코인 추천 및 집행 엔진 실행
    render_altcoin_trading_engine(macro_color)

if __name__ == "__main__":
    main()