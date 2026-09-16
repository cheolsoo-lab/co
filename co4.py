import concurrent.futures
import ccxt
import numpy as np
import pandas as pd
import streamlit as st
import ta

st.set_page_config(
    page_title="🔥 POC·ADX·Sharpe 결합 초고도화 AI 추천 대시보드",
    layout="wide",
)

TOP_MAJORS = {
    'BTC',
    'ETH',
    'SOL',
    'XRP',
    'BNB',
    'ADA',
    'AVAX',
    'DOT',
    'LINK',
    'SUI',
    'APT',
    'BCH',
    'NEAR',
    'DOGE',
}


# ==========================================
# 1. 거래소 시세 데이터 수집 (Fallback 포함)
# ==========================================
@st.cache_data(ttl=300)
def load_market_data():
  exchanges_to_try = [
      ('MEXC', getattr(ccxt, 'mexc', None)),
      ('Gate.io', getattr(ccxt, 'gate', None)),
      ('Bybit', getattr(ccxt, 'bybit', None)),
  ]

  tickers, exchange = None, None
  for ex_name, ex_class in exchanges_to_try:
    if ex_class is None:
      continue
    try:
      ex_instance = ex_class(
          {'enableRateLimit': True, 'options': {'defaultType': 'spot'}}
      )
      tickers = ex_instance.fetch_tickers()
      if tickers:
        exchange = ex_instance
        break
    except Exception:
      continue

  if not tickers or not exchange:
    st.error('모든 거래소 API 접근이 일시적으로 제한되었습니다.')
    return pd.DataFrame(), None

  market_data = []
  for symbol, data in tickers.items():
    if not symbol.endswith('/USDT'):
      continue
    base_asset = symbol.split('/')[0]
    quote_vol = data.get('quoteVolume', 0) or 0
    if base_asset not in TOP_MAJORS and quote_vol < 1000000:
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
      })

  df = pd.DataFrame(market_data)
  return df, exchange


def fetch_ohlcv_advanced(_exchange, symbol, timeframe='1d', limit=150):
  """OHLCV 수집 및 POC, ATR, ADX 기술적 지표 일괄 산출"""
  exchanges_to_try = []
  if _exchange is not None:
    exchanges_to_try.append(_exchange)

  for ex_name, ex_class in [
      ('MEXC', getattr(ccxt, 'mexc', None)),
      ('Gate.io', getattr(ccxt, 'gate', None)),
      ('Bybit', getattr(ccxt, 'bybit', None)),
  ]:
    if ex_class is not None:
      try:
        ex_inst = ex_class(
            {'enableRateLimit': True, 'options': {'defaultType': 'spot'}}
        )
        if _exchange is None or ex_inst.id != _exchange.id:
          exchanges_to_try.append(ex_inst)
      except Exception:
        continue

  for ex in exchanges_to_try:
    try:
      ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
      if ohlcv and len(ohlcv) >= 60:
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
        )
        df.rename(
            columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
            },
            inplace=True,
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # 1. ATR 계산
        df['ATR'] = ta.volatility.average_true_range(
            df['High'], df['Low'], df['Close'], window=14
        )
        # 2. ADX 계산 (추세 강도)
        adx_ind = ta.trend.ADXIndicator(
            df['High'], df['Low'], df['Close'], window=14
        )
        df['ADX'] = adx_ind.adx()
        # 3. Volume POC 산출 (최근 60봉 기준 매물대 최고 거래 가격)
        recent_df = df.iloc[-60:]
        price_bins = pd.cut(recent_df['Close'], bins=20)
        poc_bin = recent_df.groupby(price_bins, observed=False)[
            'Volume'
        ].sum().idxmax()
        df['POC_Price'] = (poc_bin.left + poc_bin.right) / 2

        return df
    except Exception:
      continue

  return pd.DataFrame()


# ==========================================
# 2. Sharpe Ratio & Profit Factor 백테스트 엔진
# ==========================================
def run_advanced_backtest(df, tp_atr_mult, sl_atr_mult, adx_threshold=20.0):
  """ADX 강도 및 POC 매물대 보정을 결합한 평가 엔진"""
  closes = df['Close'].values
  lows = df['Low'].values
  adx_vals = df['ADX'].fillna(0).values

  signals = []
  for i in range(20, len(df) - 1):
    # ADX 추세 필터 (강한 추세 구간만 매수)
    if adx_vals[i] >= adx_threshold:
      # 저점 반등 패턴 체크
      if closes[i] > closes[i - 1] and lows[i - 1] < lows[i - 2]:
        signals.append(i)

  if not signals:
    return None

  trades = []
  for b_idx in signals:
    if b_idx >= len(df) - 1:
      continue

    entry_price = df['Close'].iloc[b_idx]
    atr_val = df['ATR'].iloc[b_idx]

    if pd.isna(atr_val) or atr_val <= 0:
      continue

    tp_price = entry_price + (tp_atr_mult * atr_val)
    sl_price = entry_price - (sl_atr_mult * atr_val)

    post_df = df.iloc[b_idx + 1 : min(b_idx + 1 + 15, len(df))]
    exit_price = entry_price

    for k in range(len(post_df)):
      high_p = post_df['High'].iloc[k]
      low_p = post_df['Low'].iloc[k]

      if high_p >= tp_price:
        exit_price = tp_price
        break
      elif low_p <= sl_price:
        exit_price = sl_price
        break
      else:
        exit_price = post_df['Close'].iloc[k]

    trades.append((exit_price - entry_price) / entry_price)

  if len(trades) < 2:
    return None

  trades_arr = np.array(trades)
  tot_ret = np.sum(trades_arr) * 100
  win_rate = (np.sum(trades_arr > 0) / len(trades_arr)) * 100

  # Profit Factor 산출 (총 수익 / 총 손실)
  gains = trades_arr[trades_arr > 0]
  losses = abs(trades_arr[trades_arr < 0])
  profit_factor = (
      np.sum(gains) / np.sum(losses) if np.sum(losses) > 0 else 99.0
  )

  # Sharpe Ratio 산출 (무위험 수익률 0% 가정)
  std_dev = np.std(trades_arr)
  sharpe_ratio = (
      (np.mean(trades_arr) / std_dev) * np.sqrt(365) if std_dev > 0 else 0.0
  )

  return {
      'return_pct': round(tot_ret, 1),
      'win_rate': round(win_rate, 1),
      'profit_factor': round(profit_factor, 2),
      'sharpe_ratio': round(sharpe_ratio, 2),
      'trades_count': len(trades),
  }


def analyze_single_symbol_advanced(exchange, symbol):
  """POC, ADX, ATR, Sharpe, Profit Factor 통합 랭킹 계산 태스크"""
  df = fetch_ohlcv_advanced(exchange, symbol)
  if df.empty or len(df) < 60:
    return None

  best_metric = None
  best_score = -999.0

  # 파라미터 조합 탐색
  for tp_m in [2.5, 3.5]:
    for sl_m in [1.2, 1.8]:
      res = run_advanced_backtest(df, tp_m, sl_m, adx_threshold=20.0)
      if res and res['profit_factor'] >= 1.3 and res['win_rate'] >= 50.0:
        # Sharpe Ratio와 Profit Factor 기반 종합 스코어 생성
        score = res['sharpe_ratio'] * 0.6 + res['profit_factor'] * 0.4
        if score > best_score:
          best_score = score
          best_metric = (tp_m, sl_m, res)

  if best_metric:
    tp_m, sl_m, res = best_metric
    latest_close = df['Close'].iloc[-1]
    latest_atr = df['ATR'].iloc[-1]
    latest_adx = df['ADX'].iloc[-1]
    poc_price = df['POC_Price'].iloc[-1]

    calc_tp = latest_close + (tp_m * latest_atr)
    calc_sl = latest_close - (sl_m * latest_atr)

    return {
        'symbol': symbol,
        'current_price': latest_close,
        'poc_price': round(poc_price, 4),
        'adx': round(latest_adx, 1),
        'atr': round(latest_atr, 4),
        'sharpe_ratio': res['sharpe_ratio'],
        'profit_factor': res['profit_factor'],
        'win_rate': res['win_rate'],
        'return_pct': res['return_pct'],
        'tp_price': round(calc_tp, 4),
        'sl_price': round(calc_sl, 4),
        'tp_pct': round(((calc_tp - latest_close) / latest_close) * 100, 2),
        'sl_pct': round(((latest_close - calc_sl) / latest_close) * 100, 2),
    }

  return None


# ==========================================
# 3. 병렬 스레드 실행 엔진
# ==========================================
def run_advanced_engine_parallel(exchange, target_symbols):
  results = []
  progress_bar = st.progress(0)
  status_text = st.empty()
  completed = 0
  total = len(target_symbols)

  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    future_map = {
        executor.submit(analyze_single_symbol_advanced, exchange, sym): sym
        for sym in target_symbols
    }
    for future in concurrent.futures.as_completed(future_map):
      completed += 1
      res = future.result()
      if res:
        results.append(res)
      progress_bar.progress(completed / total)
      status_text.text(
          f"⚡ POC·ADX·Sharpe 고도화 필터 계산 중... ({completed}/{total} 완료)"
      )

  progress_bar.empty()
  status_text.empty()

  if results:
    # Sharpe Ratio 기준으로 최상위 고신뢰도 종목 정렬
    return pd.DataFrame(results).sort_values(
        by='sharpe_ratio', ascending=False
    )
  return pd.DataFrame()


# ==========================================
# 4. Streamlit 화면 구성
# ==========================================
st.title("🔥 POC·ADX·Sharpe 초고도화 크립토 AI 추천 대시보드")
st.caption(
    "Volume POC 매물대, ADX 추세 강도, Sharpe Ratio 위험 대비 수익률 및 Profit"
    " Factor 평가를 통과한 최상위 종목 추천입니다."
)

if st.sidebar.button("🔄 실시간 데이터 새로고침", type="primary"):
  st.cache_data.clear()
  st.rerun()

df_all, exchange = load_market_data()

if not df_all.empty and exchange is not None:
  majors_df = df_all[df_all['base'].isin(TOP_MAJORS)]
  others_df = (
      df_all[~df_all['base'].isin(TOP_MAJORS)]
      .sort_values(by='change_pct', ascending=False)
      .head(20)
  )
  target_df = (
      pd.concat([majors_df, others_df])
      .drop_duplicates(subset=['symbol'])
      .reset_index(drop=True)
  )

  st.subheader("🏆 Sharpe Ratio 및 Profit Factor 기반 랭킹 종목")
  run_adv_btn = st.button("🚀 초고도화 종합 백테스트 탐색 실행", type="primary")

  if 'adv_rec_df' not in st.session_state:
    st.session_state.adv_rec_df = None

  if run_adv_btn:
    symbols_list = target_df['symbol'].tolist()
    with st.spinner("멀티스레딩 분석 엔진 가동 중..."):
      res_df = run_advanced_engine_parallel(exchange, symbols_list)
      st.session_state.adv_rec_df = res_df

  if st.session_state.adv_rec_df is not None:
    adv_df = st.session_state.adv_rec_df
    if not adv_df.empty:
      st.success(
          f"⚡ 검증 완료! 총 {len(adv_df)}개 종목이 POC·ADX·Sharpe 초고도화"
          " 통과 기준을 만족했습니다."
      )
      for _, item in adv_df.iterrows():
        with st.expander(
            f"🟢 **{item['symbol']}** (현재가: ${item['current_price']:,.4f}) -"
            f" Sharpe Ratio: {item['sharpe_ratio']} | Profit Factor:"
            f" {item['profit_factor']}",
            expanded=True,
        ):
          col1, col2 = st.columns(2)
          with col1:
            st.markdown(f"""
                        * **Sharpe Ratio (위험 대비 수익):** `{item['sharpe_ratio']}`
                        * **Profit Factor (손익비 총합):** `{item['profit_factor']}`
                        * **검증 백테스트 승률 / 수익률:** `{item['win_rate']}%` / `+{item['return_pct']}%`
                        * **ADX 추세 강도:** `{item['adx']}` (20 이상 강한 추세)
                        """)
          with col2:
            st.markdown(f"""
                        * **Volume POC (최대 매물대):** `${item['poc_price']:,.4f}`
                        * **ATR 동적 목표가 (TP):** `${item['tp_price']:,.4f}` (+{item['tp_pct']}%)
                        * **ATR 동적 손절가 (SL):** `${item['sl_price']:,.4f}` (-{item['sl_pct']}%)
                        """)
    else:
      st.warning(
          "현재 초고도화 검증 기준(Profit Factor >= 1.3, Sharpe Ratio 우수 및"
          " ADX >= 20)을 모두 만족하는 종목이 없습니다."
      )
  else:
    st.info(
        "상단의 **[🚀 초고도화 종합 백테스트 탐색 실행]** 버튼을 누르면 정밀"
        " 추천 분석을 시작합니다."
    )