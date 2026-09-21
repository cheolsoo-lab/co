import concurrent.futures
import ccxt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import streamlit as st
import ta

st.set_page_config(
    page_title="🔥 추천 대시보드 (고도화 v2)",
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
# 0. 거시 유동성 및 마켓 레짐 날씨 판넬[cite: 3]
# ==========================================
def fetch_advanced_macro_data():
  exchanges_to_try = [
      ('Bybit', getattr(ccxt, 'bybit', None)),
      ('MEXC', getattr(ccxt, 'mexc', None)),
      ('Binance', getattr(ccxt, 'binance', None)),
  ]

  for ex_name, ex_class in exchanges_to_try:
    if ex_class is None:
      continue
    try:
      ex_instance = ex_class({
          'enableRateLimit': True,
          'options': {'defaultType': 'spot'},
      })
      btc_ohlcv = ex_instance.fetch_ohlcv('BTC/USDT', timeframe='1d', limit=60)
      if btc_ohlcv and len(btc_ohlcv) >= 50:
        btc_df = pd.DataFrame(
            btc_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
        )
        return btc_df, None
    except Exception:
      continue

  return None, None


def analyze_advanced_market_regime(btc_df, usdt_df):
  if btc_df is None or len(btc_df) < 50:
    return {
        'btc_status': '🟡 데이터 수집 대기/부족',
        'status_color': 'orange',
        'btc_phase': '🔄 분석 불가',
        'alt_phase': '🔄 분석 불가',
        'strategy': '네트워크 연결 또는 API 상태를 확인하세요.',
    }

  btc_df['sma20'] = btc_df['close'].rolling(window=20).mean()
  btc_df['sma50'] = btc_df['close'].rolling(window=50).mean()

  current_price = btc_df['close'].iloc[-1]
  sma20 = btc_df['sma20'].iloc[-1]
  sma50 = btc_df['sma50'].iloc[-1]

  recent_vol = btc_df['volume'].iloc[-5:].mean()
  avg_vol = btc_df['volume'].iloc[-30:].mean()
  price_change = btc_df['close'].iloc[-1] - btc_df['close'].iloc[-5]
  recent_lows_held = current_price >= btc_df['low'].iloc[-5:].min()

  if price_change >= 0 and recent_vol < avg_vol * 0.9:
    btc_phase = '⚠️ 설거지 / 개미 꼬시기 국면 (Bull Trap)'
    alt_phase = '⚠️ 알트 윗꼬리 설거지 위험'
  elif recent_lows_held and recent_vol >= avg_vol * 0.95:
    btc_phase = '🟢 진짜 매집 / 저가 방어 국면 (Accumulation)'
    alt_phase = '🟢 알트 저점 매수세(순환매집) 포착'
  else:
    btc_phase = '🔄 물량 소화 및 매물대 다지기'
    alt_phase = '🔄 알트 눈치보기 박스권'

  if current_price > sma20 and sma20 > sma50:
    btc_status = '🟢 BTC 진짜 상승 / 자금 유입 (SAFE)'
    status_color = 'green'
    strategy = '✅ 알트 롱 포지션 적극 실행 (풀 비중 / 숏 금지)'
  elif current_price < sma50:
    btc_status = '🔴 BTC 하락 / 현금 도피·탈출 (DANGER)'
    status_color = 'red'
    strategy = '🛑 알트 롱 전면 중단 / 100% 현금(테더) 방어'
  else:
    btc_status = '🟡 BTC 횡보 / 변동성 주의 (CAUTION)'
    status_color = 'orange'
    strategy = '⚠️ 비중 50% 축소 / 보수적 스캘핑 및 관망'

  return {
      'btc_status': btc_status,
      'status_color': status_color,
      'btc_phase': btc_phase,
      'alt_phase': alt_phase,
      'strategy': strategy,
  }


def render_advanced_macro_weather_panel():
  st.markdown('### 🌤️ 실시간 거시 유동성 및 마켓 레짐 날씨 패널')
  with st.spinner('비트코인 멀티 타임프레임 및 세력 수급 분석 중...'):
    btc_df, usdt_df = fetch_advanced_macro_data()
    result = analyze_advanced_market_regime(btc_df, usdt_df)

  col1, col2, col3 = st.columns(3)
  with col1:
    st.markdown('**🦁 BTC 대장 날씨**')
    if result['status_color'] == 'green':
      st.success(result['btc_status'])
    elif result['status_color'] == 'red':
      st.error(result['btc_status'])
    else:
      st.warning(result['btc_status'])

  with col2:
    st.markdown('**🦅 세력 수급 성격**')
    st.info(f"**BTC:** {result['btc_phase']}")
    st.info(f"**ALT:** {result['alt_phase']}")

  with col3:
    st.markdown('**🛡️ 최종 대응 전략**')
    st.warning(f"**{result['strategy']}**")
  st.markdown('---')
  return result['status_color']


# ==========================================
# 1. 시세 데이터 및 거래소 API 로드 (1달 유동성 기준 도입)[cite: 3]
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
    if base_asset not in TOP_MAJORS and quote_vol < 2000000:  # 최소 거래대금 상향
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
          'quote_vol': quote_vol,
          'high_24h': high_24h,
          'low_24h': low_24h,
          'last_price': last_price,
      })

  df = pd.DataFrame(market_data)
  return df, exchange


def fetch_ohlcv_full(_exchange, symbol, timeframe='1d', limit=100):
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
      if ohlcv and len(ohlcv) >= 40:
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

        df['ATR'] = ta.volatility.average_true_range(
            df['High'], df['Low'], df['Close'], window=14
        )
        adx_ind = ta.trend.ADXIndicator(
            df['High'], df['Low'], df['Close'], window=14
        )
        df['ADX'] = adx_ind.adx()

        # 🌟 [신규 추가] 최근 30일간의 평균 거래대금(USDT 기준 가치 추정) 및 30일 변동성(표준편차) 계산
        df['Quote_Volume_Val'] = df['Close'] * df['Volume']
        df['Vol_30d_Mean'] = df['Quote_Volume_Val'].rolling(window=30).mean()
        df['Volatility_30d'] = (
            df['Close'].pct_change().rolling(window=30).std()
        )

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
# 2. 백테스트 엔진 (승률 50% 강제 필터 완화 및 기대값 중심 설계)
# ==========================================
def detect_pattern_signals(df, sma_period, adx_min=18.0):
  df_calc = df.copy()
  df_calc['SMA'] = df_calc['Close'].rolling(window=sma_period).mean()

  lows = df_calc['Low'].values
  closes = df_calc['Close'].values
  smas = df_calc['SMA'].values
  adx_vals = df_calc['ADX'].fillna(0).values

  prominence = np.mean(lows) * 0.015
  peaks_idx, _ = find_peaks(-lows, distance=5, prominence=prominence)

  signals = []
  for i in range(len(peaks_idx) - 1):
    idx1, idx2 = peaks_idx[i], peaks_idx[i + 1]
    if 5 <= (idx2 - idx1) <= 35 and abs(
        lows[idx1] - lows[idx2]
    ) / min(lows[idx1], lows[idx2]) <= 0.02:
      neckline = df_calc['High'].values[idx1 : idx2 + 1].max()
      post_df = df_calc.iloc[idx2:]
      breakout = post_df[post_df['Close'] > neckline]
      if not breakout.empty:
        b_idx = df_calc.index.get_loc(breakout.index[0])
        if adx_vals[b_idx] >= adx_min and closes[b_idx] >= smas[b_idx]:
          signals.append(b_idx)

  return sorted(list(set(signals)))


def backtest_atr_engine(
    df, sma_period, tp_atr_mult, sl_atr_mult, max_holding=15, is_short=False
):
  signals = detect_pattern_signals(df, sma_period)
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

    if not is_short:
      tp_price = entry_price + (tp_atr_mult * atr_val)
      sl_price = entry_price - (sl_atr_mult * atr_val)
    else:
      tp_price = entry_price - (tp_atr_mult * atr_val)
      sl_price = entry_price + (sl_atr_mult * atr_val)

    post_df = df.iloc[b_idx + 1 : min(b_idx + 1 + max_holding, len(df))]
    exit_price = entry_price

    for k in range(len(post_df)):
      high_p = post_df['High'].iloc[k]
      low_p = post_df['Low'].iloc[k]

      if not is_short:
        if high_p >= tp_price:
          exit_price = tp_price
          break
        elif low_p <= sl_price:
          exit_price = sl_price
          break
        else:
          exit_price = post_df['Close'].iloc[k]
      else:
        if low_p <= tp_price:
          exit_price = tp_price
          break
        elif high_p >= sl_price:
          exit_price = sl_price
          break
        else:
          exit_price = post_df['Close'].iloc[k]

    if not is_short:
      trades.append((exit_price - entry_price) / entry_price)
    else:
      trades.append((entry_price - exit_price) / entry_price)

  if len(trades) < 2:
    return None

  trades_arr = np.array(trades)
  tot_ret = np.sum(trades_arr) * 100
  win_rate = (np.sum(trades_arr > 0) / len(trades_arr)) * 100

  gains = trades_arr[trades_arr > 0]
  losses = abs(trades_arr[trades_arr < 0])
  profit_factor = (
      np.sum(gains) / np.sum(losses) if np.sum(losses) > 0 else 99.0
  )

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


def analyze_single_symbol_full(
    exchange, symbol, train_ratio=0.7, is_short=False
):
  df = fetch_ohlcv_full(exchange, symbol)
  if df.empty or len(df) < 40:
    return None

  split_idx = int(len(df) * train_ratio)
  train_df = df.iloc[:split_idx].copy()
  test_df = df.iloc[split_idx:].copy()

  if len(train_df) < 25 or len(test_df) < 15:
    return None

  best_score = -999.0
  best_params = None
  best_is_metric = None

  for sma_p in [10, 15, 20]:
    for tp_m in [2.0, 3.0, 4.0]:
      for sl_m in [1.0, 1.5, 2.0]:
        res_is = backtest_atr_engine(
            train_df, sma_p, tp_m, sl_m, is_short=is_short
        )

        # 🌟 승률 50% 강제 필터를 풀고, Profit Factor와 Sharpe 중심의 기대값 기준으로 최적화
        if res_is and res_is['return_pct'] > 0 and res_is['profit_factor'] >= 1.2:
          score = res_is['sharpe_ratio'] * 0.5 + res_is['profit_factor'] * 0.5
          if score > best_score:
            best_score = score
            best_params = (sma_p, tp_m, sl_m)
            best_is_metric = res_is

  if not best_params:
    return None

  opt_sma, opt_tp_m, opt_sl_m = best_params
  res_oos = backtest_atr_engine(
      test_df, opt_sma, opt_tp_m, opt_sl_m, is_short=is_short
  )

  if res_oos and res_oos['return_pct'] > 0:
    latest_close = df['Close'].iloc[-1]
    latest_atr = df['ATR'].iloc[-1]
    latest_adx = df['ADX'].iloc[-1]
    poc_price = df['POC_Price'].iloc[-1]
    vol_30d = df['Vol_30d_Mean'].iloc[-1]
    vola_30d = df['Volatility_30d'].iloc[-1]

    # 🌟 [신규 적용] 30일 변동성이 너무 극심한 잡코인(상위 5% 초고위험)은 페널티 부여 후 제외 가능
    if pd.isna(vola_30d) or vola_30d > 0.12:  # 일일 표준편차 12% 이상은 과도한 널뛰기로 판단
      return None

    if not is_short:
      calc_tp = latest_close + (opt_tp_m * latest_atr)
      calc_sl = latest_close - (opt_sl_m * latest_atr)
      tp_pct = ((calc_tp - latest_close) / latest_close) * 100
      sl_pct = ((latest_close - calc_sl) / latest_close) * 100
    else:
      calc_tp = latest_close - (opt_tp_m * latest_atr)
      calc_sl = latest_close + (opt_sl_m * latest_atr)
      tp_pct = ((latest_close - calc_tp) / latest_close) * 100
      sl_pct = ((calc_sl - latest_close) / latest_close) * 100

    # 🌟 최종 순위 스코어 계산 시 30일 평균 거래대금 가중치 반영
    liquidity_score = np.log10(vol_30d + 1) if not pd.isna(vol_30d) else 1.0
    final_ranking_score = (
        res_oos['sharpe_ratio'] * 0.4
        + res_oos['profit_factor'] * 0.3
        + (liquidity_score * 0.3)
    )

    return {
        'symbol': symbol,
        'current_price': latest_close,
        'poc_price': round(poc_price, 4),
        'adx': round(latest_adx, 1),
        'opt_sma': opt_sma,
        'opt_tp_m': opt_tp_m,
        'opt_sl_m': opt_sl_m,
        'is_return': best_is_metric['return_pct'],
        'is_win': best_is_metric['win_rate'],
        'oos_return': res_oos['return_pct'],
        'oos_win': res_oos['win_rate'],
        'sharpe_ratio': res_oos['sharpe_ratio'],
        'profit_factor': res_oos['profit_factor'],
        'final_score': round(final_ranking_score, 2),
        'tp_price': round(calc_tp, 4),
        'sl_price': round(calc_sl, 4),
        'tp_pct': round(tp_pct, 2),
        'sl_pct': round(sl_pct, 2),
    }
  return None


# ==========================================
# 3. 멀티스레드 병렬 탐색기[cite: 3]
# ==========================================
def run_pipeline_parallel(exchange, target_symbols, is_short=False):
  results = []
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    future_map = {
        executor.submit(
            analyze_single_symbol_full, exchange, sym, 0.7, is_short
        ): sym
        for sym in target_symbols
    }
    for future in concurrent.futures.as_completed(future_map):
      res = future.result()
      if res:
        results.append(res)

  if results:
    # 🌟 30일 거래대금과 안정성이 반영된 final_score 기준으로 정렬
    return pd.DataFrame(results).sort_values(
        by='final_score', ascending=False
    )
  return pd.DataFrame()


# ==========================================
# 4. Streamlit UI 메인 화면[cite: 3]
# ==========================================
st.title("🔥 추천 대시보드 (1달 유동성 & 변동성 제어 v2)")
st.caption(
    "30일간의 누적 거래대금과 가격 변동성 안정성 지수를 반영하여 노이즈 코인을"
    " 걸러냅니다."
)

render_advanced_macro_weather_panel()

df_all, exchange = load_market_data()

if not df_all.empty and exchange is not None:
  majors_df = df_all[df_all['base'].isin(TOP_MAJORS)]
  # 🌟 24시간 변동률 기준이 아닌 24시간 '거래대금(quote_vol)' 기준으로 상위 코인 추출
  others_df = (
      df_all[~df_all['base'].isin(TOP_MAJORS)]
      .sort_values(by='quote_vol', ascending=False)
      .head(25)
  )
  target_df = (
      pd.concat([majors_df, others_df])
      .drop_duplicates(subset=['symbol'])
      .reset_index(drop=True)
  )

  all_symbols = target_df['symbol'].tolist()

  st.markdown('---')
  if st.button(
      '🚀 AI 스마트 추천 분석 실행 (1달 유동성 필터 적용)',
      use_container_width=True,
      type='primary',
  ):
    with st.spinner(
        '⏳ 30일간의 수급, 변동성, WFO 검증 엔진 병렬 가동 중...'
    ):
      st.session_state['long_res'] = run_pipeline_parallel(
          exchange, all_symbols, is_short=False
      )
      st.session_state['short_res'] = run_pipeline_parallel(
          exchange, all_symbols, is_short=True
      )
      st.session_state['analyzed'] = True
    st.success('✨ 분석이 완료되었습니다!')
  st.markdown('---')

  col_long, col_short = st.columns(2)

  with col_long:
    st.markdown('### 📈 롱 (Long) 추천 종목')
    if not st.session_state.get('analyzed', False):
      st.info('상단의 분석 실행 버튼을 눌러주세요.')
    else:
      res_long = st.session_state.get('long_res', pd.DataFrame())
      if not res_long.empty:
        st.success(f'총 {len(res_long)}개 우량 롱 후보 도출')
        for _, item in res_long.iterrows():
          with st.expander(
              f"🟢 **{item['symbol']}** | 스코어: `{item['final_score']}` | WFO:"
              f" `+{item['oos_return']}%` (승률 {item['oos_win']}%)",
              expanded=False,
          ):
            st.markdown(f"""
                        * **현재가:** `${item['current_price']:,.4f}`
                        * **최적 SMA / ADX:** `{item['opt_sma']}일` / `{item['adx']}`
                        * **WFO 검증 수익률:** `+{item['oos_return']}%` (승률 {item['oos_win']}%)
                        * **샤프 / 팩터:** `{item['sharpe_ratio']}` / `{item['profit_factor']}`
                        * **매물대 (POC):** `${item['poc_price']:,.4f}`
                        * **목표가 (TP):** `${item['tp_price']:,.4f}` (+{item['tp_pct']}%)
                        * **손절가 (SL):** `${item['sl_price']:,.4f}` (-{item['sl_pct']}%)
                        """)
      else:
        st.warning(
            '조건에 맞는 롱 종목이 없습니다 (변동성이 과도한 코인 필터링됨).'
        )

  with col_short:
    st.markdown('### 📉 숏 (Short) 추천 종목')
    if not st.session_state.get('analyzed', False):
      st.info('상단의 분석 실행 버튼을 눌러주세요.')
    else:
      res_short = st.session_state.get('short_res', pd.DataFrame())
      if not res_short.empty:
        st.error(f'총 {len(res_short)}개 우량 숏 후보 도출')
        for _, item in res_short.iterrows():
          with st.expander(
              f"🔴 **{item['symbol']}** | 스코어: `{item['final_score']}` | WFO:"
              f" `+{item['oos_return']}%` (승률 {item['oos_win']}%)",
              expanded=False,
          ):
            st.markdown(f"""
                        * **현재가:** `${item['current_price']:,.4f}`
                        * **최적 SMA / ADX:** `{item['opt_sma']}일` / `{item['adx']}`
                        * **WFO 검증 수익률:** `+{item['oos_return']}%` (승률 {item['oos_win']}%)
                        * **샤프 / 팩터:** `{item['sharpe_ratio']}` / `{item['profit_factor']}`
                        * **매물대 (POC):** `${item['poc_price']:,.4f}`
                        * **목표가 (TP):** `${item['tp_price']:,.4f}` (-{item['tp_pct']}%)
                        * **손절가 (SL):** `${item['sl_price']:,.4f}` (+{item['sl_pct']}%)
                        """)
      else:
        st.warning('조건에 맞는 숏 종목이 없습니다.')