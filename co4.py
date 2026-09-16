import concurrent.futures
import ccxt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import streamlit as st
import ta

st.set_page_config(
    page_title="🔥 크립토 AI 알고리즘 추천 대시보드",
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
# 1. 시세 데이터 및 거래소 API 로드 (Fallback)
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


def fetch_ohlcv_full(_exchange, symbol, timeframe='1d', limit=150):
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

        df['ATR'] = ta.volatility.average_true_range(
            df['High'], df['Low'], df['Close'], window=14
        )
        adx_ind = ta.trend.ADXIndicator(
            df['High'], df['Low'], df['Close'], window=14
        )
        df['ADX'] = adx_ind.adx()

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
# 2-1. 기존 탭 1용 패턴 탐지 엔진
# ==========================================
def detect_pattern_signals(df, sma_period, adx_min=20.0):
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
    df, sma_period, tp_atr_mult, sl_atr_mult, max_holding=15
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

    tp_price = entry_price + (tp_atr_mult * atr_val)
    sl_price = entry_price - (sl_atr_mult * atr_val)

    post_df = df.iloc[b_idx + 1 : min(b_idx + 1 + max_holding, len(df))]
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


def analyze_single_symbol_full(exchange, symbol, train_ratio=0.7):
  df = fetch_ohlcv_full(exchange, symbol)
  if df.empty or len(df) < 60:
    return None

  split_idx = int(len(df) * train_ratio)
  train_df = df.iloc[:split_idx].copy()
  test_df = df.iloc[split_idx:].copy()

  if len(train_df) < 40 or len(test_df) < 20:
    return None

  best_score = -999.0
  best_params = None
  best_is_metric = None

  for sma_p in [10, 15, 20]:
    for tp_m in [2.0, 3.0, 4.0]:
      for sl_m in [1.0, 1.5, 2.0]:
        res_is = backtest_atr_engine(train_df, sma_p, tp_m, sl_m)
        if res_is and res_is['return_pct'] > 0 and res_is['win_rate'] >= 50.0:
          score = res_is['sharpe_ratio'] * 0.6 + res_is['profit_factor'] * 0.4
          if score > best_score:
            best_score = score
            best_params = (sma_p, tp_m, sl_m)
            best_is_metric = res_is

  if not best_params:
    return None

  opt_sma, opt_tp_m, opt_sl_m = best_params
  res_oos = backtest_atr_engine(test_df, opt_sma, opt_tp_m, opt_sl_m)

  if res_oos and res_oos['return_pct'] > 0 and res_oos['win_rate'] >= 50.0:
    latest_close = df['Close'].iloc[-1]
    latest_atr = df['ATR'].iloc[-1]
    latest_adx = df['ADX'].iloc[-1]
    poc_price = df['POC_Price'].iloc[-1]

    calc_tp = latest_close + (opt_tp_m * latest_atr)
    calc_sl = latest_close - (opt_sl_m * latest_atr)

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
        'tp_price': round(calc_tp, 4),
        'sl_price': round(calc_sl, 4),
        'tp_pct': round(((calc_tp - latest_close) / latest_close) * 100, 2),
        'sl_pct': round(((latest_close - calc_sl) / latest_close) * 100, 2),
    }
  return None


# ==========================================
# 2-2. 탭 2, 3용 전면 고도화된 모멘텀·POC 지지 엔진
# ==========================================
def detect_momentum_signals(df, sma_period, adx_min=12.0):
  df_calc = df.copy()
  df_calc['SMA'] = df_calc['Close'].rolling(window=sma_period).mean()

  signals = []
  closes = df_calc['Close'].values
  smas = df_calc['SMA'].values
  adx_vals = df_calc['ADX'].fillna(0).values
  pocs = df_calc['POC_Price'].values

  for i in range(20, len(df_calc)):
    is_above_sma = closes[i] >= smas[i] and closes[i - 1] < smas[i - 1]
    is_near_poc = abs(closes[i] - pocs[i]) / pocs[i] <= 0.05
    has_trend = adx_vals[i] >= adx_min

    if (is_above_sma or is_near_poc) and has_trend:
      signals.append(i)

  return sorted(list(set(signals)))


def backtest_momentum_engine(
    df, sma_period, tp_atr_mult, sl_atr_mult, max_holding=15
):
  signals = detect_momentum_signals(df, sma_period)
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

    post_df = df.iloc[b_idx + 1 : min(b_idx + 1 + max_holding, len(df))]
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

  if len(trades) < 1:
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


def analyze_single_symbol_momentum(exchange, symbol, train_ratio=0.7):
  df = fetch_ohlcv_full(exchange, symbol)
  if df.empty or len(df) < 60:
    return None

  split_idx = int(len(df) * train_ratio)
  train_df = df.iloc[:split_idx].copy()
  test_df = df.iloc[split_idx:].copy()

  if len(train_df) < 40 or len(test_df) < 20:
    return None

  best_score = -999.0
  best_params = None
  best_is_metric = None

  for sma_p in [10, 15, 20]:
    for tp_m in [2.0, 3.0, 4.0]:
      for sl_m in [1.0, 1.5, 2.0]:
        res_is = backtest_momentum_engine(train_df, sma_p, tp_m, sl_m)
        if res_is and res_is['return_pct'] >= 0:
          score = res_is['sharpe_ratio'] * 0.6 + res_is['profit_factor'] * 0.4
          if score > best_score:
            best_score = score
            best_params = (sma_p, tp_m, sl_m)
            best_is_metric = res_is

  if not best_params:
    return None

  opt_sma, opt_tp_m, opt_sl_m = best_params
  res_oos = backtest_momentum_engine(test_df, opt_sma, opt_tp_m, opt_sl_m)

  if res_oos and res_oos['return_pct'] >= -2.0:
    latest_close = df['Close'].iloc[-1]
    latest_atr = df['ATR'].iloc[-1]
    latest_adx = df['ADX'].iloc[-1]
    poc_price = df['POC_Price'].iloc[-1]

    calc_tp = latest_close + (opt_tp_m * latest_atr)
    calc_sl = latest_close - (opt_sl_m * latest_atr)

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
        'tp_price': round(calc_tp, 4),
        'sl_price': round(calc_sl, 4),
        'tp_pct': round(((calc_tp - latest_close) / latest_close) * 100, 2),
        'sl_pct': round(((latest_close - calc_sl) / latest_close) * 100, 2),
    }

  return None


# ==========================================
# 3. 멀티스레드 병렬 탐색기들
# ==========================================
def run_pipeline_parallel(exchange, target_symbols, is_momentum=False):
  results = []
  completed = 0
  total = len(target_symbols)

  analyzer_func = (
      analyze_single_symbol_momentum
      if is_momentum
      else analyze_single_symbol_full
  )

  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    future_map = {
        executor.submit(analyzer_func, exchange, sym): sym
        for sym in target_symbols
    }
    for future in concurrent.futures.as_completed(future_map):
      completed += 1
      res = future.result()
      if res:
        results.append(res)

  if results:
    return pd.DataFrame(results).sort_values(
        by='sharpe_ratio', ascending=False
    )
  return pd.DataFrame()


# ==========================================
# 4. Streamlit UI 메인 화면
# ==========================================
st.title("🔥 크립토 AI 알고리즘 추천 대시보드")
st.caption(
    "메인 상단의 **[🚀 전체 통합 분석 탐색 실행]** 버튼 하나로 모든 탭의"
    " 정밀 분석이 동시에 수행됩니다."
)

col_top1, col_top2 = st.columns([6, 4])
with col_top1:
  run_all_btn = st.button(
      "🚀 전체 통합 분석 탐색 실행", type="primary", use_container_width=True
  )
with col_top2:
  if st.button("🔄 데이터 새로고침", use_container_width=True):
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

  all_symbols = target_df['symbol'].tolist()
  major_symbols = (
      target_df[target_df['base'].isin(TOP_MAJORS)]['symbol'].tolist()
  )

  # 상단 통합 버튼이 눌렸거나 세션에 결과가 없을 때 일괄 실행
  if run_all_btn:
    with st.spinner(
        "⚡ 멀티스레드 통합 엔진 가동 중... (모든 탭 분석 일괄 처리)"
    ):
      st.session_state['res_tab1'] = run_pipeline_parallel(
          exchange, all_symbols, is_momentum=False
      )
      st.session_state['res_tab2'] = run_pipeline_parallel(
          exchange, major_symbols, is_momentum=True
      )
      st.session_state['res_tab3'] = run_pipeline_parallel(
          exchange, all_symbols, is_momentum=True
      )
    st.success("🎉 모든 탭의 정밀 분석이 성공적으로 완료되었습니다!")

  tab_full, tab_major, tab_all = st.tabs([
      "🎯 통합 AI 추천 (기본 검증)",
      "👑 메이저 정밀분석 추천 (고도화 엔진)",
      "🤖 전체 코인 정밀분석 추천 (고도화 엔진)",
  ])


  def render_results_view(key_name, section_title):
    st.subheader(section_title)
    if key_name not in st.session_state or st.session_state[key_name] is None:
      st.info(
          "상단의 **[🚀 전체 통합 분석 탐색 실행]** 버튼을 눌러 분석을"
          " 시작하세요."
      )
      return

    res_df = st.session_state[key_name]
    if not res_df.empty:
      st.success(f"🎉 총 {len(res_df)}개 추천 종목이 도출되었습니다.")
      for _, item in res_df.iterrows():
        with st.expander(
            f"🟢 **{item['symbol']}** (현재가: ${item['current_price']:,.4f}) -"
            f" 검증 수익률: +{item['oos_return']}% | Sharpe:"
            f" {item['sharpe_ratio']}",
            expanded=True,
        ):
          col1, col2 = st.columns(2)
          with col1:
            st.markdown(f"""
                        * **최적 이평선(SMA) / ADX:** `{item['opt_sma']}일` / `{item['adx']}`
                        * **학습 구간(In-Sample):** `+{item['is_return']}%` (승률 {item['is_win']}%)
                        * **WFO 검증(Out-of-Sample):** `+{item['oos_return']}%` (승률 {item['oos_win']}%)
                        * **Sharpe / Profit Factor:** `{item['sharpe_ratio']}` / `{item['profit_factor']}`
                        """)
          with col2:
            st.markdown(f"""
                        * **Volume POC (핵심 매물대):** `${item['poc_price']:,.4f}`
                        * **ATR 동적 목표가 (TP):** `${item['tp_price']:,.4f}` (+{item['tp_pct']}%, ATR {item['opt_tp_m']}배)
                        * **ATR 동적 손절가 (SL):** `${item['sl_price']:,.4f}` (-{item['sl_pct']}%, ATR {item['opt_sl_m']}배)
                        """)
    else:
      st.warning("조건에 부합하는 종목을 찾지 못했습니다.")


  with tab_full:
    render_results_view('res_tab1', "전체 대상 통합 AI 추천")

  with tab_major:
    render_results_view('res_tab2', "메이저 코인 정밀분석 추천")

  with tab_all:
    render_results_view('res_tab3', "전체 코인 정밀분석 추천")