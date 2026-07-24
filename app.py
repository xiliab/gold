import os
import time
import threading
import atexit
import logging
import sqlite3
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from src.utils import get_trading_date

from src.db_init import init_db, DB_PATH, get_db_connection
from src.data_fetcher import (
    fetch_real_gold_1m_data, save_prices_to_db, load_clean_continuous_series,
    get_cached_continuous_series, clear_mock_data, fetch_spdr_holdings_bias,
    calculate_volume_imbalance
)
from src.matcher import CurveMatcher
from src.predictor import TrendPredictor
from src.corrector import AdaptiveFeedbackCorrector
from src.charts import format_chart_payload
from src.tiny_model import extract_features as _extract_features
from src.config import (
    WINDOW_SIZE, FUTURE_STEPS, MIN_POINTS_REQUIRED, EARLY_SESSION_MIN_POINTS,
    CLOSE_HOUR, TRADING_DAY_START_OFFSET_HOURS, CANDLE_INTERVAL_MINUTES,
    FETCH_INTERVAL_SECONDS, NMS_RADIUS
)

import json
from flask.json.provider import DefaultJSONProvider

class NumpyJSONProvider(DefaultJSONProvider):
    """扩展 Flask 默认 JSON provider，支持 numpy 数值/布尔/数组类型序列化"""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app = Flask(__name__)
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)
from src.data_fetcher import get_realtime_usd_cny_rate

# 静态资源缓存版本号：使用服务启动时间戳，重启后强制刷新，运行期间不变
APP_START_TIME = int(time.time())
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# ─── 分步启动计时器与服务初始化 ───────────────────────
t_start_total = time.time()
print("=" * 60)
print(" 🚀 正在启动 黄金智能走势预测与波段风控系统...")
print("=" * 60)

# [步骤 1/4] 初始化数据库并清理非 1m 行情
t0 = time.time()
init_db()
clear_mock_data()
t_step1 = (time.time() - t0) * 1000.0
print(f"⏱️  [步骤 1/4] 数据库结构初始化与纯净清洗完成！ 耗时: {t_step1:.2f} ms")

# [步骤 2/4] 多源权威外汇实时汇率校准（非阻塞热缓存极速加载）
t0 = time.time()
cny_rate = get_realtime_usd_cny_rate(force_refresh=False)
t_step2 = (time.time() - t0) * 1000.0
print(f"⏱️  [步骤 2/4] 动态权威外汇 (USD/CNY={cny_rate:.4f}) 极速就绪！ 耗时: {t_step2:.2f} ms")

# [步骤 3/4] 5 年历史 5m 极值保留趋势骨架按需懒加载 (Lazy Loading 0 冗余)
t0 = time.time()
from src.data_fetcher import load_skeleton_continuous_series_5m
df_5m = load_skeleton_continuous_series_5m()
t_step3 = (time.time() - t0) * 1000.0
skel_count = len(df_5m) if df_5m is not None else 0
print(f"⏱️  [步骤 3/4] 5 年历史 5m 极值保留趋势骨架 ({skel_count} 条) 懒加载完成！ 耗时: {t_step3:.2f} ms")

# [步骤 4/4] 匹配算法与神经网络预测引擎就绪
t0 = time.time()
matcher   = CurveMatcher(nms_radius=NMS_RADIUS)
predictor = TrendPredictor(future_steps=FUTURE_STEPS)
corrector = AdaptiveFeedbackCorrector()
t_step4 = (time.time() - t0) * 1000.0
print(f"⏱️  [步骤 4/4] 宏观趋势匹配与神经网络预测引擎初始化完成！ 耗时: {t_step4:.2f} ms")

t_total = (time.time() - t_start_total) * 1000.0
print("-" * 60)
print(f"🎉 所有引擎服务全效就绪！系统总启动耗时: {t_total:.2f} ms")
print("=" * 60)

_ready = False
_fetch_lock = threading.Lock()
_prediction_cache = None
_prediction_cache_lock = threading.Lock()

def compute_prediction_payload(force_reload_data=False):
    """从 RAM 常驻内存提取 DataFrame，计算预测并返回零延迟 Payload。"""
    df = get_cached_continuous_series(force_reload=force_reload_data)
    if df is None or len(df) < MIN_POINTS_REQUIRED:
        return None

    prices = df['close'].values
    timestamps = df['timestamp'].values
    is_gap = df['is_gap'].values

    # 【日内锚定匹配】：计算今天（06:00 以来）走出的点数 N
    dt_series    = pd.to_datetime(timestamps)
    trading_dates = get_trading_date(dt_series)
    last_date     = trading_dates[-1]
    today_points  = (trading_dates == last_date).sum()

    # 早盘容错：不足 EARLY_SESSION_MIN_POINTS 时借用昨晚数据凑齐，避免噪音
    current_N = max(EARLY_SESSION_MIN_POINTS, int(today_points))

    # 1. 精确抽取当前这 current_N 个点
    current_query = prices[-current_N:]
    current_last_price = current_query[-1]
    current_timestamp = timestamps[-1]

    # 2.1 计算机构行为影响因子 (SPDR 黄金 ETF 持仓偏向 + 日内大单量能偏差)
    spdr_bias = fetch_spdr_holdings_bias()
    vol_imbalance = calculate_volume_imbalance(df)
    inst_factor = float(np.clip(0.6 * spdr_bias + 0.4 * vol_imbalance, -1.0, 1.0))

    # 【动态闭环步骤 1】：用最新实测值比对未验证的历史预测，更新胜率、防御系数及极小模型在线训练
    win_rate, dynamic_k = corrector.evaluate_pending_predictions(
        current_timestamp, current_last_price,
        tiny_model=predictor.tiny_model, df=df, inst_factor=inst_factor
    )

    # 2. 按交易日构建历史全量矩阵并提取 Top 3 最相似交易天 (利用 CurveMatcher 内存矩阵缓存)
    H_matrix, valid_indices, weights = matcher.build_history_matrix(timestamps, prices, current_N)
    top_matches = matcher.find_top_matches(current_query, H_matrix, valid_indices, weights, top_k=3)

    # 3. 相对收益率复利还原、极小模型双模型集成与置信区间生成
    pred_result = predictor.generate_prediction(
        current_last_price=current_last_price,
        top_matches=top_matches,
        prices=prices,
        timestamps=timestamps,
        current_N=current_N,
        dynamic_k=dynamic_k,
        win_rate=win_rate,
        inst_factor=inst_factor,
        df=df,
        step_bias_ema=corrector.step_bias_ema
    )

    # 获取当前极值残差 EMA 偏置
    high_bias_ema, low_bias_ema = corrector.get_extrema_biases()

    # 预测今日剩余时间 (严格限制在今日 23:59:59 之前，不超出今天跨到次日)
    now       = datetime.now()
    today_end = now.replace(hour=23, minute=59, second=59, microsecond=0)
    steps_left = max(FUTURE_STEPS, int((today_end - now).total_seconds() / (CANDLE_INTERVAL_MINUTES * 60)))

    long_pred_result = predictor.generate_prediction(
        current_last_price=current_last_price,
        top_matches=top_matches,
        prices=prices,
        timestamps=timestamps,
        current_N=current_N,
        dynamic_k=dynamic_k,
        win_rate=win_rate,
        high_bias_ema=high_bias_ema,
        low_bias_ema=low_bias_ema,
        inst_factor=inst_factor,
        df=df,
        override_future_steps=steps_left
    )

    rest_of_day_low = None
    rest_of_day_high = None
    rest_of_day_high_time = '待计算'
    rest_of_day_low_time  = '待计算'
    if long_pred_result:
        rest_of_day_low       = long_pred_result.get('rest_of_day_low')
        rest_of_day_high      = long_pred_result.get('rest_of_day_high')
        rest_of_day_high_time = long_pred_result.get('rest_of_day_high_time', '待计算')
        rest_of_day_low_time  = long_pred_result.get('rest_of_day_low_time', '待计算')

    # 更新极值自学反馈
    corrector.update_extrema_feedback(current_last_price, rest_of_day_high, rest_of_day_low)

    # 4. 一次性查询今日历史预测价格与波段建议用于图表对比色带
    historical_pred_map = {}
    historical_advice_map = {}
    try:
        with get_db_connection(DB_PATH) as _conn:
            _c = _conn.cursor()
            _c.execute(
                "SELECT created_at, target_time, predicted_price, advice_type FROM prediction_history "
                "WHERE created_at >= ? OR target_time >= ? ORDER BY created_at ASC",
                (str(timestamps[max(0, -current_N)]), str(timestamps[max(0, -current_N)]))
            )
            for row in _c.fetchall():
                c_at, t_at, p_val, a_type = row[0], row[1], row[2], row[3]
                if t_at and p_val is not None:
                    historical_pred_map[t_at] = p_val
                if c_at and a_type:
                    historical_advice_map[c_at] = a_type
                if t_at and a_type:
                    historical_advice_map[t_at] = a_type
    except Exception:
        pass

    # 格式化图表数据 Payload (按真实 1m 分钟数 real_window_size 渲染，防止前置 700 点被腰斩)
    real_window_size = current_N * 5 if current_N < 300 else current_N
    chart_payload = format_chart_payload(df, pred_result, window_size=real_window_size, historical_pred_map=historical_pred_map, historical_advice_map=historical_advice_map)

    # 记录预测轨迹
    if chart_payload and 'future_full_timestamps' in chart_payload and pred_result and pred_result.get('predicted_prices'):
        future_times = chart_payload['future_full_timestamps']
        if len(future_times) >= 6 and len(pred_result['predicted_prices']) >= 6:
            full_target_time = future_times[-1]
            try:
                current_feat_snap = _extract_features(df, inst_factor) if df is not None and len(df) >= 20 else None
            except Exception:
                current_feat_snap = None
            corrector.record_prediction(
                target_time=full_target_time,
                base_price=current_last_price,
                predicted_price=pred_result['predicted_prices'][-1],
                lower_bound=pred_result['lower_bound'][-1],
                upper_bound=pred_result['upper_bound'][-1],
                action_advice=pred_result.get('action_advice'),
                advice_type=pred_result.get('advice_type'),
                created_at=current_timestamp,
                feature_snapshot=current_feat_snap
            )

    # 5. 校验当前残差状态
    steps = predictor.future_steps
    recent_actual = prices[-steps - 1:-1].tolist() if len(prices) > steps + 1 else []
    pred_prices_list = pred_result.get('predicted_prices') if pred_result else None
    last_pred = pred_prices_list[:len(recent_actual)] if pred_prices_list is not None else []
    need_rematch, status_msg, current_mape, last_error, avg_price_error = corrector.check_mape(recent_actual, last_pred)

    bias_msg = (f"上轮误差 {last_error:+.2f}元，均值偏差 {avg_price_error:+.2f}元"
                if abs(last_error) > 0.001 else "实时反馈校正引擎正常")

    return {
        "status": "success",
        "chart_data": chart_payload,
        "metrics": {
            "current_price": round(float(current_last_price), 2),
            "mape": round(float(current_mape * 100), 2),
            "status_msg": f"{status_msg} ({bias_msg})",
            "need_rematch": bool(need_rematch),
            "last_error": round(float(last_error), 2),
            "current_bias": round(float(avg_price_error), 2),
            "valid_count": int(pred_result.get('valid_count', 0)) if pred_result else 0,
            "direction_confidence": float(pred_result.get('direction_confidence', 0.0)) if pred_result else 0.0,
            "direction_label": pred_result.get('direction_label') if pred_result else "分析中",
            "action_advice": pred_result.get('action_advice') if pred_result else "模型初始化中",
            "advice_type": pred_result.get('advice_type') if pred_result else "NEUTRAL",
            "advice": pred_result.get('advice') if pred_result else None,
            "win_rate": float(win_rate),
            "dynamic_k": float(dynamic_k),
            "ai_logs": corrector.get_logs(),
            "up_votes": int(pred_result.get('up_votes', 0)) if pred_result else 0,
            "down_votes": int(pred_result.get('down_votes', 0)) if pred_result else 0,
            "flat_votes": int(pred_result.get('flat_votes', 0)) if pred_result else 0,
            "low_confidence_match": bool(pred_result.get('low_confidence_match', False)) if pred_result else False,
            "rest_of_day_low": round(rest_of_day_low, 2) if rest_of_day_low is not None else round(float(current_last_price), 2),
            "rest_of_day_high": round(rest_of_day_high, 2) if rest_of_day_high is not None else round(float(current_last_price), 2),
            "rest_of_day_high_time": rest_of_day_high_time,
            "rest_of_day_low_time": rest_of_day_low_time,
            "high_bias_ema": round(float(high_bias_ema), 2),
            "low_bias_ema": round(float(low_bias_ema), 2),
            "tiny_status": pred_result.get('tiny_status') if pred_result else None,
            "regime": pred_result.get('regime', 'RANGING') if pred_result else 'RANGING',
            "regime_score": float(pred_result.get('regime_score', 0.5)) if pred_result else 0.5
        }
    }


def _background_fetch_loop():
    """每 FETCH_INTERVAL_SECONDS 秒在后台静默拉取最新 1m 行情，更新 RAM 矩阵并预先算好推断结果，零延迟响应。"""
    global _ready, _prediction_cache
    while True:
        try:
            real_df = fetch_real_gold_1m_data()
            if real_df is not None and not real_df.empty:
                with _fetch_lock:
                    save_prices_to_db(real_df)
                
                # 预计算零延迟 Payload
                payload = compute_prediction_payload(force_reload_data=True)
                if payload:
                    with _prediction_cache_lock:
                        _prediction_cache = payload
                    if not _ready:
                        _ready = True
        except Exception as e:
            logging.warning(f"[后台抓取] 行情更新异常: {e}")
        time.sleep(FETCH_INTERVAL_SECONDS)

# 首次异步启动拉取，不再阻塞主线程
_fetch_thread = threading.Thread(target=_background_fetch_loop, daemon=True)
_fetch_thread.start()

def cleanup_resources():
    logging.info("Shutting down... cleaning up resources.")
atexit.register(cleanup_resources)

@app.route("/")
def index():
    return render_template("index.html", cache_bust=APP_START_TIME)

@app.route("/api/gold/predict")
def get_prediction_data():
    global _prediction_cache
    # 实时调用 compute_prediction_payload 获取绝对最新算出的图形 Payload
    payload = compute_prediction_payload(force_reload_data=False)
    if payload is not None:
        with _prediction_cache_lock:
            _prediction_cache = payload
        return jsonify(payload)

    with _prediction_cache_lock:
        if _prediction_cache is not None:
            return jsonify(_prediction_cache)

    return jsonify({"status": "error", "message": "历史数据准备中..."}), 503

@app.route("/api/gold/refresh", methods=["POST"])
def refresh_data():
    """手动触发一次立即行情拉取并落库。"""
    real_df = fetch_real_gold_1m_data()
    if real_df is not None and not real_df.empty:
        with _fetch_lock:
            save_prices_to_db(real_df)
    return jsonify({"status": "success", "message": "纯净 1m 真实行情获取与刷新完成"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)
