import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.db_init import init_db
from src.data_fetcher import load_clean_continuous_series
from src.matcher import CurveMatcher
from src.predictor import TrendPredictor, detect_market_regime
from src.config import WINDOW_SIZE, FUTURE_STEPS

def run_historical_backtest(test_samples=50):
    """
    事件驱动型历史连续滚动回测引擎：
    评估模型在历史行情切片上的点预测精度 (MAPE)、涨跌方向胜率 (Win Rate) 及夏普比率 (Sharpe Ratio)。
    """
    print("=" * 60)
    print(" 黄金价格预测模型 — 历史滚动回测评估引擎 (Backtest Engine)")
    print("=" * 60)

    init_db()
    clean_df = load_clean_continuous_series()
    if clean_df is None or len(clean_df) < WINDOW_SIZE + FUTURE_STEPS + 100:
        print("[错误] 数据库中行情点数不足，无法执行有效回测！")
        return

    prices = clean_df['close'].values
    timestamps = clean_df['timestamp'].values
    total_len = len(prices)

    matcher = CurveMatcher()
    predictor = TrendPredictor()

    mapes = []
    direction_hits = 0
    total_signals = 0
    strategy_returns = []

    # 随机抽样 test_samples 个滑动历史切片做滚动测试
    sample_indices = np.linspace(WINDOW_SIZE, total_len - FUTURE_STEPS - 1, test_samples, dtype=int)

    t0 = time.time()
    for step_count, idx in enumerate(sample_indices):
        hist_prices = prices[:idx]
        hist_ts = timestamps[:idx]
        actual_future = prices[idx : idx + FUTURE_STEPS]

        # 1. 执行形态匹配
        H_matrix, valid_indices, weights = matcher.build_history_matrix(hist_ts, hist_prices, current_N=WINDOW_SIZE)
        current_query = hist_prices[-WINDOW_SIZE:]
        top_matches = matcher.find_top_matches(current_query, H_matrix, valid_indices, weights, top_k=3)

        # 2. 生成预测
        pred_res = predictor.generate_prediction(
            current_last_price=hist_prices[-1],
            top_matches=top_matches,
            prices=hist_prices,
            timestamps=hist_ts,
            current_N=WINDOW_SIZE
        )

        if not pred_res or 'predicted_prices' not in pred_res or pred_res['predicted_prices'] is None:
            continue

        pred_p = np.array(pred_res['predicted_prices'])
        if len(pred_p) != len(actual_future):
            continue

        # 3. 计算 MAPE
        mape = np.mean(np.abs(pred_p - actual_future) / actual_future) * 100.0
        mapes.append(mape)

        # 4. 计算方向命中率
        pred_dir = pred_p[-1] - hist_prices[-1]
        actual_dir = actual_future[-1] - hist_prices[-1]
        if abs(pred_dir) > 0.05:  # 只有信号幅度大于 0.05 元时计入信号
            total_signals += 1
            if (pred_dir > 0 and actual_dir > 0) or (pred_dir < 0 and actual_dir < 0):
                direction_hits += 1

            # 模拟简单策略单步收益率
            sig_sign = np.sign(pred_dir)
            step_ret = sig_sign * (actual_future[-1] - hist_prices[-1]) / hist_prices[-1]
            strategy_returns.append(step_ret)

    elapsed = time.time() - t0
    avg_mape = np.mean(mapes) if mapes else 0.0
    win_rate = (direction_hits / total_signals * 100.0) if total_signals > 0 else 0.0
    
    ret_arr = np.array(strategy_returns)
    sharpe = (np.mean(ret_arr) / (np.std(ret_arr) + 1e-8)) * np.sqrt(252 * 1440) if len(ret_arr) > 1 else 0.0

    print(f"回测样本数: {len(mapes)} 个切片")
    print(f"平均点预测 MAPE: {avg_mape:.4f} %")
    print(f"有效交易信号数: {total_signals} 次, 信号胜率: {win_rate:.2f} %")
    print(f"策略夏普比率 (Sharpe Ratio): {sharpe:.2f}")
    print(f"评估耗时: {elapsed:.2f} 秒")
    print("=" * 60)

if __name__ == "__main__":
    run_historical_backtest(test_samples=30)
