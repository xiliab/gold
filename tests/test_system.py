import sys
import os
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.db_init import init_db
from src.data_fetcher import fetch_real_gold_1m_data, save_prices_to_db, load_clean_continuous_series
from src.matcher import CurveMatcher
from src.predictor import TrendPredictor
from src.corrector import AdaptiveFeedbackCorrector
from src.charts import format_chart_payload

class TestGoldPredictorSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        df = fetch_real_gold_1m_data()
        if df is not None and not df.empty:
            save_prices_to_db(df)

    def test_01_data_loading(self):
        clean_df = load_clean_continuous_series()
        self.assertIsNotNone(clean_df)
        self.assertGreater(len(clean_df), 1000, "加载行情数据量应大于 1000 条")

    def test_02_matcher_performance_and_nms(self):
        clean_df = load_clean_continuous_series()
        prices = clean_df['close'].values
        timestamps = clean_df['timestamp'].values
        is_gap = clean_df['is_gap'].values

        matcher = CurveMatcher(nms_radius=36, min_similarity=70.0)
        
        # 预建矩阵 (第一次)
        H_matrix, valid_indices, weights = matcher.build_history_matrix(timestamps, prices, current_N=144)
        current_query = prices[-144:]
        
        # 测量缓存命中后的矩阵检索速度 (第二次)
        t0 = time.time()
        top_matches = matcher.find_top_matches(current_query, H_matrix, valid_indices, weights, top_k=3)
        cost_ms = (time.time() - t0) * 1000.0

        print(f"\n[性能测试] 20万点矩阵检索耗时: {cost_ms:.2f} ms，符合 ≥70% 门槛数: {len(top_matches)}")
        self.assertLess(cost_ms, 1500.0, "51万点大矩阵比对计算响应耗时应低于 1.5 秒")
        for item in top_matches:
            self.assertGreaterThanOrEqual(item['similarity_pct'], 70.0)

    def assertGreaterThanOrEqual(self, a, b):
        self.assertTrue(a >= b, f"{a} 应大于等于 {b}")

    def test_03_prediction_generation(self):
        clean_df = load_clean_continuous_series()
        prices = clean_df['close'].values
        timestamps = clean_df['timestamp'].values
        is_gap = clean_df['is_gap'].values

        matcher = CurveMatcher(nms_radius=36, min_similarity=70.0)
        predictor = TrendPredictor()

        H_matrix, valid_indices, weights = matcher.build_history_matrix(timestamps, prices, current_N=144)
        top_matches = matcher.find_top_matches(prices[-144:], H_matrix, valid_indices, weights, top_k=3)

        result = predictor.generate_prediction(
            current_last_price=prices[-1],
            top_matches=top_matches,
            prices=prices,
            timestamps=timestamps,
            current_N=144
        )

        self.assertIsNotNone(result)
        self.assertIn('valid_count', result)
        self.assertIn('msg', result)

if __name__ == "__main__":
    unittest.main()
