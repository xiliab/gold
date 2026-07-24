import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.db_init import init_db
from src.data_fetcher import load_clean_continuous_series
from src.matcher import CurveMatcher
from src.tiny_model import TinyResidualPredictor
from src.predictor import TrendPredictor
from src.corrector import AdaptiveFeedbackCorrector
from src.config import WINDOW_SIZE, FUTURE_STEPS

class TestSystemRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.clean_df = load_clean_continuous_series()

    def test_01_fastdtw_computation(self):
        """测试带 Sakoe-Chiba 窗约束的 FastDTW 计算正确性"""
        matcher = CurveMatcher(use_fast_dtw=True, dtw_radius=5)
        seq1 = np.sin(np.linspace(0, 2 * np.pi, 50))
        # 构造带有 2 个时间步相位滞后的序列
        seq2 = np.sin(np.linspace(0, 2 * np.pi, 50) - 0.2)
        
        dist = matcher.compute_constrained_dtw_distance(seq1, seq2, window_radius=5)
        self.assertIsInstance(dist, float)
        self.assertGreater(dist, 0.0)
        self.assertLess(dist, 0.5, "轻微相位错位下的 DTW 距离应当很小")

    def test_02_matcher_with_fastdtw_and_acceleration(self):
        """测试整合 FastDTW 和加速索引后的匹配效率与结构"""
        if self.clean_df is None or len(self.clean_df) < WINDOW_SIZE * 2:
            self.skipTest("行情数据不足，跳过匹配测试")
            
        prices = self.clean_df['close'].values
        timestamps = self.clean_df['timestamp'].values

        matcher = CurveMatcher(use_fast_dtw=True, dtw_radius=10)
        H_matrix, valid_indices, weights = matcher.build_history_matrix(timestamps, prices, current_N=WINDOW_SIZE)
        
        self.assertIsNotNone(H_matrix)
        self.assertGreater(len(H_matrix), 0)

        current_query = prices[-WINDOW_SIZE:]
        matches = matcher.find_top_matches(current_query, H_matrix, valid_indices, weights, top_k=3)
        
        self.assertIsInstance(matches, list)
        for item in matches:
            self.assertIn('similarity_pct', item)
            self.assertIn('dtw_bonus', item)
            self.assertFalse(np.isnan(item['similarity_pct']))

    def test_03_tiny_model_online_learning_stability(self):
        """测试 TinyModel 在 20 维输入下在线 SGD 梯度的稳定性，防 NaN/爆炸"""
        model = TinyResidualPredictor(n_features=20)
        dummy_features = np.random.randn(20)
        
        pred_before = model.predict_returns(dummy_features)
        self.assertEqual(len(pred_before), FUTURE_STEPS)
        self.assertFalse(np.any(np.isnan(pred_before)))

        # 模拟在线反向传播增量更新
        actual_target = pred_before + 0.0001
        model.fit_online(dummy_features, actual_target)
        
        pred_after = model.predict_returns(dummy_features)
        self.assertFalse(np.any(np.isnan(pred_after)))

    def test_04_predictor_data_fault_tolerance(self):
        """测试 Predictor 对断层、极限边界情况的容错断言"""
        predictor = TrendPredictor()
        prices = np.full(WINDOW_SIZE, 500.0)  # 平线序列，无波动
        timestamps = pd.date_range("2026-07-24", periods=WINDOW_SIZE, freq="1min").astype(str).values

        matcher = CurveMatcher()
        H_matrix, valid_indices, weights = matcher.build_history_matrix(timestamps, prices, current_N=WINDOW_SIZE)
        matches = matcher.find_top_matches(prices, H_matrix, valid_indices, weights, top_k=3)

        result = predictor.generate_prediction(
            current_last_price=500.0,
            top_matches=matches,
            prices=prices,
            timestamps=timestamps,
            current_N=WINDOW_SIZE
        )
        self.assertIsNotNone(result)
        self.assertIn('predicted_prices', result)
        self.assertIn('valid_count', result)

    def test_05_extract_20d_features(self):
        """测试 20 维多维量价与跨市场高频特征抽取的规范性"""
        from src.tiny_model import extract_features
        df_mock = pd.DataFrame({
            'open': np.linspace(880, 882, 30),
            'high': np.linspace(880.5, 882.5, 30),
            'low': np.linspace(879.5, 881.5, 30),
            'close': np.linspace(880.1, 882.1, 30),
            'volume': np.random.randint(10, 100, 30)
        })
        feats = extract_features(df_mock, spdr_bias=0.2, dxy_bias=-0.15, us10y_bias=-0.05)
        self.assertEqual(len(feats), 20, "提取的特征维数应当正好为 20 维")
        self.assertFalse(np.any(np.isnan(feats)), "归一化特征不能包含 NaN 异常值")
        self.assertTrue(np.all(feats >= -3.0) and np.all(feats <= 3.0), "归一化特征应受限于 [-3.0, 3.0] 截断范围")

    def test_06_regime_and_quantile_bounds(self):
        """测试 Regime Switching 市场状态识别与 Quantile Loss 不对称风控包络带生成"""
        from src.predictor import detect_market_regime
        from src.corrector import AdaptiveFeedbackCorrector

        prices_trending = np.sin(np.linspace(0, 10, 50)) * 5.0 + 880.0
        regime, score = detect_market_regime(prices_trending)
        self.assertIn(regime, ["TRENDING", "RANGING"])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        corrector = AdaptiveFeedbackCorrector()
        pred_p = np.array([880.0, 880.2, 880.5, 880.8, 881.0, 881.2])
        hist_tracks = [
            [880.0, 880.4, 880.8, 881.2, 881.5, 882.0],
            [880.0, 879.8, 879.5, 879.2, 879.0, 878.5],
            [880.0, 880.1, 880.3, 880.4, 880.5, 880.6]
        ]
        lower_b, upper_b = corrector.compute_quantile_bounds(pred_p, hist_tracks, 880.0)
        self.assertEqual(len(lower_b), 6)
        self.assertEqual(len(upper_b), 6)
        for i in range(6):
            self.assertLessEqual(lower_b[i], pred_p[i], "下轨应不高于预测中轨")
            self.assertGreaterEqual(upper_b[i], pred_p[i], "上轨应不低于预测中轨")

if __name__ == "__main__":
    unittest.main()
