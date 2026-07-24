import numpy as np
import pandas as pd
import time
import logging


from src.config import NMS_RADIUS, MIN_SIMILARITY, DECAY_RATE, MATCH_ALPHA, USE_FAST_DTW, DTW_WINDOW_RADIUS, SEARCH_INDEX_ACCELERATION
from src.utils import get_trading_date

class CurveMatcher:
    def __init__(self, nms_radius=None, decay_rate=None, alpha=None, min_similarity=None, use_fast_dtw=None, dtw_radius=None):
        self.nms_radius = nms_radius if nms_radius is not None else NMS_RADIUS
        self.decay_rate = decay_rate if decay_rate is not None else DECAY_RATE
        self.alpha = alpha if alpha is not None else MATCH_ALPHA
        self.min_similarity = min_similarity if min_similarity is not None else MIN_SIMILARITY
        self.use_fast_dtw = use_fast_dtw if use_fast_dtw is not None else USE_FAST_DTW
        self.dtw_radius = dtw_radius if dtw_radius is not None else DTW_WINDOW_RADIUS
        
        # 内存常驻历史矩阵缓存 (0ms 极速碰撞)
        self._cache_key = None
        self._cache_H = None
        self._cache_v_idx = None
        self._cache_w = None

        # 日内动态淘汰剪枝池 (Dynamic Candidate Pruning Pool)
        self._pool_trading_date = None

    def compute_constrained_dtw_distance(self, seq1, seq2, window_radius=None):
        """
        带 Sakoe-Chiba 窗约束的极速 DTW (Dynamic Time Warping) 算法。
        专门纠正行情走势中由于节奏变频导致的“时间相位错位/微幅延迟与提前”。
        """
        radius = window_radius if window_radius is not None else self.dtw_radius
        N = len(seq1)
        M = len(seq2)
        window = max(radius, abs(N - M))

        dtw_matrix = np.full((N + 1, M + 1), np.inf)
        dtw_matrix[0, 0] = 0.0

        for i in range(1, N + 1):
            j_start = max(1, i - window)
            j_end = min(M + 1, i + window + 1)
            for j in range(j_start, j_end):
                cost = (seq1[i - 1] - seq2[j - 1]) ** 2
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],
                    dtw_matrix[i, j - 1],
                    dtw_matrix[i - 1, j - 1]
                )

        dist = np.sqrt(dtw_matrix[N, M]) / float(N)
        return float(dist)

    def z_score_normalize(self, series):
        std = np.std(series)
        if std == 0:
            return np.zeros_like(series), 0.0, np.mean(series)
        mean = np.mean(series)
        return (series - mean) / std, std, mean

    def smooth_series_1d(self, series, window_size=5):
        """
        自适应 1D 卷积平滑，抹平 1m 高频无意义噪点，提取清晰的趋势骨架
        """
        if len(series) < window_size or window_size <= 1:
            return series
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(series, kernel, mode='same')
        pad_half = window_size // 2
        smoothed[:pad_half] = series[:pad_half]
        smoothed[-pad_half:] = series[-pad_half:]
        return smoothed

    def build_history_matrix(self, timestamps, prices, current_N):
        """
        基于“日内自然周期”提取历史矩阵，支持日内动态剪枝淘汰。
        每天早上 06:00 (Asia/Shanghai) 视为新交易日的开始。
        """
        n_points = len(prices)
        if n_points == 0 or current_N <= 0:
            return None, [], []

        dt_series = pd.to_datetime(timestamps)
        # 用新提取的工具类处理交易日
        trading_dates = get_trading_date(dt_series)
        last_date = trading_dates[-1]

        # 如果进入新的交易日，清空上一交易日的剪枝池
        if self._pool_trading_date != last_date:
            self._pool_trading_date = last_date
            self._active_indices_set = None
            self._cache_key = None

        cache_key = (n_points, current_N, len(self._active_indices_set) if self._active_indices_set else 0)
        if self._cache_key == cache_key and self._cache_H is not None:
            return self._cache_H, self._cache_v_idx, self._cache_w

        windows = []
        valid_indices = []
        valid_timestamps = []

        grouped = pd.Series(range(len(prices))).groupby(trading_dates)
        
        for date, indices in grouped:
            if date == last_date:
                continue # 不拿今天和今天匹配
                
            indices = indices.values
            if len(indices) >= current_N:
                start_idx = indices[0]

                # 日内剪枝淘汰过滤器：若当前历史切片已被淘汰，直接跳过
                if self._active_indices_set is not None and start_idx not in self._active_indices_set:
                    continue

                w_raw = prices[start_idx : start_idx + current_N]
                z_w, std_w, mean_w = self.z_score_normalize(w_raw)
                
                if std_w > 0:
                    windows.append(z_w)
                    valid_indices.append(start_idx)
                    valid_timestamps.append(str(timestamps[start_idx]))

        if not windows:
            return None, [], []

        H_matrix = np.array(windows)  # Shape: (M, current_N)
        
        # 权重分配：如果 decay_rate > 0 则施加指数衰减；若 decay_rate == 0 则全序列均匀等权重
        if self.decay_rate > 0:
            x = np.linspace(-self.decay_rate, 0, current_N)
            weights = np.exp(x)
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(current_N) / float(current_N)

        # 写入 RAM 缓存
        self._cache_key = cache_key
        self._cache_H = H_matrix
        self._cache_v_idx = valid_indices
        self._cache_v_ts = valid_timestamps
        self._cache_w = weights

        return H_matrix, valid_indices, weights

    def find_top_matches(self, current_query, H_matrix, valid_indices, weights, top_k=3, valid_timestamps=None):
        t0 = time.time()
        if H_matrix is None or len(H_matrix) == 0:
            return []

        Q_z, Q_std, Q_mean = self.z_score_normalize(current_query)
        w = weights
        current_N = len(current_query)

        # 1. 骨架形状相关性 (Pearson correlation)
        H_wmean = H_matrix @ w             # (M,)
        Q_wmean = float(Q_z @ w)           # scalar

        H_c = H_matrix - H_wmean[:, np.newaxis]  # (M, W)
        Q_c = Q_z - Q_wmean                      # (W,)

        cov_HQ = H_c @ (w * Q_c)                 # (M,)
        var_H  = (H_c ** 2) @ w                  # (M,)
        var_Q  = float((Q_c ** 2) @ w)           # scalar

        denom  = np.sqrt(var_H * var_Q) + 1e-10
        r_corr = np.clip(cov_HQ / denom, -1.0, 1.0)  # (M,)

        # 2. 端点整体斜率相似度 (Overall Slope Similarity)
        Q_slope = Q_z[-1] - Q_z[0]
        H_slopes = H_matrix[:, -1] - H_matrix[:, 0]
        slope_cos = np.cos(np.arctan(Q_slope) - np.arctan(H_slopes)) # [-1, 1]
        slope_sim = np.clip((slope_cos + 1.0) / 2.0, 0.0, 1.0)        # 归一化到 [0, 1]

        # 3. 分段趋势与关键转折点契合度 (Segmented Trend & Pivot Similarity)
        # 将 06:00 至当前时刻按时间块拆分，评估主要时间段内的走向一致性，允许消除微观毛刺抖动
        n_segments = 4 if current_N >= 60 else (2 if current_N >= 20 else 1)
        if n_segments > 1:
            segment_edges = np.linspace(0, current_N - 1, n_segments + 1, dtype=int)
            seg_sims = []
            for s in range(n_segments):
                i_start, i_end = segment_edges[s], segment_edges[s+1]
                if i_end > i_start:
                    q_seg_delta = Q_z[i_end] - Q_z[i_start]
                    h_seg_deltas = H_matrix[:, i_end] - H_matrix[:, i_start]
                    cos_val = np.cos(np.arctan(q_seg_delta) - np.arctan(h_seg_deltas))
                    seg_sims.append(np.clip((cos_val + 1.0) / 2.0, 0.0, 1.0))
            segment_trend_sim = np.mean(seg_sims, axis=0) # (M,)
        else:
            segment_trend_sim = slope_sim

        # 4. 波峰波谷极大极小值时刻对齐惩罚 (Pivot Timing Alignment)
        # 寻找今日走势的关键最高点与最低点相对位置，强制历史切片的关键拐点时刻精准重合
        argmax_Q = int(np.argmax(Q_z))
        argmin_Q = int(np.argmin(Q_z))
        argmax_H = np.argmax(H_matrix, axis=1)  # (M,)
        argmin_H = np.argmin(H_matrix, axis=1)  # (M,)

        max_offset_rel = np.abs(argmax_H - argmax_Q) / float(current_N)
        min_offset_rel = np.abs(argmin_H - argmin_Q) / float(current_N)
        pivot_timing_sim = np.clip(1.0 - 0.6 * (max_offset_rel + min_offset_rel), 0.0, 1.0) # (M,)

        # 5. 真实严谨的复合趋势相似度打分 (0.60 真实 Pearson 相关 + 0.25 分段趋势 + 0.15 拐点时序)
        # 聚焦全天波浪骨架的主体形态契合，防止末端极值对齐过拟合导致的误匹配
        real_r_sim = np.maximum(0.0, r_corr)  # [0, 1]
        composite_sim = 0.60 * real_r_sim + 0.25 * segment_trend_sim + 0.15 * pivot_timing_sim

        # 转换为真实百份比相似度 (0 ~ 100%)
        similarity_pcts = composite_sim * 100.0
        scores = 1.0 - composite_sim  # 越小越相似

        sorted_indices = np.argsort(-composite_sim)  # 降序
        selected_candidates = []

        for attempt in range(4):
            effective_threshold = max(50.0, self.min_similarity - attempt * 5.0)
            suppressed = np.zeros(len(scores), dtype=bool)
            selected_candidates = []

            for idx in sorted_indices:
                if suppressed[idx]:
                    continue

                dtw_sim_bonus = 0.0
                if self.use_fast_dtw:
                    dtw_dist = self.compute_constrained_dtw_distance(Q_z, H_matrix[idx])
                    # DTW 距离越小，相似度加分越高（映射为 0~5% 的相位修正奖励）
                    dtw_sim_bonus = np.clip((0.20 - dtw_dist) * 25.0, -5.0, 5.0)

                sim_pct = round(float(np.clip(similarity_pcts[idx] + dtw_sim_bonus, 0.0, 100.0)), 2)
                score_val = float(1.0 - sim_pct / 100.0)

                if sim_pct < effective_threshold:
                    continue

                start_pos = valid_indices[idx]
                v_ts_arr = valid_timestamps if valid_timestamps is not None else getattr(self, '_cache_v_ts', None)
                st_time_str = str(v_ts_arr[idx]) if v_ts_arr is not None and idx < len(v_ts_arr) else ""
                selected_candidates.append({
                    'matrix_idx': idx,
                    'start_index': start_pos,
                    'start_time': st_time_str,
                    'score': score_val,
                    'similarity_pct': sim_pct,
                    'corr': float(r_corr[idx]),
                    'dtw_bonus': round(float(dtw_sim_bonus), 2),
                    'low_confidence': attempt > 0
                })

                # NMS 剔除相邻交易日
                for candidate_idx in sorted_indices:
                    if not suppressed[candidate_idx]:
                        other_pos = valid_indices[candidate_idx]
                        if abs(other_pos - start_pos) <= self.nms_radius:
                            suppressed[candidate_idx] = True

                if len(selected_candidates) >= top_k:
                    break

            if selected_candidates:
                if attempt > 0:
                    logging.warning(f"相似度阈值自动降至 {effective_threshold:.0f}% 后找到候选")
                break

        # 日内动态淘汰剪枝：保留得分在合理防守阈值以上的候选日，剔除明显走向偏离的历史日
        if len(valid_indices) > top_k * 4:
            prune_threshold = max(55.0, self.min_similarity * 0.75)
            survived_indices = {
                valid_indices[i] for i in range(len(valid_indices))
                if similarity_pcts[i] >= prune_threshold
            }
            # 安全熔断防空仓：确保淘汰后有效样本数不少于 top_k * 2
            if len(survived_indices) >= top_k * 2:
                self._active_indices_set = survived_indices
                self._cache_key = None
            else:
                self._active_indices_set = None

        t_cost = (time.time() - t0) * 1000.0
        logging.info(f"复合加权日内锚定匹配完成！耗时: {t_cost:.2f} ms，剪枝后池大小: {len(self._active_indices_set) if self._active_indices_set else '全量'}，找到 {len(selected_candidates)} 个候选。")
        return selected_candidates
