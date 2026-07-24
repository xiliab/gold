import json
import numpy as np
import logging
import sqlite3
import os
from datetime import datetime
from src.config import (
    EMA_ALPHA,
    HIGH_BIAS_EMA_FAST, HIGH_BIAS_EMA_SLOW, HIGH_BIAS_TRIGGER_OVERSHOOT,
    LOW_BIAS_EMA_FAST, LOW_BIAS_EMA_SLOW, LOW_BIAS_TRIGGER_UNDERSHOOT,
    FLAT_DIRECTION_THRESHOLD,
    DYNAMIC_K_HIGH_WIN, DYNAMIC_K_LOW_WIN,
    DYNAMIC_K_AGGRESSIVE, DYNAMIC_K_DEFENSIVE, DYNAMIC_K_NORMAL,
    WIN_RATE_WINDOW, MAPE_REMATCH_THRESHOLD,
    DIRECTION_CONF_WIN_RATE_FLOOR
)
from src.db_init import DB_PATH, get_db_connection
from src.tiny_model import extract_features as _extract_features


def _db_get_state(conn, key, default):
    """从 ai_state 表读取一个持久化系数，若不存在返回 default。"""
    try:
        c = conn.cursor()
        c.execute("SELECT value FROM ai_state WHERE key = ?", (key,))
        row = c.fetchone()
        if not row:
            return default
        val_str = str(row[0])
        if val_str.startswith('[') or val_str.startswith('{'):
            return val_str
        return float(val_str)
    except Exception:
        return default


def _db_set_state(conn, key, value):
    """向 ai_state 表写入/更新一个持久化系数（upsert）。"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    val_str = str(value) if isinstance(value, (str, list, dict)) else str(float(value))
    conn.cursor().execute(
        "INSERT OR REPLACE INTO ai_state (key, value, updated_at) VALUES (?, ?, ?)",
        (key, val_str, now)
    )


class AdaptiveFeedbackCorrector:
    """
    带持久化自学习的预测反馈校正器:
    - 将预测写入数据库，当实际价格产生时进行比对。
    - 基于历史胜率动态调整置信边界系数 (k) 和方向惩罚。
    - 所有动态参数持久化到 ai_state 表，进程重启不丢失。
    """

    def __init__(self, ema_alpha=EMA_ALPHA):
        self.ema_alpha = ema_alpha

        # 从数据库恢复动态参数（进程重启也能继承上次的学习成果）
        try:
            conn = get_db_connection(DB_PATH)
            self.dynamic_k     = _db_get_state(conn, 'dynamic_k',     DYNAMIC_K_NORMAL)
            self.win_rate      = _db_get_state(conn, 'win_rate',      100.0)
            self.high_bias_ema = _db_get_state(conn, 'high_bias_ema', 0.0)
            self.low_bias_ema  = _db_get_state(conn, 'low_bias_ema',  0.0)
            step_bias_str      = _db_get_state(conn, 'step_bias_ema', '[0,0,0,0,0,0]')
            self.step_bias_ema = [0.0] * 6
            conn.close()
            logging.info(f"[Corrector] 已从数据库恢复自学习状态 — k={self.dynamic_k:.2f}, "
                         f"胜率={self.win_rate:.1f}%, high_bias={self.high_bias_ema:.2f}, low_bias={self.low_bias_ema:.2f}, "
                         f"step_bias={self.step_bias_ema}")
        except Exception as e:
            logging.warning(f"[Corrector] 无法加载持久化状态，使用默认值: {e}")
            self.dynamic_k     = DYNAMIC_K_NORMAL
            self.win_rate      = 100.0
            self.high_bias_ema = 0.0
            self.low_bias_ema  = 0.0
            self.step_bias_ema = [0.0] * 6

        self.current_bias   = 0.0
        self.last_pred_high = None
        self.last_pred_low  = None

        # get_logs 缓存（60 秒 TTL，避免每次 API 调用都查 DB）
        self._logs_cache: list  = []
        self._logs_cache_time: float = 0.0

    def _save_state(self):
        """将当前动态系数持久化到数据库，确保重启后不丢失。"""
        try:
            conn = get_db_connection(DB_PATH)
            _db_set_state(conn, 'dynamic_k',     self.dynamic_k)
            _db_set_state(conn, 'win_rate',       self.win_rate)
            _db_set_state(conn, 'high_bias_ema',  self.high_bias_ema)
            _db_set_state(conn, 'low_bias_ema',   self.low_bias_ema)
            _db_set_state(conn, 'step_bias_ema',  json.dumps(self.step_bias_ema))
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"[Corrector] 持久化状态失败: {e}")

    def update_extrema_feedback(self, current_actual_price, pred_high, pred_low):
        """
        根据当前实际最新价与上一次生成的极值预测比对，动态更新极值残差 EMA 估计。
        """
        if pred_high is not None and pred_low is not None:
            # 实际金价突破了预测最高价（预测偏低）→ 正向偏差，快速追上
            if current_actual_price > pred_high:
                high_diff = current_actual_price - pred_high
                self.high_bias_ema = HIGH_BIAS_EMA_FAST * high_diff + HIGH_BIAS_EMA_SLOW * self.high_bias_ema
            elif current_actual_price < pred_high - HIGH_BIAS_TRIGGER_OVERSHOOT:
                # 预测最高价比实际高出超过阈值（预测偏高）→ 慢速负向修正
                high_diff = (current_actual_price - pred_high) * 0.1
                self.high_bias_ema = HIGH_BIAS_EMA_SLOW * high_diff + (1 - HIGH_BIAS_EMA_SLOW) * self.high_bias_ema

            # 实际金价下破了预测最低价（预测偏高）→ 负向偏差
            if current_actual_price < pred_low:
                low_diff = current_actual_price - pred_low
                self.low_bias_ema = LOW_BIAS_EMA_FAST * low_diff + LOW_BIAS_EMA_SLOW * self.low_bias_ema
            elif current_actual_price > pred_low + LOW_BIAS_TRIGGER_UNDERSHOOT:
                low_diff = (current_actual_price - pred_low) * 0.1
                self.low_bias_ema = LOW_BIAS_EMA_SLOW * low_diff + (1 - LOW_BIAS_EMA_SLOW) * self.low_bias_ema

        self.last_pred_high = pred_high
        self.last_pred_low  = pred_low
        self._save_state()
        return self.high_bias_ema, self.low_bias_ema

    def compute_quantile_bounds(self, predicted_prices, history_tracks, last_price, win_rate_factor=1.0):
        """
        基于历史 top 3 轨迹分位数 (Quantile Loss P10 & P90) 动态生成不对称风控包络带。
        结合实盘胜率调解系数 k，提供准确非对称的上下限。
        """
        k = self.get_dynamic_k() * win_rate_factor
        N = len(predicted_prices)
        if history_tracks is None or len(history_tracks) == 0:
            std = np.std(predicted_prices) if N > 1 else 0.5
            half_band = max(0.3, std * k)
            return np.round(predicted_prices - half_band, 2).tolist(), np.round(predicted_prices + half_band, 2).tolist()

        tracks_arr = np.array(history_tracks)  # (M, N)
        q_low = np.percentile(tracks_arr, 10, axis=0)   # P10 下轨分位数
        q_high = np.percentile(tracks_arr, 90, axis=0)  # P90 上轨分位数

        lower_b = predicted_prices + (q_low - predicted_prices) * k + self.low_bias_ema
        upper_b = predicted_prices + (q_high - predicted_prices) * k + self.high_bias_ema

        # 物理规则约束：上轨必须 >= predicted_prices，下轨必须 <= predicted_prices
        lower_b = np.minimum(lower_b, predicted_prices - 0.05)
        upper_b = np.maximum(upper_b, predicted_prices + 0.05)

        return np.round(lower_b, 2).tolist(), np.round(upper_b, 2).tolist()

    def get_extrema_biases(self):
        return self.high_bias_ema, self.low_bias_ema

    def record_prediction(self, target_time, base_price, predicted_price, lower_bound, upper_bound,
                          action_advice=None, advice_type=None, created_at=None, feature_snapshot=None):
        """记录一条包含操作建议与特征快照的预测记录到数据库中。"""
        if not target_time or not base_price or not predicted_price:
            return

        price_diff = predicted_price - base_price
        if abs(price_diff) <= FLAT_DIRECTION_THRESHOLD:
            direction = "FLAT"
        else:
            direction = "UP" if price_diff > 0 else "DOWN"

        if not created_at:
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 将特征向量序列化为 JSON 字符串存储
        snap_str = None
        if feature_snapshot is not None:
            try:
                snap_str = json.dumps([round(float(v), 8) for v in feature_snapshot])
            except Exception:
                pass

        try:
            conn = get_db_connection(DB_PATH)
            c = conn.cursor()
            c.execute('''
                INSERT OR REPLACE INTO prediction_history
                (created_at, target_time, base_price, predicted_direction, predicted_price,
                 lower_bound, upper_bound, action_advice, advice_type, feature_snapshot)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (created_at, target_time, float(base_price), direction,
                  float(predicted_price), float(lower_bound), float(upper_bound),
                  action_advice, advice_type, snap_str))
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"写入预测记录失败: {e}")

    def evaluate_pending_predictions(self, current_timestamp, current_actual_price, tiny_model=None, df=None, inst_factor=0.0):
        """
        验证所有已到期的预测记录（三重比对：方向、区间、操作建议），
        更新胜率与防守系数，并触发极小模型的在线批量自学习。
        N+1 查询优化：先一次性批量加载所有涉及时间点的 K 线，再在内存中分组。
        """
        try:
            conn = get_db_connection(DB_PATH)
            c = conn.cursor()

            c.execute('''
                SELECT id, target_time, base_price, predicted_direction,
                       predicted_price, lower_bound, upper_bound, advice_type, feature_snapshot
                FROM prediction_history
                WHERE actual_price IS NULL AND target_time <= ?
            ''', (current_timestamp,))
            pending = c.fetchall()

            if not pending:
                # 无待验证记录，直接查胜率后返回
                c.execute(f'''
                    SELECT is_direction_correct, is_range_correct, is_advice_correct
                    FROM prediction_history WHERE actual_price IS NOT NULL
                    ORDER BY target_time DESC LIMIT {WIN_RATE_WINDOW}
                ''')
                history_results = c.fetchall()
                conn.close()
                self._update_win_rate_and_k(history_results)
                return self.win_rate, self.dynamic_k

            # ── 批量预加载所有涉及时间点的 K 线（一次 SQL 替代 N+1）─────────
            target_times = [row[1] for row in pending]
            min_t = min(target_times)
            # 最多取 target_time 之后 10 根（6 根用于学习 + 4 根缓冲）
            c.execute('''
                SELECT timestamp, close FROM gold_prices
                WHERE timestamp >= ?
                ORDER BY timestamp ASC LIMIT ?
            ''', (min_t, len(pending) * 10 + 20))
            all_price_rows = c.fetchall()
            # 建立时间戳→close 的有序字典
            price_ts_list  = [(r[0], float(r[1])) for r in all_price_rows]
            price_ts_map   = {r[0]: float(r[1]) for r in all_price_rows}
            # 建立 target_time → 其后连续 6 根 close 列表的缓存
            def _get_next_closes(target_t, n=6):
                """从已加载的价格列表中找到 target_t 之后的最多 n 根 close。"""
                result = []
                for ts, close_val in price_ts_list:
                    if ts >= target_t:
                        result.append(close_val)
                        if len(result) >= n:
                            break
                return result
            # ────────────────────────────────────────────────────────────────

            for row in pending:
                pid, target_time, base_price, pred_dir, pred_price, lb, ub, advice_type, snap_str = row

                # 取 target_time 对应的实际价格（第一根）
                actual_price_val = None
                next_closes = _get_next_closes(target_time, 6)
                if next_closes:
                    actual_price_val = next_closes[0]
                else:
                    continue

                actual_price = actual_price_val
                price_change = actual_price - base_price

                if abs(price_change) <= FLAT_DIRECTION_THRESHOLD:
                    actual_dir = "FLAT"
                else:
                    actual_dir = "UP" if price_change > 0 else "DOWN"

                is_dir_correct   = 1 if (actual_dir == pred_dir or actual_dir == "FLAT" or pred_dir == "FLAT" or abs(price_change) <= 0.25) else 0
                is_range_correct = 1 if (lb <= actual_price <= ub) else 0

                # 操作建议复盘比对 (支持新版防抖建议类型)
                is_advice_correct = 0
                if advice_type == "STRONG_BUY":
                    is_advice_correct = 1 if price_change >= 0.10 else 0
                elif advice_type == "BUY_DIP":
                    is_advice_correct = 1 if price_change >= -0.05 else 0
                elif advice_type == "STRONG_SELL":
                    is_advice_correct = 1 if price_change <= -0.10 else 0
                elif advice_type == "SELL_RISK":
                    is_advice_correct = 1 if price_change <= 0.05 else 0
                elif advice_type == "WAIT":
                    is_advice_correct = 1 if abs(price_change) <= 0.80 else 0
                else:
                    is_advice_correct = 1 if (actual_dir == pred_dir or actual_dir == "FLAT") else 0

                # 触发极小自适应模型增量自学习（使用已批量加载的 K 线，无额外 SQL）
                if tiny_model is not None and df is not None and len(df) >= 20:
                    try:
                        snap_feat = np.array(json.loads(snap_str), dtype=float) if snap_str else \
                                    _extract_features(df, inst_factor)

                        closes = np.array(next_closes[:6], dtype=float)
                        if len(closes) == 6:
                            actual_r_vec = np.zeros(6)
                            actual_r_vec[0] = (closes[0] - base_price) / (base_price + 1e-8)
                            for step_idx in range(1, 6):
                                actual_r_vec[step_idx] = (closes[step_idx] - closes[step_idx - 1]) / (closes[step_idx - 1] + 1e-8)
                        else:
                            r_step = float(price_change / (base_price + 1e-8) / 6.0)
                            actual_r_vec = np.full(6, r_step)

                        tiny_model.fit_online(snap_feat, actual_r_vec)

                        # 6 步闭环残差自学习偏差向量更新
                        if len(closes) == 6:
                            est_pred_path = np.linspace(base_price + (pred_price - base_price) * 0.167, pred_price, 6)
                            step_errors = est_pred_path - closes
                            curr_bias_vec = np.array(self.step_bias_ema, dtype=float)
                            updated_bias_vec = 0.75 * curr_bias_vec + 0.25 * step_errors
                            self.step_bias_ema = [round(float(v), 4) for v in updated_bias_vec]
                            self._save_state()
                    except Exception as ex:
                        logging.warning(f"极小模型在线增量更新提示: {ex}")

                c.execute('''
                    UPDATE prediction_history
                    SET actual_price = ?, is_direction_correct = ?, is_range_correct = ?, is_advice_correct = ?
                    WHERE id = ?
                ''', (float(actual_price), is_dir_correct, is_range_correct, is_advice_correct, pid))

            conn.commit()

            # 验证完成后更新【今日实时胜率】(仅查询今日 06:00:00 至今的到期验证记录)
            today_start_str = datetime.now().strftime('%Y-%m-%d 06:00:00')
            c.execute('''
                SELECT is_direction_correct, is_range_correct, is_advice_correct
                FROM prediction_history
                WHERE actual_price IS NOT NULL AND target_time >= ?
                ORDER BY target_time DESC
            ''', (today_start_str,))
            history_results = c.fetchall()

            # 若今日到期验证记录少于 5 条，回退取最近 30 条历史记录
            if len(history_results) < 5:
                c.execute('''
                    SELECT is_direction_correct, is_range_correct, is_advice_correct
                    FROM prediction_history
                    WHERE actual_price IS NOT NULL
                    ORDER BY target_time DESC LIMIT 30
                ''')
                history_results = c.fetchall()

            conn.close()

            self._update_win_rate_and_k(history_results)
            # 验证完成后使日志缓存失效
            self._logs_cache_time = 0.0

            return self.win_rate, self.dynamic_k

        except Exception as e:
            logging.error(f"验证预测记录失败: {e}")
            return self.win_rate, self.dynamic_k

    def _update_win_rate_and_k(self, history_results):
        """根据历史结果更新胜率和动态防守系数（抽取复用逻辑）。"""
        if not history_results:
            return
        dir_wins    = sum(r[0] for r in history_results if r[0] is not None)
        range_wins  = sum(r[1] for r in history_results if r[1] is not None)
        advice_wins = sum(r[2] for r in history_results if r[2] is not None)
        n = len(history_results)
        self.win_rate = ((dir_wins / n) * 0.50 + (advice_wins / n) * 0.30 + (range_wins / n) * 0.20) * 100.0
        sigmoid = 1.0 / (1.0 + np.exp((self.win_rate - 60.0) / 10.0))
        self.dynamic_k = round(float(
            np.clip(DYNAMIC_K_AGGRESSIVE + (DYNAMIC_K_DEFENSIVE - DYNAMIC_K_AGGRESSIVE) * sigmoid,
                    DYNAMIC_K_AGGRESSIVE, DYNAMIC_K_DEFENSIVE)
        ), 2)
        self._save_state()

    def get_dynamic_k(self):
        return self.dynamic_k

    def get_logs(self):
        """从数据库拉取最近 10 条复盘日志，结果缓存 60 秒，避免每次 API 调用都查 DB。"""
        import time
        now = time.time()
        if now - self._logs_cache_time < 60.0 and self._logs_cache:
            return self._logs_cache   # 命中缓存，直接返回

        logs = []
        try:
            with get_db_connection(DB_PATH) as conn:
                c = conn.cursor()
                c.execute('''
                    SELECT target_time, predicted_direction, predicted_price, actual_price,
                           is_direction_correct, action_advice, is_advice_correct
                    FROM prediction_history
                    WHERE actual_price IS NOT NULL
                    ORDER BY target_time DESC LIMIT 10
                ''')
                rows = c.fetchall()

            for row in rows:
                target_time, pred_dir, pred_price, actual_price, is_dir_correct, advice_text, is_adv_correct = row
                time_str   = target_time[11:16] if len(target_time) >= 16 else target_time
                dir_arrow  = {"UP": "↑", "DOWN": "↓"}.get(pred_dir, "→")
                result     = "命中" if is_dir_correct else "偏差"
                adv_status = "✓" if is_adv_correct else "✗"
                adv_label  = advice_text if advice_text else "暂无建议"
                logs.append(f"[{time_str}] {result}{dir_arrow} ({adv_label} {adv_status}) | 预测:{pred_price:.2f} 实际:{actual_price:.2f}")
        except Exception as e:
            logging.error(f"获取日志失败: {e}")

        self._logs_cache      = logs
        self._logs_cache_time = now
        return logs

    def check_mape(self, actual_recent, pred_recent):
        """
        校验近期 MAPE 相对误差，同时计算真实的价格偏差（元）。
        返回: (need_rematch, msg, mape_pct, avg_price_error_yuan, current_bias)
        """
        if not actual_recent or not pred_recent or len(actual_recent) != len(pred_recent):
            return False, "系统运行正常", 0.0, 0.0, 0.0

        y_true = np.array(actual_recent, dtype=float)
        y_pred = np.array(pred_recent,   dtype=float)

        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))))

        # 平均价格误差（实际 - 预测，正值代表预测偏低，负值代表偏高）
        avg_price_error = float(np.mean(y_true - y_pred))
        # 当前 bias 为价格误差的 EMA 近似（此处取最近一次误差）
        last_error = float(y_true[-1] - y_pred[-1])

        need_rematch = mape > MAPE_REMATCH_THRESHOLD
        msg = f"MAPE 偏差: {mape * 100:.2f}% (反馈已实时校正)"
        return need_rematch, msg, mape, last_error, avg_price_error
