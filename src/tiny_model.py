import numpy as np
import pandas as pd
import sqlite3
import os
import json
import logging
from datetime import datetime


import time
from src.db_init import DB_PATH, get_db_connection

# 每个特征维度的规范化统计量（均值/标准差），基于真实黄金市场数据经验值
# 用于将不同量纲的特征归一化到均匀的 [-1, 1] 分布，避免大数值特征主导梯度
_FEAT_MEAN = np.array([0.0,    0.0,    0.05,  0.0,    0.0,    0.0,    0.0,   0.0,   0.70,   0.0,   0.0], dtype=float)
_FEAT_STD  = np.array([0.5,    0.15,   0.04,  0.08,   0.12,   0.15,   0.5,   0.5,   0.15,   0.5,   0.10], dtype=float)


def _calculate_rsi(series, period=14):
    """计算 RSI 指标 (0~100)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty and not np.isnan(rsi.iloc[-1]) else 50.0


def extract_features(df, spdr_bias=0.0, top_matches=None):
    """
    从近期 K 线序列提取 11 维微观+历史深度融合特征向量 x：
    - 前 8 维：RSI, EMA乖离率, ATR波动率, 3m/6m/15m动量, 成交量偏向, SPDR持仓偏向
    - 后 3 维：历史切片平均相似度, 历史共识比例, 历史加权预期收益率
    """
    if df is None or len(df) < 20:
        return np.zeros(11)

    prices = df['close'].values
    vols = df['volume'].values if 'volume' in df.columns else np.ones(len(prices))

    p_last = prices[-1]

    # 1. RSI (14) -> 归一化到 [-1, 1]
    rsi_val = _calculate_rsi(df['close'], 14)
    f_rsi = (rsi_val - 50.0) / 50.0

    # 2. EMA20 乖离率 (%)
    ema20 = pd.Series(prices).ewm(span=20).mean().iloc[-1]
    f_ema_dist = float((p_last - ema20) / (ema20 + 1e-8) * 100.0)

    # 3. 近期波动率 (ATR14 相对比例)
    highs = df['high'].values if 'high' in df.columns else prices
    lows = df['low'].values if 'low' in df.columns else prices
    tr = np.maximum(highs[-14:] - lows[-14:], 1e-4)
    f_atr = float(np.mean(tr) / (p_last + 1e-8) * 100.0)

    # 4. 短线动量 (3m & 6m 相对变幅 %)
    f_mom3 = float((prices[-1] - prices[-4]) / (prices[-4] + 1e-8) * 100.0) if len(prices) >= 4 else 0.0
    f_mom6 = float((prices[-1] - prices[-7]) / (prices[-7] + 1e-8) * 100.0) if len(prices) >= 7 else 0.0

    # 5. 趋势端点斜率 (15m 变幅 %)
    f_slope = float((prices[-1] - prices[-15]) / (prices[-15] + 1e-8) * 100.0) if len(prices) >= 15 else 0.0

    # 6. 主力资金量能偏向
    diffs = np.diff(prices[-15:])
    sub_vols = vols[-14:]
    buy_v = np.sum(sub_vols[diffs > 0]) if np.any(diffs > 0) else 0
    sell_v = np.sum(sub_vols[diffs < 0]) if np.any(diffs < 0) else 0
    f_vol_imb = float((buy_v - sell_v) / (buy_v + sell_v + 1e-8))

    # 7. 机构持仓偏向
    f_spdr = float(np.clip(spdr_bias, -1.0, 1.0))

    # 8. 9. 10. 新增 3 维历史匹配置信度与形态共识特征
    f_hist_sim = 0.70
    f_hist_consensus = 0.0
    f_hist_return = 0.0

    if top_matches and len(top_matches) > 0:
        sims = [m.get('similarity_pct', 70.0) / 100.0 for m in top_matches]
        f_hist_sim = float(np.mean(sims))

        # 匹配切片的历史未来多空共识比例 (-1.0 ~ +1.0)
        future_ends = [m.get('track', [p_last])[-1] - m.get('track', [p_last])[0] for m in top_matches if 'track' in m and len(m['track']) > 0]
        if future_ends:
            up_c = sum(1 for e in future_ends if e > 0)
            down_c = sum(1 for e in future_ends if e < 0)
            f_hist_consensus = float((up_c - down_c) / max(len(future_ends), 1))
            f_hist_return = float(np.mean(future_ends) / (p_last + 1e-8) * 100.0)

    raw = np.array([f_rsi, f_ema_dist, f_atr, f_mom3, f_mom6, f_slope, f_vol_imb, f_spdr,
                    f_hist_sim, f_hist_consensus, f_hist_return], dtype=float)

    # Z-Score 归一化：对齐各维度量纲，防止梯度方向偏斜
    normalized = (raw - _FEAT_MEAN) / (_FEAT_STD + 1e-8)
    return np.clip(normalized, -3.0, 3.0)


from src.config import TINY_MOMENTUM, TINY_GRAD_CLIP, TINY_MSE_BASE, TINY_MAX_CLIP_BOUND


def _tanh(x):
    """Tanh 激活函数，保留双向正负信号"""
    return np.tanh(x)


def _tanh_grad(h):
    """Tanh 的导数 1.0 - h^2 (直接利用前向传播计算出的 h)"""
    return 1.0 - h ** 2


class TinyResidualPredictor:
    """
    纯 NumPy 实现的极小自适应单隐藏层网络预测模型:
    - 输入:  11 维 Z-Score 归一化微观+历史融合特征向量 x
    - 隐藏层: W1 (11→16) + Tanh，提供对称非线性表达能力
    - 输出层: W2 (16→6)，预测未来 6 步收益率残差
    - 在线学习: Momentum SGD + 全局 L2 梯度裁剪 + 渐进式输出裁剪 + 自适应学习率衰减
    """

    HIDDEN = 16  # 隐藏层神经元数量

    def __init__(self, n_features=11, future_steps=6, lr=0.02, l2_reg=0.001):
        self.n_features = n_features
        self.future_steps = future_steps
        self.lr_init = lr        # 初始学习率
        self.lr = lr
        self.l2_reg = l2_reg

        # 隐藏层参数：Xavier 初始化
        scale1 = np.sqrt(2.0 / n_features)
        self.W1 = np.random.randn(n_features, self.HIDDEN) * scale1
        self.b1 = np.zeros(self.HIDDEN, dtype=float)

        # 输出层参数：零位偏置初始化 (避免模型未训练前盲目看涨/看跌)
        self.W2 = np.random.randn(self.HIDDEN, future_steps) * 0.001
        self.b2 = np.zeros(future_steps, dtype=float)

        # 动量速度张量初始化
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)

        self.train_count = 0
        self.last_mse = 0.0
        self.ema_mse = 0.0

        self.load_model()

    def load_model(self, db_path=DB_PATH):
        """从 SQLite ai_state 表读取模型权重（支持 JSON 新格式与旧版逐行兼容）。"""
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()

            # 1. 尝试读取新版 JSON 格式
            c.execute("SELECT value FROM ai_state WHERE key = 'tiny_model_weights'")
            row = c.fetchone()
            if row:
                data = json.loads(row[0])
                w1_arr = np.array(data['W1'])
                if w1_arr.size == self.n_features * self.HIDDEN:
                    self.W1 = w1_arr.reshape(self.n_features, self.HIDDEN)
                    self.b1 = np.array(data['b1'])
                    self.W2 = np.array(data['W2']).reshape(self.HIDDEN, self.future_steps)
                    self.b2 = np.array(data['b2'])
                    self.train_count = data.get('train_count', 0)
                    self.ema_mse = data.get('ema_mse', 0.0)
                    if self.train_count > 0:
                        self.lr = self.lr_init / np.sqrt(self.train_count + 1)
                    
                    self.v_W1 = np.zeros_like(self.W1)
                    self.v_b1 = np.zeros_like(self.b1)
                    self.v_W2 = np.zeros_like(self.W2)
                    self.v_b2 = np.zeros_like(self.b2)
                    conn.close()
                    return
                else:
                    logging.info(f"[TinyModel] 自动平滑升级至 11 维历史-实时双核特征矩阵...")

            # 2. 回退：读取旧版单行存储格式并自动迁移
            migrated = False
            # 加载 W1
            c.execute("SELECT key, value FROM ai_state WHERE key LIKE 'tiny_w1_%'")
            for k, val in c.fetchall():
                idx = int(k.replace('tiny_w1_', ''))
                r, col = divmod(idx, self.HIDDEN)
                if r < self.n_features and col < self.HIDDEN:
                    self.W1[r, col] = float(val)
                    migrated = True

            # 加载 b1
            c.execute("SELECT key, value FROM ai_state WHERE key LIKE 'tiny_b1_%'")
            for k, val in c.fetchall():
                idx = int(k.replace('tiny_b1_', ''))
                if idx < self.HIDDEN:
                    self.b1[idx] = float(val)

            # 加载 W2
            c.execute("SELECT key, value FROM ai_state WHERE key LIKE 'tiny_w2_%'")
            for k, val in c.fetchall():
                idx = int(k.replace('tiny_w2_', ''))
                r, col = divmod(idx, self.future_steps)
                if r < self.HIDDEN and col < self.future_steps:
                    self.W2[r, col] = float(val)

            # 加载 b2
            c.execute("SELECT key, value FROM ai_state WHERE key LIKE 'tiny_b2_%'")
            for k, val in c.fetchall():
                idx = int(k.replace('tiny_b2_', ''))
                if idx < self.future_steps:
                    self.b2[idx] = float(val)

            # 加载训练次数
            c.execute("SELECT value FROM ai_state WHERE key = 'tiny_train_count'")
            r_cnt = c.fetchone()
            if r_cnt:
                self.train_count = int(r_cnt[0])
                self.lr = self.lr_init / np.sqrt(self.train_count + 1)

            # 初始化动量矩阵
            self.v_W1 = np.zeros_like(self.W1)
            self.v_b1 = np.zeros_like(self.b1)
            self.v_W2 = np.zeros_like(self.W2)
            self.v_b2 = np.zeros_like(self.b2)

            conn.close()
            
            if migrated and self.train_count > 0:
                logging.info(f"[TinyModel] 旧格式模型加载成功，启动自动迁移至 JSON...")
                self.save_model(db_path)
                
        except Exception as e:
            logging.warning(f"[TinyModel] 参数加载提示: {e}")

    def save_model(self, db_path=DB_PATH):
        """将模型所有参数持久化保存至 SQLite ai_state 表（启用 WAL 模式、5000ms 超时与忙锁指数避让重试）。"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data = {
            'W1': self.W1.tolist(),
            'b1': self.b1.tolist(),
            'W2': self.W2.tolist(),
            'b2': self.b2.tolist(),
            'train_count': self.train_count,
            'ema_mse': self.ema_mse
        }
        json_str = json.dumps(data)

        # 忙锁退避重试最多 3 次，彻底防御高频数据库锁碰撞
        for attempt in range(3):
            try:
                with get_db_connection(db_path) as conn:
                    c = conn.cursor()
                    c.execute("INSERT OR REPLACE INTO ai_state (key, value, updated_at) VALUES (?, ?, ?)",
                              ('tiny_model_weights', json_str, now))
                    c.execute("DELETE FROM ai_state WHERE key LIKE 'tiny_%' AND key != 'tiny_model_weights'")
                    conn.commit()
                    return
            except sqlite3.OperationalError as e:
                if 'locked' in str(e).lower() and attempt < 2:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    logging.error(f"[TinyModel] 参数保存重试告警: {e}")
                    break
            except Exception as e:
                logging.error(f"[TinyModel] 参数保存异常: {e}")
                break

    def _forward(self, x_vec):
        """前向传播: x → 隐藏层 Tanh → 输出层"""
        h_pre = x_vec @ self.W1 + self.b1   # (1, HIDDEN)
        h     = _tanh(h_pre)                 # (1, HIDDEN)
        out   = h @ self.W2 + self.b2        # (1, future_steps)
        return h_pre, h, out

    def predict_returns(self, x):
        """
        根据 11 维特征向量 x，推断未来 6 步预期收益率残差向量。
        采用渐进式裁剪边界：早期 (train_count=0) 限制在 ±0.3%，随训练次数扩增至上限 ±0.5%。
        """
        x_vec = np.asarray(x, dtype=float).flatten()
        if len(x_vec) < self.n_features:
            x_vec = np.pad(x_vec, (0, self.n_features - len(x_vec)), mode='constant')
        elif len(x_vec) > self.n_features:
            x_vec = x_vec[:self.n_features]

        x_mat = x_vec.reshape(1, -1)
        _, _, out = self._forward(x_mat)
        # 单分钟收益率严格限制在物理合理区间 (±0.03% ~ ±0.08%，即单分钟 ±0.25 ~ ±0.70 元/克)
        clip_bound = min(0.0003 + self.train_count * 0.00001, TINY_MAX_CLIP_BOUND)
        return np.clip(out.flatten(), -clip_bound, clip_bound)

    def predict_price_path(self, current_last_price, x):
        """
        根据当前金价与特征 x，推出极小模型的未来 6 点绝对价格路径
        """
        r_pred = self.predict_returns(x)
        cum_r = np.cumprod(1.0 + r_pred)
        return current_last_price * cum_r

    def fit_online(self, x, actual_returns):
        """
        EWA-Ridge 在线学习 (带 Exponential Decay 遗忘因子与 L2 脊正则):
        当历史预测对应的实际收益率到期并对齐特征快照后，触发此函数更新权重参数
        """
        if x is None or actual_returns is None or len(actual_returns) != self.future_steps:
            return

        x_vec = np.asarray(x, dtype=float).flatten()
        if len(x_vec) < self.n_features:
            x_vec = np.pad(x_vec, (0, self.n_features - len(x_vec)), mode='constant')
        elif len(x_vec) > self.n_features:
            x_vec = x_vec[:self.n_features]

        x_mat  = x_vec.reshape(1, -1)
        y_true = np.asarray(actual_returns, dtype=float).reshape(1, -1)

        # 指数遗忘衰减微调权重（赋予最新 K 线更大权重）
        decay_factor = 0.995
        self.W1 *= decay_factor
        self.W2 *= decay_factor

        # 前向传播
        h_pre, h, y_pred = self._forward(x_mat)
        err = y_pred - y_true  # (1, future_steps)

        # 自适应学习率衰减: lr_t = lr_init / sqrt(t+1)
        self.train_count += 1
        self.lr = float(self.lr_init / np.sqrt(self.train_count + 1))

        # 输出层梯度
        grad_W2 = h.T @ err + self.l2_reg * self.W2   # (HIDDEN, future_steps)
        grad_b2 = err.flatten()                         # (future_steps,)

        # 隐藏层梯度（Tanh 反向传播，利用已算好的 h）
        delta_h = (err @ self.W2.T) * _tanh_grad(h)     # (1, HIDDEN)
        grad_W1 = x_mat.T @ delta_h + self.l2_reg * self.W1  # (n_features, HIDDEN)
        grad_b1 = delta_h.flatten()                            # (HIDDEN,)

        # 全局 L2 梯度裁剪
        total_norm = np.sqrt(
            np.sum(grad_W1 ** 2) + np.sum(grad_b1 ** 2) +
            np.sum(grad_W2 ** 2) + np.sum(grad_b2 ** 2) + 1e-12
        )
        if total_norm > TINY_GRAD_CLIP:
            scale = TINY_GRAD_CLIP / total_norm
            grad_W1 *= scale
            grad_b1 *= scale
            grad_W2 *= scale
            grad_b2 *= scale

        # Momentum SGD 动量更新
        if not hasattr(self, 'v_W1'):
            self.v_W1 = np.zeros_like(self.W1)
            self.v_b1 = np.zeros_like(self.b1)
            self.v_W2 = np.zeros_like(self.W2)
            self.v_b2 = np.zeros_like(self.b2)

        self.v_W1 = TINY_MOMENTUM * self.v_W1 - self.lr * grad_W1
        self.v_b1 = TINY_MOMENTUM * self.v_b1 - self.lr * grad_b1
        self.v_W2 = TINY_MOMENTUM * self.v_W2 - self.lr * grad_W2
        self.v_b2 = TINY_MOMENTUM * self.v_b2 - self.lr * grad_b2

        self.W1 += self.v_W1
        self.b1 += self.v_b1
        self.W2 += self.v_W2
        self.b2 += self.v_b2

        # 记录当前与 EMA 平滑残差 MSE
        curr_mse = float(np.mean(err ** 2))
        self.last_mse = curr_mse
        if self.ema_mse == 0.0:
            self.ema_mse = curr_mse
        else:
            self.ema_mse = 0.8 * self.ema_mse + 0.2 * curr_mse

        # 每 3 次更新做一次数据库持久化
        if self.train_count % 3 == 0:
            self.save_model()

        logging.info(f"[TinyModel] 在线学习完成 (第 {self.train_count} 次, lr={self.lr:.5f})，残差 MSE: {self.last_mse:.8f} (EMA: {self.ema_mse:.8f})")

    def get_model_status(self):
        """返回模型自学习状态，供 API 与前端透出呈现。"""
        if self.train_count >= 5 and self.ema_mse > 0:
            quality = max(0.0, min(1.0, 1.0 - self.ema_mse / TINY_MSE_BASE))
            ensemble_w = int(round(10 + 40 * quality))
        else:
            ensemble_w = 10

        return {
            "train_count": self.train_count,
            "last_mse":    round(float(self.last_mse if self.last_mse > 0 else self.ema_mse), 8),
            "current_lr":  round(float(self.lr), 6),
            "ensemble_weight": ensemble_w
        }


if __name__ == "__main__":
    df_dummy = pd.DataFrame({
        'open':   np.linspace(880, 882, 30),
        'high':   np.linspace(880.5, 882.5, 30),
        'low':    np.linspace(879.5, 881.5, 30),
        'close':  np.linspace(880.1, 882.1, 30),
        'volume': np.random.randint(10, 100, 30)
    })
    feat = extract_features(df_dummy, 0.5)
    print("提取特征（Z-Score 归一化后）:", feat)
    model = TinyResidualPredictor()
    pred_r = model.predict_returns(feat)
    print("推断未来 6 步收益率:", pred_r)
    pred_p = model.predict_price_path(882.1, feat)
    print("推断未来 6 点金价:", pred_p)
    model.fit_online(feat, np.array([0.0001, 0.0002, 0.0001, -0.0001, 0.0, 0.0001]))
    print("在线学习测试完成，模型状态:", model.get_model_status())
