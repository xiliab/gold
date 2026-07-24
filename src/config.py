# 系统全局配置文件 — 所有超参数和魔法数字统一管理于此

# ── 数据源 ────────────────────────────────────────────
SOURCE_TAG = 'YFinance_GC_CNY_1M'       # 数据库中 1m 行情的来源标记
FETCH_INTERVAL_SECONDS = 60             # 后台自动抓取行情的时间间隔（秒）

# ── 时间节点 ──────────────────────────────────────────
CANDLE_INTERVAL_MINUTES = 1             # K 线时间粒度（分钟）
FUTURE_STEPS = 6                        # 预测未来步数 = 6 分钟
WINDOW_SIZE = 284                       # 图表展示的最近点数（约半个交易日）
MIN_POINTS_REQUIRED = 284              # 启动预测所需最少历史点数
EARLY_SESSION_MIN_POINTS = 60          # 早盘容错最少点数（避免噪音）
CLOSE_HOUR = 5                          # 次日凌晨 5 点为收盘时间（Asia/Shanghai）
TRADING_DAY_START_OFFSET_HOURS = 6     # 用于区分交易日的时间偏移（06:00 为新日开始）
GAP_THRESHOLD_MINUTES = 15             # 相邻数据点时间差超过此值则认为是断层

# ── 匹配算法 ──────────────────────────────────────────
NMS_RADIUS = 284                        # 非极大值抑制半径（同等于 1 个完整交易日点数）
MIN_SIMILARITY = 70.0                   # 最低相似度门槛（%）
DECAY_RATE = 0.0                        # 时间衰减率（0.0 为全序列均匀等权比对，从06:00至今全过程对齐）
MATCH_ALPHA = 0.5                       # 匹配权重中历史权重占比
SMOOTH_WINDOW_SIZE = 5                  # 趋势平滑窗口（抹平 1m 噪音，防过度拘泥于微观毛刺）

# ── 预测模型 ──────────────────────────────────────────
SOFTMAX_TEMPERATURE = 10.0              # Softmax 温度参数，控制权重集中程度
FLAT_DIRECTION_THRESHOLD = 0.15        # 价格变动小于此值（元）时方向判定为 FLAT
UP_DOWN_RETURN_THRESHOLD = 0.0001      # 累计收益率高于此值才计票为 UP/DOWN

# ── 自学习系统 ────────────────────────────────────────
EMA_ALPHA = 0.6                         # EMA 平滑因子（越大越快响应新数据）
HIGH_BIAS_EMA_FAST = 0.3               # 极值高点超调后的快速 EMA 权重
HIGH_BIAS_EMA_SLOW = 0.7               # 极值高点未超调时的慢速 EMA 权重
HIGH_BIAS_TRIGGER_OVERSHOOT = 3.0      # 预测最高价高于实际超过此值（元）才触发慢速修正
LOW_BIAS_EMA_FAST = 0.3
LOW_BIAS_EMA_SLOW = 0.7
LOW_BIAS_TRIGGER_UNDERSHOOT = 3.0
DIRECTION_CONF_WIN_RATE_FLOOR = 0.3    # 胜率因子下限（防止置信度被过度惩罚）
DYNAMIC_K_HIGH_WIN = 75.0              # 胜率高于此值时缩小防守系数
DYNAMIC_K_LOW_WIN = 50.0               # 胜率低于此值时扩大防守系数
DYNAMIC_K_AGGRESSIVE = 0.9             # 激进系数
DYNAMIC_K_DEFENSIVE = 1.4              # 防守系数
DYNAMIC_K_NORMAL = 1.0                 # 常规系数

# --- 新增：操作建议风控参数 ---
ADVICE_RISK_REWARD_HIGH = 2.0          # 高盈亏比阈值
ADVICE_RISK_REWARD_LOW = 1.5           # 低盈亏比阈值
ADVICE_POSITION_HIGH = 0.75            # 高位区阈值
ADVICE_POSITION_LOW = 0.25             # 低位区阈值
ADVICE_MIN_DIRECTION_CONF = 0.50       # 短线信号最低置信度
ADVICE_STRONG_CONF = 0.70              # 强信号置信度门槛
ADVICE_MIN_WIN_RATE_WEIGHT = 0.35      # 胜率熔断阈值（direction_confidence * win_rate）
ADVICE_FLAT_DIRECTION = 0.00015        # 震荡识别门槛，波动门槛 (对应黄金约 0.13 元/克的物理震荡噪声界限)
WIN_RATE_WINDOW = 20                   # 计算胜率时参考最近 N 条记录
MAPE_REMATCH_THRESHOLD = 0.005         # MAPE 超过此值时触发即时重匹配

# --- TinyModel 增量自学习优化参数 ---
TINY_MOMENTUM = 0.9                    # SGD 动量衰减系数
TINY_GRAD_CLIP = 1.0                   # 全局 Total L2 梯度裁剪阈值
TINY_MSE_BASE = 1e-4                   # 评估 TinyModel 质量的基准 MSE (对应 100% 质量得分)
TINY_MAX_CLIP_BOUND = 0.0008           # 微观模型单分钟渐进式收益率离散裁剪上限 (±0.08% ≈ ±0.70 元/克/分钟，符合国内黄金物理波幅上限)

# ── 机构行为影响因子 ────────────────────────────────────
INSTITUTIONAL_FACTOR_WEIGHT = 0.35      # 机构行为对预测曲线的综合影响权重
VOLUME_IMBALANCE_WINDOW = 30           # 计算日内主力买卖量能偏向的 K 线窗口数
SPDR_WEIGHT = 0.6                      # SPDR 黄金持仓变动的相对权重
VOLUME_FLOW_WEIGHT = 0.4               # 日内主力成交量偏向的相对权重

# ── 预测模型魔法参数 (风控与平滑) ───────────────────────
INST_DRIFT_FACTOR = 0.04               # 机构漂移因子系数
BIG_MOVE_THRESH_BASE = 1.20            # 大行情判断基准阈值（元/克）
BIG_MOVE_MIN_MULTIPLIER = 0.5          # 动态阈值最小乘数
BIG_MOVE_MAX_MULTIPLIER = 2.0          # 动态阈值最大乘数
