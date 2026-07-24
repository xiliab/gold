# 黄金行情预测与风控系统 Bug 诊断、修复与防误犯手册

本手册汇总记录系统开发迭代过程中出现的所有典型 Bug 故障现象、底层产生根源、修正方案及防误犯代码规范，避免后续二次踩坑。

---

## 📌 BUG-001：5m 骨架点数与 1m 分钟数单位混淆（长度缩水 5 倍导致图表断裂与陡插）

- **故障现象**：匹配相似度高达 95.6%，但在前端图表中紫色历史线左侧大面积断裂空白（缺失 700 个点），最右侧仅有十几个稀疏折点陡峭下插。
- **底层根源**：`current_N` 在 5m 骨架比对中代表的是 **5m 根数**（例如 176 点），代码按时间戳穿透查询时，误将 `176` 直接作为 1m 分钟数查数据库（仅查了 176 分钟 = 2.93 小时的数据，缺了 700 个点）。
- **终极解法**：在调取时间跨度时换算为真实的 1m 分钟数 `real_1m_N = current_N * 5 if current_N < 300 else current_N`，精准调取 892 个 1m 密集细节点。
- **代码规范约束**：
  在 [src/predictor.py](file:///Users/xiliab/Desktop/%E5%BC%80%E5%8F%91/gold/src/predictor.py) 中，任何接收 `current_N` 参数进行 1m 数据范围调取的逻辑，必须显式进行 `real_1m_N = current_N * 5` 单位换算！

---

## 📌 BUG-002：5m 骨架数组索引与 1m 原始行情数组索引错配（时间脱节 2.5 年）

- **故障现象**：紫色历史线高高悬挂在天花板上，整体走势与今日价格完全不符。
- **底层根源**：拿着 5m 骨架数组中的 `start_index`（如 328008）直接去 1m 数组切片，查出了 2022 年 3 月完全无关的历史价格。
- **终极解法**：匹配阶段导出 ISO 时间戳 `start_time`（如 `'2024-09-04 06:00:00'`），在 [src/predictor.py](file:///Users/xiliab/Desktop/%E5%BC%80%E5%8F%91/gold/src/predictor.py) 中通过 `fetch_1m_slice_by_range(start_time, end_time)` 按时间戳精准穿透。
- **代码规范约束**：
  严禁跨越 5m 骨架与 1m 原始行情两个不同的 NumPy 数组维度直接复用整数索引 `idx`；必须统一使用 ISO 时间戳做唯一语义索引！

---

## 📌 BUG-003：末端单点强行锚定导致的全局垂直拉扯变形（悬挂斜拉线）

- **故障现象**：紫色历史线前半段高悬天花板，后半段斜向暴跌链接到最新点。
- **底层根源**：末端单点强行锚定公式 `current_last_price + (t1_raw - anchor_hist_price)` 导致历史切片整体波幅被硬生生拉扯斜变形。
- **终极解法**：采用全天均值中枢平移对齐 (Global Mean-Level Alignment)：`aligned_arr = mu_today + (t1_raw - mu_hist)`，偏置控制在 0.0001 元。
- **代码规范约束**：
  在 [src/predictor.py](file:///Users/xiliab/Desktop/%E5%BC%80%E5%8F%91/gold/src/predictor.py) 中，对齐历史全天轨迹时必须使用全天均值中枢平移（`mu_today`），禁止使用末端单点强制死锁。

---

## 📌 BUG-004：Chart.js 默认 `spanGaps: true` 与前置 `None` 导致 45 度魔鬼斜线

- **故障现象**：图表最左侧出现一条直冲天际的 45 度紫/绿色斜拉线。
- **底层根源**：开端包含 `None` 占位时，Chart.js 默认连线会从画布 `(0, y_min)` 斜着连到第一个非空数值。
- **终极解法**：在 [static/app.js](file:///Users/xiliab/Desktop/%E5%BC%80%E5%8F%91/gold/static/app.js) 中所有 Line Dataset 显式配置 `spanGaps: false`。
- **代码规范约束**：
  所有 Chart.js Line Dataset 配置项中，一律必须显式声明 `spanGaps: false`。

---

## 📌 BUG-005：阴跌下探中的“粘性防抖死锁”与“逆势叫接刀”

- **故障现象**：价格持续阴跌，建议卡片却一直提示 `逢低吸纳 (BUY_DIP)`。
- **底层根源**：防抖缓冲区 `last_state == "BUY_DIP"` 锁死了状态，且缺少 15m 下跌斜率熔断。
- **终极解法**：拔除粘性死锁，引入 `Downward Slope Guard`（当 `realtime_15m_slope < -0.15` 时强行禁止 `BUY_DIP`）。
- **代码规范约束**：
  在 [src/predictor.py](file:///Users/xiliab/Desktop/%E5%BC%80%E5%8F%91/gold/src/predictor.py) 的 `action_advice` 引擎中，只要 `realtime_15m_slope < -0.15`，一律阻断 `BUY_DIP` 输出。

---

## 📌 BUG-006：单边主升浪中的“高位误判减仓”

- **故障现象**：单边拉升行情中，波段建议一路提示 `逢高减仓`。
- **底层根源**：`k_pos > 0.75`（处于日内高位区）盲目判定为“顶背离”。
- **终极解法**：部署 `Bullish Momentum Guard`（当 15m 斜率 >= +0.15 或三轨多头强共振时，主升浪拉升期绝对禁止叫卖减仓）。
- **代码规范约束**：
  在强多头格局（`realtime_15m_slope >= +0.15`）下，禁止输出 `SELL_RISK` / `STRONG_SELL`，保持顺势做多。

---

## 📌 BUG-007：1m 原始高频杂波导致 Pearson 相关系数打分虚低

- **故障现象**：波段形态很像，但匹配得分只有 65%~75%。
- **底层根源**：1m 原始行情充斥高频微观杂波毛刺，逐点计算 Pearson 系数时拉低了数学相关度。
- **终极解法**：引入 5m 极值保留趋势骨架 `gold_prices_skeleton_5m_cache.pkl`，消去高频杂波，匹配度跃升至 **93% ~ 95.6%**。
- **代码规范约束**：
  在 [src/matcher.py](file:///Users/xiliab/Desktop/%E5%BC%80%E5%8F%91/gold/src/matcher.py) 中，形态匹配统一基于 5m 平滑骨架 `close_smooth` 进行。

---

## 📌 BUG-008：app.py 渲染口 window_size 误传 5m 点数致历史轨迹被腰斩 745 点

- **故障现象**：后端算法生成的全脉络轨迹明明包含了 941 个点，但前端显示的紫色线开头 06:00~20:00 依然大面积断裂空白，只留尾部 180 点悬挂斜下插。
- **底层根源**：[app.py](file:///Users/xiliab/Desktop/%E5%BC%80%E5%8F%91/gold/app.py) 调用 `format_chart_payload` 时误将 5m 骨架点数 `current_N` (186 点) 作为 `window_size` 传参，致使 [src/charts.py](file:///Users/xiliab/Desktop/%E5%BC%80%E5%8F%91/gold/src/charts.py) 执行 `top1_raw[-192:]` 硬生生腰斩切掉了前面 745 个点（整整 14 小时真实数据）。
- **终极解法**：在 [app.py](file:///Users/xiliab/Desktop/%E5%BC%80%E5%8F%91/gold/app.py) 中修正传参为真正的 1m 分钟数 `real_window_size = current_N * 5`，使前端全脉络 941 个 1m 精细点零腰斩完整展现。
- **代码规范约束**：
  调用 `format_chart_payload` 时，`window_size` 必须与 `clean_df` 的采样维度 (1m) 保持绝对一致，严禁混用 5m 点数！

---

## 📌 BUG-009：app.route /api/gold/predict 内存全局变量 _prediction_cache 锁死旧数据

- **故障现象**：修改后端代码并重启服务后，前端界面拿到的依旧是上一个旧内存缓存打出的旧 JSON 响应。
- **底层根源**：[app.py](file:///Users/xiliab/Desktop/%E5%BC%80%E5%8F%91/gold/app.py) 的 `get_prediction_data()` 路由直接无条件解包 `_prediction_cache` 全局变量返回，导致即便修改了计算算法，前端请求依然命中旧的内存对象。
- **终极解法**：在 `get_prediction_data()` 中实时调用 `compute_prediction_payload(force_reload_data=False)` 获取绝对最新算出的图像 Payload 并同步写缓存。
- **代码规范约束**：
  API 路由不得硬读取未过期的旧全局缓存对象；必须确保算子能够响应实时计算逻辑！

---

## 📌 BUG-010：Chart.js responsive:true 冲刷覆盖 canvas 动态宽度致横向滚动失灵

- **故障现象**：虽在 JS 中设置了 `canvas.style.width`，但到了前台页面随着时间拉长，横坐标点数增多时依然越来越挤，水平滑动滚动条无法使用。
- **底层根源**：Chart.js 在初始化时开启了 `responsive: true`，每次调用 `goldChart.update()` 时会自动冲刷重置 `<canvas>` 的行内 CSS 宽度为 `100%`（即父容器的屏幕物理宽度 1200px），强行覆盖了 JS 设定的动态宽度。
- **终极解法**：在 [templates/index.html](file:///Users/xiliab/Desktop/%E5%BC%80%E5%8F%91/gold/templates/index.html) 中给 `<canvas>` 外层包裹独立容器 `#chartInnerContainer`；在 [static/app.js](file:///Users/xiliab/Desktop/%E5%BC%80%E5%8F%91/gold/static/app.js) 中通过控制 `#chartInnerContainer` 的物理宽度避开 Chart.js 冲刷，并按 `labelCount / 15` 动态生成整齐整点刻度。
- **代码规范约束**：
  控制 Chart.js 的响应式物理滚动，严禁直接给 `<canvas>` 设置行内 `style.width`；必须通过控制外层包裹 DOM 容器的物理 Width 实现！
