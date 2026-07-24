let goldChart = null;
let countdownTimer = null;
let timeLeft = 60; // 60 秒自动实时轮询
let lastRenderedTime = null;

// ── 平滑动画状态 ────────────────────────────────────────────
// 保存上一帧各数据集的数据，用于插值过渡
let prevDatasets = null;
let animFrameId = null;
const SMOOTH_DURATION = 800; // 平滑过渡总时长 (ms)
const SMOOTH_FPS = 60;       // 目标帧率
// ────────────────────────────────────────────────────────────

// ── 全局 DOM 缓存（避免 updateUI 每次调用都重复遍历 DOM 树）──────────
let ELEMS = {};

document.addEventListener('DOMContentLoaded', () => {
    // 一次性缓存所有常用 DOM 元素
    ELEMS = {
        countdown:         document.getElementById('countdown'),
        currentPrice:      document.getElementById('currentPrice'),
        currentTime:       document.getElementById('currentTime'),
        mapeText:          document.getElementById('mapeText'),
        statusText:        document.getElementById('statusText'),
        todayLow:          document.getElementById('todayLow'),
        todayLowSub:       document.getElementById('todayLowSub'),
        todayHigh:         document.getElementById('todayHigh'),
        todayHighSub:      document.getElementById('todayHighSub'),
        directionLabel:    document.getElementById('directionLabel'),
        directionConf:     document.getElementById('directionConfidence'),
        upVotes:           document.getElementById('upVotes'),
        downVotes:         document.getElementById('downVotes'),
        flatVotes:         document.getElementById('flatVotes'),
        actionAdvice:      document.getElementById('actionAdvice'),
        adviceBadges:      document.getElementById('adviceBadges'),
        waveTimeWindow:    document.getElementById('waveTimeWindow'),
        waveProfitSpace:   document.getElementById('waveProfitSpace'),
        waveRiskReward:    document.getElementById('waveRiskReward'),
        adviceSummaryText: document.getElementById('adviceSummaryText'),
        winRate:           document.getElementById('winRateText'),
        dynamicK:          document.getElementById('dynamicKText'),
        tinyTrainCount:    document.getElementById('tinyTrainCount'),
        tinyMse:           document.getElementById('tinyMse'),
        tinyWeightText:    document.getElementById('tinyWeightText'),
        aiLogsContainer:   document.getElementById('aiLogsContainer'),
    };
    initChart();
    fetchData();
    startCountdown();

    // Tab 切换时正确管理动画帧，防止帧积压和 GC 抖动
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            if (animFrameId) {
                cancelAnimationFrame(animFrameId);
                animFrameId = null;
            }
        }
        // 切回时不需要手动重启，下次 fetchData 会自然触发动画
    });
});

function startCountdown() {
    if (countdownTimer) clearInterval(countdownTimer);
    timeLeft = 60;
    countdownTimer = setInterval(() => {
        timeLeft--;
        if (timeLeft <= 0) {
            timeLeft = 60;
            fetchData();
            return;
        }
        const s = String(timeLeft).padStart(2, '0');
        if (ELEMS.countdown) ELEMS.countdown.innerText = `00:${s}`;
    }, 1000);
}

function initChart() {
    const ctx = document.getElementById('goldChart').getContext('2d');
    goldChart = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            // 禁用 Chart.js 内置动画，改由我们自己的平滑插值逐帧驱动
            animation: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#f8fafc',
                    bodyColor: '#cbd5e1',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label === '波段建议色带') {
                                const types = context.chart._adviceTypes || [];
                                const type = types[context.dataIndex];
                                if (type === 'STRONG_BUY')  return '💡 波段建议历史: 🟢 强烈建议买入';
                                if (type === 'BUY_DIP')     return '💡 波段建议历史: 🟢 建议逢低吸纳';
                                if (type === 'STRONG_SELL') return '💡 波段建议历史: 🔴 强烈建议卖出';
                                if (type === 'SELL_RISK')   return '💡 波段建议历史: 🔴 建议逢高减仓';
                                return '💡 波段建议历史: 🟡 建议观望等待';
                            }
                            if (label) label += ': ';
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(2) + ' 元/克';
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8', maxTicksLimit: 12, font: { size: 11 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: {
                        color: '#94a3b8',
                        font: { size: 11 },
                        callback: function(value) {
                            return value.toFixed(2) + ' 元';
                        }
                    }
                },
                yAdvice: {
                    display: false,
                    min: -2,
                    max: 2,
                    position: 'right',
                    grid: { display: false }
                }
            }
        }
    });
}

async function fetchData() {
    try {
        // Reset countdown if fetched manually
        startCountdown();
        
        // Add cache-busting parameter to prevent browser from returning old data
        const resp = await fetch(`/api/gold/predict?_t=${new Date().getTime()}`);
        const json = await resp.json();
        if (json.status === 'success') {
            updateUI(json);
        }
    } catch (e) {
        console.error('获取预测数据失败:', e);
    }
}

/**
 * 对单条数据线做数值插值（跳过 null）
 * @param {Array} from 上一帧数据
 * @param {Array} to   目标数据
 * @param {number} t   插值进度 [0, 1]
 */
function lerpDataset(from, to, t) {
    const len = Math.max(from.length, to.length);
    const result = new Array(len);
    for (let i = 0; i < len; i++) {
        const a = i < from.length ? from[i] : null;
        const b = i < to.length ? to[i] : null;
        if (a !== null && b !== null && typeof a === 'number' && typeof b === 'number') {
            result[i] = a + (b - a) * t;
        } else {
            // 只要有一端是 null，直接用目标端（或 null）
            result[i] = b;
        }
    }
    return result;
}

/**
 * 平滑过渡到新数据集
 * 若标签数量或数据集数量发生变化（如新交易日开始），直接无动画切换；
 * 否则对每条线做逐帧数值插值，实现平滑滑入效果。
 */
function adjustChartWidthAndScroll(labelCount) {
    const wrapper = document.getElementById('chartScrollWrapper');
    const innerContainer = document.getElementById('chartInnerContainer') || document.getElementById('goldChart');
    if (!wrapper || !innerContainer) return;

    // 每一个 1m 时间点维持至少 4 像素间距，呈现紧凑高密度宏观视野
    const minPointSpacing = 4;
    const neededWidth = Math.max(wrapper.clientWidth, labelCount * minPointSpacing);
    innerContainer.style.width = neededWidth + 'px';

    // 自动平滑滚动最右侧（展示最新价格与预测）
    setTimeout(() => {
        wrapper.scrollLeft = wrapper.scrollWidth;
    }, 50);
}

function smoothTransitionTo(newLabels, newDatasets, yMin, yMax) {
    adjustChartWidthAndScroll(newLabels.length);

    // 动态按 15 分钟间隔计算 maxTicksLimit，保证滚动时时间刻度清晰整齐
    if (goldChart && goldChart.options && goldChart.options.scales && goldChart.options.scales.x) {
        goldChart.options.scales.x.ticks.maxTicksLimit = Math.max(12, Math.ceil(newLabels.length / 15));
    }

    // 取消上一个进行中的动画帧
    if (animFrameId !== null) {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
    }

    const isFirstRender = prevDatasets === null;
    const labelsChanged = !goldChart.data.labels.length ||
        goldChart.data.labels.length !== newLabels.length;
    const dsCountChanged = goldChart.data.datasets.length !== newDatasets.length;

    // 首次渲染 / 结构变化：直接写入，无动画
    if (isFirstRender || labelsChanged || dsCountChanged) {
        goldChart.data.labels = newLabels;
        goldChart.data.datasets = newDatasets;
        goldChart.options.scales.y.min = yMin;
        goldChart.options.scales.y.max = yMax;
        goldChart.update('none');
        prevDatasets = newDatasets.map(ds => ({ data: [...(ds.data || [])] }));
        adjustChartWidthAndScroll(newLabels.length);
        return;
    }

    // 记录插值起点 (当前图表实际数据，避免上一帧动画还没完就被打断产生跳变)
    const fromDatasets = goldChart.data.datasets.map((ds, i) => ({
        data: [...(ds.data || [])]
    }));
    const fromYMin = goldChart.options.scales.y.min;
    const fromYMax = goldChart.options.scales.y.max;

    const startTime = performance.now();

    function frame(now) {
        const elapsed = now - startTime;
        // easeOutCubic 缓动：先快后慢，像曲线自然滑动到目标位
        const rawT = Math.min(elapsed / SMOOTH_DURATION, 1.0);
        const t = 1 - Math.pow(1 - rawT, 3);

        // 插值各数据集
        goldChart.data.labels = newLabels;
        newDatasets.forEach((ds, i) => {
            if (i < goldChart.data.datasets.length) {
                goldChart.data.datasets[i].data = lerpDataset(
                    fromDatasets[i] ? fromDatasets[i].data : [],
                    ds.data || [],
                    t
                );
            }
        });

        // Y 轴范围平滑过渡
        goldChart.options.scales.y.min = fromYMin + (yMin - fromYMin) * t;
        goldChart.options.scales.y.max = fromYMax + (yMax - fromYMax) * t;

        goldChart.update('none');

        if (rawT < 1.0) {
            animFrameId = requestAnimationFrame(frame);
        } else {
            // 动画结束，写入精确目标值
            newDatasets.forEach((ds, i) => {
                if (i < goldChart.data.datasets.length) {
                    goldChart.data.datasets[i].data = ds.data || [];
                }
            });
            goldChart.options.scales.y.min = yMin;
            goldChart.options.scales.y.max = yMax;
            goldChart.update('none');
            prevDatasets = newDatasets.map(ds => ({ data: [...(ds.data || [])] }));
            animFrameId = null;
        }
    }

    animFrameId = requestAnimationFrame(frame);
}

function updateUI(payload) {
    if (!payload || !payload.chart_data) return;
    const data = payload.chart_data;
    const metrics = payload.metrics || {};

    // 1. 更新顶部与侧边栏指标卡片
    // 操作建议大字渲染与配色
    
    // 卡片5: 操作建议 (波段获利/避险导向)
    if (ELEMS.actionAdvice) {
        ELEMS.actionAdvice.textContent = metrics.action_advice || '暂无建议';
        const adviceType = metrics.advice_type || 'WAIT';
        const adviceElem = ELEMS.actionAdvice;
        
        if (adviceType === 'STRONG_BUY' || adviceType === 'BUY_DIP') {
            adviceElem.style.background = 'rgba(16, 185, 129, 0.2)';
            adviceElem.style.color = '#10b981';
            adviceElem.style.border = '1px solid rgba(16, 185, 129, 0.4)';
            adviceElem.style.boxShadow = '0 0 14px rgba(16, 185, 129, 0.25)';
        } else if (adviceType === 'STRONG_SELL' || adviceType === 'SELL_RISK') {
            adviceElem.style.background = 'rgba(239, 68, 68, 0.2)';
            adviceElem.style.color = '#ef4444';
            adviceElem.style.border = '1px solid rgba(239, 68, 68, 0.4)';
            adviceElem.style.boxShadow = '0 0 14px rgba(239, 68, 68, 0.25)';
        } else {
            adviceElem.style.background = 'rgba(245, 158, 11, 0.15)';
            adviceElem.style.color = '#f59e0b';
            adviceElem.style.border = '1px solid rgba(245, 158, 11, 0.3)';
            adviceElem.style.boxShadow = 'none';
        }

        // 解析波段 advice 结构
        if (metrics.advice && ELEMS.adviceBadges) {
            ELEMS.adviceBadges.style.display = 'flex';
            
            const adv = metrics.advice;
            if (ELEMS.waveTimeWindow) {
                ELEMS.waveTimeWindow.innerText = adv.time_window || '未来 1 小时内';
            }
            if (ELEMS.waveProfitSpace) {
                ELEMS.waveProfitSpace.innerText = adv.expected_profit || '--';
                if (adviceType.includes('BUY')) {
                    ELEMS.waveProfitSpace.style.color = 'var(--green-glow)';
                } else if (adviceType.includes('SELL')) {
                    ELEMS.waveProfitSpace.style.color = 'var(--red-glow)';
                } else {
                    ELEMS.waveProfitSpace.style.color = '#f59e0b';
                }
            }
            if (ELEMS.waveRiskReward) {
                const rr = adv.risk_reward_ratio;
                ELEMS.waveRiskReward.innerText = rr !== undefined ? `${rr.toFixed(1)}x` : '--';
                if (rr >= 2.0) {
                    ELEMS.waveRiskReward.style.color = 'var(--green-glow)';
                } else if (rr < 1.0) {
                    ELEMS.waveRiskReward.style.color = 'var(--red-glow)';
                } else {
                    ELEMS.waveRiskReward.style.color = '#94a3b8';
                }
            }
            if (ELEMS.adviceSummaryText && adv.summary) {
                ELEMS.adviceSummaryText.innerText = adv.summary;
            }
        } else if (ELEMS.adviceBadges) {
            ELEMS.adviceBadges.style.display = 'none';
        }
    }

    // 方向置信度
    if (ELEMS.directionLabel) ELEMS.directionLabel.textContent = metrics.direction_label || '无数据';
    if (ELEMS.directionConf)  ELEMS.directionConf.textContent  = `${(metrics.direction_confidence * 100).toFixed(0)}%`;
    if (ELEMS.upVotes)    ELEMS.upVotes.textContent   = metrics.up_votes   || 0;
    if (ELEMS.downVotes)  ELEMS.downVotes.textContent = metrics.down_votes || 0;
    if (ELEMS.flatVotes)  ELEMS.flatVotes.textContent = metrics.flat_votes || 0;

    // 胜率与自学防守
    if (ELEMS.winRate  && metrics.win_rate  !== undefined) ELEMS.winRate.textContent  = `${metrics.win_rate.toFixed(1)}%`;
    if (ELEMS.dynamicK && metrics.dynamic_k !== undefined) ELEMS.dynamicK.textContent = metrics.dynamic_k.toFixed(2);

    // 极小自适应模型状态 rendering
    if (metrics.tiny_status) {
        const ts = metrics.tiny_status;
        if (ELEMS.tinyTrainCount) ELEMS.tinyTrainCount.textContent = ts.train_count || 0;
        if (ELEMS.tinyMse) {
            const lr = ts.current_lr !== undefined ? `  lr=${ts.current_lr}` : '';
            ELEMS.tinyMse.textContent = `${(ts.last_mse || 0).toFixed(4)}${lr}`;
        }
        if (ELEMS.tinyWeightText) ELEMS.tinyWeightText.textContent = `主导权重 ${ts.ensemble_weight || 60}%`;
    }
    if (metrics.ai_logs && metrics.ai_logs.length > 0) {
        const logsHtml = metrics.ai_logs.map(log =>
            `<div style="padding: 8px; background: rgba(0,0,0,0.2); border-radius: 6px; border-left: 2px solid var(--purple-glow, #a855f7);">${log}</div>`
        ).join('');
        if (ELEMS.aiLogsContainer) ELEMS.aiLogsContainer.innerHTML = logsHtml;
    }

    if (ELEMS.currentPrice && metrics.current_price) {
        ELEMS.currentPrice.innerText = `${metrics.current_price.toFixed(2)} 元/克`;
    }

    if (data.current_time) {
        if (ELEMS.currentTime) {
            // 保留月日与时间 (例如 07-22 07:49:00)，避免隐去日期导致用户误以为停留在昨日
            let t_fmt = data.current_time;
            if (data.current_time.length >= 19) {
                t_fmt = data.current_time.substring(5, 19); // 截取 07-22 07:49:00
            }
            ELEMS.currentTime.innerText = `最新时间: ${t_fmt}`;
        }
    }

    if (ELEMS.mapeText && metrics.mape !== undefined) {
        ELEMS.mapeText.innerText = `近15m 偏差 MAPE: ${metrics.mape}%`;
    }

    // 2. statusText更新
    if (ELEMS.statusText) {
        if (metrics.need_rematch) {
            ELEMS.statusText.innerText = '触发即时重匹配中...';
            ELEMS.statusText.style.color = '#ef4444';
        } else {
            ELEMS.statusText.innerText = '运算引擎运行正常';
            ELEMS.statusText.style.color = '#10b981';
        }
    }

    // 3. 卡片3 & 4: 今日剩余时间预测最低 / 最高价
    if (metrics.rest_of_day_low !== undefined && metrics.rest_of_day_high !== undefined && metrics.current_price) {
        const projLow  = metrics.rest_of_day_low;
        const projHigh = metrics.rest_of_day_high;

        // 卡片3: 最低价
        if (ELEMS.todayLow) ELEMS.todayLow.innerText = `${projLow.toFixed(2)} 元/克`;
        if (ELEMS.todayLowSub) {
            const lowDiff = metrics.current_price - projLow;
            const timeStr = metrics.rest_of_day_low_time ? `⏰ 预估 ${metrics.rest_of_day_low_time}` : '';
            const spaceStr = lowDiff > 0.01 ? `下行 ${lowDiff.toFixed(2)}元` : '接近最低点';
            ELEMS.todayLowSub.innerText = `${timeStr} · ${spaceStr}`;
        }

        // 卡片4: 最高价
        if (ELEMS.todayHigh) ELEMS.todayHigh.innerText = `${projHigh.toFixed(2)} 元/克`;
        if (ELEMS.todayHighSub) {
            const highDiff = projHigh - metrics.current_price;
            const timeStr = metrics.rest_of_day_high_time ? `⏰ 预估 ${metrics.rest_of_day_high_time}` : '';
            const spaceStr = highDiff > 0.01 ? `上行 ${highDiff.toFixed(2)}元` : '接近最高点';
            ELEMS.todayHighSub.innerText = `${timeStr} · ${spaceStr}`;
        }
    }

    // 4. 预测趋势卡片2
    const trendElem = document.getElementById('forecastTrend');
    const targetElem = document.getElementById('forecastTarget');
    if (trendElem && targetElem) {
        if (!data.predicted_prices || metrics.valid_count === 0) {
            trendElem.innerText = '暂无参考 (无 ≥70% 匹配)';
            trendElem.className = 'card-value';
            trendElem.style.fontSize = '18px';
            trendElem.style.color = '#94a3b8';
            targetElem.innerText = '历史样本中缺乏足够高相似度轨迹';
        } else if (data.predicted_prices && data.predicted_prices.length > 0) {
            const lastReal = metrics.current_price || data.current_price;
            const lastPred = data.predicted_prices[data.predicted_prices.length - 1];
            const diffAmt  = lastPred - lastReal;
            const diffPct  = (diffAmt / lastReal) * 100.0;
            trendElem.style.fontSize = '22px';
            targetElem.innerText = `未来 6m 目标价: ${lastPred.toFixed(2)} 元/克`;
            if (diffAmt > 0.05) {
                trendElem.innerText = `▲ 看涨 (+${diffAmt.toFixed(2)}元)`;
                trendElem.className = 'card-value price-up';
            } else if (diffAmt < -0.05) {
                trendElem.innerText = `▼ 看跌 (${diffAmt.toFixed(2)}元)`;
                trendElem.className = 'card-value price-down';
            } else {
                trendElem.innerText = `► 横盘震荡 (${diffAmt >= 0 ? '+' : ''}${diffAmt.toFixed(2)}元)`;
                trendElem.className = 'card-value';
            }
        }
    }

    let focusValues = [];
    if (data.real_prices) focusValues.push(...data.real_prices.filter(v => v !== null && typeof v === 'number'));
    if (data.predicted_prices) focusValues.push(...data.predicted_prices.filter(v => v !== null && typeof v === 'number'));
    if (data.historical_pred_prices) focusValues.push(...data.historical_pred_prices.filter(v => v !== null && typeof v === 'number'));

    let yMin = goldChart.options.scales.y.min;
    let yMax = goldChart.options.scales.y.max;
    if (focusValues.length > 0) {
        const todayMin = Math.min(...focusValues);
        const todayMax = Math.max(...focusValues);
        const todayDiff = todayMax - todayMin;
        const margin = Math.max(todayDiff * 0.08, 0.15);
        yMin = Number((todayMin - margin).toFixed(2));
        yMax = Number((todayMax + margin).toFixed(2));
    }

    // 4. 组装数据集 (结构与 label 不变，只更新 data 用于平滑插值)
    lastRenderedTime = data.current_time;
    if (goldChart) {
        goldChart._adviceTypes = data.historical_advice_types || [];
    }

    const datasets = [
        {
            label: '波段建议色带',
            data: (data.historical_advice_types || []).map(() => -1.85),
            borderColor: 'rgba(16, 185, 129, 0.5)',
            segment: {
                borderColor: ctx => {
                    const types = data.historical_advice_types || [];
                    const type = types[ctx.p0DataIndex];
                    if (type === 'STRONG_BUY' || type === 'BUY_DIP') return 'rgba(16, 185, 129, 0.65)';
                    if (type === 'STRONG_SELL' || type === 'SELL_RISK') return 'rgba(239, 68, 68, 0.65)';
                    return 'rgba(245, 158, 11, 0.30)';
                }
            },
            borderWidth: 10,
            tension: 0,
            pointRadius: 0,
            yAxisID: 'yAdvice',
            fill: false
        },
        {
            label: '今日实际金价',
            data: data.real_prices || [],
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            borderWidth: 2.5,
            tension: 0.2,
            pointRadius: 0,
            spanGaps: false,
            fill: true
        },
        {
            label: 'Top 1 历史同频走势 (起点对齐)',
            data: data.top1_aligned_prices || [],
            borderColor: 'rgba(168, 85, 247, 0.85)',
            borderWidth: 2.0,
            borderDash: [4, 4],
            tension: 0.2,
            pointRadius: 0,
            spanGaps: false,
            fill: false
        },
        {
            label: '历史预测曲线',
            data: data.historical_pred_prices || [],
            borderColor: 'rgba(234, 179, 8, 0.6)', // 黄色虚线，半透明不影响真实曲线
            borderWidth: 1.5,
            borderDash: [4, 4],
            tension: 0.2,
            pointRadius: 0,
            spanGaps: false,
            fill: false
        },
        {
            label: '未来预测走势',
            data: data.predicted_prices || [],
            borderColor: '#ef4444',
            borderWidth: 2.5,
            borderDash: [5, 5],
            tension: 0.2,
            pointRadius: 3,
            pointBackgroundColor: '#ef4444',
            fill: false
        },
        {
            label: '置信区间上轨',
            data: data.upper_bound || [],
            borderColor: 'rgba(239, 68, 68, 0.25)',
            borderWidth: 1,
            pointRadius: 0,
            fill: '+1',
            backgroundColor: 'rgba(239, 68, 68, 0.08)'
        },
        {
            label: '置信区间下轨',
            data: data.lower_bound || [],
            borderColor: 'rgba(239, 68, 68, 0.25)',
            borderWidth: 1,
            pointRadius: 0,
            fill: false
        }
    ];

    if (data.history_tracks) {
        const colors = ['#f59e0b', '#3b82f6', '#8b5cf6'];
        data.history_tracks.forEach((track, i) => {
            datasets.push({
                label: `参考历史天 #${track.rank} (${track.score}%)`,
                data: track.data || [],
                borderColor: colors[i % colors.length],
                borderWidth: 1.5,
                borderDash: [3, 3],
                pointRadius: 0,
                fill: false
            });
        });
    }

    // 5. 触发平滑过渡动画（不重建图表，只对数值做帧级插值）
    if (goldChart) {
        smoothTransitionTo(data.labels || [], datasets, yMin, yMax);
    }

    // 6. 渲染 Top 3 历史轨迹卡片
    renderHistoryCards(data.history_tracks);
}

function renderHistoryCards(tracks) {
    const grid = document.getElementById('historyGrid');
    if (!grid) return;
    if (!tracks || tracks.length === 0) {
        grid.innerHTML = '<div style="color: #94a3b8;">暂无历史匹配数据</div>';
        return;
    }

    grid.innerHTML = tracks.map(t => {
        const dateStr = t.start_time ? t.start_time.split(' ')[0] : '';
        return `
        <div class="history-card">
            <div class="history-header">
                <span class="rank-badge">Top ${t.rank} 最相似历史日</span>
                <span class="similarity-score">相似度 ${t.score}%</span>
            </div>
            <div class="history-time">历史交易日: ${dateStr}</div>
        </div>
        `;
    }).join('');
}
