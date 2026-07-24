from datetime import datetime, timedelta

def format_chart_payload(clean_df, prediction_result, window_size=284, historical_pred_map=None, historical_advice_map=None):
    """
    将按交易日（284 个 1m 点 = 1 完整交易日）对齐的实测数据、预测曲线及 Top 3 历史交易天轨迹格式化为 Chart.js。
    historical_pred_map: dict[timestamp_str -> predicted_price]
    historical_advice_map: dict[timestamp_str -> advice_type]
    """
    if clean_df is None or clean_df.empty:
        return {}

    # 取最近 1 个完整交易日 (284 个 1m 点) 作为当前查询天
    history_slice = clean_df.tail(window_size)
    history_timestamps = history_slice['timestamp'].tolist()
    history_prices = history_slice['close'].round(2).tolist()

    last_time_str = history_timestamps[-1]
    try:
        last_dt = datetime.strptime(last_time_str, '%Y-%m-%d %H:%M:%S')
    except Exception:
        last_dt = datetime.now()

    # 未来 6 个 1m 时间戳标签 (%H:%M，覆盖未来 6 分钟)
    future_timestamps = [
        (last_dt + timedelta(minutes=1 * i)).strftime('%H:%M')
        for i in range(1, 7)
    ]
    
    # 格式化历史点的时间轴刻度
    history_time_labels = []
    for t in history_timestamps:
        try:
            dt_obj = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
            if dt_obj.date() == last_dt.date():
                history_time_labels.append(dt_obj.strftime('%H:%M'))
            else:
                history_time_labels.append(dt_obj.strftime('%m-%d %H:%M'))
        except Exception:
            history_time_labels.append(t)

    all_labels = history_time_labels + future_timestamps

    # 填充实测价格线 (未来 6 点填充 None)
    real_line = history_prices + [None] * 6

    # 历史预测价格对比线（由调用层传入，此处无 IO 操作）
    pred_map = historical_pred_map or {}
    historical_pred_line = [pred_map.get(ts, None) for ts in history_timestamps]

    # 历史预测价格线同样补齐未来 6 个点
    historical_pred_line.extend([None] * 6)

    # 预测价格线
    if prediction_result and prediction_result.get('predicted_prices') is not None:
        pred_line = [None] * (len(history_prices) - 1) + [history_prices[-1]] + prediction_result['predicted_prices']
        upper_line = [None] * (len(history_prices) - 1) + [history_prices[-1]] + prediction_result['upper_bound']
        lower_line = [None] * (len(history_prices) - 1) + [history_prices[-1]] + prediction_result['lower_bound']
        history_tracks = prediction_result.get('history_tracks', [])
    else:
        pred_line = [None] * (len(history_prices) + 6)
        upper_line = [None] * (len(history_prices) + 6)
        lower_line = [None] * (len(history_prices) + 6)
        history_tracks = []

    track_lines = []
    for track in history_tracks:
        track_line = [None] * (len(history_prices) - 1) + [history_prices[-1]] + track['future_prices']
        track_lines.append({
            'rank': track['rank'],
            'score': track['score'],
            'start_time': track['start_time'],
            'end_time': track['end_time'],
            'data': track_line
        })

    # 未来 6 个 1m 完整的 ISO 时间戳 (%Y-%m-%d %H:%M:%S)
    future_full_timestamps = [
        (last_dt + timedelta(minutes=1 * i)).strftime('%Y-%m-%d %H:%M:%S')
        for i in range(1, 7)
    ]

    top1_raw = prediction_result.get('top1_aligned_track', []) if prediction_result else []
    target_len = len(history_prices) + 6
    if top1_raw and len(top1_raw) >= target_len:
        top1_line = top1_raw[-target_len:]
    elif top1_raw:
        # 06:00 之前的非匹配区间统一填充 None，配合前端 spanGaps: false 实现 06:00 前干脆断开、06:00 后呈现逼真波浪线
        pad_len = target_len - len(top1_raw)
        top1_line = [None] * pad_len + top1_raw
    else:
        top1_line = [None] * target_len

    advice_map = historical_advice_map or {}
    historical_advice_types = [advice_map.get(ts, 'WAIT') for ts in history_timestamps]
    historical_advice_types.extend(['WAIT'] * 6)

    return {
        'labels': all_labels,
        'future_full_timestamps': future_full_timestamps,
        'real_prices': real_line,
        'predicted_prices': pred_line,
        'historical_pred_prices': historical_pred_line,
        'top1_aligned_prices': top1_line,
        'historical_advice_types': historical_advice_types,
        'upper_bound': upper_line,
        'lower_bound': lower_line,
        'history_tracks': track_lines,
        'current_price': history_prices[-1],
        'current_time': last_time_str
    }
