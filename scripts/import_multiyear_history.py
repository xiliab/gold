import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.config import SOURCE_TAG, CANDLE_INTERVAL_MINUTES
from src.db_init import get_db_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gold_prices.db")
OUNCES_TO_GRAMS = 31.1034768

def import_5year_history(db_path=DB_PATH):
    """
    通过 yfinance 拉取过去 5 年 (2020年至今) 的黄金行情 (GC=F) 和 美元/人民币汇率 (CNY=X)，
    换算为国内 元/克，对齐重采样后增量灌入 SQLite 数据库。
    """
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=5 * 365)
    
    start_str = start_dt.strftime('%Y-%m-%d')
    end_str = end_dt.strftime('%Y-%m-%d')
    
    logging.info(f"正在从 Yahoo Finance 下载过去 5 年 ({start_str} 至 {end_str}) 黄金及汇率历史数据...")
    
    # 1. 优先下载过去 2 年的小时级数据 (1h)
    df_hourly = yf.download(tickers=['GC=F', 'CNY=X'], period='2y', interval='1h', progress=False)
    
    # 2. 下载过去 5 年的日线数据 (1d) 补充 2020 - 2024 年的数据空白
    df_daily = yf.download(tickers=['GC=F', 'CNY=X'], start=start_str, end=end_str, interval='1d', progress=False)
    
    processed_dfs = []
    
    # 处理函数
    def process_raw_yf(df, freq_name):
        if df is None or df.empty:
            return None
        try:
            gold_c = df['Close']['GC=F'].ffill()
            gold_o = df['Open']['GC=F'].ffill()
            gold_h = df['High']['GC=F'].ffill()
            gold_l = df['Low']['GC=F'].ffill()
            gold_v = df['Volume']['GC=F'].fillna(0)
            cny_c  = df['Close']['CNY=X'].ffill().bfill()
        except KeyError:
            return None
            
        data = pd.DataFrame({
            'gold_open': gold_o,
            'gold_high': gold_h,
            'gold_low': gold_l,
            'gold_close': gold_c,
            'gold_vol': gold_v,
            'cny': cny_c
        }).dropna(subset=['gold_close', 'gold_open'])
        
        factor = data['cny'] / OUNCES_TO_GRAMS
        data['open']   = data['gold_open']  * factor
        data['high']   = data['gold_high']  * factor
        data['low']    = data['gold_low']   * factor
        data['close']  = data['gold_close'] * factor
        data['volume'] = data['gold_vol']
        
        return data[['open', 'high', 'low', 'close', 'volume']]

    df_h_proc = process_raw_yf(df_hourly, "1h")
    df_d_proc = process_raw_yf(df_daily, "1d")
    
    # 合并日线与小时线 (小时线优先)
    if df_d_proc is not None and not df_d_proc.empty:
        processed_dfs.append(df_d_proc)
    if df_h_proc is not None and not df_h_proc.empty:
        processed_dfs.append(df_h_proc)
        
    if not processed_dfs:
        logging.error("获取 5 年历史行情失败，请检查网络！")
        return 0
        
    combined = pd.concat(processed_dfs)
    combined = combined.reset_index()
    
    # 统一转换时区至 Asia/Shanghai
    dt_col = None
    for col in ['Datetime', 'Date', 'index', 'timestamp']:
        if col in combined.columns:
            dt_col = col
            break
    if dt_col:
        combined['dt'] = pd.to_datetime(combined[dt_col], utc=True)
    else:
        combined['dt'] = pd.to_datetime(combined.index, utc=True)

    combined['dt'] = combined['dt'].dt.tz_convert('Asia/Shanghai')
        
    combined = combined.sort_values('dt').drop_duplicates('dt').set_index('dt')
    
    logging.info(f"成功整理基础历史切片 {len(combined)} 条。正在执行 1m 连贯插值平滑...")
    
    # 优化 yfinance 拉取: 采用 period='2y'，避免偶发 730d 标签拒绝
    # 插值重采样至连续 1 分钟点
    resample_rule = f'{CANDLE_INTERVAL_MINUTES}min'
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    resampled = combined[numeric_cols].resample(resample_rule).interpolate(method='time').reset_index()
    resampled['timestamp'] = resampled['dt'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    rows = list(zip(
        resampled['timestamp'].astype(str),
        resampled['open'].round(4).astype(float),
        resampled['high'].round(4).astype(float),
        resampled['low'].round(4).astype(float),
        resampled['close'].round(4).astype(float),
        resampled['volume'].fillna(0).astype(float),
        [SOURCE_TAG] * len(resampled)
    ))
    
    logging.info(f"生成 5 年高密度 1m 历史 K 线 {len(rows)} 条！开始高速落库...")
    
    inserted = 0
    with get_db_connection(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT OR IGNORE INTO gold_prices (timestamp, open, high, low, close, volume, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows
        )
        inserted = cursor.rowcount
        
    logging.info(f"5 年历史行情导入完成！数据库新增 {inserted} 条记录！")
    return inserted

if __name__ == "__main__":
    import_5year_history()
