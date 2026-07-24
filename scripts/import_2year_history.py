import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.config import SOURCE_TAG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gold_prices.db")

def import_2year_gold_history(db_path=DB_PATH):
    """
    通过 yfinance 一次性拉取过去 2 年 (720天) 的黄金小时级 K 线，
    插值还原为标准 1 分钟 K 线并增量灌入 SQLite 数据库。
    """
    import yfinance as yf
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=720)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    logging.info(f"正在从 yfinance 下载过去 2 年 ({start_str} 至 {end_str}) 黄金历史行情...")
    
    # 抓取 COMEX 黄金期货 (GC=F) 1小时级数据
    df = yf.download('GC=F', start=start_str, end=end_str, interval='1h')
    
    if df is None or df.empty:
        logging.error("拉取 2 年历史黄金数据失败！")
        return 0
        
    logging.info(f"成功获取 2 年 1 小时级别数据 {len(df)} 条。开始处理格式与 1m 插值重采样...")
    
    # 提取 OHLC 价格
    if isinstance(df.columns, pd.MultiIndex):
        ohlc = df[['Open', 'High', 'Low', 'Close']].xs('GC=F', level=1, axis=1)
    else:
        ohlc = df[['Open', 'High', 'Low', 'Close']]
        
    ohlc = ohlc.rename(columns={'Open': 'open_usd', 'High': 'high_usd', 'Low': 'low_usd', 'Close': 'close_usd'})
    ohlc['dt'] = pd.to_datetime(ohlc.index).tz_localize(None)
    ohlc = ohlc.sort_values('dt').drop_duplicates('dt').dropna()
    
    # 汇率与金价转换映射 (COMEX 美元/盎司 -> 国内人民币 元/克)
    latest_usd_price = ohlc['close_usd'].iloc[-1]
    
    # 从 SQLite 中读取最新真实国内金价做基准对齐
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT close FROM gold_prices ORDER BY timestamp DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    
    latest_cny_price = row[0] if row else 877.0
    scale_factor = latest_cny_price / latest_usd_price
    
    for col in ['open', 'high', 'low', 'close']:
        ohlc[col] = ohlc[f'{col}_usd'] * scale_factor
    
    # 1 分钟插值重采样 (Resample 1min)
    ohlc = ohlc.set_index('dt')
    # 对 OHLC 分别进行插值
    resampled = ohlc[['open', 'high', 'low', 'close']].resample('1min').interpolate(method='time').reset_index()
    resampled['timestamp'] = resampled['dt'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    logging.info(f"插值扩充完成！产生连续标准 1m K 线点数: {len(resampled)} 条。灌入 SQLite 数据库中...")
    
    # 批量增量写入 SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    records = []
    for _, r in resampled.iterrows():
        records.append((str(r['timestamp']), float(r['open']), float(r['high']), float(r['low']), float(r['close']), 0, SOURCE_TAG))
        
    cursor.executemany("""
        INSERT OR IGNORE INTO gold_prices (timestamp, open, high, low, close, volume, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, records)
    
    inserted = cursor.rowcount
    conn.commit()
    conn.close()
    
    logging.info(f"完成 2 年历史数据导入！增量写入数据库 {inserted} 条。")
    return len(resampled)

if __name__ == "__main__":
    import_2year_gold_history()
