import os
import sys
import sqlite3
import pandas as pd
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gold_prices.db")

def reload_clean_data(db_path=DB_PATH):
    """
    彻底清空杂音数据，纯建 AU0 官方 1 分钟标准行情数据库
    """
    from src.config import SOURCE_TAG
    import akshare as ak
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # 彻底清空表数据
    cur.execute("DELETE FROM gold_prices")
    conn.commit()
    conn.close()
    
    logging.info("已清空旧数据库杂音记录。开始抓取纯净 1m 真实行情...")
    from src.data_fetcher import fetch_real_gold_1m_data
    df = fetch_real_gold_1m_data()
    
    if df is not None and not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        records = []
        for _, r in df.iterrows():
            records.append((str(r['timestamp']), float(r['open']), float(r['high']), float(r['low']), float(r['close']), float(r['volume']), SOURCE_TAG))
            
        cur.executemany("""
            INSERT OR IGNORE INTO gold_prices (timestamp, open, high, low, close, volume, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, records)
        conn.commit()
        conn.close()
        
        logging.info(f"纯净数据库重建成功！共写入 {len(records)} 条标准 1m K 线，最新金价: {df['close'].iloc[-1]} 元/克。")

if __name__ == "__main__":
    reload_clean_data()
