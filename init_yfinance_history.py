import yfinance as yf
import pandas as pd
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "gold_prices.db")

def init_history():
    print("Downloading 7 days of 1-minute XAUUSD and CNY=X data...")
    df = yf.download(tickers=['GC=F', 'CNY=X'], period='7d', interval='1m', progress=False)
    
    if df.empty:
        print("Failed to download data!")
        return
        
    print("Processing data...")
    try:
        gold_c = df['Close']['GC=F'].ffill()
        gold_o = df['Open']['GC=F'].ffill()
        gold_h = df['High']['GC=F'].ffill()
        gold_l = df['Low']['GC=F'].ffill()
        gold_v = df['Volume']['GC=F'].fillna(0)
        
        cny_c = df['Close']['CNY=X'].ffill()
    except KeyError as e:
        print(f"Missing columns: {e}")
        return
        
    data = pd.DataFrame({
        'gold_open': gold_o,
        'gold_high': gold_h,
        'gold_low': gold_l,
        'gold_close': gold_c,
        'gold_vol': gold_v,
        'cny': cny_c
    }).dropna()
    
    print("Converting to CNY/gram...")
    factor = data['cny'] / 31.1034768
    data['open'] = data['gold_open'] * factor
    data['high'] = data['gold_high'] * factor
    data['low'] = data['gold_low'] * factor
    data['close'] = data['gold_close'] * factor
    data['volume'] = data['gold_vol']
    
    print("Resampling to 1-minute candles...")
    data = data.resample('1min', label='left', closed='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    data = data.reset_index()
    # yfinance returns timezone-aware datetime (usually America/New_York or UTC)
    # Convert to Asia/Shanghai time
    if data['Datetime'].dt.tz is None:
        data['Datetime'] = data['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
    else:
        data['Datetime'] = data['Datetime'].dt.tz_convert('Asia/Shanghai')
        
    data['Datetime'] = data['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    data = data.rename(columns={'Datetime': 'timestamp'})
    data['source'] = 'YFinance_GC_CNY_1M'
    
    print(f"Generated {len(data)} rows of 1-minute candles.")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("Clearing old data from gold_prices...")
    cursor.execute("DELETE FROM gold_prices")
    
    inserted = 0
    for _, row in data.iterrows():
        try:
            cursor.execute('''
                INSERT INTO gold_prices (timestamp, open, high, low, close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (row['timestamp'], row['open'], row['high'], row['low'], row['close'], row['volume'], row['source']))
            inserted += 1
        except sqlite3.IntegrityError:
            pass
            
    conn.commit()
    conn.close()
    print(f"Successfully inserted {inserted} rows into database.")

if __name__ == "__main__":
    init_history()
