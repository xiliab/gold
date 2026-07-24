import sqlite3
from datetime import datetime
conn = sqlite3.connect('/Users/xiliab/Desktop/开发/gold/gold_prices.db')
c = conn.cursor()
c.execute('''
INSERT OR REPLACE INTO prediction_history 
(created_at, target_time, base_price, predicted_direction, predicted_price, lower_bound, upper_bound)
VALUES (?, ?, ?, ?, ?, ?, ?)
''', (str(datetime.now()), '2026-07-20 22:00:00', 870.0, 'UP', 875.0, 868.0, 880.0))
conn.commit()
print("Success")
conn.close()
