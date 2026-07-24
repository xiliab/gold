import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gold_prices.db")

def get_db_connection(db_path=DB_PATH):
    """获取 SQLite 连接，启用 WAL busy_timeout 保护。"""
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn

def init_db(db_path=DB_PATH):
    """
    初始化 SQLite 数据库及黄金行情数据表价格 schema
    包含: timestamp (ISO 8601字符串/秒), open, high, low, close, volume, source
    """
    conn = sqlite3.connect(db_path, timeout=10)
    cursor = conn.cursor()
    # 启用 WAL 模式：允许一写多读并发，解决后台写线程与 API 读线程竞态锁死问题
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gold_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT UNIQUE NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL DEFAULT 0,
            source TEXT DEFAULT 'SGE_Au99.99'
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            target_time TEXT UNIQUE,
            base_price REAL,
            predicted_direction TEXT,
            predicted_price REAL,
            lower_bound REAL,
            upper_bound REAL,
            actual_price REAL,
            is_direction_correct INTEGER,
            is_range_correct INTEGER,
            error_amt REAL,
            action_advice TEXT,
            advice_type TEXT,
            is_advice_correct INTEGER,
            feature_snapshot TEXT
        )
    """)
    # 动态为旧数据库增加新列（先读取 PRAGMA table_info 校验，避开重复 ALTER TABLE 的 schema 锁定开销）
    cursor.execute("PRAGMA table_info(prediction_history)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    for col_name, col_def in [
        ("action_advice", "action_advice TEXT"),
        ("advice_type", "advice_type TEXT"),
        ("is_advice_correct", "is_advice_correct INTEGER"),
        ("feature_snapshot", "feature_snapshot TEXT")
    ]:
        if col_name not in existing_cols:
            try:
                cursor.execute(f"ALTER TABLE prediction_history ADD COLUMN {col_def}")
            except Exception:
                pass
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp ON gold_prices(timestamp)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_source ON gold_prices(source)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_pred_target ON prediction_history(target_time)
    """)
    # AI 自学习动态系数持久化表（单行 upsert 模式）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_state (
            key TEXT PRIMARY KEY,
            value REAL NOT NULL,
            updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print(f"Database initialized at: {DB_PATH}")
