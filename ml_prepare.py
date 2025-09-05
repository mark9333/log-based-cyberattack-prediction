import sqlite3
import pandas as pd
from db_utils import get_db_path, init_db

ATTACK_TYPES = [
    'brute_force', 'port_scan', 'sql_injection', 'ddos',
    'malware', 'ip_blocked', 'unauthorized_access', 'intrusion'
]

def prepare_ml_data_from_db(db_path: str | None = None, limit_days: int | None = None):
    init_db() 
    path = db_path or get_db_path()

    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS log_analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                file_name TEXT,
                attack_type TEXT,
                occurrences INTEGER
            )
        """)
        conn.commit()

        q = """
        SELECT date, attack_type, SUM(occurrences) AS cnt
        FROM log_analysis_results
        GROUP BY date, attack_type
        ORDER BY date
        """
        df_long = pd.read_sql_query(q, conn)

    if df_long.empty:
        return pd.DataFrame(columns=['date'] + ATTACK_TYPES)

    df_wide = (
        df_long
        .pivot_table(index='date', columns='attack_type', values='cnt', fill_value=0)
        .reset_index()
        .sort_values('date')
        .reset_index(drop=True)
    )

    for c in ATTACK_TYPES:
        if c not in df_wide.columns:
            df_wide[c] = 0

    if limit_days is not None and limit_days > 0:
        df_wide = df_wide.tail(limit_days).reset_index(drop=True)

    return df_wide[['date'] + ATTACK_TYPES]