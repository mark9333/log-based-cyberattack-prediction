import sqlite3
import os
import re
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "log_analysis.db"

def get_db_path() -> str:
    return str(DB_PATH)

def init_db():
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS log_analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        file_name TEXT,
        attack_type TEXT,
        occurrences INTEGER
    )
    """)
    conn.commit()
    conn.close()

def save_to_db(file_path, results: dict):
    init_db()
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()

    filename = os.path.basename(file_path)
    m = re.match(r"simulated_log_(\d{8})\.log", filename)
    if m:
        date_str = m.group(1)
        try:
            date = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            date = datetime.now().strftime("%Y-%m-%d")
    else:
        date = datetime.now().strftime("%Y-%m-%d")

    for attack_type, count in results.items():
        cur.execute("""
            INSERT INTO log_analysis_results (date, file_name, attack_type, occurrences)
            VALUES (?, ?, ?, ?)
        """, (date, filename, attack_type, count))

    conn.commit()
    conn.close()

def check_db_dates(db_path: str | None = None):
    path = db_path or get_db_path()

    if not os.path.exists(path):
        return []

    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT date FROM log_analysis_results ORDER BY date")
        dates = [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        dates = []
    conn.close()
    return dates

def clear_database(soft: bool = False):
    try:
        db_path = get_db_path()
    except Exception as e:
        return False, f"Cannot resolve DB path: {e}"

    if soft:
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("PRAGMA foreign_keys=OFF;")
            tables = cur.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
            cur.execute("BEGIN;")
            for (tname,) in tables:
                cur.execute(f'DELETE FROM "{tname}";')
            conn.commit()
            cur.execute("VACUUM;")
            conn.close()
            init_db() 
            return True, "All rows deleted (schema preserved). Database reinitialized."
        except Exception as e:
            try:
                conn.close()
            except Exception:
                pass
            return False, f"Soft clear failed: {e}"

    try:
        if os.path.exists(db_path):
            try:
                c = sqlite3.connect(db_path)
                c.close()
            except Exception:
                pass
            os.remove(db_path)
        return True, "Database file removed. (Will be recreated on next write/init.)"
    except Exception as e:
        return False, f"Hard clear failed: {e}"