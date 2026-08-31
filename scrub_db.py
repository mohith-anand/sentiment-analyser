import os
import sqlite3
import shutil
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'predictions.db')

if not os.path.exists(DB_PATH):
    print("No predictions.db found; nothing to scrub.")
    raise SystemExit(0)

bak_name = f"{DB_PATH}.bak.{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
shutil.copy2(DB_PATH, bak_name)
print(f"Backup created at: {bak_name}")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("PRAGMA table_info(predictions)")
cols = [r[1] for r in cur.fetchall()]
if 'text' not in cols:
    print("No 'text' column found; nothing to scrub.")
else:
    # If the column is NOT NULL, replace text with a fixed placeholder to avoid constraint errors
    try:
        cur.execute("UPDATE predictions SET text = NULL")
    except sqlite3.IntegrityError:
        cur.execute("UPDATE predictions SET text = '<REDACTED>'")
    conn.commit()
    cur.execute("SELECT COUNT(*) FROM predictions")
    total = cur.fetchone()[0]
    print(f"Scrubbed 'text' from {total} rows (set to NULL or '<REDACTED>')")
conn.close()
print('Done')
