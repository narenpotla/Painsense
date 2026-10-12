"""
PainSense - database logging.

SQLite is used for the prototype.

The database stores:
    - timestamp
    - pain probability
    - prediction
    - alert status
    - alert reason

The score column stores the 0-10 intensity estimate.
probability stores that score divided by 10, for older dashboard reads.
"""

import sqlite3
import time
import os

import config


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_conn():
    os.makedirs(
        os.path.dirname(config.DB_PATH),
        exist_ok=True
    )

    # timeout: wait instead of failing with "database is locked" while
    # the dashboard reads during a write.
    conn = sqlite3.connect(
        config.DB_PATH,
        timeout=5.0
    )

    return conn


# =========================================================
# INITIALIZE DATABASE
# =========================================================

def init_db():

    conn = get_conn()

    # WAL lets the dashboard read while realtime_infer writes ~4 rows/s.
    conn.execute("PRAGMA journal_mode=WAL")

    # Existing table is preserved.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pain_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            score REAL NOT NULL,
            alert_flag INTEGER NOT NULL,
            alert_reason TEXT
        )
    """)

    # -----------------------------------------------------
    # Add new columns if they do not already exist.
    # -----------------------------------------------------

    columns = conn.execute(
        "PRAGMA table_info(pain_scores)"
    ).fetchall()

    column_names = {
        column[1]
        for column in columns
    }

    if "probability" not in column_names:

        conn.execute("""
            ALTER TABLE pain_scores
            ADD COLUMN probability REAL
        """)

    if "prediction" not in column_names:

        conn.execute("""
            ALTER TABLE pain_scores
            ADD COLUMN prediction TEXT
        """)

    conn.commit()

    conn.close()

    prune(config.DB_MAX_ROWS)


def prune(max_rows=None):
    """Keep only the newest max_rows rows. Returns rows deleted."""

    if max_rows is None:
        max_rows = config.DB_MAX_ROWS

    conn = get_conn()

    cur = conn.execute(
        """
        DELETE FROM pain_scores
        WHERE id <= (
            SELECT id FROM pain_scores
            ORDER BY id DESC
            LIMIT 1 OFFSET ?
        )
        """,
        (int(max_rows),)
    )

    deleted = cur.rowcount
    conn.commit()
    conn.close()

    return deleted


# =========================================================
# LOG PREDICTION
# =========================================================

def log_score(
    score,
    alert_flag=False,
    alert_reason=None,
    ts=None,
    prediction=None
):
    """
    Store one intensity observation.

    score:
        Expression intensity from 0 to 10 (synthetic morph scale).

    prediction:
        'PAIN' when the alert is on, otherwise 'NO PAIN'. When not
        given, falls back to the single-frame threshold.
    """

    score = max(
        float(config.PAIN_SCORE_MIN),
        min(
            float(config.PAIN_SCORE_MAX),
            float(score)
        )
    )

    probability = score / float(config.PAIN_SCORE_MAX)

    if prediction is None:
        prediction = (
            "PAIN"
            if score >= config.ALERT_THRESHOLD
            else "NO PAIN"
        )

    conn = get_conn()

    conn.execute(
        """
        INSERT INTO pain_scores
        (
            timestamp,
            score,
            alert_flag,
            alert_reason,
            probability,
            prediction
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            ts or time.time(),
            score,
            int(alert_flag),
            alert_reason,
            probability,
            prediction
        )
    )

    conn.commit()

    conn.close()


# =========================================================
# GET RECENT HISTORY
# =========================================================

def get_recent(n=50):

    conn = get_conn()

    rows = conn.execute(
        """
        SELECT
            timestamp,
            score,
            probability,
            prediction,
            alert_flag,
            alert_reason
        FROM pain_scores
        ORDER BY id DESC
        LIMIT ?
        """,
        (n,)
    ).fetchall()

    conn.close()

    # Database returns newest first.
    # Reverse so dashboard displays oldest -> newest.
    rows.reverse()

    history = []

    for row in rows:

        history.append(
            {
                "timestamp": row[0],
                "score": row[1],
                "probability": row[2],
                "prediction": row[3],
                "alert": bool(row[4]),
                "reason": row[5]
            }
        )

    return history


# =========================================================
# CLEAR DATABASE
# =========================================================

def clear_db():

    conn = get_conn()

    conn.execute(
        "DELETE FROM pain_scores"
    )

    conn.commit()

    conn.close()


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    init_db()

    print(
        f"Initialized DB at {config.DB_PATH}"
    )