"""
PainSense - web dashboard.

Run alongside realtime_infer.py:
    python dashboard_web.py
Then open http://127.0.0.1:5000 (bound to localhost only).
"""

import time

from flask import Flask, render_template, jsonify, request

import config
import dashboard_state
import db


app = Flask(__name__)


# =========================================================
# DASHBOARD HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        seq_len=config.SEQ_LEN,
        interval=config.CAPTURE_INTERVAL_SEC,
        alert_on=config.ALERT_THRESHOLD,
        alert_off=config.ALERT_OFF_THRESHOLD,
        alert_window=config.ALERT_MEDIAN_WINDOW,
        history_points=config.DASHBOARD_HISTORY,
        stale_sec=config.DASHBOARD_STALE_SEC
    )


# =========================================================
# LIVE DASHBOARD STATE
# =========================================================

@app.route("/api/state")
def api_state():

    state = dashboard_state.read_state()

    # "live" = the camera loop wrote a state recently.
    ts = state.get("updated_ts")
    state["live"] = (
        ts is not None
        and (time.time() - float(ts)) <= config.DASHBOARD_STALE_SEC
    )

    return jsonify(state)


# =========================================================
# RECENT OBSERVATIONS / HISTORY
# =========================================================

@app.route("/api/history")
def api_history():

    n = request.args.get("n", default=config.DASHBOARD_HISTORY, type=int)
    n = max(1, min(int(n), 1000))

    return jsonify(db.get_recent(n=n))


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    db.init_db()

    print("PainSense web dashboard: http://127.0.0.1:5000")
    print("History API: http://127.0.0.1:5000/api/history")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
