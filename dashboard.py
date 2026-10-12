"""
dashboard.py - web dashboard for the live demo (http://127.0.0.1:5000).

Shows the current score, alert, capture status (OFFLINE when
live_camera.py is not running) and a 30 s trend from the SQLite log.
Bound to 127.0.0.1 only. Page: templates/index.html, static/.

    GET /             the page
    GET /api/state    latest live result (logs/dashboard_state.json)
    GET /api/history  recent scores from logs/painsense.db (?n=, max 1000)

Run (in a second terminal, next to live_camera.py):
    python dashboard.py
"""

import time

from flask import Flask, render_template, jsonify, request

from painsense import config, db, live_state


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

    state = live_state.read_state()

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
