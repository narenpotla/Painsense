"""
painsense.live_state - the latest live result, shared with the dashboard.

The camera loop writes logs/dashboard_state.json (atomically, via a
temp file); dashboard.py reads it every second. updated_ts tells the
dashboard whether the camera is still running.
"""

import json
import os
import tempfile
import time
from datetime import datetime

from painsense import config


STATE_FILE = config.LIVE_STATE_PATH


def _empty_state():
    return {
        "score": None,
        "probability": None,
        "prediction": None,
        "sequence_current": 0,
        "sequence_total": config.SEQ_LEN,
        "alert": False,
        "alert_reason": None,
        "updated_at": None,
        "updated_ts": None
    }


def write_state(
    score=None,
    probability=None,
    prediction=None,
    sequence_current=0,
    sequence_total=config.SEQ_LEN,
    alert=False,
    alert_reason=None
):
    """
    Save the latest prediction so the Flask dashboard can read it.

    updated_ts (epoch seconds) lets the dashboard tell a live feed
    from a stopped one.
    """

    os.makedirs(
        os.path.dirname(STATE_FILE),
        exist_ok=True
    )

    now = time.time()

    state = {
        "score": score,
        "probability": probability,
        "prediction": prediction,
        "sequence_current": sequence_current,
        "sequence_total": sequence_total,
        "alert": alert,
        "alert_reason": alert_reason,
        "updated_at": datetime.fromtimestamp(now).strftime("%H:%M:%S"),
        "updated_ts": now
    }

    # Write to a temporary file first.
    # This prevents Flask from reading a half-written file.

    fd, temp_path = tempfile.mkstemp(
        suffix=".json",
        dir=os.path.dirname(STATE_FILE)
    )

    try:

        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(state, file, indent=2)

        os.replace(temp_path, STATE_FILE)

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


def read_state():
    """
    Read the latest prediction.
    """

    if not os.path.exists(STATE_FILE):
        return _empty_state()

    try:

        with open(STATE_FILE, "r", encoding="utf-8") as file:
            state = json.load(file)

    except (json.JSONDecodeError, OSError):
        return _empty_state()

    merged = _empty_state()
    merged.update(state)
    return merged
