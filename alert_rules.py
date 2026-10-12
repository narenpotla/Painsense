"""
PainSense - alert rules.

The live alert is stateful (AlertState):
    1. The score history is smoothed with a median over the last
       ALERT_MEDIAN_WINDOW scores.
    2. The alert turns ON when that median reaches ALERT_THRESHOLD
       (only once the window is full, so one frame can never trigger it).
    3. The alert turns OFF only when the median falls below
       ALERT_OFF_THRESHOLD (hysteresis), so it does not flicker around
       the threshold.
    4. No face / no score resets everything.
"""

from collections import deque
from statistics import median

import config


def _clamp(score):
    return max(
        float(config.PAIN_SCORE_MIN),
        min(float(config.PAIN_SCORE_MAX), float(score))
    )


class AlertState:
    """Median filter + hysteresis over the 0-10 intensity score."""

    def __init__(
        self,
        window=None,
        on_threshold=None,
        off_threshold=None
    ):
        self.window = (
            config.ALERT_MEDIAN_WINDOW if window is None else int(window)
        )
        self.on_threshold = (
            config.ALERT_THRESHOLD if on_threshold is None
            else float(on_threshold)
        )
        self.off_threshold = (
            config.ALERT_OFF_THRESHOLD if off_threshold is None
            else float(off_threshold)
        )

        if self.off_threshold > self.on_threshold:
            raise ValueError("off_threshold must be <= on_threshold")

        self.scores = deque(maxlen=self.window)
        self.active = False
        self.median = None

    def reset(self):
        """Forget all history and clear the alert (call on no face)."""
        self.scores.clear()
        self.active = False
        self.median = None

    def update(self, score, face_detected=True):
        """
        Feed one score. Returns (alert, reason).
        """

        if not face_detected or score is None:
            self.reset()
            return False, None

        self.scores.append(_clamp(score))
        self.median = float(median(self.scores))

        if self.active:
            if self.median < self.off_threshold:
                self.active = False
        elif (
            len(self.scores) >= self.window
            and self.median >= self.on_threshold
        ):
            self.active = True

        if self.active:
            return True, (
                f"Median of last {len(self.scores)} scores "
                f"{self.median:.1f} reached threshold "
                f"{self.on_threshold:.1f}"
            )

        return False, None


def evaluate(score, face_detected=True):
    """
    Stateless single-frame rule, kept for older callers.

    The live demo uses AlertState instead.
    Return (alert, reason) for a 0-10 intensity score.
    """

    if not face_detected or score is None:
        return False, None

    score = _clamp(score)

    if score >= config.ALERT_THRESHOLD:
        return True, (
            f"Pain score {score:.1f} reached "
            f"threshold {config.ALERT_THRESHOLD:.1f}"
        )

    return False, None
