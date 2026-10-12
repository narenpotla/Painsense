"""
painsense.photo - score one still photo.

    score_photo(path_or_bytes) -> dict with the score, face box, crop

Steps: open the image (any format Pillow reads), apply the phone EXIF
rotation, shrink so the longest side is at most PHOTO_MAX_SIDE, find the
largest face, crop it exactly like the live demo (painsense.face), then
score that single face with the photo model:

    checkpoints/pain_intensity_image.pt   frame-only head(s): sees one
        face at a time, which is what a photo is. Made by
        cross_validate.py --final (CV MAE is in the README).
    fallback: the live LSTM fed the same face SEQ_LEN times, with a
        warning, if the photo model has not been built yet.

The image is processed in memory and never written to disk. Used by
score_image.py and the dashboard's "Score a photo" card.
"""

import base64
import io
import os
import threading

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from painsense import config, face
from painsense.data import frame_transform
from painsense.models import load_feature_checkpoint

MAX_PIXELS = 60_000_000      # refuse absurdly large images (decompression bombs)

_scorer = None
_lock = threading.Lock()


class PhotoScorer:
    """Backbone + head for single faces. Load once, reuse."""

    def __init__(self):
        if os.path.exists(config.IMAGE_CKPT):
            path, self.repeat = config.IMAGE_CKPT, 1
        elif os.path.exists(config.FEATURE_CKPT):
            path, self.repeat = config.FEATURE_CKPT, config.SEQ_LEN
            print(
                f"WARNING: {config.IMAGE_CKPT} not found; scoring photos with "
                "the live LSTM on a repeated frame. Build the photo model "
                "with: python cross_validate.py --final"
            )
        else:
            raise FileNotFoundError(
                "No model found. Build it with: "
                "python cross_validate.py --epochs 30 --seeds 3 --final"
            )

        self.backbone, self.head, payload = load_feature_checkpoint(path)
        self.scale = float(payload.get("scale", 1.0))
        self.bias = float(payload.get("bias", 0.0))
        n_heads = len(payload.get("state_dicts") or [None])
        self.description = (
            f"{payload.get('variant', '?')} x{n_heads}"
            + ("" if self.repeat == 1 else " (live model, repeated frame)")
        )

    @torch.inference_mode()
    def score_crop(self, crop_bgr):
        """0-10 score of one 128x128 BGR face crop."""

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        x = frame_transform(Image.fromarray(rgb)).unsqueeze(0)
        feat = self.backbone(x.to(config.DEVICE))            # (1, 512)
        window = feat.unsqueeze(1).repeat(1, self.repeat, 1)   # (1, T, 512)
        raw = float(self.head(window)[0, -1].item())

        return max(
            config.PAIN_SCORE_MIN,
            min(config.PAIN_SCORE_MAX, self.scale * raw + self.bias)
        )


def get_scorer():
    """The shared PhotoScorer (created on first use)."""

    global _scorer

    if _scorer is None:
        _scorer = PhotoScorer()

    return _scorer


def load_image(source):
    """
    Path or raw bytes -> BGR numpy image, upright (EXIF applied).
    Raises ValueError if it is not a readable image.
    """

    try:
        if isinstance(source, (bytes, bytearray)):
            image = Image.open(io.BytesIO(source))
        else:
            image = Image.open(source)

        if image.width * image.height > MAX_PIXELS:
            raise ValueError(
                f"Image too large ({image.width}x{image.height})."
            )

        image = ImageOps.exif_transpose(image).convert("RGB")

    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ValueError("Not a readable image file.")

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _shrink(image, max_side):
    h, w = image.shape[:2]
    scale = max_side / float(max(h, w))

    if scale >= 1.0:
        return image

    return cv2.resize(
        image, (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA
    )


def _png_base64(image_bgr):
    ok, buf = cv2.imencode(".png", image_bgr)
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def score_photo(source):
    """
    Score the largest face in a photo.

    Returns a dict:
        face_found      bool
        score           0-10 (None without a face)
        above_alert     score >= ALERT_THRESHOLD (the live alert line)
        box             [x, y, w, h] of the face in the (shrunk) image
        image_size      [w, h] of the (shrunk) image
        crop_png        base64 PNG of the 128x128 face crop the model saw
        model           which model scored it
        message         one line for people
    """

    image = _shrink(load_image(source), config.PHOTO_MAX_SIDE)
    h, w = image.shape[:2]

    with _lock:
        scorer = get_scorer()
        box = face.detect_face_box(image)

        if box is None:
            return {
                "face_found": False,
                "score": None,
                "above_alert": False,
                "box": None,
                "image_size": [w, h],
                "crop_png": None,
                "model": scorer.description,
                "message": "No face found. Use a clear, front-facing photo "
                           "where the face is at least 40 px tall.",
            }

        crop = face.crop_from_box(image, box)
        score = scorer.score_crop(crop)

    above = score >= config.ALERT_THRESHOLD

    return {
        "face_found": True,
        "score": round(score, 2),
        "above_alert": bool(above),
        "box": list(box),
        "image_size": [w, h],
        "crop_png": _png_base64(crop),
        "model": scorer.description,
        "message": (
            f"{'At or above' if above else 'Below'} the live alert line "
            f"({config.ALERT_THRESHOLD:.0f}). Expression intensity on a "
            "synthetic morph scale, not a pain measurement."
        ),
    }
