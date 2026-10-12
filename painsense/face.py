"""
painsense.face - face detection and the single face-crop recipe.

Used by crop_faces.py (to make the stored training crops), by the live
camera and by score_video.py, so every face the model sees is cut the
same way:

    largest Haar face box
    -> margin 0.15 * max(w, h) on every side (clipped to the image)
    -> cv2.resize to CROP_SIZE x CROP_SIZE (128)

If no face is found, None is returned. A background is never passed to
the model as if it were a face.
"""

import cv2

from painsense import config

FACE_MARGIN = 0.15          # margin = FACE_MARGIN * max(w, h) on every side
DETECT_MIN_SIZE = (40, 40)  # Haar minSize used when the training crops were made

_face_cascade = None


def get_face_detector():
    """Load and cache the OpenCV Haar frontal-face detector."""

    global _face_cascade

    if _face_cascade is None:

        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(
                f"OpenCV {cv2.__version__} has no CascadeClassifier "
                "(it was removed in OpenCV 5). Install OpenCV 4.x:\n"
                "    pip uninstall -y opencv-python opencv-python-headless\n"
                '    pip install "opencv-python>=4.8,<5"'
            )

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        _face_cascade = cv2.CascadeClassifier(cascade_path)

        if _face_cascade.empty():
            raise RuntimeError(
                "Could not load OpenCV Haar face detector.\n"
                f"Expected cascade file at:\n{cascade_path}"
            )

    return _face_cascade


def detect_face_box(img_bgr, min_size=DETECT_MIN_SIZE):
    """Return the largest detected face as (x, y, w, h), or None."""

    if img_bgr is None or img_bgr.size == 0:
        return None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = get_face_detector().detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=min_size
    )

    if len(faces) == 0:
        return None

    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(fw), int(fh)


def crop_from_box(img_bgr, box, size=config.CROP_SIZE):
    """
    Crop a face box with margin and resize to size x size.

    Reproduces the crops in data/videos/face_frames. The model transform
    later upsamples them to IMG_SIZE = 224, exactly as for training.

    Returns a (size, size, 3) array, or None if the crop is empty.
    """

    x, y, w, h = box
    margin = int(FACE_MARGIN * max(w, h))

    ih, iw = img_bgr.shape[:2]

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(iw, x + w + margin)
    y2 = min(ih, y + h + margin)

    crop = img_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return cv2.resize(crop, (size, size))


def crop_face(img_bgr, size=config.CROP_SIZE, min_size=DETECT_MIN_SIZE):
    """Detect the largest face and return its crop, or None (no face)."""

    box = detect_face_box(img_bgr, min_size=min_size)

    if box is None:
        return None

    return crop_from_box(img_bgr, box, size=size)
