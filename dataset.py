"""
PainSense - PyTorch datasets using REAL SynPAIN video frames.

FrameDataset:
    Reads real cropped face frames from:
    data/videos/face_frames/<video_name>/*.jpg

SequenceDataset:
    Builds real consecutive 8-frame sequences from the same video.
    Each sequence receives the video's real Pain / NoPain label.

Labels:
    0 = NoPain
    1 = Pain
"""

import os
import glob
import re
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

import config


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------
# Image transforms
# ---------------------------------------------------------

frame_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


train_frame_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def get_video_label(video_name):
    """
    Get the real SynPAIN label from the video folder name.

    Examples:
        1000026870_NoPain_man_Young -> 0
        1100031270_Pain_man_Young   -> 1
    """

    name = video_name.lower()

    if "_nopain_" in name:
        return 0

    if "_pain_" in name:
        return 1

    raise ValueError(
        f"Could not determine Pain/NoPain label from video name: {video_name}"
    )


def get_frame_number(path):
    """
    Extract frame number from names such as:
        frame_000001.jpg
        frame_000125.jpg
    """

    match = re.search(r"frame_(\d+)", os.path.basename(path))

    if match:
        return int(match.group(1))

    return -1


def list_labeled_videos(frames_dir=None):
    """
    SynPAIN videos whose folder name contains Pain or NoPain.

    Unlabeled folders such as video_001 are skipped.
    """

    if frames_dir is None:
        frames_dir = config.VIDEO_FACE_DIR

    labeled = []

    for folder in sorted(glob.glob(os.path.join(frames_dir, "*"))):
        if not os.path.isdir(folder):
            continue

        video_name = os.path.basename(folder)

        try:
            label = get_video_label(video_name)
        except ValueError:
            continue

        labeled.append((video_name, label, folder))

    return labeled


def frame_intensity(label, frame_number, frame_min, frame_max):
    """
    Generated intensity for one frame.

    NoPain videos stay at 0. Those clips transition into a
    non-pain expression, not a pain face.

    Pain videos map the first frame to 0 and the last frame
    to PAIN_SCORE_MAX. SynPAIN built these clips as a morph
    from neutral to the pain expression.
    """

    if label == 0:
        return 0.0

    span = max(frame_max - frame_min, 1)
    alpha = (frame_number - frame_min) / span
    alpha = max(0.0, min(1.0, float(alpha)))
    return alpha * float(config.PAIN_SCORE_MAX)


def export_intensity_manifest(frames_dir=None, manifest_path=None):
    """
    Write one row per labeled face frame:

        video, frame, frame_number, intensity, label_source
    """

    if frames_dir is None:
        frames_dir = config.VIDEO_FACE_DIR

    if manifest_path is None:
        manifest_path = config.INTENSITY_MANIFEST

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    rows = []

    for video_name, label, folder in list_labeled_videos(frames_dir):
        frame_files = sorted(
            glob.glob(os.path.join(folder, "*.jpg")),
            key=get_frame_number
        )

        if not frame_files:
            continue

        numbers = [get_frame_number(path) for path in frame_files]
        frame_min = numbers[0]
        frame_max = numbers[-1]
        source = (
            "non_pain_expression"
            if label == 0
            else "neutral_to_pain_progress"
        )

        for path, number in zip(frame_files, numbers):
            intensity = frame_intensity(
                label,
                number,
                frame_min,
                frame_max
            )
            rows.append((
                video_name,
                os.path.basename(path),
                number,
                f"{intensity:.4f}",
                source
            ))

    with open(manifest_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(
            "video,frame,frame_number,intensity,label_source\n"
        )
        for row in rows:
            handle.write(",".join(str(part) for part in row) + "\n")

    print(
        f"Intensity manifest: {len(rows)} frames -> {manifest_path}"
    )

    return manifest_path


# ---------------------------------------------------------
# Frame Dataset
# ---------------------------------------------------------

class FrameDataset(Dataset):
    """
    Reads real cropped SynPAIN face frames.

    Expected structure:

    data/videos/face_frames/
        1000026870_NoPain_man_Young/
            frame_000001.jpg
            frame_000002.jpg
            ...

        1100031270_Pain_man_Young/
            frame_000001.jpg
            frame_000002.jpg
            ...
    """

    def __init__(
        self,
        frames_dir=None,
        train=True
    ):

        if frames_dir is None:
            frames_dir = config.VIDEO_FACE_DIR

        self.samples = []
        self.groups = []

        # Only folders whose name carries a Pain / NoPain label.
        # video_001 ... video_040 are label-free copies and are skipped.
        labeled = list_labeled_videos(frames_dir)

        if not labeled:
            raise RuntimeError(
                f"No labeled video face folders found in {frames_dir}"
            )

        for video_name, label, video_folder in labeled:

            frame_files = sorted(
                glob.glob(os.path.join(video_folder, "*.jpg")),
                key=get_frame_number
            )

            for frame_path in frame_files:

                self.samples.append(
                    (frame_path, label)
                )

                # Store the video name.
                # This is important because validation must be
                # separated by video rather than by individual frame.
                self.groups.append(video_name)

        if not self.samples:
            raise RuntimeError(
                f"No face frames found in {frames_dir}"
            )

        self.transform = (
            train_frame_transform
            if train
            else frame_transform
        )

        print(
            f"FrameDataset: {len(self.samples)} real face frames "
            f"from {len(set(self.groups))} videos"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        img = Image.open(path).convert("RGB")

        return (
            self.transform(img),
            torch.tensor(label, dtype=torch.long)
        )


# ---------------------------------------------------------
# Sequence Dataset
# ---------------------------------------------------------

class SequenceDataset(Dataset):
    """
    Builds REAL temporal sequences directly from SynPAIN
    video face frames.

    Example:

        frame_000001
        frame_000002
        frame_000003
        ...
        frame_000008

        becomes one sequence.

    Sequence shape:

        (T, C, H, W)

    Target shape:

        (T,)

    Since SynPAIN provides a video-level Pain/NoPain label,
    the label is repeated for every timestep.

    Example for a Pain video:

        target = [1,1,1,1,1,1,1,1]

    Example for a NoPain video:

        target = [0,0,0,0,0,0,0,0]
    """

    def __init__(
        self,
        seq_dir=None,
        train=True
    ):

        if seq_dir is None:
            seq_dir = config.VIDEO_FACE_DIR

        self.samples = []
        self.groups = []

        labeled = list_labeled_videos(seq_dir)

        if not labeled:
            raise RuntimeError(
                f"No labeled video face folders found in {seq_dir}"
            )

        sequence_length = config.SEQ_LEN

        for video_name, label, video_folder in labeled:

            frame_files = sorted(
                glob.glob(os.path.join(video_folder, "*.jpg")),
                key=get_frame_number
            )

            # Build sequences ONLY from truly consecutive frame numbers.
            #
            # This prevents us from joining frame 20 and frame 22
            # when frame 21 was skipped by face detection.
            for start in range(
                0,
                len(frame_files) - sequence_length + 1
            ):

                candidate = frame_files[
                    start:start + sequence_length
                ]

                numbers = [
                    get_frame_number(path)
                    for path in candidate
                ]

                expected = list(
                    range(
                        numbers[0],
                        numbers[0] + sequence_length
                    )
                )

                if numbers != expected:
                    continue

                self.samples.append(
                    candidate
                )

                self.groups.append(video_name)

        if not self.samples:
            raise RuntimeError(
                f"No real {sequence_length}-frame sequences could "
                f"be created from {seq_dir}"
            )

        self.transform = frame_transform

        print(
            f"SequenceDataset: {len(self.samples)} real "
            f"{sequence_length}-frame sequences "
            f"from {len(set(self.groups))} videos"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        frame_paths = self.samples[idx]

        video_name = self.groups[idx]

        label = get_video_label(video_name)

        tensors = []

        for frame_path in frame_paths:

            img = Image.open(frame_path).convert("RGB")

            tensors.append(
                self.transform(img)
            )

        seq_tensor = torch.stack(tensors)

        # Binary Pain/NoPain target repeated for each frame.
        target_tensor = torch.full(
            (len(frame_paths),),
            float(label),
            dtype=torch.float32
        )

        return seq_tensor, target_tensor


# ---------------------------------------------------------
# Window helpers (shared with cv_intensity.py)
# ---------------------------------------------------------

def valid_window_starts(frame_numbers, seq_len=None, stride=None):
    """
    Frame numbers at which a window can start.

    A window is seq_len frames, `stride` video frames apart. Every
    frame in it must exist (face detection can drop frames), so a
    window never silently bridges a gap.

    Returns a sorted list of start frame numbers.
    """

    if seq_len is None:
        seq_len = config.SEQ_LEN

    if stride is None:
        stride = config.SEQ_STRIDE

    present = set(frame_numbers)

    return [
        n for n in sorted(present)
        if all((n + k * stride) in present for k in range(seq_len))
    ]


def window_numbers(start, seq_len=None, stride=None):
    if seq_len is None:
        seq_len = config.SEQ_LEN

    if stride is None:
        stride = config.SEQ_STRIDE

    return [start + k * stride for k in range(seq_len)]


# ---------------------------------------------------------
# Sequence-level augmentation
# ---------------------------------------------------------

AUGMENT_OPS = ("flip", "brightness", "contrast", "rotation", "crop", "blur")


def augment_clip(images, rng=random):
    """
    Apply ONE randomly chosen augmentation to a whole clip.

    The same op with the same parameters is applied to every frame,
    so the temporal signal is not corrupted by per-frame jitter.

    images: list of PIL.Image (RGB).
    """

    op = rng.choice(AUGMENT_OPS)

    if op == "flip":
        return [ImageOps.mirror(im) for im in images]

    if op == "brightness":
        factor = rng.uniform(0.75, 1.25)
        return [ImageEnhance.Brightness(im).enhance(factor) for im in images]

    if op == "contrast":
        factor = rng.uniform(0.75, 1.25)
        return [ImageEnhance.Contrast(im).enhance(factor) for im in images]

    if op == "rotation":
        angle = rng.uniform(-10.0, 10.0)
        return [
            im.rotate(angle, resample=Image.BILINEAR) for im in images
        ]

    if op == "crop":
        w, h = images[0].size
        scale = rng.uniform(0.85, 1.0)
        cw, ch = int(w * scale), int(h * scale)
        x0 = rng.randint(0, w - cw)
        y0 = rng.randint(0, h - ch)
        return [
            im.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BILINEAR)
            for im in images
        ]

    radius = rng.uniform(0.5, 1.5)
    return [im.filter(ImageFilter.GaussianBlur(radius)) for im in images]


# ---------------------------------------------------------
# Intensity sequence dataset
# ---------------------------------------------------------

class IntensitySequenceDataset(Dataset):
    """
    Windows of SEQ_LEN face frames SEQ_STRIDE video frames apart, with
    a generated 0-10 target ("expression intensity on a synthetic morph
    scale").

    Pain video: target rises linearly with frame index, 0 at the first
    frame and 10 at the last (a neutral-to-pain morph).
    NoPain video: target is 0 on every frame.

    train=True enables sequence-level augmentation (one op per clip).
    """

    def __init__(
        self,
        frames_dir=None,
        sequences_per_video=6,
        train=True,
        stride=None
    ):
        if frames_dir is None:
            frames_dir = config.VIDEO_FACE_DIR

        if stride is None:
            stride = config.SEQ_STRIDE

        self.samples = []
        self.groups = []
        self.stride = stride
        self.augment = bool(train)
        sequence_length = config.SEQ_LEN

        for video_name, label, folder in list_labeled_videos(frames_dir):
            frame_files = sorted(
                glob.glob(os.path.join(folder, "*.jpg")),
                key=get_frame_number
            )

            if not frame_files:
                continue

            by_number = {get_frame_number(p): p for p in frame_files}
            numbers = sorted(by_number)
            frame_min = numbers[0]
            frame_max = numbers[-1]

            valid_starts = valid_window_starts(
                numbers, sequence_length, stride
            )

            if not valid_starts:
                continue

            pick_count = min(sequences_per_video, len(valid_starts))
            chosen = []

            if pick_count == 1:
                chosen = [valid_starts[len(valid_starts) // 2]]
            else:
                for index in range(pick_count):
                    position = round(
                        index * (len(valid_starts) - 1) / (pick_count - 1)
                    )
                    chosen.append(valid_starts[position])

            for start in sorted(set(chosen)):
                nums = window_numbers(start, sequence_length, stride)
                candidate = [by_number[n] for n in nums]
                intensities = [
                    frame_intensity(label, n, frame_min, frame_max)
                    for n in nums
                ]
                self.samples.append((candidate, intensities))
                self.groups.append(video_name)

        if not self.samples:
            raise RuntimeError(
                f"No intensity sequences could be built from {frames_dir}"
            )

        self.transform = frame_transform

        print(
            f"IntensitySequenceDataset: {len(self.samples)} sequences "
            f"from {len(set(self.groups))} videos "
            f"(stride {stride} frames = {stride / config.VIDEO_FPS:.2f} s)"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, intensities = self.samples[idx]
        images = [Image.open(p).convert("RGB") for p in frame_paths]

        if self.augment:
            images = augment_clip(images)

        seq_tensor = torch.stack([self.transform(im) for im in images])
        target_tensor = torch.tensor(intensities, dtype=torch.float32)
        return seq_tensor, target_tensor


# ---------------------------------------------------------
# Video-level splits
# ---------------------------------------------------------

def _eval_copy(dataset):
    """Shallow copy with augmentation off and the plain transform."""

    import copy

    clone = copy.copy(dataset)
    clone.transform = frame_transform

    if hasattr(clone, "augment"):
        clone.augment = False

    return clone


def stratified_group_split(groups, val_frac, test_frac=0.0, seed=42):
    """
    Split video names into (train, val, test) lists, stratified by the
    Pain / NoPain label. Each class contributes
    round(n * frac) videos (at least 1 when frac > 0) to val and test.
    """

    by_label = {0: [], 1: []}

    for group in sorted(set(groups)):
        by_label[get_video_label(group)].append(group)

    rng = random.Random(seed)
    train, val, test = [], [], []

    for label in (0, 1):
        items = by_label[label]
        rng.shuffle(items)

        n = len(items)
        n_test = max(1, round(n * test_frac)) if test_frac > 0 else 0
        n_val = max(1, round(n * val_frac)) if val_frac > 0 else 0

        if n_test + n_val >= n:
            raise RuntimeError(
                f"Not enough videos ({n}) of label {label} for the split."
            )

        test += items[:n_test]
        val += items[n_test:n_test + n_val]
        train += items[n_test + n_val:]

    return sorted(train), sorted(val), sorted(test)


def split_dataset_3way(
    dataset,
    val_frac=config.VAL_SPLIT,
    test_frac=config.TEST_SPLIT,
    seed=42
):
    """
    Video-level train / val / test split (stratified by label).

    Use val to choose the checkpoint and fit the calibration, then
    score test once. Augmentation is on for train only.

    Returns (train_subset, val_subset, test_subset, names) where
    names = {"train": [...], "val": [...], "test": [...]}.
    """

    if not hasattr(dataset, "groups"):
        raise RuntimeError(
            "Dataset does not contain video group information."
        )

    train_v, val_v, test_v = stratified_group_split(
        dataset.groups, val_frac, test_frac, seed
    )

    def indices(names):
        wanted = set(names)
        return [i for i, g in enumerate(dataset.groups) if g in wanted]

    eval_ds = _eval_copy(dataset)

    train_subset = torch.utils.data.Subset(dataset, indices(train_v))
    val_subset = torch.utils.data.Subset(eval_ds, indices(val_v))
    test_subset = torch.utils.data.Subset(eval_ds, indices(test_v))

    print(
        f"Video split: {len(train_v)} train / {len(val_v)} val / "
        f"{len(test_v)} test videos"
    )
    print(
        f"Samples: {len(train_subset)} train / {len(val_subset)} val / "
        f"{len(test_subset)} test"
    )

    return (
        train_subset,
        val_subset,
        test_subset,
        {"train": train_v, "val": val_v, "test": test_v}
    )


def stratified_kfold_videos(videos, n_folds=5, seed=0):
    """
    Stratified k-fold over VIDEOS (never over windows).

    Returns a list of (train_videos, test_videos) of length n_folds.
    Each class is shuffled with `seed` and dealt round-robin so every
    fold gets an equal number of Pain and NoPain videos.
    """

    rng = random.Random(seed)
    folds = [[] for _ in range(n_folds)]

    for label in (0, 1):
        items = sorted(v for v in set(videos) if get_video_label(v) == label)
        rng.shuffle(items)

        for i, video in enumerate(items):
            folds[i % n_folds].append(video)

    splits = []

    for k in range(n_folds):
        test = sorted(folds[k])
        train = sorted(v for j, f in enumerate(folds) if j != k for v in f)
        splits.append((train, test))

    return splits


def split_dataset(
    dataset,
    val_split=config.VAL_SPLIT,
    seed=42
):
    """
    Split the dataset by VIDEO into train / validation.

    Kept for the binary stages and for reproducing the original
    6-video validation set. Training data keeps its augmentation;
    validation data uses the plain transform and no augmentation.
    """

    if not hasattr(dataset, "groups"):
        raise RuntimeError(
            "Dataset does not contain video group information."
        )

    groups = sorted(set(dataset.groups))

    pain_groups = [g for g in groups if get_video_label(g) == 1]
    nopain_groups = [g for g in groups if get_video_label(g) == 0]

    rng = random.Random(seed)

    rng.shuffle(pain_groups)
    rng.shuffle(nopain_groups)

    def split_groups(group_list):

        if len(group_list) <= 1:
            return group_list, []

        n_val = max(1, int(len(group_list) * val_split))
        n_val = min(n_val, len(group_list) - 1)

        return group_list[n_val:], group_list[:n_val]

    train_pain, val_pain = split_groups(pain_groups)
    train_nopain, val_nopain = split_groups(nopain_groups)

    train_groups = set(train_pain + train_nopain)
    val_groups = set(val_pain + val_nopain)

    train_indices = [
        i for i, group in enumerate(dataset.groups)
        if group in train_groups
    ]

    val_indices = [
        i for i, group in enumerate(dataset.groups)
        if group in val_groups
    ]

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(_eval_copy(dataset), val_indices)

    print(
        f"Video split: "
        f"{len(train_groups)} train videos, "
        f"{len(val_groups)} validation videos"
    )

    print(
        f"Samples: "
        f"{len(train_subset)} train, "
        f"{len(val_subset)} validation"
    )

    return train_subset, val_subset


# ---------------------------------------------------------
# Webcam-style degradation (feature-cache augmentation)
# ---------------------------------------------------------
#
# SynPAIN faces are clean, sharp and studio-lit. A laptop webcam gives
# small, soft, noisy, JPEG-compressed faces with odd white balance and a
# face box that jitters between frames. These functions simulate that.
#
# One parameter set is drawn per (video, variant): a clip is "filmed"
# by one simulated camera. Box jitter and sensor noise change per frame.
#
# held_out=True draws from a DIFFERENT family (harsher ranges plus gamma
# and motion blur, which training never uses). It is only for the
# "simulated webcam" test set, so that test does not simply reward
# training on the exact same distortion.

def sample_webcam_params(rng, held_out=False):
    """rng: numpy RandomState. Returns a dict of degradation params."""

    u = rng.uniform

    if not held_out:
        return {
            "flip": bool(rng.rand() < 0.5),
            "angle": u(-8, 8),
            "scale": u(0.88, 1.0),
            "box_jitter": u(0.0, 0.03),
            "brightness": u(0.65, 1.35),
            "contrast": u(0.7, 1.3),
            "saturation": u(0.6, 1.3),
            "gain_r": u(0.9, 1.1),
            "gain_b": u(0.9, 1.1),
            "lowres": u(0.4, 0.9),
            "blur": u(0.0, 1.0),
            "noise": u(0.0, 6.0),
            "jpeg": int(rng.randint(40, 90)),
            "gamma": 1.0,
            "motion": 0,
        }

    return {
        "flip": bool(rng.rand() < 0.5),
        "angle": u(-10, 10),
        "scale": u(0.85, 1.0),
        "box_jitter": u(0.01, 0.04),
        "brightness": u(0.55, 1.45),
        "contrast": u(0.6, 1.2),
        "saturation": u(0.5, 1.2),
        "gain_r": u(0.85, 1.15),
        "gain_b": u(0.85, 1.15),
        "lowres": u(0.25, 0.6),
        "blur": u(0.3, 1.4),
        "noise": u(3.0, 10.0),
        "jpeg": int(rng.randint(20, 55)),
        "gamma": u(0.7, 1.4),
        "motion": int(rng.choice([0, 3, 5])),
    }


def apply_webcam(img, p, rng):
    """
    Degrade one PIL RGB face crop with params p.
    rng: numpy RandomState for the per-frame parts (box jitter, noise).
    Returns a PIL image of the same size.
    """

    import io

    import numpy as np

    w, h = img.size

    if p["flip"]:
        img = ImageOps.mirror(img)

    # Face-box geometry: fixed rotation/scale, per-frame jitter.
    img = img.rotate(p["angle"], resample=Image.BILINEAR)
    cw, ch = w * p["scale"], h * p["scale"]
    jx = rng.uniform(-1, 1) * p["box_jitter"] * w
    jy = rng.uniform(-1, 1) * p["box_jitter"] * h
    x0 = min(max((w - cw) / 2 + jx, 0), w - cw)
    y0 = min(max((h - ch) / 2 + jy, 0), h - ch)
    img = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BILINEAR)

    # Exposure and white balance.
    img = ImageEnhance.Brightness(img).enhance(p["brightness"])
    img = ImageEnhance.Contrast(img).enhance(p["contrast"])
    img = ImageEnhance.Color(img).enhance(p["saturation"])

    arr = np.asarray(img).astype(np.float32)
    arr[..., 0] *= p["gain_r"]
    arr[..., 2] *= p["gain_b"]

    if p["gamma"] != 1.0:
        arr = 255.0 * (np.clip(arr, 0, 255) / 255.0) ** p["gamma"]

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # Small, soft face: downscale then upscale, blur, motion blur.
    lw, lh = max(8, int(w * p["lowres"])), max(8, int(h * p["lowres"]))
    img = img.resize((lw, lh), Image.BILINEAR).resize((w, h), Image.BILINEAR)

    if p["blur"] > 0:
        img = img.filter(ImageFilter.GaussianBlur(p["blur"]))

    if p["motion"]:
        k = p["motion"]
        kernel = [0.0] * (k * k)
        for i in range(k):
            kernel[(k // 2) * k + i] = 1.0 / k
        img = img.filter(ImageFilter.Kernel((k, k), kernel, scale=1.0))

    # Sensor noise (per frame), then JPEG.
    if p["noise"] > 0:
        arr = np.asarray(img).astype(np.float32)
        arr += rng.normal(0.0, p["noise"], arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=p["jpeg"])
    buf.seek(0)
    return Image.open(buf).convert("RGB")
