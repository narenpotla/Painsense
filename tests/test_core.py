"""
PainSense core tests (stdlib unittest, no extra dependencies).

Run from the project root:
    python -m unittest discover -s tests -v
"""

import os
import random
import sys
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402
import alert_rules  # noqa: E402
import dataset  # noqa: E402
import model  # noqa: E402


def _videos():
    pain = [f"{1100000000 + i}_Pain_man_Young" for i in range(20)]
    nopain = [f"{1000000000 + i}_NoPain_man_Young" for i in range(20)]
    return pain + nopain


class AlertStateTest(unittest.TestCase):

    def test_needs_full_window_and_median(self):
        state = alert_rules.AlertState(window=5, on_threshold=6, off_threshold=4)

        # One very high frame never triggers on its own.
        self.assertFalse(state.update(10)[0])

        for _ in range(3):
            self.assertFalse(state.update(7)[0])

        # Window full, median 7 -> on.
        self.assertTrue(state.update(7)[0])

    def test_hysteresis(self):
        state = alert_rules.AlertState(window=5, on_threshold=6, off_threshold=4)

        for _ in range(5):
            state.update(8)

        self.assertTrue(state.active)

        # Median drops to 5: between off (4) and on (6) -> stays on.
        for _ in range(3):
            state.update(5)

        self.assertTrue(state.active)

        # Median below 4 -> off.
        for _ in range(3):
            state.update(1)

        self.assertFalse(state.active)

    def test_reset_on_no_face(self):
        state = alert_rules.AlertState(window=5)

        for _ in range(5):
            state.update(9)

        self.assertTrue(state.active)
        self.assertEqual(state.update(9, face_detected=False), (False, None))
        self.assertEqual(len(state.scores), 0)
        self.assertFalse(state.active)

    def test_legacy_evaluate_still_works(self):
        self.assertTrue(alert_rules.evaluate(7.0)[0])
        self.assertFalse(alert_rules.evaluate(2.0)[0])
        self.assertFalse(alert_rules.evaluate(9.0, face_detected=False)[0])


class WindowTest(unittest.TestCase):

    def test_stride_and_gaps(self):
        numbers = list(range(1, 61))
        numbers.remove(20)   # face detection dropped frame 20

        starts = dataset.valid_window_starts(numbers, seq_len=8, stride=6)

        for s in starts:
            window = dataset.window_numbers(s, seq_len=8, stride=6)
            self.assertEqual(window[1] - window[0], 6)
            self.assertNotIn(20, window)
            self.assertTrue(set(window) <= set(numbers))

        # 2 (2, 8, 14, 20...) needs frame 20 -> excluded.
        self.assertNotIn(2, starts)
        self.assertIn(1, starts)

    def test_too_short(self):
        self.assertEqual(
            dataset.valid_window_starts(list(range(1, 40)), seq_len=8, stride=6),
            []
        )


class SplitTest(unittest.TestCase):

    def test_kfold_is_stratified_and_disjoint(self):
        videos = _videos()
        folds = dataset.stratified_kfold_videos(videos, n_folds=5, seed=3)
        seen = []

        for train, test in folds:
            self.assertFalse(set(train) & set(test))
            self.assertEqual(len(train) + len(test), len(videos))
            labels = [dataset.get_video_label(v) for v in test]
            self.assertEqual(labels.count(1), 4)
            self.assertEqual(labels.count(0), 4)
            seen += test

        self.assertEqual(sorted(seen), sorted(videos))

    def test_three_way_split_sizes(self):
        train, val, test = dataset.stratified_group_split(
            _videos(), val_frac=0.15, test_frac=0.15, seed=42
        )
        self.assertEqual((len(train), len(val), len(test)), (28, 6, 6))
        self.assertFalse(set(train) & set(val))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(val) & set(test))

        for part in (val, test):
            labels = [dataset.get_video_label(v) for v in part]
            self.assertEqual(labels.count(1), 3)


class AugmentTest(unittest.TestCase):

    def test_same_op_on_every_frame(self):
        rng_image = np.random.RandomState(0)
        frame = Image.fromarray(
            rng_image.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        clip = [frame.copy() for _ in range(8)]

        for seed in range(30):
            out = dataset.augment_clip(clip, rng=random.Random(seed))
            first = np.asarray(out[0])

            for image in out[1:]:
                np.testing.assert_array_equal(first, np.asarray(image))


class ModelTest(unittest.TestCase):

    def test_feature_models_are_causal(self):
        torch.manual_seed(0)
        x = torch.randn(2, 8, config.FEATURE_DIM)
        y = x.clone()
        y[:, 5:] = torch.randn(2, 3, config.FEATURE_DIM)

        for variant in model.FeatureIntensityModel.VARIANTS:
            m = model.FeatureIntensityModel(variant).eval()

            with torch.no_grad():
                a, b = m(x), m(y)

            self.assertEqual(tuple(a.shape), (2, 8))
            torch.testing.assert_close(a[:, :5], b[:, :5])

    def test_ensemble_averages_heads(self):
        torch.manual_seed(0)
        heads = [model.FeatureIntensityModel("lstm").eval() for _ in range(3)]
        ens = model.FeatureEnsemble(heads).eval()
        x = torch.randn(2, 8, config.FEATURE_DIM)

        with torch.no_grad():
            expected = torch.stack([h(x) for h in heads]).mean(0)
            torch.testing.assert_close(ens(x), expected)

    def test_backbone_bn_stays_in_eval(self):
        m = model.PainIntensityLSTM()
        m.train()
        bns = [
            layer for layer in m.encoder.backbone.modules()
            if isinstance(layer, torch.nn.BatchNorm2d)
        ]
        self.assertTrue(bns)
        self.assertTrue(all(not layer.training for layer in bns))
        self.assertTrue(m.lstm.training)

    def test_split_forward_matches_full_forward(self):
        m = model.PainIntensityLSTM().eval()
        x = torch.randn(1, 8, 3, 64, 64)

        with torch.no_grad():
            full = m(x)
            feats = m.encode_backbone(x[0]).unsqueeze(0)
            split = m.score_features(feats)

        torch.testing.assert_close(full, split)

    def test_feature_checkpoint_is_self_contained(self):
        if not os.path.exists(config.FEATURE_CKPT):
            self.skipTest("feature checkpoint not built")

        payload = torch.load(config.FEATURE_CKPT, map_location="cpu")

        for key in ("state_dict", "variant", "seq_len", "stride",
                    "scale", "bias", "backbone_state_dict"):
            self.assertTrue(
                key in payload,
                f"{key} missing; rebuild with cv_intensity.py --final"
            )

        self.assertEqual(payload["seq_len"], config.SEQ_LEN)
        self.assertEqual(payload["stride"], config.SEQ_STRIDE)

        backbone, head, _ = model.load_feature_checkpoint(
            config.FEATURE_CKPT, device="cpu"
        )

        with torch.no_grad():
            feats = backbone(torch.randn(8, 3, 224, 224)).unsqueeze(0)
            out = head(feats)

        self.assertEqual(tuple(out.shape), (1, 8))


class CropTest(unittest.TestCase):

    def test_crop_size_and_border_clip(self):
        try:
            import data_prep
        except Exception as error:  # pragma: no cover
            self.skipTest(f"data_prep import failed: {error}")

        image = np.zeros((200, 300, 3), dtype=np.uint8)

        crop = data_prep.crop_from_box(image, (100, 50, 80, 80))
        self.assertEqual(crop.shape, (config.CROP_SIZE, config.CROP_SIZE, 3))

        # Box touching the corner: margin is clipped, still a valid crop.
        crop = data_prep.crop_from_box(image, (0, 0, 40, 40))
        self.assertEqual(crop.shape, (config.CROP_SIZE, config.CROP_SIZE, 3))


class WebcamAugTest(unittest.TestCase):

    def test_shape_and_determinism(self):
        image = Image.fromarray(
            np.random.RandomState(1).randint(0, 255, (128, 128, 3), dtype=np.uint8)
        )

        for held_out in (False, True):
            params = dataset.sample_webcam_params(
                np.random.RandomState(7), held_out=held_out
            )
            a = dataset.apply_webcam(image, params, np.random.RandomState(3))
            b = dataset.apply_webcam(image, params, np.random.RandomState(3))
            self.assertEqual(a.size, image.size)
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_held_out_family_differs(self):
        rng = np.random.RandomState(0)
        train = [dataset.sample_webcam_params(rng) for _ in range(200)]
        held = [dataset.sample_webcam_params(rng, held_out=True) for _ in range(200)]

        self.assertTrue(all(p["gamma"] == 1.0 and p["motion"] == 0 for p in train))
        self.assertTrue(any(p["motion"] > 0 for p in held))
        self.assertLess(
            np.mean([p["jpeg"] for p in held]),
            np.mean([p["jpeg"] for p in train])
        )


class StorageTest(unittest.TestCase):

    def test_db_prune_and_order(self):
        import db

        old = config.DB_PATH

        with tempfile.TemporaryDirectory() as tmp:
            config.DB_PATH = os.path.join(tmp, "test.db")

            try:
                db.init_db()

                for i in range(30):
                    db.log_score(i % 10, ts=1000 + i)

                self.assertEqual(db.prune(max_rows=10), 20)

                rows = db.get_recent(n=50)
                self.assertEqual(len(rows), 10)
                self.assertEqual(rows[0]["timestamp"], 1020)
                self.assertEqual(rows[-1]["timestamp"], 1029)
            finally:
                config.DB_PATH = old

    def test_dashboard_state_roundtrip(self):
        import dashboard_state

        old = dashboard_state.STATE_FILE

        with tempfile.TemporaryDirectory() as tmp:
            dashboard_state.STATE_FILE = os.path.join(tmp, "state.json")

            try:
                empty = dashboard_state.read_state()
                self.assertIsNone(empty["score"])
                self.assertIsNone(empty["updated_ts"])

                dashboard_state.write_state(score=4.2, alert=True)
                state = dashboard_state.read_state()
                self.assertEqual(state["score"], 4.2)
                self.assertTrue(state["alert"])
                self.assertIsNotNone(state["updated_ts"])
            finally:
                dashboard_state.STATE_FILE = old


if __name__ == "__main__":
    unittest.main()
