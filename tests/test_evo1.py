from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bench.backends.base import load_export_info
from bench.obs import make_obs


class Evo1BundleTests(unittest.TestCase):
    def test_bundle_json_is_an_export_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {"model": "evo1", "chunk_size": 50}
            (root / "bundle.json").write_text(json.dumps(expected))
            self.assertEqual(load_export_info(root), expected)


class Evo1ObservationTests(unittest.TestCase):
    def test_uniform_noise_is_deterministic_and_bounded(self):
        source = make_obs(
            "synthetic",
            "test",
            chunk_size=50,
            state_dim=24,
            max_action_dim=24,
            seed=7,
            n_views=1,
            noise_distribution="uniform",
        )
        first = source[3].noise
        second = source[3].noise
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, (1, 50, 24))
        self.assertGreaterEqual(float(first.min()), -1.0)
        self.assertLessEqual(float(first.max()), 1.0)
        self.assertEqual(source.describe()["noise_distribution"], "uniform")

    def test_unknown_noise_distribution_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown noise distribution"):
            make_obs("synthetic", "test", 1, 1, noise_distribution="triangle")


if __name__ == "__main__":
    unittest.main()
