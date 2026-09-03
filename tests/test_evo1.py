from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bench import models
from bench.backends.base import load_export_info
from bench.evo1_parity import _compare
from bench.obs import make_obs
from bench.report import validity_table


class Evo1RegistryTests(unittest.TestCase):
    def test_bootstrap_is_explicitly_nondeployable(self):
        spec = models.get("evo1-bootstrap")
        self.assertEqual(spec.family, "evo1")
        self.assertFalse(spec.extras["deployable"])
        self.assertIsNone(spec.torch_repo)
        self.assertIsNone(spec.split_repo)
        self.assertEqual(spec.noise_distribution, "uniform")

    def test_report_exposes_nondeployable_parity(self):
        rendered = validity_table([{
            "label": "evo1-bootstrap.ort",
            "model": {"deployable": False},
            "validity": {
                "execution": "pass",
                "instrumentation": "pass",
                "placement": "not_checked",
                "parity": "pass",
                "provenance": "pass",
            },
        }])
        self.assertIn("evo1-bootstrap.ort", rendered)
        self.assertIn("**no**", rendered)
        self.assertIn("| pass | pass | not_checked | pass | pass |", rendered)

    def test_bundle_json_is_an_export_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {"model": "evo1", "chunk_size": 50}
            (root / "bundle.json").write_text(json.dumps(expected))
            self.assertEqual(load_export_info(root), expected)


class Evo1ObservationTests(unittest.TestCase):
    def test_fixture_comparison_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "fixture shape mismatch"):
            _compare(np.zeros((2, 3)), np.zeros((3, 2)))

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
