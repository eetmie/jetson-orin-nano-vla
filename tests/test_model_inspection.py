"""Small checkpoint fixtures exercise host inspection and the exported runtime contract."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from export import inspect_model as inspector
from export import cli as exporter
from bench import cli, model_contract, run
from bench.vendor.pi05_numeric import transform

ROOT = Path(__file__).resolve().parents[1]


def config(family="smolvla", views=1, state=3, action=4, horizon=16):
    return {"type": family, "input_features": {
                **{f"observation.images.camera{i}": {"type": "VISUAL", "shape": [3, 224, 224]} for i in range(views)},
                "observation.state": {"type": "STATE", "shape": [state]}},
            "output_features": {"action": {"type": "ACTION", "shape": [action]}},
            "max_state_dim": 32, "max_action_dim": 32, "chunk_size": horizon,
            inspector.FAMILIES[family]: 7}


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def write(self, cfg=None, recipe=None):
        if cfg is not None:
            (self.path / "config.json").write_text(json.dumps(cfg))
        if recipe is not None:
            (self.path / "export_config.json").write_text(json.dumps(recipe))

    def weights(self, width=32, name="model.safetensors"):
        header = json.dumps({"model.action_in_proj.weight": {
            "dtype": "F32", "shape": [1, width], "data_offsets": [0, 4 * width]}}).encode()
        (self.path / name).write_bytes(struct.pack("<Q", len(header)) + header + bytes(4 * width))

    def test_all_supported_lerobot_families_use_checkpoint_shapes(self):
        for family in ("smolvla", "evo1", "pi05", "xvla"):
            cfg = config(family)
            if family == "xvla":
                cfg["action_mode"] = "ee6d"
            self.write(cfg)
            spec = inspector.inspect_checkpoint(self.path)
            self.assertEqual((spec["views"], spec["state_dim"], spec["action_dim"], spec["chunk_size"], spec["num_steps"]),
                             (1, 3, 4, 16, 7))
            self.assertEqual(spec["max_action_dim"], 20 if family == "xvla" else 32)

    def test_camera_order_and_subset_are_preserved(self):
        cfg = config(views=3)
        cfg["input_features"] = dict(reversed(list(cfg["input_features"].items())))
        self.write(cfg)
        spec = inspector.inspect_checkpoint(self.path, views=2)
        self.assertEqual(spec["image_keys"], ["observation.images.camera2", "observation.images.camera1"])
        with self.assertRaisesRegex(ValueError, "declared cameras"):
            inspector.inspect_checkpoint(self.path, views=4)

    def test_missing_feature_dimension_is_not_guessed(self):
        cfg = config()
        del cfg["input_features"]["observation.state"]
        self.write(cfg)
        with self.assertRaisesRegex(ValueError, "observation.state"):
            inspector.inspect_checkpoint(self.path)

    def test_unsupported_policy_is_rejected_before_export(self):
        self.write({"type": "act"})
        with self.assertRaisesRegex(ValueError, "unsupported LeRobot"):
            inspector.inspect_checkpoint(self.path)

    def test_padded_width_must_cover_real_dimensions(self):
        self.write(config(state=33))
        with self.assertRaisesRegex(ValueError, "state_dim exceeds"):
            inspector.inspect_checkpoint(self.path)

    def test_header_verifies_configuration_without_reading_tensor_data(self):
        self.write(config())
        self.weights()
        self.assertEqual(inspector.inspect_checkpoint(self.path)["inspection"]["weight_tensors_checked"], 1)
        self.weights(width=31)
        with self.assertRaisesRegex(ValueError, "shape disagrees"):
            inspector.inspect_checkpoint(self.path)

    def test_sharded_header_and_missing_shard(self):
        self.write(config())
        (self.path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "model-01.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "missing weights shard"):
            inspector.inspect_checkpoint(self.path)
        self.weights(name="model-01.safetensors")
        self.assertEqual(inspector.inspect_checkpoint(self.path)["inspection"]["weight_tensors_checked"], 1)

    def test_normal_hub_cache_symlink_is_accepted(self):
        self.write(config())
        self.weights(name="blob")
        (self.path / "model.safetensors").symlink_to(self.path / "blob")
        self.assertEqual(inspector.inspect_checkpoint(self.path)["inspection"]["weight_tensors_checked"], 1)

    def test_adapter_only_checkpoint_has_actionable_error(self):
        self.write(config())
        (self.path / "adapter_config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "merge the adapter"):
            inspector.inspect_checkpoint(self.path)

    def test_invalid_header_length_is_bounded(self):
        self.write(config())
        (self.path / "model.safetensors").write_bytes(struct.pack("<Q", 2**40))
        with self.assertRaisesRegex(ValueError, "header size"):
            inspector.inspect_checkpoint(self.path)

    def test_checkpoint_step_resolves_pretrained_model(self):
        child = self.path / "pretrained_model"
        child.mkdir()
        (child / "config.json").write_text(json.dumps(config()))
        source = inspector.parse_source(str(self.path))
        self.assertEqual(source.location, str(child))

    def test_openpi_requires_configuration_instead_of_guessing_family(self):
        (self.path / "params").mkdir()
        with self.assertRaisesRegex(ValueError, "--openpi-config"):
            inspector.inspect_checkpoint(self.path)

    def test_openpi_named_libero_recipe_and_custom_metadata(self):
        spec = inspector.inspect_checkpoint(self.path, openpi_config="pi05_libero")
        self.assertEqual((spec["framework"], spec["views"], spec["state_dim"], spec["action_dim"], spec["chunk_size"]),
                         ("openpi", 2, 8, 7, 10))
        recipe = {"framework": "openpi", "openpi_config": "pi05_custom", "state_dim": 3, "action_dim": 4,
                  "max_state_dim": 32, "max_action_dim": 32, "chunk_size": 50, "num_steps": 5,
                  "image_keys": ["base_0_rgb"], "raw_image_keys": ["observation/image"],
                  "state_key": "observation/state", "discrete_state_input": True, "output_transform": "slice"}
        self.write(recipe=recipe)
        spec = inspector.inspect_checkpoint(self.path)
        self.assertEqual((spec["views"], spec["state_dim"], spec["action_dim"], spec["chunk_size"], spec["num_steps"]),
                         (1, 3, 4, 50, 5))

    def test_openpi_discovers_real_dimensions_from_its_normalization_assets(self):
        self.write({"action_dim": 32, "action_horizon": 50}, {
            "framework": "openpi", "openpi_config": "pi05_custom", "num_steps": 10,
            "image_keys": ["base_0_rgb"], "raw_image_keys": ["observation/image"],
            "state_key": "observation/state"})
        asset = self.path / "assets/my_robot"
        asset.mkdir(parents=True)
        (asset / "norm_stats.json").write_text(json.dumps({"norm_stats": {
            "state": {"mean": [1, 2, 3]}, "actions": {"mean": [0, 0, 0, 0, 0]}}}))
        spec = inspector.inspect_checkpoint(self.path)
        self.assertEqual((spec["state_dim"], spec["action_dim"], spec["max_action_dim"], spec["chunk_size"]), (3, 5, 32, 50))

    def test_shard_index_cannot_escape_selected_checkpoint(self):
        self.write(config())
        (self.path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "../weights.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "outside the checkpoint"):
            inspector.inspect_checkpoint(self.path)

    def test_nonfinite_fps_is_rejected(self):
        self.write(config())
        with self.assertRaisesRegex(ValueError, "fps must be positive"):
            inspector.inspect_checkpoint(self.path, fps=float("nan"))

    def test_inspect_cli_does_not_create_output_or_install_model_packages(self):
        self.write(config())
        process = subprocess.run([sys.executable, str(ROOT / "export/cli.py"), str(self.path), "--inspect"],
                                 capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 0, process.stderr)
        spec = json.loads(process.stdout)
        self.assertEqual(spec["state_dim"], 3)
        self.assertEqual(list(self.path.iterdir()), [self.path / "config.json"])

    def test_bundle_contract_overrides_every_base_profile_dimension(self):
        self.write(config())
        spec = inspector.inspect_checkpoint(self.path)
        (self.path / model_contract.FILENAME).write_text(json.dumps(spec))
        (self.path / "export_info.json").write_text("{}")
        args = SimpleNamespace(model=None, task=None, views=None)
        resolved = cli.Resolved(args, self.path)
        self.assertEqual((resolved.views, resolved.state_dim, resolved.action_dim, resolved.noise_width,
                          resolved.chunk_size), (1, 3, 4, 32, 16))
        self.assertEqual(run.detect_model(self.path), "smolvla-base")
        self.assertIsNotNone(model_contract.aot_unavailable(spec, {}))
        args.model = "pi05-base"
        with self.assertRaisesRegex(ValueError, "disagrees"):
            cli.Resolved(args, self.path)

    def test_launcher_does_not_force_base_camera_count(self):
        _, command = run.plan("smolvla-base", self.path, self.path / "cache", False, 1, self.path / "result")
        self.assertNotIn("--views", command)

    def test_native_fallback_can_use_a_non_gpt2_tokenizer_and_custom_label(self):
        self.write(config())
        spec = inspector.inspect_checkpoint(self.path)
        spec["name"] = "my-finetune"
        spec["runtime"] = {"lean_tokenizer": False}
        (self.path / model_contract.FILENAME).write_text(json.dumps(spec))
        stages, command = run.plan("smolvla-base", self.path, self.path / "cache", False, 1, self.path / "result")
        self.assertNotIn("-c", stages[0][2])
        self.assertEqual(command[command.index("--label") + 1], "my-finetune.trt")

    def test_saved_contract_is_manifested_and_shape_mismatches_fail(self):
        self.write(config())
        spec = inspector.inspect_checkpoint(self.path)
        (self.path / "export_info.json").write_text(json.dumps({"chunk_size": 16, "max_action_dim": 32, "num_steps": 7, "n_cam_slots": 1}))
        exporter.finalize_bundle(self.path, spec)
        manifest = (self.path / "MANIFEST.sha256").read_text()
        expected = hashlib.sha256((self.path / model_contract.FILENAME).read_bytes()).hexdigest()
        self.assertIn(expected + "  model_spec.json", manifest)
        spec["chunk_size"] = 17
        with self.assertRaisesRegex(ValueError, "disagrees"):
            exporter.finalize_bundle(self.path, spec)


class HubSourceTests(unittest.TestCase):
    def test_ids_and_urls_resolve_revision_and_checkpoint_subfolder(self):
        source = inspector.parse_source("https://huggingface.co/owner/model/tree/v2/checkpoints/100/pretrained_model")
        self.assertEqual((source.location, source.revision, source.subfolder), ("owner/model", "v2", "checkpoints/100/pretrained_model"))
        self.assertEqual(inspector.parse_source("owner/model").kind, "hub")
        self.assertEqual(inspector.parse_source("https://hf.co/owner/model").location, "owner/model")

    def test_revision_conflict_and_non_model_urls_fail(self):
        for value, kwargs in (("https://huggingface.co/owner/model/tree/main", {"revision": "v1"}),
                              ("https://example.com/owner/model", {}),
                              ("https://huggingface.co/datasets/owner/model", {}),
                              ("./missing-checkpoint", {}),
                              ("owner/model", {"subfolder": "../escape"})):
            with self.subTest(source=value), self.assertRaises(ValueError):
                inspector.parse_source(value, **kwargs)

    def test_metadata_fetch_resolves_once_and_never_fetches_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(config()))
            calls = []
            def fetch(repo, name, revision):
                calls.append((repo, name, revision))
                return str(root / name)
            api = SimpleNamespace(model_info=lambda *args, **kwargs: SimpleNamespace(
                sha="a" * 40, siblings=[SimpleNamespace(rfilename=n) for n in ("config.json", "model.safetensors", "optimizer.pt")]))
            hub = SimpleNamespace(HfApi=lambda: api, hf_hub_download=fetch)
            with patch.dict(sys.modules, {"huggingface_hub": hub}):
                local, resolved = inspector.resolve_metadata(inspector.parse_source("owner/model"))
            self.assertEqual(local, root)
            self.assertEqual(resolved.revision, "a" * 40)
            self.assertEqual(calls, [("owner/model", "config.json", "a" * 40)])


class NormalizationTests(unittest.TestCase):
    def test_openpi_and_lerobot_quantiles_preserve_their_own_operation_order(self):
        stats = {"q01": [1.0, -2.0], "q99": [1.0, 5.0]}
        x = np.array([[2.0, 0.0]], np.float32)
        lo, hi = np.array(stats["q01"], np.float32), np.array(stats["q99"], np.float32)
        expected = (x - np.asarray(stats["q01"])) / (np.asarray(stats["q99"]) - np.asarray(stats["q01"]) + 1e-6) * 2.0 - 1.0
        np.testing.assert_array_equal(transform(x, stats, implementation="openpi", eps=1e-6), expected)
        expected = np.float32(2) * (x - lo) / np.where(hi == lo, np.float32(1e-8), hi - lo) - np.float32(1)
        np.testing.assert_array_equal(transform(x, stats), expected)

    def test_openpi_preserves_statistics_precision_for_nearly_constant_state(self):
        stats = {"q01": [14.024799029541], "q99": [14.024799049541]}
        x = np.array([14.024799], np.float32)
        expected = (x - np.asarray(stats["q01"])) / (np.asarray(stats["q99"]) - np.asarray(stats["q01"]) + 1e-6) * 2 - 1
        np.testing.assert_array_equal(transform(x, stats, implementation="openpi", eps=1e-6), expected)

    def test_mean_std_and_identity_normalization(self):
        x = np.array([[3.0, 4.0]], np.float32)
        stats = {"mean": [2.0, 1.0], "std": [1.0, 2.0]}
        normalized = transform(x, stats, mode="MEAN_STD", implementation="openpi", eps=1e-6)
        np.testing.assert_allclose(transform(normalized, stats, mode="MEAN_STD", implementation="openpi", eps=1e-6, inverse=True), x)
        np.testing.assert_array_equal(transform(x, None), x)


if __name__ == "__main__":
    unittest.main()
