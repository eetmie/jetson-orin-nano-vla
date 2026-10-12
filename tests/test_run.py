"""./run.sh: model detection, bundle checks and the stage plan."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from bench import run


def write_bundle(root: Path, meta_name: str, meta: dict, files: dict[str, bytes]) -> None:
    (root / meta_name).write_text(json.dumps(meta))
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    names = [meta_name, *files]
    (root / "MANIFEST.sha256").write_text("".join(
        f"{hashlib.sha256((root / n).read_bytes()).hexdigest()}  {n}\n" for n in names))


XVLA = {"action_mode": "ee6d", "fixture": {"file": "fixture.npz"}, "graphs": []}


class DetectTests(unittest.TestCase):
    def test_each_bundle_kind(self):
        cases = {"smolvla-base": ("export_info.json", {}),
                 "pi05-libero": ("bundle.json", {"policy": "pi05_libero"}),
                 "pi05-base": ("bundle.json", {"policy": "pi05_lerobot"}),
                 "evo1-libero": ("bundle.json", {"model": "evo1"}),
                 "groot-n17-base": ("bundle.json", {"embodiment": "x", "deepstack": 3}),
                 "groot-n16-base": ("bundle.json", {"embodiment": "x"}),
                 "xvla-base": ("bundle.json", XVLA)}
        for model, (name, meta) in cases.items():
            with tempfile.TemporaryDirectory() as d:
                (Path(d) / name).write_text(json.dumps(meta))
                self.assertEqual(run.detect_model(Path(d)), model)

    def test_unknown_bundle_asks_for_model(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "bundle.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "--model"):
                run.detect_model(Path(d))


class ValidateTests(unittest.TestCase):
    def test_complete_bundle_passes(self):
        with tempfile.TemporaryDirectory() as d:
            write_bundle(Path(d), "bundle.json", XVLA, {"fixture.npz": b"x", "g/a.onnx": b"y"})
            self.assertEqual(run.validate_bundle("xvla-base", Path(d), True)["action_mode"], "ee6d")

    def test_changed_file_fails(self):
        with tempfile.TemporaryDirectory() as d:
            write_bundle(Path(d), "bundle.json", XVLA, {"fixture.npz": b"x"})
            (Path(d) / "fixture.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                run.validate_bundle("xvla-base", Path(d), True)

    def test_unlisted_file_fails(self):
        with tempfile.TemporaryDirectory() as d:
            write_bundle(Path(d), "bundle.json", XVLA, {"fixture.npz": b"x"})
            (Path(d) / "stray.onnx").write_bytes(b"z")
            with self.assertRaisesRegex(ValueError, "not in the bundle manifest"):
                run.validate_bundle("xvla-base", Path(d), True)

    def test_smolvla_aot_needs_hoisted_two_camera_export(self):
        with tempfile.TemporaryDirectory() as d:
            meta = {"fixture": {"file": "f.npz"}, "embed_tokens": "e.npy", "n_cam_slots": 1,
                    "kv": {"cross": "expert-projected"}}
            write_bundle(Path(d), "export_info.json", meta, {"f.npz": b"x", "e.npy": b"y"})
            with self.assertRaisesRegex(ValueError, "--baseline"):
                run.validate_bundle("smolvla-base", Path(d), True)
            run.validate_bundle("smolvla-base", Path(d), False)


class PlanTests(unittest.TestCase):
    def plan(self, model, optimized=True):
        cache = Path("/c")
        return run.plan(model, Path("/b"), cache, optimized, 300, cache / "out.json")

    def test_aot_stages_chain_their_caches(self):
        stages, bench = self.plan("xvla-base")
        self.assertEqual([s[0] for s in stages],
                         ["TensorRT engines", "AOT Triton denoise", "AOT Triton vision"])
        self.assertIn("/c/baseline-cache", stages[1][2])
        self.assertIn("/c/denoise-cache", stages[2][2])
        self.assertEqual(stages[2][1], Path("/c/optimized-cache"))
        self.assertIn("bench/aot/xvla/run_candidate.py", bench)
        self.assertEqual(bench[bench.index("--cache-dir") + 1], "/c/optimized-cache")

    def test_baseline_and_native_only_skip_aot(self):
        for model, optimized in (("xvla-base", False), ("pi05-base", True)):
            stages, bench = self.plan(model, optimized)
            self.assertEqual(len(stages), 1)
            self.assertEqual(bench[bench.index("--cache-dir") + 1], "/c/baseline-cache")


if __name__ == "__main__":
    unittest.main()
