"""Guard all-model setup against damaged inputs, cache reuse and floating sources."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import types

from bench.tools import vla_setup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from vla_common.run_pinned import revision_for


class ModelSetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def bundle(self, model="evo1-libero", **extra):
        path = self.root / model
        path.mkdir()
        info = {"model": "evo1", "fixture": {"file": "fixture.npz"},
                "graphs": [{"file": "vision.onnx"}], **extra}
        (path / "bundle.json").write_text(json.dumps(info))
        for name in ["fixture.npz", "vision.onnx", "vision.onnx.data"]:
            (path / name).write_bytes(b"fixture/weights")
        manifest = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in path.iterdir()]
        (path / "MANIFEST.sha256").write_text("".join(manifest))
        return path

    def test_missing_external_weights_and_changed_fixture_rejected(self):
        path = self.bundle()
        vla_setup.validate_bundle("evo1-libero", path)
        (path / "vision.onnx.data").unlink()
        with self.assertRaisesRegex(ValueError, "artifact"):
            vla_setup.validate_bundle("evo1-libero", path)
        (path / "vision.onnx.data").write_bytes(b"fixture/weights")
        (path / "fixture.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            vla_setup.validate_bundle("evo1-libero", path)

    def test_random_head_and_wrong_groot_checkpoint_rejected(self):
        path = self.bundle(random_action_head=True)
        with self.assertRaisesRegex(ValueError, "trained weights"):
            vla_setup.validate_bundle("evo1-libero", path)
        with self.assertRaisesRegex(ValueError, "GR00T-N1.7"):
            vla_setup.validate_bundle("groot-n17-base", path)

    def test_manifest_traversal_rejected(self):
        path = self.bundle()
        (path / "MANIFEST.sha256").write_text("abc  ../outside\n")
        with self.assertRaisesRegex(ValueError, "artifact"):
            vla_setup.validate_bundle("evo1-libero", path)

    def test_manifest_must_cover_model_metadata(self):
        path = self.bundle()
        manifest = path / "MANIFEST.sha256"
        manifest.write_text("\n".join(line for line in manifest.read_text().splitlines()
                                      if not line.endswith("bundle.json")))
        with self.assertRaisesRegex(ValueError, "missing from manifest"):
            vla_setup.validate_bundle("evo1-libero", path)

    def test_each_selected_recipe_consumes_its_newly_built_cache(self):
        for model in vla_setup.LISTED_MODELS:
            with self.subTest(model=model):
                work = self.root / "fresh" / model
                stages = vla_setup.plan(model, self.root / model, work, Path("/venv/python"),
                                        Path("/builder/python"), 10)
                builds = [command for name, command in stages if "--base-cache" in command]
                final = stages[-1][1]
                self.assertTrue(builds, 'every selected recipe must build its AOT cache')
                if builds:
                    last = builds[-1]
                    self.assertEqual(final[final.index("--cache-dir") + 1], last[last.index("--out") + 1])
                    self.assertIn("run_candidate.py", " ".join(final))
                self.assertIn("fp32", final)
                self.assertTrue(all(str(work) in str(c) for _, c in stages))

    def test_fp32_attention_scores_only_where_selected(self):
        expect = {'xvla-base': ('denoise', True), 'pi05-libero': ('attention', True),
                  'smolvla-base': ('vision', False), 'evo1-libero': ('step', False)}
        for model, (stage, wanted) in expect.items():
            stages = dict(vla_setup.plan(model, self.root / 'bundle', self.root / model,
                                         Path('/runtime/python'), Path('/builder/python'), 10))
            self.assertEqual('--score-fp32' in stages[stage], wanted, model)

    def test_groot_selected_recipe_uses_all_attention_and_baseline_needs_no_compiler(self):
        for model in ('groot-n16-base', 'groot-n17-base'):
            work = self.root / model
            stages = vla_setup.plan(model, self.root / 'bundle', work, Path('/runtime/python'),
                                   Path('/builder/python'), 10)
            attention = dict(stages)['attention']
            self.assertEqual(attention[attention.index('--mode') + 1], 'all')
            baseline = vla_setup.plan(model, self.root / 'bundle', work, Path('/runtime/python'),
                                     Path('/builder/python'), 10, optimized=False)
            self.assertEqual([name for name, _ in baseline], ['runtime-doctor', 'benchmark'])
            self.assertNotIn('/builder/python', str(baseline))
            self.assertIn(str(work / 'baseline-cache'), baseline[-1][1])

    def test_failed_stage_keeps_logs_and_does_not_run_benchmark(self):
        bundle = self.bundle()
        work = self.root / "new run"
        args = type("Args", (), dict(mode="optimized", runtime_python=Path(sys.executable),
                    build_python=Path(sys.executable), duration_s=10, accumulate="fp32"))()
        result = type("Result", (), dict(returncode=2))()
        with patch("bench.tools.vla_setup.subprocess.run", return_value=result) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "runtime-doctor failed"):
                vla_setup.run_model(args, "evo1-libero", bundle, work)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(json.loads((work / "setup.json").read_text())["status"], "fail")
        self.assertTrue((work / "runtime-doctor.log").exists())
        self.assertFalse((work / "baseline-cache").exists())

    def test_hub_guard_pins_nested_loads_and_rejects_floating_custom_sources(self):
        pins = {"aux/model": "a" * 40}
        self.assertEqual(revision_for("aux/model", "main", pins), "a" * 40)
        self.assertEqual(revision_for("custom/model", "b" * 40, pins), "b" * 40)
        with self.assertRaisesRegex(ValueError, "Floating"):
            revision_for("aux/model", "v1", pins)
        with self.assertRaisesRegex(ValueError, "Unpinned"):
            revision_for("unknown/model", None, pins)

    def test_transformers_download_and_followup_cache_lookup_use_same_revision(self):
        from vla_common.run_pinned import install
        hub = types.ModuleType("huggingface_hub")
        hub.hf_hub_download = lambda *args, **kw: kw["revision"]
        hub.snapshot_download = lambda *args, **kw: kw["revision"]
        transformers = types.ModuleType("transformers")
        utils = types.ModuleType("transformers.utils")
        cache = types.ModuleType("transformers.utils.hub")
        cache.cached_file = lambda *args, **kw: kw.get("revision")
        cache.cached_files = lambda *args, **kw: kw.get("revision")
        transformers.utils, utils.hub = utils, cache
        with patch.dict(sys.modules, {"huggingface_hub": hub, "transformers": transformers,
                                      "transformers.utils": utils, "transformers.utils.hub": cache}):
            install()
            repo = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
            fetched = hub.hf_hub_download(repo, "config.json", revision="main")
            self.assertEqual(cache.cached_files(repo, ["config.json"], revision="main"), fetched)
            self.assertEqual(cache.cached_file(repo, "config.json", revision="main"), fetched)
            self.assertEqual(len(fetched), 40)
            self.assertIsNone(cache.cached_file(str(self.root), "config.json"))

    def test_benchmark_fetch_pins_revision_and_separates_snapshot_directories(self):
        from bench.models import fetch
        calls = []
        def fake_download(**kwargs):
            calls.append(kwargs)
            return kwargs["local_dir"]
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = fake_download
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            a = fetch("lerobot/smolvla_base", cache=self.root)
            b = fetch("lerobot/smolvla_base", cache=self.root, revision="b" * 40)
        self.assertNotEqual(a, b)
        self.assertEqual(calls[0]["revision"], "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6")
        with self.assertRaisesRegex(ValueError, "full commit SHA"):
            fetch("custom/model", revision="main")


if __name__ == "__main__":
    unittest.main()
