"""Failures a clean user setup must detect before expensive GPU builds."""
import contextlib
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench.tools.smolvla_setup import GRAPHS, main, validate_bundle
from bench.tools.trt_env import main as create_env


class UsageSetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = self.root / "bundle with spaces"
        self.bundle.mkdir()
        self.info = dict(n_cam_slots=2, fixture={"file": "fixture.npz"}, embed_tokens="embed_tokens.npy",
                         kv={"cross": "expert-projected"})
        (self.bundle / "export_info.json").write_text(json.dumps(self.info))
        files = [f"{g}.onnx" for g in GRAPHS] + ["stats.json", "fixture.npz", "embed_tokens.npy",
                 "smolvlm_vision.onnx.data", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
        for name in files:
            path = self.bundle / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"artifact")
        (self.bundle / "tokenizer/tokenizer_config.json").write_text('{"tokenizer_class": "GPT2Tokenizer"}')
        self.manifest()

    def manifest(self):
        entries = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(self.bundle)}"
                   for p in self.bundle.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"]
        (self.bundle / "MANIFEST.sha256").write_text("\n".join(entries) + "\n")

    def test_complete_bundle_and_missing_external_weights(self):
        validate_bundle(self.bundle)
        (self.bundle / "smolvlm_vision.onnx.data").unlink()
        with self.assertRaisesRegex(ValueError, "manifest artifact"):
            validate_bundle(self.bundle)

    def test_changed_weights_fail_before_engine_build(self):
        (self.bundle / "smolvlm_vision.onnx.data").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            validate_bundle(self.bundle)

    def test_omitting_existing_weights_from_manifest_does_not_bypass_integrity(self):
        manifest = self.bundle / "MANIFEST.sha256"
        manifest.write_text("\n".join(line for line in manifest.read_text().splitlines()
                                       if not line.endswith("smolvlm_vision.onnx.data")))
        with self.assertRaisesRegex(ValueError, "missing from manifest"):
            validate_bundle(self.bundle)

    def test_ort_only_bundle_rejected(self):
        self.info.pop("fixture")
        (self.bundle / "export_info.json").write_text(json.dumps(self.info))
        with self.assertRaisesRegex(ValueError, "stock-policy fixture"):
            validate_bundle(self.bundle)

    def test_unhoisted_bundle_does_not_silently_change_selected_workload(self):
        self.info.pop("kv")
        (self.bundle / "export_info.json").write_text(json.dumps(self.info))
        with self.assertRaisesRegex(ValueError, "hoist-cross-kv"):
            validate_bundle(self.bundle)

    def test_manifest_cannot_read_outside_bundle(self):
        (self.bundle / "MANIFEST.sha256").write_text("abc  ../outside\n")
        with self.assertRaisesRegex(ValueError, "invalid manifest artifact"):
            validate_bundle(self.bundle)

    def test_existing_directory_refused_without_mutation(self):
        sentinel = self.root / "keep"
        sentinel.write_text("keep")
        with contextlib.redirect_stderr(io.StringIO()):
            status = main(["--bundle", str(self.bundle), "--work-dir", str(self.root)])
        self.assertEqual(status, 1)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_dry_run_without_dependencies_preserves_interpreter_symlink(self):
        executable = self.root / "venv python"
        executable.symlink_to("/usr/bin/python3")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(["--bundle", str(self.bundle), "--work-dir", str(self.root / "new"),
                           "--runtime-python", str(executable), "--dry-run"])
        self.assertEqual(status, 0)
        self.assertIn(str(executable), output.getvalue())
        self.assertIn("--accumulate fp32", output.getvalue())
        self.assertIn("--base-cache", output.getvalue())
        self.assertFalse((self.root / "new").exists())

    def test_export_missing_environment_is_actionable(self):
        exporter = self.root / "export.sh"
        exporter.write_bytes((Path(__file__).resolve().parents[1] / "export/export.sh").read_bytes())
        result = subprocess.run(["bash", str(exporter), "org/model", str(self.root / "out")],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("run export/setup.sh first", result.stdout)

    def test_export_revision_is_forwarded_and_snapshots_are_separate(self):
        exporter = self.root / "export.sh"
        exporter.write_bytes((Path(__file__).resolve().parents[1] / "export/export.sh").read_bytes())
        hf = self.root / ".venv-mock/bin/hf"
        hf.parent.mkdir(parents=True)
        hf.write_text(f"#!{sys.executable}\n" +
                      "import json, sys\nfrom pathlib import Path\n" +
                      "p=Path(sys.argv[sys.argv.index('--local-dir')+1]); p.mkdir(parents=True)\n" +
                      "(p/'config.json').write_text(json.dumps({'type':'unsupported','input_features':{'observation.images.a':{}}}))\n" +
                      "(p/'args.json').write_text(json.dumps(sys.argv[1:]))\n")
        hf.chmod(0o755)
        # The public wrapper uses the selected venv's verified HTTP downloader.
        python = hf.with_name("python")
        python.write_bytes(hf.read_bytes())
        python.chmod(0o755)
        for revision in ("a" * 40, "b" * 40):
            result = subprocess.run(["bash", str(exporter), "org/model", str(self.root / "out"),
                                     "--revision", revision], capture_output=True, text=True)
            self.assertIn("unsupported policy type", result.stdout)
        snapshots = list((self.root / ".checkpoints").glob("*/args.json"))
        self.assertEqual(len(snapshots), 2)
        revisions = {json.loads(p.read_text())[4] for p in snapshots}
        self.assertEqual(revisions, {"a" * 40, "b" * 40})

    def test_environment_creation_preserves_existing_directory(self):
        sentinel = self.root / "keep"
        sentinel.write_text("keep")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            create_env([str(self.root)])
        self.assertEqual(sentinel.read_text(), "keep")

    def test_missing_system_bindings_fail_before_environment_creation(self):
        target = self.root / "new-env"
        output = io.StringIO()
        with patch("bench.tools.trt_env.importlib.util.find_spec", return_value=None), \
                contextlib.redirect_stderr(output), self.assertRaises(SystemExit):
            create_env([str(target)])
        self.assertIn("install JetPack's Python bindings", output.getvalue())
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
