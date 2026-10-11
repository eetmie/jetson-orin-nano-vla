"""Pinned downloads and trained-weight identity without requiring Torch or CUDA."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / "export/pi05"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


weights = load("weight_digest")
downloads = load("download")
conversion = load("convert")


class Pi05ArtifactsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_conversion_uses_model_flag_instead_of_checkpoint_directory_name(self):
        source = '\n'.join('if "pi05" in checkpoint_dir: selected.append(True)\nelse: selected.append(False)' for _ in range(3))
        for flag, directory in ((True, '/work/jax-checkpoint'), (False, '/old/pi05-named-dir')):
            scope = dict(pi05=flag, checkpoint_dir=directory, selected=[])
            exec(conversion.patch_selector(source), scope)
            self.assertEqual(scope['selected'], [flag] * 3)

    def test_conversion_refuses_changed_upstream_selector(self):
        with self.assertRaisesRegex(RuntimeError, 'selector changed'):
            conversion.patch_selector('if "pi05" in checkpoint_dir: pass')

    def model(self, name, active=b"12", unused=b"34", reverse=False):
        entries = [("active", {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}),
                   ("unused", {"dtype": "BF16", "shape": [1], "data_offsets": [2, 4]})]
        header = json.dumps(dict(reversed(entries) if reverse else entries)).encode()
        path = self.root / name
        path.write_bytes(struct.pack("<Q", len(header)) + header + active + unused)
        return path

    def test_header_order_and_unused_head_do_not_change_trained_weight_identity(self):
        a = self.model("a")
        b = self.model("b", unused=b"56", reverse=True)
        self.assertEqual(weights.tensor_digest(a, ["unused"]), weights.tensor_digest(b, ["unused"]))
        c = self.model("c", active=b"99")
        self.assertNotEqual(weights.tensor_digest(a, ["unused"]), weights.tensor_digest(c, ["unused"]))
        self.assertNotEqual(weights.tensor_digest(a), weights.tensor_digest(b))

    def test_download_pins_generation_and_checks_bytes_before_publishing(self):
        data = b"pinned checkpoint"
        lock = {"bucket": "public", "prefix": "checkpoint/", "objects": [
            {"file": "weights/file", "generation": "123", "size": len(data),
             "sha256": hashlib.sha256(data).hexdigest()}]}
        with patch.object(downloads, "urlopen", return_value=io.BytesIO(data)) as request:
            downloads.download(lock, self.root / "good")
        self.assertIn("generation=123", request.call_args.args[0])
        self.assertEqual((self.root / "good/weights/file").read_bytes(), data)
        with patch.object(downloads, "urlopen", return_value=io.BytesIO(b"changed")):
            with self.assertRaisesRegex(ValueError, "checksum"):
                downloads.download(lock, self.root / "bad")
        self.assertFalse((self.root / "bad/weights/file").exists())

    def test_download_refuses_existing_destination(self):
        with self.assertRaises(FileExistsError):
            downloads.download({"objects": []}, self.root)

    def ranges(self, target, change=None, fail_first=False):
        data = b"pinned checkpoint"
        lock = {"bucket": "public", "prefix": "checkpoint/", "objects": [
            {"file": "weights/file", "generation": "123", "size": len(data),
             "sha256": hashlib.sha256(data).hexdigest()}]}

        failed = False

        def response(request, timeout):
            nonlocal failed
            self.assertIn("generation=123", request.full_url)
            span = request.get_header("Range").removeprefix("bytes=")
            start, end = map(int, span.split("-"))
            body = data[start:end + 1]
            headers = {"Content-Range": f"bytes {start}-{end}/{len(data)}",
                       "x-goog-generation": "123"}
            status = 206
            if change:
                body, headers, status = change(body, headers, status)
            stream = io.BytesIO(body)
            stream.status, stream.headers = status, headers
            if fail_first and start == 0 and not failed:
                failed = True
                original_read, reads = stream.read, 0
                def read(amount=-1):
                    nonlocal reads
                    reads += 1
                    if reads > 1:
                        raise ConnectionResetError("connection dropped after a partial write")
                    return original_read(1)
                stream.read = read
            return stream

        with patch.object(downloads, "RANGE_BYTES", 4), \
             patch.object(downloads.time, "sleep"), \
             patch.object(downloads, "urlopen", side_effect=response):
            downloads.download(lock, target, workers=3)
        return data

    def test_parallel_ranges_publish_only_verified_complete_bytes(self):
        target = self.root / "parallel"
        expected = self.ranges(target)
        self.assertEqual((target / "weights/file").read_bytes(), expected)
        self.assertFalse((target / "weights/file.partial").exists())

    def test_parallel_ranges_reject_ignored_range_header(self):
        target = self.root / "ignored"
        with self.assertRaisesRegex(ValueError, "range response"):
            self.ranges(target, lambda b, h, s: (b, h, 200))
        self.assertFalse((target / "weights/file").exists())

    def test_parallel_ranges_reject_wrong_span_or_generation(self):
        for field, value in (("Content-Range", "bytes 0-3/999"), ("x-goog-generation", "124")):
            with self.subTest(field=field):
                target = self.root / field
                with self.assertRaises(ValueError):
                    self.ranges(target, lambda b, h, s: (b, {**h, field: value}, s))
                self.assertFalse((target / "weights/file").exists())

    def test_parallel_ranges_reject_truncated_body(self):
        target = self.root / "truncated"
        with self.assertRaisesRegex(ValueError, "Truncated"):
            self.ranges(target, lambda b, h, s: (b[:-1], h, s))
        self.assertFalse((target / "weights/file").exists())

    def test_parallel_ranges_still_reject_changed_checksum(self):
        target = self.root / "corrupted"
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.ranges(target, lambda b, h, s: (b"X" * len(b), h, s))
        self.assertFalse((target / "weights/file").exists())

    def test_retry_overwrites_failed_slice_and_keeps_other_ranges_exact(self):
        target = self.root / "retry"
        expected = self.ranges(target, fail_first=True)
        self.assertEqual((target / "weights/file").read_bytes(), expected)

    def test_resume_reuses_only_checksum_verified_complete_files(self):
        data = b"pinned checkpoint"
        target = self.root / "resume"
        (target / "weights").mkdir(parents=True)
        (target / "weights/file").write_bytes(data)
        lock = {"bucket": "public", "prefix": "checkpoint/", "objects": [
            {"file": "weights/file", "generation": "123", "size": len(data),
             "sha256": hashlib.sha256(data).hexdigest()}]}
        with patch.object(downloads, "urlopen") as request:
            downloads.download(lock, target, resume=True)
        request.assert_not_called()
        self.assertEqual((target / "weights/file").read_bytes(), data)

    def test_resume_rejects_changed_complete_file_without_overwriting(self):
        target = self.root / "changed"
        target.mkdir()
        (target / "file").write_bytes(b"changed")
        lock = {"objects": [{"file": "file", "size": 7, "sha256": "0" * 64}]}
        with patch.object(downloads, "urlopen") as request:
            with self.assertRaisesRegex(ValueError, "checksum"):
                downloads.download(lock, target, resume=True)
        request.assert_not_called()
        self.assertEqual((target / "file").read_bytes(), b"changed")

    def test_resume_rejects_symlink_artifact_before_reading_or_fetching(self):
        target = self.root / "linked"
        target.mkdir()
        outside = self.root / "outside"
        outside.write_bytes(b"private")
        (target / "file").symlink_to(outside)
        with patch.object(downloads, "urlopen") as request:
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                downloads.download({"objects": [{"file": "file"}]}, target, resume=True)
        request.assert_not_called()
        self.assertEqual(outside.read_bytes(), b"private")


if __name__ == "__main__":
    unittest.main()
