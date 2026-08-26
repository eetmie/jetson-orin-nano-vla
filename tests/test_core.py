from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bench.backends.base import Backend, InferResult
from bench.obs import Observation
from bench.parity import (ResultLoadError, attach_comparison_signature, compare,
                          cosine, load_results, parity_report)
from bench.runner import latency_stats, run_benchmark, write_result


def signed_result(label: str, chunks, indices=(1, 2), *, seed=1234,
                  input_hashes=("input-1", "input-2"),
                  noise_hashes=("noise-1", "noise-2")):
    result = {
        "label": label,
        "backend": "torch-smolvla" if label == "reference" else "ort-split-trt",
        "status": "ok",
        "model": {
            "key": "smolvla-base",
            "family": "smolvla",
            "task": "pick",
            "views": 2,
            "cam_slots": 2,
            "state_dim": 6,
            "action_dim": 2,
        },
        "obs": {"kind": "synthetic", "hw": [2, 2], "views": 2, "seed": seed},
        "meta": {
            "chunk_size": 2,
            "action_dim": 2,
            "num_steps": 10,
            "resize": [512, 512],
            "tokenizer": "tokenizer",
            "stats_sha256": "stats",
        },
        "chunk_shape": [2, 2],
        "saved_chunks": {
            "noise_injected": True,
            "obs_indices": list(indices),
            "seed": seed,
            "input_sha256": list(input_hashes),
            "noise_sha256": list(noise_hashes),
            "chunks": chunks,
        },
    }
    return attach_comparison_signature(result)


class StatisticsTests(unittest.TestCase):
    def test_ordered_quartiles_keep_remainder(self):
        stats = latency_stats([1, 1, 1, 1, 100])
        self.assertEqual(stats["quartile_means_in_order"], [1.0, 1.0, 1.0, 100.0])
        self.assertGreater(stats["drift_q4_vs_q1_pct"], 0)

    def test_percentile_is_interpolated(self):
        self.assertEqual(latency_stats([0, 100])["p50"], 50.0)

    def test_empty_latency_rejected(self):
        with self.assertRaises(ValueError):
            latency_stats([])


class ParityTests(unittest.TestCase):
    def test_equal_zero_vectors_pass(self):
        zeros = np.zeros((2, 2, 2), dtype=np.float32).tolist()
        reference = signed_result("reference", zeros)
        candidate = signed_result("candidate", zeros)
        verdict = compare(reference, candidate)
        self.assertEqual(cosine(np.zeros(2), np.zeros(2)), 1.0)
        self.assertEqual(verdict["verdict"], "PASS")
        self.assertEqual(verdict["cosine_min"], 1.0)

    def test_chunks_align_by_observation_index(self):
        chunks = [
            np.full((2, 2), 1, dtype=np.float32).tolist(),
            np.full((2, 2), 2, dtype=np.float32).tolist(),
        ]
        reference = signed_result("reference", chunks)
        candidate = signed_result(
            "candidate", list(reversed(chunks)), indices=(2, 1),
            input_hashes=("input-2", "input-1"),
            noise_hashes=("noise-2", "noise-1"))
        verdict = compare(reference, candidate)
        self.assertEqual(verdict["verdict"], "PASS")
        self.assertEqual(verdict["obs_indices"], [1, 2])

    def test_identity_mismatch_fails_closed(self):
        zeros = np.zeros((2, 2, 2), dtype=np.float32).tolist()
        reference = signed_result("reference", zeros)
        candidate = signed_result("candidate", zeros, seed=9)
        self.assertEqual(compare(reference, candidate)["verdict"], "IDENTITY MISMATCH")

    def test_nonfinite_action_fails(self):
        chunks = np.zeros((2, 2, 2), dtype=np.float32)
        candidate_chunks = chunks.copy()
        candidate_chunks[0, 0, 0] = np.nan
        reference = signed_result("reference", chunks.tolist())
        candidate = signed_result("candidate", candidate_chunks.tolist())
        self.assertEqual(compare(reference, candidate)["verdict"], "FAIL")

    def test_report_requires_explicit_reference_and_candidate(self):
        zeros = np.zeros((2, 2, 2), dtype=np.float32).tolist()
        reference = signed_result("reference", zeros)
        self.assertIn("error", parity_report([reference]))
        self.assertIn("error", parity_report([reference], "reference"))

    def test_result_loading_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ResultLoadError):
                load_results([root / "missing"])
            (root / "bad.json").write_text("{")
            with self.assertRaises(ResultLoadError):
                load_results([root])


class FakeObs:
    seed = 7

    def __init__(self, events):
        self.events = events

    def describe(self):
        return {"kind": "fake", "views": 1, "seed": self.seed}

    def __getitem__(self, index):
        self.events.append(("obs", index))
        return Observation(
            index=index,
            images=[np.full((2, 2, 3), index, dtype=np.uint8)],
            state=np.asarray([index], dtype=np.float32),
            task="test",
            noise=np.full((1, 2, 2), index, dtype=np.float32),
        )


class FakeBackend(Backend):
    name = "fake"
    noise_injected = True

    def __init__(self, events, bad=False):
        self.events = events
        self.bad = bad
        self.calls = 0

    def load(self):
        self.events.append(("load", None))

    def meta(self):
        return {"chunk_size": 2, "num_steps": 1, "action_dim": 2}

    def infer(self, obs):
        self.events.append(("infer", obs.index))
        self.calls += 1
        chunk = np.full((2, 2), obs.index, dtype=np.float32)
        if self.bad:
            chunk[0, 0] = np.nan
        return InferResult(chunk, {"total": 1.0, "model": 0.8})


class FakeMonitor:
    def __init__(self):
        self.windows = []

    def start(self):
        pass

    def stop(self):
        pass

    @contextmanager
    def window(self, name):
        yield
        self.windows.append(name)

    def summary(self):
        return {"windows": {name: {"n": 1} for name in self.windows}}


class FakeProcWatch:
    def __init__(self, pids):
        self.windows = []

    def start(self):
        pass

    def stop(self):
        pass

    def rss_now_mb(self):
        return 1.0

    def mark(self, name, t0, t1):
        self.windows.append(name)

    def summary(self):
        return {"windows": {name: {"n": 1} for name in self.windows}}


class RunnerTests(unittest.TestCase):
    def run_fake(self, *, bad=False):
        events = []
        backend = FakeBackend(events, bad=bad)
        with (patch("bench.runner.make_monitor", return_value=FakeMonitor()),
              patch("bench.runner.ProcWatch", FakeProcWatch),
              patch("bench.runner.collect_env", return_value={}),
              patch("bench.runner.time.sleep", return_value=None)):
            result = run_benchmark(
                backend, FakeObs(events), iters=3, warmup=1, idle_s=0,
                save_chunks=1, fps=30, obs_ring_size=3)
        return result, events, backend

    def test_observations_materialize_before_load_and_counts_are_exact(self):
        result, events, backend = self.run_fake()
        self.assertEqual(events[:4], [("obs", 0), ("obs", 1), ("obs", 2),
                                      ("load", None)])
        self.assertEqual(backend.calls, 5)  # first + warmup + 3 measured
        self.assertEqual(result["measurement"]["completed_calls"], 3)
        self.assertEqual(result["saved_chunks"]["obs_indices"], [2])
        self.assertEqual(len(result["saved_chunks"]["input_sha256"]), 1)
        self.assertEqual(result["validity"]["execution"], "pass")
        self.assertEqual(result["validity"]["instrumentation"], "pass")

    def test_nonfinite_output_fails_execution_contract(self):
        result, _, _ = self.run_fake(bad=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["validity"]["execution"], "fail")
        self.assertIn("NaN or infinity", result["error"])

    def test_zero_iterations_rejected(self):
        with self.assertRaisesRegex(ValueError, "iters must be positive"):
            run_benchmark(FakeBackend([]), FakeObs([]), iters=0)

    def test_atomic_writer_rejects_nonstandard_nan_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text("old")
            with self.assertRaises(ValueError):
                write_result({"bad": float("nan")}, output)
            self.assertEqual(output.read_text(), "old")
            write_result({"ok": True}, output)
            self.assertEqual(json.loads(output.read_text()), {"ok": True})


if __name__ == "__main__":
    unittest.main()
