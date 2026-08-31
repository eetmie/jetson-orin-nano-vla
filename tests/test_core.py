from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bench.backends.base import Backend, InferResult
from bench.backends.torch_xvla import _processor_tokenizer_contract
from bench.obs import Observation, make_obs
from bench.ort_profile import placement_verdict, summarize_profile
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


class ORTProfileTests(unittest.TestCase):
    def test_profile_summary_uses_measured_provider_events(self):
        events = [
            {"cat": "Session", "name": "model_run", "dur": 100},
            {
                "cat": "Node", "name": "trt_kernel_time", "dur": 80,
                "args": {
                    "provider": "TensorrtExecutionProvider",
                    "op_name": "TRTKernel",
                },
            },
            {
                "cat": "Node", "name": "shape_kernel_time", "dur": 20,
                "args": {
                    "provider": "CPUExecutionProvider",
                    "op_name": "Shape",
                },
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(events))
            summary = summarize_profile(path)
        self.assertEqual(
            summary["providers"]["TensorrtExecutionProvider"]["duration_ms"],
            0.08)
        verdict = placement_verdict({"graph": summary}, ["graph"])
        self.assertEqual(verdict["status"], "pass")
        self.assertIn("graph", verdict["fallbacks"])

    def test_placement_fails_when_expected_graph_has_no_trt_node(self):
        verdict = placement_verdict({
            "graph": {
                "providers": {
                    "CUDAExecutionProvider": {"event_count": 1},
                },
            },
        }, ["graph", "missing"])
        self.assertEqual(verdict["status"], "fail")
        self.assertEqual(verdict["missing_profiles"], ["missing"])
        self.assertEqual(
            verdict["graphs_without_trt_nodes"], ["graph", "missing"])


class ObservationFixtureTests(unittest.TestCase):
    def test_fixture_replays_aligned_pixels_and_raw_state(self):
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
            cv2.imwrite(str(images / "frame.png"), rgb[:, :, ::-1])
            state = np.array([1.5, -2.0, 3.25], dtype=np.float32)
            (root / "fixture.json").write_text(json.dumps({
                "schema_version": 1,
                "task": "dig",
                "views": 1,
                "image_hw": [4, 5],
                "state_dim": 3,
                "source_dataset": {"repo_id": "local/test"},
                "records": [{
                    "episode_index": 5,
                    "frame_index": 7,
                    "dataset_index": 11,
                    "state": state.tolist(),
                    "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                    "images": [{
                        "path": "images/frame.png",
                        "array_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                    }],
                }],
            }))
            source = make_obs(
                f"fixture:{root}", "dig", chunk_size=2, state_dim=3,
                max_action_dim=4, seed=9, n_views=1)
            observation = source[3]
            np.testing.assert_array_equal(observation.image, rgb)
            np.testing.assert_array_equal(observation.state, state)
            self.assertEqual(observation.index, 3)
            self.assertEqual(source.describe()["record_ids"][0]["episode_index"], 5)
            with self.assertRaisesRegex(ValueError, "does not match requested task"):
                make_obs(
                    f"fixture:{root}", "wrong", chunk_size=2, state_dim=3,
                    max_action_dim=4, seed=9, n_views=1)


class XVLAContractTests(unittest.TestCase):
    def test_engine_cache_manifest_rejects_stale_or_mixed_contents(self):
        import sys
        from types import ModuleType

        with patch.dict(sys.modules, {"onnxruntime": ModuleType("onnxruntime")}):
            from bench.vendor.xvla_split_ort import (
                _validate_engine_cache_manifest,
                _write_engine_cache_manifest,
            )

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache / "graph.engine").write_bytes(b"engine")
            identity = {"bundle": "abc", "precision": "fp16"}
            _write_engine_cache_manifest(cache, identity)
            self.assertEqual(
                _validate_engine_cache_manifest(cache, identity)["identity"],
                identity)

            (cache / "mixed.timing").write_bytes(b"timing")
            with self.assertRaisesRegex(ValueError, "missing, truncated, or mixed"):
                _validate_engine_cache_manifest(cache, identity)
            (cache / "mixed.timing").unlink()
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                _validate_engine_cache_manifest(
                    cache, {"bundle": "different", "precision": "fp16"})

    def test_saved_processor_contract_wins_over_raw_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({
                "tokenizer_max_length": 1024,
            }))
            (root / "policy_preprocessor.json").write_text(json.dumps({
                "steps": [{
                    "registry_name": "tokenizer_processor",
                    "config": {
                        "tokenizer_name": "facebook/bart-large",
                        "max_length": 50,
                        "padding": "max_length",
                        "padding_side": "right",
                        "truncation": True,
                    },
                }],
            }))
            contract = _processor_tokenizer_contract(root)
            self.assertEqual(contract["max_length"], 50)
            self.assertEqual(contract["tokenizer_name"], "facebook/bart-large")

    def test_schema_v2_bundle_resolves_physical_and_model_widths(self):
        from types import SimpleNamespace
        from bench.cli import Resolved

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bundle.json").write_text(json.dumps({
                "chunk_size": 50, "real_state_dim": 3, "real_action_dim": 4,
                "max_action_dim": 20, "valid_views": 1, "num_image_views": 3,
            }))
            args = SimpleNamespace(
                model=None, family="xvla", task="dig", fps=None, chunk_size=None,
                state_dim=None, action_dim=None, views=None, cam_slots=None,
                noise_width=None,
            )
            resolved = Resolved(args, root)
            self.assertEqual(resolved.state_dim, 3)
            self.assertEqual(resolved.action_dim, 4)
            self.assertEqual(resolved.noise_width, 20)
            self.assertEqual(resolved.views, 1)
            self.assertEqual(resolved.cam_slots, 3)

    def test_split_runtime_crosses_physical_boundary_once(self):
        import sys
        from types import ModuleType

        with patch.dict(sys.modules, {"onnxruntime": ModuleType("onnxruntime")}):
            from bench.vendor.xvla_split_ort import XVLASplitPolicy

        contract = {
            "state": {
                "dim": 3, "model_dim": 20,
                "normalization": {
                    "mode": "MEAN_STD", "eps": 1e-8,
                    "mean": [1, 2, 3], "std": [2, 4, 8],
                },
            },
            "action": {
                "dim": 4, "model_dim": 20,
                "normalization": {
                    "mode": "MEAN_STD", "eps": 1e-8,
                    "mean": [10, 20, 30, 40], "std": [1, 2, 3, 4],
                },
            },
        }
        policy = object.__new__(XVLASplitPolicy)
        policy.state_dim = 3
        policy.model_state_dim = 20
        policy.real_action_dim = 4
        policy.action_dim = 20
        policy.chunk_size = 2
        policy.steps = 1
        policy.denoise_input_mode = "x_t"
        policy.gripper_idx = ()
        policy.processor_contract = contract
        policy.rng = np.random.default_rng(0)
        policy.last_timings = {}
        policy.encode_observation = lambda images, task: np.zeros((1, 1, 1))
        captured = {}

        def denoise(x_t, t, proprio, cond_tokens):
            captured["proprio"] = proprio.copy()
            return np.ones_like(x_t)

        policy._denoise_step = denoise
        actions = policy.sample_actions(
            [], "dig", np.array([3, 6, 11], dtype=np.float32),
            x1=np.zeros((1, 2, 20), dtype=np.float32),
        )
        np.testing.assert_allclose(captured["proprio"][0, :3], [1, 1, 1])
        np.testing.assert_allclose(captured["proprio"][0, 3:], 0)
        np.testing.assert_allclose(policy.last_normalized_action, 1)
        np.testing.assert_allclose(actions[0], [11, 22, 33, 44])
        self.assertEqual(actions.shape, (2, 4))

    def test_malformed_saved_processor_contract_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "policy_preprocessor.json").write_text(json.dumps({"steps": []}))
            with self.assertRaisesRegex(ValueError, "no tokenizer_processor"):
                _processor_tokenizer_contract(root)

    def test_device_resident_denoise_keeps_split_outputs_on_cuda(self):
        import sys
        from types import ModuleType

        with patch.dict(sys.modules, {"onnxruntime": ModuleType("onnxruntime")}):
            from bench.vendor.xvla_split_ort import XVLASplitPolicy

        class Value:
            def __init__(self, array):
                self.array = np.asarray(array)

            def numpy(self):
                return self.array

        class Output:
            def __init__(self, name):
                self.name = name

        class Binding:
            def __init__(self):
                self.inputs = {}
                self.outputs = []
                self.output_devices = []

            def clear_binding_inputs(self):
                self.inputs.clear()

            def clear_binding_outputs(self):
                self.outputs.clear()

            def bind_cpu_input(self, name, value):
                self.inputs[name] = Value(value)

            def bind_ortvalue_input(self, name, value):
                self.inputs[name] = value

            def bind_output(self, name, device, device_id):
                self.output_devices.append((name, device, device_id))

            def get_outputs(self):
                return self.outputs

        class Session:
            def __init__(self, index):
                self.index = index
                self.binding = Binding()

            def get_outputs(self):
                return [Output("action" if self.index == 3 else "hidden_out")]

            def run_with_iobinding(self, binding):
                source = (binding.inputs.get("x_t") or binding.inputs.get("action")
                          or binding.inputs["hidden_in"])
                binding.outputs = [Value(source.numpy() + 1)]

        policy = object.__new__(XVLASplitPolicy)
        policy.denoise = [Session(i) for i in range(4)]
        policy._denoise_io = [session.binding for session in policy.denoise]
        static = {
            "proprio": Value(np.zeros((1, 20), dtype=np.float32)),
            "cond_tokens": Value(np.zeros((1, 2, 3), dtype=np.float32)),
        }

        output = policy._denoise_step_device_resident(
            np.zeros((1, 2, 20), dtype=np.float32), 1.0, static)

        np.testing.assert_allclose(output, 4)
        self.assertIs(policy._denoise_io[0].inputs["proprio"], static["proprio"])
        self.assertIs(policy._denoise_io[0].inputs["cond_tokens"],
                      static["cond_tokens"])
        for binding in policy._denoise_io:
            self.assertEqual(binding.output_devices[-1][1:], ("cuda", 0))

        x1_device = Value(np.ones((1, 2, 20), dtype=np.float32))
        action_device = Value(np.zeros((1, 2, 20), dtype=np.float32))
        fused = policy._denoise_step_device_resident_fused(
            x1_device, action_device, 1.0, static)
        self.assertIs(policy._denoise_io[0].inputs["x1"], x1_device)
        self.assertIs(policy._denoise_io[0].inputs["action"], action_device)
        self.assertNotIn("x_t", policy._denoise_io[0].inputs)
        np.testing.assert_allclose(fused.numpy(), 4)


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

    def test_cli_retains_parity_report_atomically(self):
        from bench.cli import cmd_parity

        chunks = np.zeros((2, 2, 2), dtype=np.float32).tolist()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.json"
            candidate = root / "candidate.json"
            output = root / "parity.json"
            write_result(signed_result("reference", chunks), reference)
            write_result(signed_result("candidate", chunks), candidate)

            code = cmd_parity(SimpleNamespace(
                paths=[str(reference), str(candidate)], reference="reference",
                out=str(output)))

            self.assertEqual(code, 0)
            report = json.loads(output.read_text())
            self.assertEqual(report["comparisons"][0]["verdict"], "PASS")


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
