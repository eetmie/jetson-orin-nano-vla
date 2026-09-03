# VENDORED from spark-projects @ 54f247c —
# orin-nano/evo1-runtime/evo1_runtime/split_ort.py
#
# This repository-side copy adds deterministic image/prompt preprocessing for the
# benchmark Observation contract. The graph layout, provider settings, serial engine
# prebuild, device-resident action cache, and Euler loop are the tested implementation.
"""EVO1 bootstrap split-ONNX inference on ONNX Runtime + TensorRT EP.

This runtime intentionally accepts only the checksummed, nondeployable bootstrap
bundle. Its action head is deterministic random initialization. It measures EVO1's
infrastructure cost and must never provide actions to a robot.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

BOOTSTRAP_WARNING = (
    "EVO1 bootstrap bundle: deterministic random action head; never use its actions "
    "to control a robot."
)
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"


def verify_bundle(bundle_dir: str | Path) -> dict:
    """Verify checksums and fail closed unless this is the marked bootstrap."""
    root = Path(bundle_dir).resolve()
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file():
        raise ValueError(f"bundle manifest is missing: {manifest}")
    checked = set()
    for line in manifest.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError(f"manifest path escapes bundle: {relative}")
        if not path.is_file():
            raise ValueError(f"manifest artifact is missing: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"bundle identity mismatch: {relative}")
        checked.add(relative)

    bundle = json.loads((root / "bundle.json").read_text())
    required = {"bundle.json", bundle["fixture"]["file"]}
    required.update(graph["file"] for graph in bundle["graphs"])
    missing = sorted(required - checked)
    if missing:
        raise ValueError(f"bundle artifacts absent from manifest: {missing}")
    if bundle.get("model") != "evo1" or bundle.get("schema_version") != 1:
        raise ValueError("unsupported EVO1 bundle schema")
    if (bundle.get("deployable") is not False or
            bundle.get("random_action_head") is not True):
        raise ValueError("runtime accepts only a marked nondeployable bootstrap bundle")
    if bundle.get("max_views") != 1 or bundle.get("valid_views") != 1:
        raise ValueError("runtime requires the tested one-camera EVO1 export")
    tokenizer_relative = Path(bundle["tokenizer"]["path"])
    tokenizer = (root / tokenizer_relative).resolve()
    if root not in tokenizer.parents:
        raise ValueError("bundle tokenizer path escapes the bundle")
    if not tokenizer.is_dir():
        raise ValueError(f"bundle tokenizer is missing: {tokenizer}")
    tokenizer_files = {
        path.relative_to(root).as_posix()
        for path in tokenizer.rglob("*")
        if path.is_file()
    }
    if not tokenizer_files:
        raise ValueError("bundle tokenizer is empty")
    unchecksummed = sorted(tokenizer_files - checked)
    if unchecksummed:
        raise ValueError(f"bundle tokenizer files absent from manifest: {unchecksummed}")
    return bundle


def make_session_options() -> "ort.SessionOptions":
    """Disable ORT fusions which manufacture unsupported FP16 contrib ops."""
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_cpu_mem_arena = False
    return options


def build_providers(cache_dir: str | Path, precision: str = "fp16") -> list:
    """Use the measured TensorRT -> CPU stack; CUDA fallback was not beneficial."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    available = set(ort.get_available_providers())
    providers: list = []
    if "TensorrtExecutionProvider" in available:
        providers.append(
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    "trt_fp16_enable": precision == "fp16",
                    "trt_bf16_enable": precision == "bf16",
                    "trt_layer_norm_fp32_fallback": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(cache),
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": str(cache),
                    "trt_max_workspace_size": int(
                        os.environ.get("TRT_WORKSPACE_MB", "512")
                    )
                    * (1 << 20),
                    "trt_builder_optimization_level": int(
                        os.environ.get("TRT_OPT_LEVEL", "2")
                    ),
                    "trt_min_subgraph_size": 5,
                },
            )
        )
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    if not providers:
        raise RuntimeError(f"no usable ONNX Runtime provider in {sorted(available)}")
    return providers


def _cpu_session(path: Path) -> "ort.InferenceSession":
    return ort.InferenceSession(
        str(path),
        sess_options=make_session_options(),
        providers=["CPUExecutionProvider"],
    )


def causal_mask(valid: np.ndarray) -> np.ndarray:
    """The additive causal + padding mask used by native LeRobot EVO1."""
    length = valid.shape[1]
    causal = np.tril(np.ones((length, length), dtype=bool))
    allowed = causal[None, None, :, :] & valid[:, None, None, :].astype(bool)
    return np.where(allowed, 0.0, -10000.0).astype(np.float32)


def preprocess_image(image_hwc_uint8: np.ndarray, size: int) -> np.ndarray:
    """Resize one RGB image with PIL bicubic and apply ImageNet normalization."""
    from PIL import Image

    image = np.asarray(image_hwc_uint8)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected uint8 HxWx3 RGB image, got {image.shape} {image.dtype}")
    resized = Image.fromarray(image).resize((size, size), Image.Resampling.BICUBIC)
    values = np.asarray(resized, dtype=np.float32) / 255.0
    values = (values - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None])


class Evo1SplitPolicy:
    """Fixed one-camera/320-token inference for bootstrap benchmarking only."""

    def __init__(
        self,
        bundle_dir: str | Path,
        cache_dir: str | Path,
        precision: str = "fp16",
        num_steps: int | None = None,
        *,
        allow_bootstrap: bool = False,
    ) -> None:
        if not allow_bootstrap:
            raise ValueError(BOOTSTRAP_WARNING + " Pass allow_bootstrap=True for benchmarks.")
        self.root = Path(bundle_dir).resolve()
        self.bundle = verify_bundle(self.root)
        self.cache_dir = Path(cache_dir).resolve()
        self.precision = precision
        self.steps = int(
            self.bundle["num_inference_timesteps"] if num_steps is None else num_steps
        )
        if self.steps <= 0:
            raise ValueError("num_steps must be positive")
        self.graphs = {graph["name"]: graph for graph in self.bundle["graphs"]}
        self.sessions: dict[str, ort.InferenceSession] = {}
        self.load_timings_s: dict[str, float] = {}
        self.last_timings: dict[str, float] = {}

        from transformers import AutoTokenizer

        tokenizer_path = self.root / self.bundle["tokenizer"]["path"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True
        )
        self._load("token_embedding", cpu_only=True)
        for name in (
            "vision_0",
            "vision_1",
            "vision_2",
            "vision_3",
            "language_0",
            "language_1",
            "language_2",
            "action_context",
            "action_step",
            "action_output",
        ):
            self._load(name, cpu_only=False)

        gpu = {"TensorrtExecutionProvider", "CUDAExecutionProvider"}
        action_names = ("action_step", "action_output")
        if not all(
            gpu.intersection(self.sessions[name].get_providers())
            for name in action_names
        ):
            raise ValueError("device-resident action requires GPU action sessions")
        self._action_io = {
            name: self.sessions[name].io_binding() for name in action_names
        }

    def _load(self, name: str, *, cpu_only: bool) -> None:
        path = self.root / self.graphs[name]["file"]
        started = time.perf_counter()
        if cpu_only:
            session = _cpu_session(path)
        else:
            session = ort.InferenceSession(
                str(path),
                sess_options=make_session_options(),
                providers=build_providers(self.cache_dir, self.precision),
            )
        self.sessions[name] = session
        self.load_timings_s[name] = time.perf_counter() - started

    def _chain(self, names: tuple[str, ...], feed: dict) -> np.ndarray:
        value = None
        for index, name in enumerate(names):
            if index:
                inputs = self.sessions[name].get_inputs()
                accepted = {item.name for item in inputs}
                feed = {key: item for key, item in feed.items() if key in accepted}
                feed[inputs[0].name] = value
            value = self.sessions[name].run(None, feed)[0]
        assert value is not None
        return value

    def _prompt_inputs(self, task: str) -> tuple[np.ndarray, np.ndarray]:
        count = int(self.bundle["image_seq_length"])
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * count + IMG_END_TOKEN
        prompt = f"Image-1: {image_tokens}\n{task.strip()}"
        encoded = self.tokenizer(
            prompt,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=int(self.bundle["seq_len"]),
        )
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        context_mask = np.asarray(encoded["attention_mask"], dtype=bool)
        actual = int(np.count_nonzero(input_ids == int(self.bundle["image_token_id"])))
        if actual != count:
            raise ValueError(
                f"prompt contains {actual} image tokens, expected {count}; text was truncated"
            )
        return input_ids, context_mask

    def _run(
        self,
        *,
        pixel_values: np.ndarray,
        input_ids: np.ndarray,
        context_mask: np.ndarray,
        state: np.ndarray,
        initial_noise: np.ndarray,
    ) -> dict:
        timings: dict[str, float] = {}
        started = time.perf_counter()
        image_features = self._chain(
            ("vision_0", "vision_1", "vision_2", "vision_3"),
            {"pixel_values": np.asarray(pixel_values, dtype=np.float32)},
        )
        timings["vision"] = time.perf_counter() - started

        started = time.perf_counter()
        input_ids = np.asarray(input_ids, dtype=np.int64)
        merged = self.sessions["token_embedding"].run(
            None, {"input_ids": input_ids}
        )[0].copy()
        positions = np.flatnonzero(
            input_ids.reshape(-1) == int(self.bundle["image_token_id"])
        )
        merged.reshape(-1, int(self.bundle["hidden_size"]))[positions] = (
            image_features.reshape(-1, int(self.bundle["hidden_size"]))
        )
        fused_tokens = self._chain(
            ("language_0", "language_1", "language_2"),
            {"hidden_in": merged, "causal_mask": causal_mask(context_mask)},
        )
        timings["language_with_cpu_embedding"] = time.perf_counter() - started

        started = time.perf_counter()
        cached = self.sessions["action_context"].run(
            None,
            {
                "fused_tokens": fused_tokens,
                "context_mask": np.asarray(context_mask, dtype=bool),
                "state": np.asarray(state, dtype=np.float32),
            },
        )
        timings["action_context"] = time.perf_counter() - started

        step_inputs = [item.name for item in self.sessions["action_step"].get_inputs()]
        cached_names = step_inputs[2:]
        if len(cached_names) != len(cached):
            raise ValueError("action cache graph contract does not match action step")
        started = time.perf_counter()
        cached_device = {
            name: ort.OrtValue.ortvalue_from_numpy(
                np.ascontiguousarray(value), "cuda", 0
            )
            for name, value in zip(cached_names, cached, strict=True)
        }
        timings["action_cache_upload"] = time.perf_counter() - started

        action = np.asarray(initial_noise, dtype=np.float32).copy()
        step_total = 0.0
        output_total = 0.0
        for index in range(self.steps):
            time_index = np.asarray(
                [min(int((index / self.steps) * 999), 999)], dtype=np.int64
            )
            started = time.perf_counter()
            step_io = self._action_io["action_step"]
            step_io.clear_binding_inputs()
            step_io.clear_binding_outputs()
            step_io.bind_cpu_input("action", np.ascontiguousarray(action))
            step_io.bind_cpu_input("time_index", time_index)
            for name, value in cached_device.items():
                step_io.bind_ortvalue_input(name, value)
            step_io.bind_output("action_hidden", "cuda", 0)
            self.sessions["action_step"].run_with_iobinding(step_io)
            action_hidden = step_io.get_outputs()[0]
            step_total += time.perf_counter() - started

            started = time.perf_counter()
            output_io = self._action_io["action_output"]
            output_io.clear_binding_inputs()
            output_io.clear_binding_outputs()
            output_io.bind_ortvalue_input("action_hidden", action_hidden)
            output_io.bind_output("velocity", "cpu", 0)
            self.sessions["action_output"].run_with_iobinding(output_io)
            velocity = output_io.get_outputs()[0].numpy()
            output_total += time.perf_counter() - started
            action += velocity / self.steps

        timings[f"action_step_x{self.steps}"] = step_total
        timings[f"action_output_x{self.steps}"] = output_total
        timings["total"] = sum(timings.values())
        self.last_timings = timings
        return {
            "vision": image_features,
            "fused": fused_tokens,
            "action": action,
            "timings_s": timings,
        }

    def sample_actions(
        self,
        image_hwc_uint8: np.ndarray,
        task: str,
        state: np.ndarray,
        noise: np.ndarray,
    ) -> np.ndarray:
        started = time.perf_counter()
        pixel_values = preprocess_image(
            image_hwc_uint8, int(self.bundle["image_size"])
        )
        input_ids, context_mask = self._prompt_inputs(task)
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        width = int(self.bundle["max_state_dim"])
        if state.size > width:
            raise ValueError(f"state has {state.size} values, bundle supports {width}")
        padded_state = np.zeros((1, width), dtype=np.float32)
        padded_state[0, : state.size] = state
        noise = np.asarray(noise, dtype=np.float32)
        expected_noise = (
            1,
            int(self.bundle["chunk_size"]),
            int(self.bundle["max_action_dim"]),
        )
        if noise.shape != expected_noise:
            raise ValueError(f"noise has shape {noise.shape}, expected {expected_noise}")
        preprocess_s = time.perf_counter() - started
        output = self._run(
            pixel_values=pixel_values,
            input_ids=input_ids,
            context_mask=context_mask,
            state=padded_state,
            initial_noise=noise,
        )
        output["timings_s"]["preprocess"] = preprocess_s
        output["timings_s"]["total"] += preprocess_s
        self.last_timings = output["timings_s"]
        return output["action"][0]

    def run_fixture(self, fixture: "np.lib.npyio.NpzFile") -> dict:
        """Run the exact native-reference tensors through all eleven graphs."""
        return self._run(
            pixel_values=fixture["pixel_values"],
            input_ids=fixture["input_ids"],
            context_mask=fixture["context_mask"],
            state=fixture["state"],
            initial_noise=fixture["initial_noise"],
        )


_BUILD_ONE = r'''
import json, os, resource, sys, time
from pathlib import Path
import numpy as np
import onnxruntime as ort

graph, cache, precision = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
cache.mkdir(parents=True, exist_ok=True)
options = ort.SessionOptions()
options.log_severity_level = 3
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
options.enable_cpu_mem_arena = False
trt = {
    "device_id": 0,
    "trt_fp16_enable": precision == "fp16",
    "trt_bf16_enable": precision == "bf16",
    "trt_layer_norm_fp32_fallback": True,
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": str(cache),
    "trt_timing_cache_enable": True,
    "trt_timing_cache_path": str(cache),
    "trt_max_workspace_size": int(os.environ.get("TRT_WORKSPACE_MB", "512")) * (1 << 20),
    "trt_builder_optimization_level": int(os.environ.get("TRT_OPT_LEVEL", "2")),
    "trt_min_subgraph_size": 5,
}
providers = [("TensorrtExecutionProvider", trt), "CPUExecutionProvider"]
started = time.perf_counter()
session = ort.InferenceSession(str(graph), sess_options=options, providers=providers)
load_s = time.perf_counter() - started
types = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(bool)": bool,
}
feed = {}
for item in session.get_inputs():
    if any(not isinstance(dim, int) for dim in item.shape):
        raise ValueError(f"dynamic build input is unsupported: {item.name} {item.shape}")
    feed[item.name] = np.zeros(item.shape, dtype=types[item.type])
session.run(None, feed)
print("BUILD_RESULT " + json.dumps({
    "graph": graph.stem,
    "load_s": round(load_s, 3),
    "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3),
    "providers": session.get_providers(),
}))
'''


def prebuild_engines(
    bundle_dir: str | Path,
    cache_dir: str | Path,
    precision: str = "fp16",
) -> list[dict]:
    """Build/load each TRT engine in a fresh process so builder memory is returned."""
    root = Path(bundle_dir).resolve()
    bundle = verify_bundle(root)
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    results = []
    for graph in bundle["graphs"]:
        name = graph["name"]
        if name == "token_embedding":
            continue
        print(f"prebuild {name} ({graph['size_mb']} MB)...", flush=True)
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                _BUILD_ONE,
                str(root / graph["file"]),
                str(cache),
                precision,
            ],
            capture_output=True,
            text=True,
            env=dict(os.environ, TRT_DROP_CUDA_EP="1"),
        )
        if process.returncode:
            print(process.stdout, end="")
            print(process.stderr, end="", file=sys.stderr)
            raise RuntimeError(f"TensorRT prebuild failed for {name}")
        line = next(
            value
            for value in process.stdout.splitlines()
            if value.startswith("BUILD_RESULT ")
        )
        result = json.loads(line.removeprefix("BUILD_RESULT "))
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    if not list(cache.glob("*.engine")):
        raise RuntimeError("TensorRT prebuild produced no engine files")
    return results
