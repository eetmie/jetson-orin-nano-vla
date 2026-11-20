"""Benchmark the fixed LIBERO LaWAM proof-of-concept bundle on a Jetson."""

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bench.backends.base import Backend, InferResult
from bench.obs import SyntheticObs
from bench.runner import run_benchmark, write_result
from bench.vendor.groot_trt import Engines
from bench.vendor.trt_device import DevBuf, Device

TASK = "Pick up the cube."
FRONT = (
    [f"vis{i}" for i in range(4)]
    + [f"lang{i}" for i in range(4)]
    + ["dino", "latent", "kv"]
)
KV = ("kt", "vt", "ki", "vi")
FLOW = [f"g{i}" for i in range(4)]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def prepare(images):
    rgb = np.stack(images).transpose(0, 3, 1, 2).astype(np.float32) / 255
    primary = (rgb[:1] - MEAN) / STD
    patches = (2 * rgb - 1).reshape(2, 3, 8, 2, 16, 8, 2, 16)
    patches = patches.transpose(0, 2, 5, 3, 6, 1, 4, 7)
    pixels = np.repeat(patches[:, :, :, :, :, :, None], 2, axis=6).reshape(512, 1536)
    return pixels, primary


class LaWAMProbe(Backend):
    name = "lawam-libero-split-poc"
    noise_injected = True

    def __init__(self, bundle):
        self.bundle = bundle
        self.parity = {}
        self.ring_parity = {}

    def constant(self, array):
        buf = DevBuf(array.shape, array.dtype)
        self.device.set_once(buf, array)
        return buf

    def load(self):
        engines = Engines(SimpleNamespace(names=FRONT + FLOW), self.bundle)
        self.device = Device(engines)
        self.bound = {
            "pixels": self.device.buf_for("vis0", "pixels"),
            "primary": self.device.buf_for("dino", "primary"),
        }
        for stage in FRONT:
            self.bound.update(self.device.outputs(stage))
        self.hidden = [self.device.buf_for("g0", "out") for _ in range(2)]
        self.actions = [self.device.buf_for("g3", "out") for _ in range(2)]
        self.initial = self.device.buf_for("g0", "actions")
        with np.load(self.bundle / "fixture.npz") as f:
            self.pos = self.constant(f["pos"])
            self.times = [self.constant(t[None]) for t in f["times"]]
        with np.load(self.bundle / "post.npz") as f:
            self.lo, self.hi = f["lo"][:6], f["hi"][:6]
        with np.load(self.bundle / "input.npz") as f:
            for key in ("pixels", "primary"):
                self.bound[key].host()[...] = f[key]
            self.initial.host()[...] = f["noise"][:, : self.initial.shape[1]]
            reference = f["reference"].astype(np.float64)
        with np.load(self.bundle / "ring-reference.npz") as refs:
            self.references = {
                int(index): refs[index].astype(np.float64) for index in refs.files
            }
        with np.load(self.bundle / "ring-sdpa-reference.npz") as refs:
            self.sdpa_references = {
                int(index): refs[index].astype(np.float64) for index in refs.files
            }
        with np.load(self.bundle / "ring-fp32-reference.npz") as refs:
            self.fp32_references = {
                int(index): refs[index].astype(np.float64) for index in refs.files
            }
        self.enqueue()
        self.device.sync()
        # The front stays uncaptured: TensorRT's lang0 engine fails inside stream capture.
        self.graph = self.device.capture(self.enqueue_flow)
        self.enqueue_front()
        self.device.launch(self.graph)
        self.device.sync()
        actual = self.actions[1].host()[:, :8, :7].astype(np.float64)
        np.save(self.bundle / "full-actions.npy", actual)
        cosine = float(
            np.vdot(actual, reference)
            / (np.linalg.norm(actual) * np.linalg.norm(reference))
        )
        if not np.isfinite(actual).all():
            for stage in FRONT:
                for key in self.device.spec(stage):
                    if self.device.spec(stage)[key][2]:
                        continue
                    value = self.bound[key]
                    self.device.download(value)
                    self.device.sync()
                    print(
                        stage,
                        key,
                        "finite fraction",
                        np.isfinite(value.host()).mean(),
                        flush=True,
                    )
            raise ValueError("Non-finite fixture actions")
        self.parity = {
            "cosine": cosine,
            "max_abs": float(np.abs(actual - reference).max()),
            "rmse": float(np.sqrt(np.mean((actual - reference) ** 2))),
            "threshold": 0.99999,
            "pass": cosine >= 0.99999,
            "reference": "FP32 LeRobot LaWAM normalized 8x7 action chunk, observation 0",
        }
        if cosine < 0.99999:
            context = self.bound["context"]
            self.device.download(context)
            self.device.sync()
            np.save(self.bundle / "full-context.npy", context.host())
        print(self.parity, flush=True)

    def enqueue(self):
        self.enqueue_front()
        self.enqueue_flow()

    def enqueue_front(self):
        for key in ("pixels", "primary"):
            self.device.upload_staged(self.bound[key])
        self.device.upload_staged(self.initial)
        for stage in FRONT:
            self.device.enqueue(stage, self.bound)

    def enqueue_flow(self):
        actions = self.initial
        for step, timestep in enumerate(self.times):
            for i, stage in enumerate(FLOW):
                out = self.hidden[i % 2] if i < 3 else self.actions[step % 2]
                self.device.enqueue(
                    stage,
                    {
                        "x": self.hidden[(i + 1) % 2],
                        "context": self.bound["context"],
                        "t": timestep,
                        "pos": self.pos,
                        "actions": actions,
                        "out": out,
                        **{k: self.bound[f"{k}{i}"] for k in KV},
                    },
                )
            actions = out
        self.device.download(actions)

    def infer(self, obs):
        start = time.perf_counter()
        pixels, primary = prepare(obs.images)
        self.bound["pixels"].host()[...] = pixels
        self.bound["primary"].host()[...] = primary
        self.initial.host()[:, :8] = obs.noise[:, :8]
        self.initial.host()[:, 8:] = 0
        prepared = time.perf_counter()
        self.enqueue_front()
        self.device.launch(self.graph)
        self.device.sync()
        normalized = self.actions[1].host()[0, :8, :7].astype(np.float32)
        if obs.index in self.references and obs.index not in self.ring_parity:
            ref = self.references[obs.index].reshape(-1)
            actual = normalized.ravel().astype(np.float64)
            cosine = float(
                np.dot(actual, ref) / (np.linalg.norm(actual) * np.linalg.norm(ref))
            )
            fp32 = self.fp32_references[obs.index].reshape(-1)
            fp32_cosine = float(
                np.dot(actual, fp32) / (np.linalg.norm(actual) * np.linalg.norm(fp32))
            )
            sdpa = self.sdpa_references[obs.index].reshape(-1)
            sdpa_cosine = float(
                np.dot(actual, sdpa) / (np.linalg.norm(actual) * np.linalg.norm(sdpa))
            )
            self.ring_parity[obs.index] = {
                "eager_cosine": cosine,
                "eager_max_abs": float(np.max(np.abs(ref - actual))),
                "sdpa_cosine": sdpa_cosine,
                "sdpa_max_abs": float(np.max(np.abs(sdpa - actual))),
                "sdpa_pass": sdpa_cosine >= 0.99999,
                "fp32_cosine": fp32_cosine,
                "fp32_max_abs": float(np.max(np.abs(fp32 - actual))),
                "fp32_pass": fp32_cosine >= 0.99999,
            }
        actions = np.clip(normalized, -1, 1)
        actions[:, :6] = (actions[:, :6] + 1) / 2 * (self.hi - self.lo) + self.lo
        actions[:, 6] = np.where(actions[:, 6] >= 0.5, 1, -1)
        return InferResult(
            actions,
            {
                "total": (time.perf_counter() - start) * 1000,
                "preprocess": (prepared - start) * 1000,
            },
        )

    def meta(self):
        return {
            "chunk_size": 8,
            "cameras": 2,
            "flow_steps": 10,
            "task": TASK,
            "embodiment_id": 25,
            "fixture_parity": self.parity,
            "ring_parity": self.ring_parity,
            "checkpoint_revision": "f3d7b131bf612127a6e384ea84f563bc23564832",
        }

    def close(self):
        if hasattr(self, "graph"):
            self.device.destroy_graph(self.graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--duration", type=float, default=300)
    args = parser.parse_args()
    source = SyntheticObs(
        task=TASK, chunk_size=50, state_dim=32, n_views=2, h=256, w=256
    )
    result = run_benchmark(
        LaWAMProbe(args.bundle), source, duration_s=args.duration, fps=20
    )
    write_result(result, args.bundle / "full-result.json")
    if result["status"] != "ok":
        raise RuntimeError(result["error"])
