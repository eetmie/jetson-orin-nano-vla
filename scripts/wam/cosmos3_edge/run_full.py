"""Execute a fixed-prompt joint video/action Cosmos3 Edge pack."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from rgb import prepare_canvas
from bench.vendor.groot_trt import Engines
from bench.vendor.trt_device import Device
from bench.backends.base import Backend, InferResult
from bench.obs import SyntheticObs
from bench.runner import run_benchmark, write_result

GROUPS = [f"group{i:02d}" for i in range(14)]


class Edge(Backend):
    name = "cosmos3-edge-libero-trt-poc"
    noise_injected = True

    def __init__(self, root):
        self.root = root

    def load(self):
        with np.load(self.root / "fixture.npz") as f:
            self.times = f["times"]
            self.coeff = f["coeff"]
            self.grid = f["grid"][0]
        with np.load(self.root / "observations.npz") as f:
            self.resize = {k: f[k] for k in ("xi", "yi", "xw", "yw", "canvas_hw")}
            self.lo, self.hi = f["lo"], f["hi"]
        names = ["vae", "first", *GROUPS, "last"]
        eng = Engines(SimpleNamespace(names=names), self.root)
        d = self.d = Device(eng)
        self.canvas = d.buf_for("vae", "canvas")
        self.condition = d.buf_for("vae", "condition")
        self.first = {k: d.buf_for("first", k) for k in d.spec("first")}
        self.ping = [self.first["out"], d.buf_for(GROUPS[0], "out")]
        self.last = d.outputs("last")

    def infer(self, obs):
        start = time.perf_counter()
        canvas = prepare_canvas(obs.images, self.resize)
        self.d.upload(self.canvas, canvas)
        self.d.enqueue("vae", dict(canvas=self.canvas, condition=self.condition))
        self.d.download(self.condition)
        self.d.sync()
        z = self.condition.host()
        T, H, W = map(int, self.grid)
        h, w = H // 2, W // 2
        cond = (
            z[:, :, 0]
            .reshape(1, 48, h, 2, w, 2)
            .transpose(0, 2, 4, 3, 5, 1)
            .reshape(1, -1, 192)
        )
        rng = np.random.default_rng([1234, obs.index, 11])
        video = rng.standard_normal((1, T, h * w, 192), dtype=np.float32)
        video[:, 0] = cond
        act = obs.noise.copy()
        act[:, :, 10:] = 0
        split = video.size
        state = np.concatenate((video.ravel(), act.ravel()))
        shape = video.shape
        history = np.zeros((31, state.size), np.float32)
        history[0] = state
        for step, t in enumerate(self.times):
            self.d.upload(self.first["patches"], state[:split].reshape(shape))
            self.d.upload(self.first["actions"], state[split:].reshape(act.shape))
            self.d.upload(self.first["time"], t[None])
            self.d.enqueue("first", self.first)
            for i, name in enumerate(GROUPS):
                self.d.enqueue(
                    name, dict(hidden=self.ping[i % 2], out=self.ping[1 - i % 2])
                )
            self.d.enqueue("last", dict(hidden=self.ping[len(GROUPS) % 2], **self.last))
            for value in self.last.values():
                self.d.download(value)
            self.d.sync()
            v = self.last["video_velocity"].host().reshape(shape)
            a = self.last["action_velocity"].host().reshape(act.shape)
            v[:, 0] = 0
            a[:, :, 10:] = 0
            history[step + 1, :split] = v.ravel()
            history[step + 1, split:] = a.ravel()
            state = self.coeff[step, : step + 2] @ history[: step + 2]
        action = state[split:].reshape(act.shape)[0, :, :10]
        chunk = (action + 1) * 0.5 * (self.hi - self.lo) + self.lo
        return InferResult(chunk, {"total": (time.perf_counter() - start) * 1000})

    def meta(self):
        return {
            "checkpoint": "lerobot/Cosmos3-Edge-Policy-LIBERO",
            "prompt": "Pick up the cube.",
            "chunk_size": 16,
            "denoising_steps": 30,
            "prompt_KV": "precomputed for fixed text",
            "canvas": "resize, composite and normalize in timed call",
            "output": "quantile-unnormalized 10D policy actions",
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--platform", required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    src = SyntheticObs(
        task="Pick up the cube.",
        chunk_size=16,
        state_dim=8,
        max_action_dim=64,
        h=256,
        w=256,
        n_views=2,
    )
    report = run_benchmark(
        Edge(root),
        src,
        iters=args.iters,
        duration_s=300 if args.platform == "jetson" else None,
        warmup=0,
        idle_s=2,
        fps=20,
        save_chunks=2,
    )
    write_result(report, root / (args.platform + "-full.json"))
    print(
        json.dumps(
            {
                k: report.get(k)
                for k in ("status", "error", "latency_ms", "chunk_shape", "measurement")
            }
        ),
        flush=True,
    )
    if report["status"] != "ok":
        raise RuntimeError(report.get("traceback"))
