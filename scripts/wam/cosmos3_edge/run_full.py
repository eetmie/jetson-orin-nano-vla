"""Execute a fixed-prompt joint video/action Cosmos3 Edge pack."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from rgb import prepare_canvas
from bench.vendor.groot_trt import Engines
from bench.vendor.trt_device import DevBuf, Device
from bench.backends.base import Backend, InferResult
from bench.obs import SyntheticObs
from bench.runner import run_benchmark, write_result

GROUPS = [f"group{i:02d}" for i in range(14)]


class Edge(Backend):
    name = "cosmos3-edge-libero-trt-poc"
    noise_injected = True

    def __init__(self, root):
        self.root = root
        self.ring_parity = {}

    def load(self):
        with np.load(self.root / "fixture.npz") as f:
            self.times = f["times"]
            self.coeff = f["coeff"]
            self.grid = f["grid"][0]
        with np.load(self.root / "observations.npz") as f:
            self.resize = {k: f[k] for k in ("xi", "yi", "xw", "yw", "canvas_hw")}
            self.lo, self.hi = f["lo"], f["hi"]
        with np.load(self.root / "ring-reference.npz") as f:
            self.references = {k: f[k].astype(np.float64) for k in f.files}
        names = ["vae", "first", *GROUPS, "last", "combine"]
        eng = Engines(SimpleNamespace(names=names), self.root)
        d = self.d = Device(eng)
        self.canvas = d.buf_for("vae", "canvas")
        self.condition = d.buf_for("vae", "condition")
        first = d.spec("first")
        self.ping = [d.buf_for("first", "out"), d.buf_for(GROUPS[0], "out")]
        last = d.spec("last")
        # The whole 30-step trajectory stays on the GPU: row 0 of `history` is the start
        # state, row k+1 the velocity of step k, and `combine` makes the next state.
        self.history = DevBuf(d.spec("combine")["history"][0], np.float32)
        n = self.history.shape[1] * 4
        self.split = int(np.prod(first["patches"][0])) * 4
        self.state = DevBuf((1, self.history.shape[1]), np.float32)
        d.zero_once(self.history)
        steps = len(self.times)
        coeff = DevBuf((steps, steps + 1, 1), np.float32)
        d.set_once(coeff, self.coeff[:, :, None])
        times = DevBuf(self.times.shape, np.float32)
        d.set_once(times, self.times)
        self.start = DevBuf.view(self.history, 0, (1, self.history.shape[1]))

        def at(buf, offset, tensor, spec):
            return DevBuf.view(buf, offset, spec[tensor][0])

        self.steps = []
        for k in range(steps):
            src = self.history if k == 0 else self.state
            self.steps.append(
                dict(
                    first=dict(
                        patches=at(src, 0, "patches", first),
                        actions=at(src, self.split, "actions", first),
                        time=DevBuf.view(times, k * times.shape[1] * 4, first["time"][0]),
                        out=self.ping[0],
                    ),
                    last=dict(
                        hidden=self.ping[len(GROUPS) % 2],
                        video_velocity=at(self.history, (k + 1) * n, "video_velocity", last),
                        action_velocity=at(
                            self.history, (k + 1) * n + self.split, "action_velocity", last
                        ),
                    ),
                    combine=dict(
                        history=self.history,
                        coeff=DevBuf.view(coeff, k * (steps + 1) * 4, (steps + 1, 1)),
                        state=self.state,
                    ),
                )
            )
        self.actions = at(self.state, self.split, "actions", first)
        self.sample()
        d.sync()
        self.graph = d.capture(self.sample)

    def sample(self):
        for k in self.steps:
            self.d.enqueue("first", k["first"])
            for i, name in enumerate(GROUPS):
                self.d.enqueue(name, dict(hidden=self.ping[i % 2], out=self.ping[1 - i % 2]))
            self.d.enqueue("last", k["last"])
            self.d.enqueue("combine", k["combine"])
        self.d.download(self.actions)

    def close(self):
        if hasattr(self, "graph"):
            self.d.destroy_graph(self.graph)

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
        self.d.upload(self.start, np.concatenate((video.ravel(), act.ravel())))
        self.d.launch(self.graph)
        self.d.sync()
        action = self.actions.host()[0, :, :10].copy()
        key = str(obs.index)
        if key in self.references and key not in self.ring_parity:
            got = action.ravel().astype(np.float64)
            row = {}
            for name, ref in (("fp32", key), ("bf16", "bf16/" + key)):
                want = self.references[ref].ravel()
                row[name + "_cosine"] = float(
                    got @ want / (np.linalg.norm(got) * np.linalg.norm(want))
                )
                row[name + "_max_abs"] = float(np.abs(got - want).max())
            row["fp32_pass"] = row["fp32_cosine"] >= 0.99999
            self.ring_parity[key] = row
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
            "ring_parity": self.ring_parity,
            "ring_reference": "stock policy FP32 (config default) and BF16, same noise",
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
