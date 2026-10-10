# ─────────────────────────────────────────────────────────────────────────────
# Ported from the author's pi0.5 Orin Nano prototype (spark-projects,
# pi05-spark-inference/prototype/pi05_fp16_full_20261010: build_template.py,
# refit_pilot.py, run_full.py). Template builds with refit, the cudaMalloc allocator,
# shared scratch and the inference chain are that tested implementation; here they sit
# on groot_trt's Engines and trt_device's Device, so the chain is device-resident and
# replayable as a CUDA graph like the other families.
# ─────────────────────────────────────────────────────────────────────────────
"""pi0.5 split inference on the TensorRT runtime: FP16 weights, refitted templates.

Every SigLIP layer has the same graph, and so does every Gemma language layer and every
action-expert layer. The bundle ships one ONNX template per layer kind (7 kinds) plus
each of the 67 components' own weights; the board builds each template once as a
weight-stripped, refittable plan and refits a copy per component at load. So build
memory is one layer's, and the plans on disk are a few MB.

Per observation (3 image slots, 2 real cameras; 10 Euler steps):

    stem, vision_00..26, vision_tail   per slot -> 256 prefix rows each
    prompt rows                        mapped FP16 embedding, gathered once per task
    language_00..17                    prefix pass; writes each layer's K/V
    10 x: action_input, action_00..17 (cached AdaRMS modulation), action_output,
          actions += -0.1 * velocity (FP32)
    host: quantile unnormalization of the first 7 dims
"""

from __future__ import annotations

import ctypes
import functools
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .groot_trt import (Engines, _ck, _cmp, _cuda, _cudart_version, _malloc, _sha256,
                        _trt_version)

MIB = 1 << 20
MASK_NEG = np.float32(-2.3819763e38)
EULER_DT = np.float32(-0.1)


class Bundle:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.b = b = json.loads((self.root / "bundle.json").read_text())
        if b.get("policy") != "pi05_libero" or not b.get("complete"):
            raise ValueError(f"{root} is not a complete pi0.5 LIBERO split bundle")
        self.components = b["components"]
        self.names = [c["label"] for c in self.components]
        self.kinds = list(dict.fromkeys(c["kind"] for c in self.components))
        self.image_keys = b["image_keys"]
        self.slots = int(b["image_slots"])
        self.prefix = int(b["prefix_tokens"])
        self.prompt = int(b["prompt_capacity"])
        self.horizon, self.width = int(b["action_horizon"]), int(b["action_dim"])
        self.steps = int(b["num_steps"])
        self.n_vision = sum(c["kind"] == "vision" for c in self.components)
        self.n_lang = sum(c["kind"] == "language" for c in self.components)
        self.n_action = sum(c["kind"] == "action" for c in self.components)
        stats = json.loads((self.root / "norm_stats.json").read_text())["norm_stats"]["actions"]
        self.q01, self.q99 = np.asarray(stats["q01"]), np.asarray(stats["q99"])
        self.embed = np.load(self.root / "embedding_fp16.npy", mmap_mode="r")
        self.rope_cos = np.load(self.root / "rope_cos_fp16.npy")
        self.rope_sin = np.load(self.root / "rope_sin_fp16.npy")
        self.styles = np.load(self.root / "conditioning_10step_fp16.npy")
        self.tasks = b.get("tasks", {})
        # Compact layout: only the real cameras and the prompt's real tokens, packed in
        # order into a shorter static prefix; the masked camera is never computed.
        self.compact = self.prefix < self.slots * 256 + self.prompt

    def template(self, kind: str) -> Path:
        return self.root / "templates" / f"{kind}.onnx"

    def tokens(self, task: str) -> tuple[np.ndarray, np.ndarray]:
        """(ids [1,200], mask [1,200]) of a prompt the exporter tokenized."""
        if task not in self.tasks:
            raise ValueError(f"no tokens for {task!r} in this bundle (it has "
                             f"{sorted(self.tasks)}); re-export with --task")
        t = self.tasks[task]
        return (np.asarray(t["tokens"], np.int64)[None],
                np.asarray(t["mask"], bool)[None])

    def prompt_rows(self, ids: np.ndarray) -> np.ndarray:
        """Embedding rows, scaled in FP32 then rounded to FP16, as the export does."""
        return (np.asarray(self.embed[ids[0]], np.float32)
                * np.float32(2048 ** 0.5)).astype(np.float16)[None]

    def preprocess(self, image_u8: np.ndarray) -> np.ndarray:
        """uint8 HxWx3 -> [1,3,224,224] FP16 in [-1,1]: openpi's resize_with_pad (linear,
        antialiased, centered black padding) then x/255*2-1."""
        img = resize_with_pad(image_u8, 224, 224)
        x = img.astype(np.float32) / 255.0 * 2.0 - 1.0
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None]).astype(np.float16)

    def unnormalize(self, actions: np.ndarray) -> np.ndarray:
        """[10,32] normalized -> [10,7] robot units (LIBERO quantile normalization)."""
        return ((actions[:, :7] + 1) / 2) * (self.q99 - self.q01 + 1e-6) + self.q01


@functools.lru_cache(maxsize=8)
def _linear_taps(n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """jax.image.resize LINEAR (antialiased when shrinking) as per-output (index, weight)
    taps: a triangle of half-width max(1, n_in/n_out) around each output center."""
    scale = n_out / n_in
    support = max(1.0, 1.0 / scale)
    center = (np.arange(n_out) + 0.5) / scale
    lo = np.floor(center - support).astype(int)
    idx = lo[:, None] + np.arange(int(np.ceil(2 * support)) + 2)[None]
    w = np.maximum(0.0, 1.0 - np.abs(idx + 0.5 - center[:, None]) / support)
    w[(idx < 0) | (idx >= n_in)] = 0.0
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return np.clip(idx, 0, n_in - 1), w.astype(np.float32)


def _resize_axis(x: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """Weighted sum of the taps, accumulated tap by tap in the order numpy's reduction
    over the tap axis uses (bit-identical), without materializing every tap at once."""
    idx, w = _linear_taps(x.shape[axis], n_out)
    shape = [1] * x.ndim
    shape[axis] = n_out
    acc = np.take(x, idx[:, 0], axis=axis) * w[:, 0].reshape(shape)
    for k in range(1, idx.shape[1]):
        acc += np.take(x, idx[:, k], axis=axis) * w[:, k].reshape(shape)
    return acc


def resize_with_pad(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """openpi image_tools.resize_with_pad for one uint8 HxWxC image."""
    h, w, c = img.shape
    if (h, w) == (height, width):
        return img
    ratio = max(w / width, h / height)
    rh, rw = int(h / ratio), int(w / ratio)
    x = _resize_axis(_resize_axis(img.astype(np.float32), rh, 0), rw, 1)
    x = np.clip(np.round(x), 0, 255).astype(np.uint8)
    top, left = (height - rh) // 2, (width - rw) // 2
    out = np.zeros((height, width, c), np.uint8)
    out[top:top + rh, left:left + rw] = x
    return out


# ── engines ──────────────────────────────────────────────────────────────────


def build_template(onnx_path: str, descriptor_path: str, engine_path: str) -> None:
    """One weight-stripped, individually refittable plan for a layer kind."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    descriptor = json.loads(Path(descriptor_path).read_text())
    with trt.Builder(logger) as builder:
        net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        cfg = builder.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * MIB)
        cfg.builder_optimization_level = 2
        cfg.clear_flag(trt.BuilderFlag.TF32)
        cfg.set_flag(trt.BuilderFlag.REFIT_INDIVIDUAL)
        cfg.set_flag(trt.BuilderFlag.STRIP_PLAN)
        parser = trt.OnnxParser(net, logger)
        if not parser.parse_from_file(onnx_path):
            raise SystemExit("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        for wt in descriptor["weights"]:
            if not net.mark_weights_refittable(wt["name"]):
                raise SystemExit(f"cannot mark refittable: {wt['name']}")
        plan = builder.build_serialized_network(net, cfg)
        if plan is None:
            raise SystemExit(f"build failed: {onnx_path}")
        Path(engine_path).write_bytes(plan)


def prebuild_templates(bundle: Bundle, cache_dir) -> dict:
    """Build every missing or stale template plan, one subprocess each, serially."""
    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    stack = (f"tensorrt={_trt_version()} cudart={_cudart_version()} opt_level=2 "
             "workspace_mb=512 strongly_typed refit_individual strip_plan no_tf32")
    built = {}
    for kind in bundle.kinds:
        onnx = bundle.template(kind)
        want = f"{_sha256(onnx)} {stack}"
        eng, key = cache / f"{kind}.engine", cache / f"{kind}.onnx.sha256"
        if eng.exists() and key.exists() and key.read_text().strip() == want:
            continue
        t0 = time.time()
        subprocess.run(
            [sys.executable, "-c",
             "import sys; from bench.vendor.pi05_trt import build_template; "
             "build_template(sys.argv[1], sys.argv[2], sys.argv[3])",
             str(onnx), str(onnx.with_suffix(".json")), str(eng)],
            check=True, cwd=str(Path(__file__).resolve().parents[2]))
        key.write_text(want + "\n")
        built[kind] = round(time.time() - t0, 1)
        print(f"[trt] built template {kind} in {built[kind]} s", flush=True)
    return built


def _allocator_class():
    import tensorrt as trt

    class CudaMallocAllocator(trt.IGpuAllocator):
        """Plain cudaMalloc/cudaFree under TensorRT. With the default allocator the
        prototype's first full load failed at action_14 with ~1.65 GiB still available;
        this one loaded every engine (cause of the default's failure unproven)."""

        def __init__(self):
            trt.IGpuAllocator.__init__(self)
            self.live, self.peak = {}, 0

        def allocate(self, size, alignment, flags):
            if not size:
                return 0
            p = ctypes.c_void_p()
            if _cuda().cudaMalloc(ctypes.byref(p), ctypes.c_size_t(int(size))):
                return 0
            self.live[p.value] = int(size)
            self.peak = max(self.peak, sum(self.live.values()))
            return p.value

        def allocate_async(self, size, alignment, flags, stream):
            return self.allocate(size, alignment, flags)

        def deallocate(self, pointer):
            if not pointer:
                return True
            ok = _cuda().cudaFree(ctypes.c_void_p(int(pointer))) == 0
            if ok:
                self.live.pop(int(pointer), None)
            return ok

        def deallocate_async(self, pointer, stream):
            return self.deallocate(pointer)

        def reallocate(self, base, alignment, new_size):
            return 0

    return CudaMallocAllocator


class Pi05Engines(Engines):
    """groot_trt.Engines over refitted copies of the kind templates."""

    def __init__(self, bundle: Bundle, cache_dir):
        import tensorrt as trt

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.allocator = _allocator_class()()
        self.runtime.gpu_allocator = self.allocator
        cache = Path(cache_dir).expanduser()
        plans = {k: (cache / f"{k}.engine").read_bytes() for k in bundle.kinds}
        self.engines, self.contexts = {}, {}
        for c in bundle.components:
            e = self.runtime.deserialize_cuda_engine(plans[c["kind"]])
            if e is None:
                raise RuntimeError(f"could not deserialize the {c['kind']} template")
            _refit(e, bundle.root / "weights" / c["label"], self.logger)
            self.engines[c["label"]] = e
        del plans
        gc.collect()
        self.scratch_bytes = max(e.device_memory_size_v2 for e in self.engines.values())
        self.scratch = _malloc(self.scratch_bytes)
        for name, e in self.engines.items():
            ctx = e.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
            ctx.set_device_memory(self.scratch, self.scratch_bytes)
            self.contexts[name] = ctx
        self.stream = ctypes.c_void_p()
        _ck(_cuda().cudaStreamCreate(ctypes.byref(self.stream)), "cudaStreamCreate")
        self.buffers, self.last_upload = {}, {}
        self.io = {n: self._io(e) for n, e in self.engines.items()}


def _refit(engine, directory: Path, logger) -> None:
    """Load one component's own weights into a template copy, one array at a time."""
    import tensorrt as trt

    record = json.loads((directory / "index.json").read_text())
    by_name = {w["name"]: w for w in record["weights"]}
    refitter = trt.Refitter(engine, logger)
    arrays = []
    for name in refitter.get_all_weights():
        if name not in by_name:
            raise RuntimeError(f"{directory.name}: no exported weights for {name}")
        a = np.load(directory / by_name[name]["file"], mmap_mode="r")
        arrays.append(a)
        if not refitter.set_named_weights(name, trt.Weights(a)):
            raise RuntimeError(f"{directory.name}: refit rejected {name}")
    if refitter.get_missing_weights():
        raise RuntimeError(f"{directory.name}: missing {refitter.get_missing_weights()}")
    if not refitter.refit_cuda_engine():
        raise RuntimeError(f"{directory.name}: refit failed")
    del refitter, arrays
    gc.collect()


# ── the chain ────────────────────────────────────────────────────────────────


class Pi05Device:
    """The whole policy as one device-resident chain, optionally one CUDA graph.

    The prefix buffer is rewritten by the language layers, so the prompt rows are kept
    in their own buffer and copied back into place on the device before every pass.
    Masks, positions and prompt rows depend only on the task and are uploaded when it
    changes; the per-step AdaRMS modulation is a fixed table bound by offset.
    """

    def __init__(self, bundle: Bundle, engines, ops, graph: bool = False):
        from .trt_device import Device, DevBuf

        self.b, self.d, self.use_graph = bundle, Device(engines), graph
        d, b = self.d, bundle
        f16 = np.float16
        self.images = [DevBuf((1, 3, 224, 224), f16) for _ in range(b.slots)]
        self.n_real = None
        vis = d.spec("vision_00")["hidden"][0]
        self.vis = [DevBuf(vis, f16), DevBuf(vis, f16)]
        self.features = DevBuf(d.spec("vision_tail")["output"][0], f16)
        pre = d.spec("language_00")["hidden"][0]
        self.prefix = [DevBuf(pre, f16), DevBuf(pre, f16)]
        # Prompt rows (plus, compact, the zeroed tail), copied into the prefix on the
        # device before every pass, since the language layers overwrite it.
        self.rows = DevBuf((1, b.prefix if b.compact else b.prompt, pre[-1]), f16)
        self.rows_bytes = 0
        self.p_mask = DevBuf(d.spec("language_00")["mask"][0], np.float32)
        self.p_cos = DevBuf(d.spec("language_00")["cos"][0], f16)
        self.p_sin = DevBuf(d.spec("language_00")["sin"][0], f16)
        kv = d.spec("language_00")["key"][0]
        self.kv = [(DevBuf(kv, f16), DevBuf(kv, f16)) for _ in range(b.n_lang)]
        exp = d.spec("action_00")["hidden"][0]
        self.expert = [DevBuf(exp, f16), DevBuf(exp, f16)]
        self.s_mask = DevBuf(d.spec("action_00")["mask"][0], np.float32)
        self.s_cos = DevBuf(d.spec("action_00")["cos"][0], f16)
        self.s_sin = DevBuf(d.spec("action_00")["sin"][0], f16)
        a = d.spec("action_input")["actions"][0]
        self.actions = [DevBuf(a, np.float32), DevBuf(a, np.float32)]
        self.velocity = DevBuf(d.spec("action_output")["velocity"][0], np.float32)
        self.styles = DevBuf(b.styles.shape, b.styles.dtype)
        d.upload(self.styles, b.styles)
        sites, row = b.styles.shape[1], b.styles.shape[2] * b.styles.dtype.itemsize
        mod = d.spec("action_00")["mod_in"][0]
        fin = d.spec("action_output")["mod_final"][0]
        self.mods = [[DevBuf.view(self.styles, (s * sites + i) * row, mod if i < sites - 1 else fin)
                      for i in range(sites)] for s in range(b.steps)]
        self.euler = ops.get("axpy", a, float(EULER_DT))
        # Unused slots (the masked third camera) hold the all -1 image.
        for img in self.images:
            d.upload(img, -np.ones(img.shape, f16))
        self.ev = [d.event() for _ in range(4)]
        self.task, self.graph = None, None
        d.sync()

    def _set_task(self, ids: np.ndarray, mask: np.ndarray, n_real: int) -> None:
        d, b = self.d, self.b
        hidden = self.rows.shape[-1]
        if b.compact:
            tok = ids[0][mask[0]]
            valid_count = n_real * 256 + len(tok)
            if valid_count > b.prefix:
                raise ValueError(f"{valid_count} valid prefix tokens exceed this bundle's "
                                 f"capacity {b.prefix}; it does not truncate")
            pad = np.zeros((1, b.prefix), bool)
            pad[:, :valid_count] = True
            rows = np.zeros((1, b.prefix - n_real * 256, hidden), np.float16)
            rows[0, :len(tok)] = b.prompt_rows(tok[None])[0]
        else:
            valid = [np.full((1, 256), i < n_real) for i in range(b.slots)]
            pad = np.concatenate(valid + [mask], axis=1)
            rows = b.prompt_rows(ids)
        d.upload(self.p_mask, np.where(pad[:, None, :, None] & pad[:, None, None, :],
                                       np.float32(0), MASK_NEG))
        pos = np.cumsum(pad, axis=-1) - 1
        d.upload(self.p_cos, b.rope_cos[pos + 1])
        d.upload(self.p_sin, b.rope_sin[pos + 1])
        spos = np.sum(pad, axis=-1)[:, None] + np.arange(b.horizon)[None]
        d.upload(self.s_cos, b.rope_cos[spos + 1])
        d.upload(self.s_sin, b.rope_sin[spos + 1])
        sv = np.concatenate([np.broadcast_to(pad[:, None, :], (1, b.horizon, b.prefix)),
                             np.ones((1, b.horizon, b.horizon), bool)], axis=-1)
        d.upload(self.s_mask, np.where(sv[:, None], np.float32(0), MASK_NEG))
        self.rows.host()[0, :rows.shape[1]] = rows[0]
        d.upload_staged(self.rows)
        self.rows_bytes = rows.nbytes
        self.task = (ids.tobytes(), mask.tobytes(), n_real)
        if n_real != self.n_real:              # the captured copies depend on it
            self.n_real, self.graph = n_real, None
        if self.graph is None:
            self._enqueue()                    # every engine runs once before capture
            d.sync()
            if self.use_graph:
                self.graph = d.capture(self._enqueue)

    def _enqueue(self) -> None:
        d, b = self.d, self.b
        img_bytes = self.features.nbytes
        d.record(self.ev[0])
        slots = self.n_real if b.compact else b.slots
        for slot, img in enumerate(self.images[:slots]):
            d.enqueue("stem", {"image": img, "hidden": self.vis[0]})
            for i in range(b.n_vision):
                d.enqueue(f"vision_{i:02d}", {"hidden": self.vis[i % 2], "output": self.vis[1 - i % 2]})
            d.enqueue("vision_tail", {"hidden": self.vis[b.n_vision % 2], "output": self.features})
            d.d2d(self.prefix[0], self.features, img_bytes, slot * img_bytes)
        d.d2d(self.prefix[0], self.rows, self.rows_bytes, slots * img_bytes)
        d.record(self.ev[1])
        for i in range(b.n_lang):
            k, v = self.kv[i]
            d.enqueue(f"language_{i:02d}", {"hidden": self.prefix[i % 2],
                                            "output": self.prefix[1 - i % 2], "mask": self.p_mask,
                                            "cos": self.p_cos, "sin": self.p_sin,
                                            "key": k, "value": v})
        d.record(self.ev[2])
        for s in range(b.steps):
            a_in, a_out = self.actions[s % 2], self.actions[(s + 1) % 2]
            d.enqueue("action_input", {"actions": a_in, "hidden": self.expert[0]})
            for i in range(b.n_action):
                k, v = self.kv[i]
                d.enqueue(f"action_{i:02d}", {
                    "hidden": self.expert[i % 2], "output": self.expert[1 - i % 2],
                    "mask": self.s_mask, "cos": self.s_cos, "sin": self.s_sin,
                    "prefix_k": k, "prefix_v": v,
                    "mod_in": self.mods[s][2 * i], "mod_post": self.mods[s][2 * i + 1]})
            d.enqueue("action_output", {"hidden": self.expert[b.n_action % 2],
                                        "mod_final": self.mods[s][-1], "velocity": self.velocity})
            d.enqueue(self.euler, {"a": a_in, "b": self.velocity, "out": a_out})
        d.record(self.ev[3])

    @property
    def result(self):
        return self.actions[self.b.steps % 2]

    def infer(self, images: list[np.ndarray], ids: np.ndarray, mask: np.ndarray,
              noise: np.ndarray, timings: dict | None = None) -> np.ndarray:
        """images: [1,3,224,224] FP16 per real camera. Returns normalized [1,10,32]."""
        d = self.d
        t0 = time.perf_counter()
        if self.task != (ids.tobytes(), mask.tobytes(), len(images)):
            self._set_task(ids, mask, len(images))
        for buf, img in zip(self.images, images):
            d.upload(buf, img)
        d.upload(self.actions[0], noise.reshape(self.actions[0].shape))
        if self.graph is not None:
            d.launch(self.graph)
        else:
            self._enqueue()
        d.download(self.result)
        d.sync()
        if timings is not None:
            ev = self.ev
            timings.update(vision=d.elapsed_ms(ev[0], ev[1]), language=d.elapsed_ms(ev[1], ev[2]),
                           denoise=d.elapsed_ms(ev[2], ev[3]),
                           total=(time.perf_counter() - t0) * 1e3)
        return self.result.host().copy()


def validate_fixtures(bundle: Bundle, device: Pi05Device) -> dict:
    """Engines vs the exporter's full-FP16 PyTorch outputs (and the original mixed
    BF16/FP32 policy) for every fixture in the bundle, plus the host preprocessing
    against openpi's processed images. Gated on the full FP16 chunk."""
    reports, ok = [], True
    for path in sorted((bundle.root / "fixtures").glob("fixture_*.npz")):
        f = np.load(path)
        imgs = [f["image_" + k].astype(np.float16) for k in bundle.image_keys[:2]]
        act = device.infer(imgs, f["tokens"].astype(np.int64), f["token_mask"].astype(bool),
                           f["noise"].astype(np.float32))
        robot = bundle.unnormalize(act[0])
        pre = np.concatenate([bundle.preprocess(f[raw]) for raw in ("image", "wrist")])
        ref = np.concatenate([f["image_base_0_rgb"], f["image_left_wrist_0_rgb"]])
        r = {"fixture": path.stem,
             "chunk_vs_fp16": _cmp(act, f["fp16_actions_normalized"]),
             "chunk_vs_mixed": _cmp(act, f["actions"]),
             "robot_vs_fp16": _cmp(robot, f["fp16_actions_robot"]),
             "image_preprocess": _cmp(pre.astype(np.float32), ref)}
        c = r["chunk_vs_fp16"]
        r["max_abs_vs_fp16"] = c["max_abs"]
        ok &= c["cosine"] >= 0.99999 and c["max_abs"] <= 0.005
        reports.append(r)
    if not reports:
        return {"status": "MISSING"}
    worst = min(reports, key=lambda r: r["chunk_vs_fp16"]["cosine"])
    return {"status": "PASS" if ok else "FAIL",
            "gate": "full normalized chunk vs the bundle's FP16 PyTorch reference: "
                    "cosine >= 0.99999 and max |diff| <= 0.005",
            "fixtures": len(reports), "worst": worst, "reports": reports}
