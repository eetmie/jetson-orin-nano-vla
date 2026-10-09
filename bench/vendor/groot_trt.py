# ─────────────────────────────────────────────────────────────────────────────
# VENDORED from spark-projects @ 9e367cd — vla-onnx/groot/{pipeline,trt_runtime}.py
#
# GR00T N1.6 split bundle on TensorRT alone: no ONNX Runtime, no torch. Merged into one
# file and given the benchmark's preprocessing, engine-cache keying and fixture gate;
# the graph order, host glue and denoising loop are the tested implementation.
# ─────────────────────────────────────────────────────────────────────────────
"""GR00T N1.6 split-engine inference on the TensorRT runtime, numpy host glue.

Why not ORT like the other families: ORT's TensorRT EP keeps every ONNX initializer in
host memory beside the engine's own copy, and on unified memory both count (~5.5
bytes/param measured for X-VLA here). GR00T deploys ~2.3 B params, so only a runtime
that holds each weight once fits. Every execution context shares ONE scratch buffer
sized to the largest engine, since the stages run strictly one after another.

Pipeline per observation (V views, 4 denoising steps):

    host     tokens -> rows of the mmap'd FP16 embedding table (never resident)
    vision_k [V,3,252,252] -> [V,81,2048]
    host     scatter image tokens into the sequence (right-padded to a fixed S)
    llm_k    16 Qwen3 layers + final norm
    cond     vlln, embodiment-sliced state encoder
    4x:  time -> dit_0 .. dit_10   (last chunk applies the decoder and Euler step)
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

FIXTURE_MAX_PCT_RANGE = 1.0      # the README's full-chunk gate
FIXTURE_MIN_COSINE = 0.999

_cudart = None
_H2D, _D2H = 1, 2


def _cuda():
    global _cudart
    if _cudart is None:
        _cudart = ctypes.CDLL("libcudart.so.13")
    return _cudart


def _ck(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed: cudaError {rc}")


def _malloc(n: int) -> int:
    p = ctypes.c_void_p()
    _ck(_cuda().cudaMalloc(ctypes.byref(p), ctypes.c_size_t(max(n, 1))), f"cudaMalloc({n})")
    return p.value


def _trt_version() -> str:
    # Package metadata, not `import tensorrt`: the builder runs in child processes and
    # this one should not hold the library while they do.
    from importlib import metadata
    for dist in ("tensorrt", "tensorrt_cu13", "tensorrt-cu13"):
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            pass
    import tensorrt
    return tensorrt.__version__


def _cudart_version() -> int:
    v = ctypes.c_int()
    _ck(_cuda().cudaRuntimeGetVersion(ctypes.byref(v)), "cudaRuntimeGetVersion")
    return v.value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ── bundle + host glue ──────────────────────────────────────────────────────


class Bundle:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.b = json.loads((self.root / "bundle.json").read_text())
        self.embed = np.load(self.root / self.b["embed_tokens"], mmap_mode="r")
        self.names = [g["name"] for g in self.b["graphs"]]
        self.vision = [n for n in self.names if n.startswith("vision_")]
        self.llm = [n for n in self.names if n.startswith("llm_")]
        self.dit = [n for n in self.names if n.startswith("dit_")]
        ids = np.asarray(self.b["input_ids"], dtype=np.int64)
        n, neg = self.b["prompt_tokens"], np.float32(self.b["mask_neg"])
        valid = np.arange(ids.shape[0]) < n
        img = ids == self.b["image_token"]
        self.input_ids = ids
        self.image_positions = img
        self.text_bias = np.where(~img & valid, 0.0, neg).astype(np.float32)[None, None]
        self.image_bias = np.where(img & valid, 0.0, neg).astype(np.float32)[None, None]
        # The prompt never changes within a bundle, so its embedded rows are gathered once.
        self.prompt_embeds = np.asarray(self.embed[ids], dtype=np.float32)[None]

    def preprocess(self, image_u8: np.ndarray) -> np.ndarray:
        """One HxWx3 uint8 frame -> [3,H',W'] float32, as the stock eval pipeline does it.

        Gr00tN1d6Processor eval transform: letterbox to square, shortest edge to 256
        (INTER_AREA), center crop 0.95, shortest edge back to 256; then the Eagle
        processor's resize to a multiple of 28 (PIL bicubic), and (x/255 - 0.5)/0.5.
        """
        import cv2
        from PIL import Image

        h, w = image_u8.shape[:2]
        if h != w:
            m = max(h, w)
            top, left = (m - h) // 2, (m - w) // 2
            image_u8 = cv2.copyMakeBorder(image_u8, top, m - h - top, left, m - w - left,
                                          cv2.BORDER_CONSTANT, value=0)
        edge = self.b.get("shortest_image_edge", 256)
        frac = self.b.get("crop_fraction", 0.95)

        def smallest_max(img):
            hh, ww = img.shape[:2]
            s = edge / min(hh, ww)
            if s == 1:
                return img
            return cv2.resize(img, (round(ww * s), round(hh * s)), interpolation=cv2.INTER_AREA)

        img = smallest_max(image_u8)
        hh, ww = img.shape[:2]
        ch, cw = int(hh * frac), int(ww * frac)
        y0, x0 = (hh - ch) // 2, (ww - cw) // 2
        img = smallest_max(img[y0:y0 + ch, x0:x0 + cw])
        th, tw = self.b["image_hw"]
        img = np.asarray(Image.fromarray(img).resize((tw, th), Image.BICUBIC))
        x = img.astype(np.float32) / 255.0
        x = (x - self.b["image_mean"]) / self.b["image_std"]
        return np.ascontiguousarray(x.transpose(2, 0, 1))


def infer(bundle: Bundle, run, pixel_values: np.ndarray, state: np.ndarray,
          noise: np.ndarray, timings: dict | None = None) -> np.ndarray:
    """One action chunk. pixel_values [V,3,H,W], state [1,1,128], noise [1,50,128]."""
    b = bundle.b
    t = {} if timings is None else timings
    t0 = time.perf_counter()
    x = pixel_values
    for name in bundle.vision:
        x = next(iter(run(name, {"pixel_values" if name == bundle.vision[0] else "x": x}).values()))
    t1 = time.perf_counter()
    h = bundle.prompt_embeds.copy()
    h[0, bundle.image_positions] = x.reshape(-1, x.shape[-1])
    for name in bundle.llm:
        h = next(iter(run(name, {"h": h}).values()))
    cond = run("cond", {"features": h, "state": state.astype(np.float32)})
    vl, sf = cond["vl"], cond["state_features"]
    tb, ib = bundle.text_bias, bundle.image_bias
    t2 = time.perf_counter()
    actions = noise.astype(np.float32)
    for step in b["timesteps"]:
        te = run("time", {"t": np.array([step], np.float32)})
        o = run(bundle.dit[0], {"actions": actions, "t_proj": te["t_proj"], "tau": te["tau"],
                                "state_features": sf, "vl": vl, "text_bias": tb, "image_bias": ib})
        hh, temb = o["h_out"], o["temb"]
        for name in bundle.dit[1:-1]:
            hh = run(name, {"h": hh, "temb": temb, "vl": vl, "text_bias": tb,
                            "image_bias": ib})["h_out"]
        actions = run(bundle.dit[-1], {"h": hh, "temb": temb, "vl": vl, "text_bias": tb,
                                       "image_bias": ib, "actions": actions})["actions_next"]
    t3 = time.perf_counter()
    t.update(vision=(t1 - t0) * 1e3, backbone=(t2 - t1) * 1e3, denoise=(t3 - t2) * 1e3,
             total=(t3 - t0) * 1e3)
    return actions


# ── engines ──────────────────────────────────────────────────────────────────


def build_one(onnx_path: str, engine_path: str, timing_cache: str, opt_level: int,
              workspace_mb: int) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # Strongly typed: the mixed-FP16 ONNX already says which ops stay FP32 (norms,
    # softmax, the RMSNorms, the time sinusoids). A weakly typed FP16 build may re-pick
    # them in FP16, and a Qwen3 massive activation squared in FP16 overflows.
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, logger)
    if not parser.parse_from_file(onnx_path):
        raise SystemExit("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    cfg.builder_optimization_level = opt_level
    tc = Path(timing_cache)
    cache = cfg.create_timing_cache(tc.read_bytes() if tc.exists() else b"")
    cfg.set_timing_cache(cache, ignore_mismatch=False)
    blob = builder.build_serialized_network(net, cfg)
    if blob is None:
        raise SystemExit(f"build failed: {onnx_path}")
    Path(engine_path).write_bytes(blob)
    tc.write_bytes(cfg.get_timing_cache().serialize())


def prebuild_engines(bundle: Bundle, cache_dir: str | Path, opt_level: int = 2,
                     workspace_mb: int = 512) -> dict:
    """Build every missing or stale engine, one subprocess each, serially.

    An engine is keyed by the sha256 of the ONNX it came from plus the TensorRT and CUDA
    versions and the builder options (`<name>.onnx.sha256` beside it), so neither a
    re-exported bundle nor an upgraded stack can run against an old engine. Serial,
    one process per graph: two resident TensorRT builders OOM this board.
    opt_level 2 / 512 MB workspace is what every GR00T build was measured with: each
    peaked at 3.3 GB RSS with >= 3.9 GB still available.
    """
    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    stack = (f"tensorrt={_trt_version()} cudart={_cudart_version()} "
             f"opt_level={opt_level} workspace_mb={workspace_mb} strongly_typed")
    onnx_path_of = getattr(bundle, "onnx_path", lambda n: bundle.root / f"{n}.onnx")
    built = {}
    for name in bundle.names:
        onnx_path = onnx_path_of(name)
        sha = f"{_sha256(onnx_path)} {stack}"
        eng, key = cache / f"{name}.engine", cache / f"{name}.onnx.sha256"
        if eng.exists() and key.exists() and key.read_text().strip() == sha:
            continue
        t0 = time.time()
        subprocess.run(
            [sys.executable, "-c",
             "import sys; from bench.vendor.groot_trt import build_one; "
             "build_one(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))",
             str(onnx_path), str(eng), str(cache / "timing.cache"), str(opt_level),
             str(workspace_mb)],
            check=True, cwd=str(Path(__file__).resolve().parents[2]))
        key.write_text(sha + "\n")
        built[name] = round(time.time() - t0, 1)
        print(f"[trt] built {name} in {built[name]} s", flush=True)
    return built


class Engines:
    def __init__(self, bundle: Bundle, cache_dir: str | Path):
        import tensorrt as trt

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        cache = Path(cache_dir).expanduser()
        self.engines, self.contexts = {}, {}
        for name in bundle.names:
            blob = (cache / f"{name}.engine").read_bytes()
            self.engines[name] = self.runtime.deserialize_cuda_engine(blob)
            del blob
            if self.engines[name] is None:
                raise RuntimeError(f"could not deserialize {name}.engine")
        self.scratch_bytes = max(e.device_memory_size_v2 for e in self.engines.values())
        self.scratch = _malloc(self.scratch_bytes)
        for name, e in self.engines.items():
            ctx = e.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
            ctx.set_device_memory(self.scratch, self.scratch_bytes)
            self.contexts[name] = ctx
        self.stream = ctypes.c_void_p()
        _ck(_cuda().cudaStreamCreate(ctypes.byref(self.stream)), "cudaStreamCreate")
        # Device buffers shared by tensor name+shape; an input whose host array is the
        # same object as the last upload (vl, biases) is not re-copied. The object is
        # held, not its id(): a freed array's id can be reused.
        self.buffers: dict[tuple, int] = {}
        self.last_upload: dict[tuple, object] = {}
        self.io = {n: self._io(e) for n, e in self.engines.items()}

    def add(self, name: str, path) -> None:
        """Load one more engine onto the shared scratch (the chains' small op engines)."""
        trt = self.trt
        e = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if e is None:
            raise RuntimeError(f"could not deserialize {path}")
        if e.device_memory_size_v2 > self.scratch_bytes:
            raise RuntimeError(f"{name} needs more scratch than the shared buffer")
        ctx = e.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        ctx.set_device_memory(self.scratch, self.scratch_bytes)
        self.engines[name], self.contexts[name] = e, ctx
        self.io[name] = self._io(e)

    def _io(self, e):
        trt = self.trt
        out = []
        for i in range(e.num_io_tensors):
            n = e.get_tensor_name(i)
            shape = tuple(e.get_tensor_shape(n))
            dt = np.dtype(trt.nptype(e.get_tensor_dtype(n)))
            mode = e.get_tensor_mode(n)
            key = (n if mode == trt.TensorIOMode.INPUT else "out:" + n, shape, dt.str)
            if key not in self.buffers:
                self.buffers[key] = _malloc(int(np.prod(shape)) * dt.itemsize)
            out.append((n, mode, shape, dt, key))
        return out

    def run(self, name, feeds):
        trt, cu = self.trt, _cuda()
        ctx, outs = self.contexts[name], {}
        for n, mode, shape, dt, key in self.io[name]:
            ptr = self.buffers[key]
            ctx.set_tensor_address(n, ptr)
            if mode == trt.TensorIOMode.INPUT:
                src = feeds[n]
                if self.last_upload.get(key) is src:
                    continue
                x = np.ascontiguousarray(src, dtype=dt)
                assert x.shape == shape, (name, n, x.shape, shape)
                _ck(cu.cudaMemcpyAsync(ctypes.c_void_p(ptr), x.ctypes.data_as(ctypes.c_void_p),
                                       ctypes.c_size_t(x.nbytes), _H2D, self.stream), "H2D")
                self.last_upload[key] = src
            else:
                outs[n] = (ptr, np.empty(shape, dt))
        if not ctx.execute_async_v3(self.stream.value):
            raise RuntimeError(f"{name}: execute failed")
        for n, (ptr, arr) in outs.items():
            _ck(cu.cudaMemcpyAsync(arr.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(ptr),
                                   ctypes.c_size_t(arr.nbytes), _D2H, self.stream), "D2H")
        _ck(cu.cudaStreamSynchronize(self.stream), "sync")
        return {n: arr for n, (ptr, arr) in outs.items()}


# ── fixture gate ─────────────────────────────────────────────────────────────


def _cmp(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    d = np.abs(a - b)
    rng = float(b.max() - b.min()) or 1.0
    return {"cosine": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
            "max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "max_pct_range": float(d.max() / rng * 100)}


def validate_fixture(bundle: Bundle, engines: Engines) -> dict:
    """Engines vs the stock-PyTorch FP32 outputs the exporter stored in the bundle.

    Same pixels, state and noise as the reference; fails closed on the full chunk.
    `image_preprocess` checks this runtime's host preprocessing against the stock
    processor's own pixels for the same raw frames.
    """
    fx = bundle.b.get("fixture") or {}
    path = bundle.root / fx.get("file", "fixture.npz")
    if not path.exists():
        return {"status": "MISSING", "file": str(path)}
    r = np.load(path)
    act = infer(bundle, engines.run, r["pixel_values"].astype(np.float32), r["state"], r["noise"])
    ref = r["action_pred"]
    pre = np.stack([bundle.preprocess(np.ascontiguousarray(im)) for im in r["raw"]])
    reports = {
        "action": _cmp(act[:, 0], ref[:, 0]),
        "chunk": _cmp(act, ref),
        "image_preprocess": _cmp(pre, r["pixel_values"]),
    }
    ok = (reports["chunk"]["cosine"] >= FIXTURE_MIN_COSINE
          and reports["chunk"]["max_pct_range"] <= FIXTURE_MAX_PCT_RANGE)
    return {"status": "PASS" if ok else "FAIL", "threshold": FIXTURE_MIN_COSINE,
            "max_pct_range_threshold": FIXTURE_MAX_PCT_RANGE,
            "source": fx.get("source", "stock PyTorch float32"), "reports": reports}
