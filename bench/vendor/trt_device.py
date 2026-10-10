# ─────────────────────────────────────────────────────────────────────────────
# Written for this repo (not vendored). Device-resident execution on top of groot_trt's
# Engines: the same engines, contexts and shared scratch, without the host round trip.
# ─────────────────────────────────────────────────────────────────────────────
"""Device-resident engine chains, optionally replayed as one CUDA graph.

`Engines.run` uploads every input, downloads every output and synchronizes after each
engine. A chain here binds each producer's output buffer straight to its consumer's
input, keeps everything on one stream, uploads the observation once through pinned
staging, and downloads the action chunk once at the end. With `graph=True` the whole
enqueue sequence is captured once and replayed per inference.

Stage times come from CUDA events on the stream, not from the host clock: with nothing
synchronizing in between, host time measures enqueue, not execution.
"""

from __future__ import annotations

import ctypes
import time

import numpy as np

from .groot_trt import _ck, _cuda, _malloc

_H2D, _D2H, _D2D = 1, 2, 3
_CAPTURE_THREAD_LOCAL = 1


class DevBuf:
    """A device tensor: pointer, shape and dtype, plus a pinned host mirror on demand."""

    def __init__(self, shape, dtype):
        self.shape, self.dtype = tuple(int(x) for x in shape), np.dtype(dtype)
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        self.ptr = _malloc(self.nbytes)
        self._host = None

    @classmethod
    def view(cls, base: "DevBuf", offset_bytes: int, shape) -> "DevBuf":
        """A non-owning window into `base` (e.g. one camera's tokens)."""
        v = cls.__new__(cls)
        v.shape, v.dtype = tuple(int(x) for x in shape), base.dtype
        v.nbytes = int(np.prod(v.shape)) * v.dtype.itemsize
        if offset_bytes + v.nbytes > base.nbytes:
            raise ValueError("view runs past its buffer")
        v.ptr, v._host = base.ptr + offset_bytes, None
        return v

    def host(self) -> np.ndarray:
        """Pinned host array of the same shape (allocated once): a fixed address the
        copy engine can DMA from and that a captured graph may reference."""
        if self._host is None:
            p = ctypes.c_void_p()
            _ck(_cuda().cudaHostAlloc(ctypes.byref(p), ctypes.c_size_t(max(self.nbytes, 1)),
                                      ctypes.c_uint(0)), "cudaHostAlloc")
            raw = (ctypes.c_byte * self.nbytes).from_address(p.value)
            self._host = np.frombuffer(raw, self.dtype).reshape(self.shape)
        return self._host


class Device:
    """Enqueue-only helpers over an Engines instance (its contexts, scratch and stream)."""

    def __init__(self, engines):
        self.e = engines
        self.trt = engines.trt
        self.stream = engines.stream
        self.cu = _cuda()
        self.capturing = False

    # -- buffers --------------------------------------------------------------
    def spec(self, name: str) -> dict:
        """{tensor: (shape, dtype, is_input)} of one engine."""
        trt = self.trt
        return {n: (shape, dt, mode == trt.TensorIOMode.INPUT)
                for n, mode, shape, dt, _key in self.e.io[name]}

    def outputs(self, name: str) -> dict[str, DevBuf]:
        return {n: DevBuf(shape, dt) for n, (shape, dt, is_in) in self.spec(name).items()
                if not is_in}

    def buf_for(self, name: str, tensor: str) -> DevBuf:
        shape, dt, _ = self.spec(name)[tensor]
        return DevBuf(shape, dt)

    # -- stream work ------------------------------------------------------------
    def upload(self, dst: DevBuf, arr: np.ndarray) -> None:
        h = dst.host()
        h[...] = np.asarray(arr, dtype=dst.dtype).reshape(dst.shape)
        self.upload_staged(dst)

    def upload_staged(self, dst: DevBuf) -> None:
        _ck(self.cu.cudaMemcpyAsync(ctypes.c_void_p(dst.ptr),
                                    ctypes.c_void_p(dst.host().ctypes.data),
                                    ctypes.c_size_t(dst.nbytes), _H2D, self.stream), "H2D")

    def download(self, src: DevBuf) -> None:
        _ck(self.cu.cudaMemcpyAsync(ctypes.c_void_p(src.host().ctypes.data),
                                    ctypes.c_void_p(src.ptr),
                                    ctypes.c_size_t(src.nbytes), _D2H, self.stream), "D2H")

    def d2d(self, dst: DevBuf, src: DevBuf, nbytes: int, dst_off: int = 0,
            src_off: int = 0) -> None:
        _ck(self.cu.cudaMemcpyAsync(ctypes.c_void_p(dst.ptr + dst_off),
                                    ctypes.c_void_p(src.ptr + src_off),
                                    ctypes.c_size_t(nbytes), _D2D, self.stream), "D2D")

    def copy2d(self, dst: DevBuf, dpitch: int, src: DevBuf, spitch: int, width: int,
               height: int, dst_off: int = 0, src_off: int = 0) -> None:
        """Strided device copy: `height` rows of `width` bytes (e.g. one half of a concat)."""
        _ck(self.cu.cudaMemcpy2DAsync(ctypes.c_void_p(dst.ptr + dst_off), ctypes.c_size_t(dpitch),
                                      ctypes.c_void_p(src.ptr + src_off), ctypes.c_size_t(spitch),
                                      ctypes.c_size_t(width), ctypes.c_size_t(height), _D2D,
                                      self.stream), "D2D 2D")

    def enqueue(self, name: str, bind: dict[str, DevBuf]) -> None:
        """Run one engine on the stream with these buffers; no synchronization."""
        ctx = self.e.contexts[name]
        for n in self.spec(name):
            ctx.set_tensor_address(n, bind[n].ptr)
        if not ctx.execute_async_v3(self.stream.value):
            raise RuntimeError(f"{name}: enqueue failed")

    def sync(self) -> None:
        _ck(self.cu.cudaStreamSynchronize(self.stream), "sync")

    # -- events -----------------------------------------------------------------
    def event(self) -> ctypes.c_void_p:
        ev = ctypes.c_void_p()
        _ck(self.cu.cudaEventCreate(ctypes.byref(ev)), "cudaEventCreate")
        return ev

    def record(self, ev) -> None:
        # External: under stream capture this becomes an event node the graph records on
        # every launch; a plain record would only be a capture-time dependency.
        if self.capturing:
            _ck(self.cu.cudaEventRecordWithFlags(ev, self.stream, ctypes.c_uint(1)),
                "cudaEventRecordWithFlags")
        else:
            _ck(self.cu.cudaEventRecord(ev, self.stream), "cudaEventRecord")

    def elapsed_ms(self, a, b) -> float:
        ms = ctypes.c_float()
        _ck(self.cu.cudaEventElapsedTime(ctypes.byref(ms), a, b), "cudaEventElapsedTime")
        return float(ms.value)

    # -- CUDA graphs ------------------------------------------------------------
    def capture(self, enqueue_fn) -> ctypes.c_void_p:
        """Capture enqueue_fn's stream work into an instantiated graph.

        Every engine must have run once uncaptured first (TensorRT may allocate on its
        first enqueue), and the bound addresses must not change afterwards.
        """
        cu = self.cu
        _ck(cu.cudaStreamBeginCapture(self.stream, ctypes.c_int(_CAPTURE_THREAD_LOCAL)),
            "cudaStreamBeginCapture")
        self.capturing = True
        graph = ctypes.c_void_p()
        try:
            try:
                enqueue_fn()
            finally:
                self.capturing = False
                rc = cu.cudaStreamEndCapture(self.stream, ctypes.byref(graph))
            _ck(rc, "cudaStreamEndCapture")
            gexec = ctypes.c_void_p()
            _ck(cu.cudaGraphInstantiate(ctypes.byref(gexec), graph, ctypes.c_ulonglong(0)),
                "cudaGraphInstantiate")
            return gexec
        finally:
            # The executable owns its graph state after instantiation. Release the
            # source graph on success and on failed enqueue/instantiation alike.
            if graph:
                _ck(cu.cudaGraphDestroy(graph), "cudaGraphDestroy")

    def destroy_graph(self, gexec) -> None:
        """Release an executable that will no longer be launched."""
        if gexec is not None:
            _ck(self.cu.cudaGraphExecDestroy(gexec), "cudaGraphExecDestroy")

    def launch(self, gexec) -> None:
        _ck(self.cu.cudaGraphLaunch(gexec, self.stream), "cudaGraphLaunch")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of each contiguous True run."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


class DitChain:
    """GR00T's denoising loop on the device, fed by graph input names (both layouts).

    Per-step constants (time encodings, schema-2 AdaLN modulation) are uploaded once;
    schema-2 cross-attention K/V run once per observation from `vl`; actions ping-pong
    between two buffers. `static` holds vl, state_features and the attention biases.
    """

    def __init__(self, d: "Device", bundle, consts: list[dict], static: dict):
        from .groot_trt import _pool_name

        self.d, self.b, self.static, self.pool_name = d, bundle, static, _pool_name
        needed = {i for n in bundle.dit for i in bundle.graph[n]["inputs"]}
        self.steps = []
        for c in consts:
            bufs = {}
            for k, v in c.items():
                if k in needed:
                    bufs[k] = DevBuf(v.shape, v.dtype)
                    d.upload(bufs[k], v)
            self.steps.append(bufs)
        self.kv_out = [d.outputs(n) for n in bundle.kv]
        self.dit_out = {n: {k: v for k, v in d.outputs(n).items() if k != "actions_next"}
                        for n in bundle.dit}
        a = d.buf_for(bundle.dit[0], "actions")
        self.actions = [a, DevBuf(a.shape, a.dtype)]

    @property
    def result(self) -> DevBuf:
        return self.actions[len(self.steps) % 2]

    def enqueue(self) -> None:
        d, bd = self.d, self.b
        pool = dict(self.static)
        for name, out in zip(bd.kv, self.kv_out):
            d.enqueue(name, {"vl": pool["vl"], **out})
            pool.update(out)
        for i, c in enumerate(self.steps):
            a_in, a_out = self.actions[i % 2], self.actions[(i + 1) % 2]
            pool.update(c)
            pool["actions"] = a_in
            for name in bd.dit:
                outs = self.dit_out[name]
                bind = {k: pool[k] for k in bd.graph[name]["inputs"]}
                bind.update(outs)
                if "actions_next" in bd.graph[name]["outputs"]:
                    bind["actions_next"] = a_out
                d.enqueue(name, bind)
                pool.update({self.pool_name(k): v for k, v in outs.items()})


class Groot16Device:
    """GR00T N1.6 as one device-resident chain (see groot_trt.infer for the host one).

    Differences from the host chain, none numerical: image tokens reach the prompt by
    device-to-device copies; the four timestep embeddings are computed once at load
    (they depend on nothing but the fixed schedule); actions ping-pong between two
    buffers across steps.
    """

    def __init__(self, bundle, engines, graph: bool = False):
        self.b, self.d = bundle, Device(engines)
        d, b = self.d, bundle.b
        bd = bundle
        self.vision_in = d.buf_for(bd.vision[0], "pixel_values")
        self.vision_out = [d.outputs(n) for n in bd.vision]
        # The prompt's text rows never change; only the image runs are rewritten.
        self.h0 = d.buf_for(bd.llm[0], "h")
        d.upload(self.h0, bd.prompt_embeds)
        self.llm_out = [d.outputs(n) for n in bd.llm]
        self.state = d.buf_for("cond", "state")
        self.cond_out = d.outputs("cond")
        self.text_bias = DevBuf(bd.text_bias.shape, np.float32)
        self.image_bias = DevBuf(bd.image_bias.shape, np.float32)
        d.upload(self.text_bias, bd.text_bias)
        d.upload(self.image_bias, bd.image_bias)
        self.dit = DitChain(d, bd, bd.consts, {
            "vl": self.cond_out["vl"], "state_features": self.cond_out["state_features"],
            "text_bias": self.text_bias, "image_bias": self.image_bias})
        d.sync()
        hidden = self.h0.shape[-1]
        row = hidden * self.h0.dtype.itemsize
        runs = _runs(bd.image_positions)
        per_view = self.vision_out[-1]["vision_tokens"]
        if per_view.dtype != self.h0.dtype:
            raise ValueError("vision tokens and prompt embeddings differ in dtype")
        if sum(e - s for s, e in runs) != int(np.prod(per_view.shape[:-1])):
            raise ValueError("image-token runs do not match the vision output")
        self.copies, src = [], 0
        for s, e in runs:
            self.copies.append((s * row, src * row, (e - s) * row))
            src += e - s
        self.ev = [d.event() for _ in range(4)]
        self.graph = None
        self._enqueue()                      # every engine runs once before any capture
        d.sync()
        if graph:
            self.graph = d.capture(self._enqueue)

    def _enqueue(self) -> None:
        d, bd = self.d, self.b
        d.record(self.ev[0])
        x = self.vision_in
        for name, out in zip(bd.vision, self.vision_out):
            d.enqueue(name, {("pixel_values" if name == bd.vision[0] else "x"): x, **out})
            x = next(iter(out.values()))
        for dst, src, n in self.copies:
            d.d2d(self.h0, x, n, dst, src)
        d.record(self.ev[1])
        h = self.h0
        for name, out in zip(bd.llm, self.llm_out):
            d.enqueue(name, {"h": h, **out})
            h = next(iter(out.values()))
        d.enqueue("cond", {"features": h, "state": self.state, **self.cond_out})
        d.record(self.ev[2])
        self.dit.enqueue()
        d.record(self.ev[3])

    @property
    def result(self) -> DevBuf:
        return self.dit.result

    def infer(self, pixel_values: np.ndarray, state: np.ndarray, noise: np.ndarray,
              timings: dict | None = None) -> np.ndarray:
        d = self.d
        t0 = time.perf_counter()
        d.upload(self.vision_in, pixel_values)
        d.upload(self.state, state)
        d.upload(self.dit.actions[0], noise)
        if self.graph is not None:
            d.launch(self.graph)
        else:
            self._enqueue()
        d.download(self.result)
        d.sync()
        if timings is not None:
            ev = self.ev
            timings.update(vision=d.elapsed_ms(ev[0], ev[1]),
                           backbone=d.elapsed_ms(ev[1], ev[2]),
                           denoise=d.elapsed_ms(ev[2], ev[3]),
                           total=(time.perf_counter() - t0) * 1e3)
        return self.result.host().copy()


class XVLADevice:
    """X-VLA as one device-resident chain (see xvla_trt.infer for the host one).

    The interpolation x_t = x1 * t + action * (1 - t) runs as a small FP32 op engine
    with one weight buffer per step; the clean-action estimate stays in one buffer that
    denoise_3 rewrites each step. Prompt IDs are uploaded only when they change. The
    gripper sigmoid stays on the host, after the one download.
    """

    def __init__(self, bundle, engines, ops, graph: bool = False):
        self.b, self.d = bundle, Device(engines)
        d, bd = self.d, bundle
        self.vision_in = d.buf_for(bd.vision[0], "pixel_values")
        self.vision_out = [d.outputs(n) for n in bd.vision]
        feats = next(iter(self.vision_out[-1].values()))
        self.ids = d.buf_for(bd.text[0], "input_ids")
        img0 = d.spec(bd.text[0])["image_tokens"][0]
        aux = d.spec("cond")["aux_visual"][0]
        if bd.valid_views == bd.num_views:
            self.full = feats
        else:
            # Padded views read as exactly zero: a zeroed buffer the real views are
            # copied into, the rest never written.
            self.full = DevBuf((bd.num_views,) + feats.shape[1:], feats.dtype)
            d.upload(self.full, np.zeros(self.full.shape, self.full.dtype))
        per_view = int(np.prod(feats.shape[1:])) * feats.dtype.itemsize
        self.image_tokens = DevBuf.view(self.full, 0, img0)
        self.aux = DevBuf.view(self.full, per_view, aux)
        self.text_out = [d.outputs(n) for n in bd.text]
        self.cond_out = d.outputs("cond")
        x_t_spec = d.spec(bd.denoise[0])["x_t"][0]
        self.x1 = DevBuf(x_t_spec, np.float32)
        self.x_t = DevBuf(x_t_spec, np.float32)
        self.action = DevBuf(x_t_spec, np.float32)
        self.proprio = d.buf_for(bd.denoise[0], "proprio")
        self.lerp = ops.get("lerp", x_t_spec)
        self.steps = []
        for i in range(bd.steps, 0, -1):
            s = i / bd.steps
            w = DevBuf((1,) * len(x_t_spec), np.float32)
            t = d.buf_for(bd.denoise[0], "t")
            d.upload(w, np.array(s, np.float32))
            d.upload(t, np.array([s], np.float32))
            self.steps.append((w, t))
        self.den_out = [d.outputs(n) for n in bd.denoise[:-1]]
        self.zeros = DevBuf(x_t_spec, np.float32)
        d.upload(self.zeros, np.zeros(x_t_spec, np.float32))
        d.sync()
        self.feats = feats
        self.task_ids = None
        self.ev = [d.event() for _ in range(4)]
        self.graph = None
        self._enqueue()
        d.sync()
        if graph:
            self.graph = d.capture(self._enqueue)

    def _enqueue(self) -> None:
        d, bd = self.d, self.b
        d.record(self.ev[0])
        x = self.vision_in
        for name, out in zip(bd.vision, self.vision_out):
            d.enqueue(name, {("pixel_values" if name == bd.vision[0] else "hidden_in"): x,
                             **out})
            x = next(iter(out.values()))
        if self.full is not self.feats:
            d.d2d(self.full, self.feats, self.feats.nbytes)
        d.record(self.ev[1])
        h = None
        for name, out in zip(bd.text, self.text_out):
            feeds = ({"input_ids": self.ids, "image_tokens": self.image_tokens}
                     if name == bd.text[0] else {"hidden_in": h})
            d.enqueue(name, {**feeds, **out})
            h = next(iter(out.values()))
        d.enqueue("cond", {"vlm_features": h, "aux_visual": self.aux, **self.cond_out})
        cond = self.cond_out["cond_tokens"]
        d.record(self.ev[2])
        d.d2d(self.action, self.zeros, self.action.nbytes)
        for w, t in self.steps:
            d.enqueue(self.lerp, {"x1": self.x1, "a": self.action, "w": w, "out": self.x_t})
            d.enqueue(bd.denoise[0], {"x_t": self.x_t, "t": t, "proprio": self.proprio,
                                      "cond_tokens": cond, **self.den_out[0]})
            h = next(iter(self.den_out[0].values()))
            for name, out in zip(bd.denoise[1:-1], self.den_out[1:]):
                d.enqueue(name, {"hidden_in": h, **out})
                h = next(iter(out.values()))
            d.enqueue(bd.denoise[-1], {"hidden_in": h, "action": self.action})
        d.record(self.ev[3])

    def infer(self, pixel_values, input_ids, proprio, x1, timings: dict | None = None):
        d = self.d
        t0 = time.perf_counter()
        d.upload(self.vision_in, pixel_values)
        if self.task_ids is None or not np.array_equal(input_ids, self.task_ids):
            d.upload(self.ids, input_ids)
            self.task_ids = input_ids.copy()
        d.upload(self.proprio, proprio)
        d.upload(self.x1, x1)
        if self.graph is not None:
            d.launch(self.graph)
        else:
            self._enqueue()
        d.download(self.action)
        d.sync()
        action = self.action.host().copy()
        g = self.b.gripper
        action[..., g] = 1.0 / (1.0 + np.exp(-action[..., g]))
        if timings is not None:
            ev = self.ev
            timings.update(vision=d.elapsed_ms(ev[0], ev[1]), text=d.elapsed_ms(ev[1], ev[2]),
                           cond=0.0, denoise=d.elapsed_ms(ev[2], ev[3]),
                           total=(time.perf_counter() - t0) * 1e3)
        return action


class Evo1Device:
    """EVO1 as one device-resident chain (see evo1_trt.infer for the host one).

    The prompt's text rows, causal mask and context mask are uploaded when token IDs
    or the validity mask change; image features reach the prompt by device copies. The
    Euler update action + velocity / steps runs as a small FP32 op engine (1/steps is a
    power of two here, so it is the same arithmetic), ping-ponging two buffers.
    """

    def __init__(self, bundle, engines, ops, graph: bool = False):
        self.b, self.d = bundle, Device(engines)
        d, bd = self.d, bundle
        self.use_graph = graph
        self.vision_in = d.buf_for(bd.vision[0], "pixel_values")
        self.vision_out = [d.outputs(n) for n in bd.vision]
        self.h0 = d.buf_for(bd.language[0], "hidden_in")
        self.mask = d.buf_for(bd.language[0], "causal_mask")
        self.lang_out = [d.outputs(n) for n in bd.language]
        self.cmask = d.buf_for("action_context", "context_mask")
        self.state = d.buf_for("action_context", "state")
        self.cache = d.outputs("action_context")
        a_spec = d.spec("action_step")["action"][0]
        self.actions = [DevBuf(a_spec, np.float32), DevBuf(a_spec, np.float32)]
        self.step_out = d.outputs("action_step")
        self.vel = d.outputs("action_output")["velocity"]
        self.axpy = ops.get("axpy", a_spec, 1.0 / bd.steps)
        self.time_index = []
        for i in range(bd.steps):
            ti = d.buf_for("action_step", "time_index")
            d.upload(ti, np.asarray([min(int((i / bd.steps) * 999), 999)], np.int64))
            self.time_index.append(ti)
        self.ev = [d.event() for _ in range(5)]
        self.task_ids = None
        self.task_mask = None
        self.copies = None
        self.graph = None
        d.sync()

    def _set_prompt(self, input_ids: np.ndarray, context_mask: np.ndarray) -> None:
        from .evo1_trt import causal_mask

        d, bd = self.d, self.b
        d.upload(self.h0, np.asarray(bd.embed[input_ids[0]], np.float32)[None])
        d.upload(self.mask, causal_mask(context_mask))
        d.upload(self.cmask, context_mask)
        row = bd.hidden * self.h0.dtype.itemsize
        copies, src = [], 0
        for s, e in _runs(input_ids[0] == bd.image_token):
            copies.append((s * row, src * row, (e - s) * row))
            src += e - s
        self.task_ids = input_ids.copy()
        self.task_mask = context_mask.copy()
        if copies != self.copies:
            self.copies = copies
            d.destroy_graph(self.graph)
            self.graph = None
            self._enqueue()                  # run uncaptured once before any capture
            d.sync()
            if self.use_graph:
                self.graph = d.capture(self._enqueue)

    def _enqueue(self) -> None:
        d, bd = self.d, self.b
        d.record(self.ev[0])
        x = self.vision_in
        for name, out in zip(bd.vision, self.vision_out):
            d.enqueue(name, {("pixel_values" if name == bd.vision[0] else "hidden_in"): x,
                             **out})
            x = next(iter(out.values()))
        for dst, src, n in self.copies:
            d.d2d(self.h0, x, n, dst, src)
        d.record(self.ev[1])
        h = self.h0
        for name, out in zip(bd.language, self.lang_out):
            d.enqueue(name, {"hidden_in": h, "causal_mask": self.mask, **out})
            h = next(iter(out.values()))
        d.record(self.ev[2])
        d.enqueue("action_context", {"fused_tokens": h, "context_mask": self.cmask,
                                     "state": self.state, **self.cache})
        d.record(self.ev[3])
        for i, ti in enumerate(self.time_index):
            a_in, a_out = self.actions[i % 2], self.actions[(i + 1) % 2]
            d.enqueue("action_step", {"action": a_in, "time_index": ti, **self.cache,
                                      **self.step_out})
            d.enqueue("action_output", {"action_hidden": self.step_out["action_hidden"],
                                        "velocity": self.vel})
            d.enqueue(self.axpy, {"a": a_in, "b": self.vel, "out": a_out})
        d.record(self.ev[4])

    @property
    def result(self) -> DevBuf:
        return self.actions[len(self.time_index) % 2]

    def infer(self, pixel_values, input_ids, context_mask, state, noise,
              timings: dict | None = None) -> np.ndarray:
        d = self.d
        t0 = time.perf_counter()
        if (self.task_ids is None or not np.array_equal(input_ids, self.task_ids)
                or not np.array_equal(context_mask, self.task_mask)):
            self._set_prompt(input_ids, context_mask)
        d.upload(self.vision_in, pixel_values)
        d.upload(self.state, state)
        d.upload(self.actions[0], noise)
        if self.graph is not None:
            d.launch(self.graph)
        else:
            self._enqueue()
        d.download(self.result)
        d.sync()
        if timings is not None:
            ev = self.ev
            timings.update(vision=d.elapsed_ms(ev[0], ev[1]),
                           language=d.elapsed_ms(ev[1], ev[2]),
                           action_context=d.elapsed_ms(ev[2], ev[3]),
                           denoise=d.elapsed_ms(ev[3], ev[4]),
                           total=(time.perf_counter() - t0) * 1e3)
        return self.result.host().copy()


class SmolVLADevice:
    """SmolVLA as one device-resident chain (see smolvla_trt.Policy for the host one).

    The prefix [cameras, prompt, state] is one persistent buffer: each camera's scaled
    embedding and the state projection are written straight into their rows, the empty
    slot's embedding is computed once, the prompt rows and masks are uploaded when the
    task or camera count changes. Per step: action_in, a strided copy into that step's
    [action | time] buffer (the time half is precomputed), time_in, SiLU, time_out,
    decode, action_out and the Euler update, all on the GPU.
    """

    def __init__(self, bundle, engines, ops, graph: bool = False):
        from .smolvla_trt import GRAPHS, sinusoidal_time_embedding

        self.b, self.d, self.G = bundle, Device(engines), GRAPHS
        d, bd = self.d, bundle
        self.use_graph = graph
        self.pix = [d.buf_for("vision", "image") for _ in range(bd.n_cam_slots)]
        # Cameras may arrive as the uint8 [512,512,3] canvas instead: a quarter of the
        # upload, and SigLIP's float scaling runs on the GPU as an exact table gather.
        size = self.pix[0].shape[-1]
        self.canvas = [DevBuf((size, size, 3), np.uint8) for _ in range(bd.n_cam_slots)]
        self.to_pix = ops.get("siglip_u8", self.canvas[0].shape)
        self.img = d.outputs("vision")["img_embeds"]           # [1,64,960], unscaled
        hidden = self.img.shape[-1]
        self.row = hidden * 4
        self.prefix = d.buf_for("prefill", "vlm_embeds")      # [1,prefix,960]
        self.scale = ops.get("axpy", self.img.shape, float(np.float32(np.sqrt(hidden))))
        self.zeros_img = DevBuf(self.img.shape, np.float32)
        d.upload(self.zeros_img, np.zeros(self.img.shape, np.float32))
        n_img = self.img.shape[1]
        self.slots = [DevBuf.view(self.prefix, i * n_img * self.row, self.img.shape)
                      for i in range(bd.n_cam_slots)]
        self.pad_cam = DevBuf(self.img.shape, np.float32)
        if bd.n_cam_slots > 1:
            d.upload(self.pix[-1], -np.ones(self.pix[-1].shape, np.float32))
            d.enqueue("vision", {"image": self.pix[-1], "img_embeds": self.img})
            d.enqueue(self.scale, {"a": self.zeros_img, "b": self.img, "out": self.pad_cam})
        self.state = d.buf_for("state_proj", "state")
        self.state_row = DevBuf.view(self.prefix, (bd.prefix_len - 1) * self.row,
                                     d.spec("state_proj")["output"][0])
        self.p_mask = d.buf_for("prefill", "attention_mask")
        self.p_pos = d.buf_for("prefill", "position_ids")
        self.kv = d.outputs("prefill")
        self.d_mask = d.buf_for("decode", "attention_mask")
        self.d_pos = d.buf_for("decode", "position_ids")
        x_spec = d.spec("action_in")["action"][0]
        self.x = [DevBuf(x_spec, np.float32), DevBuf(x_spec, np.float32)]
        self.a = d.outputs("action_in")["output"]                       # [1,50,720]
        cat_spec = d.spec("time_in")["action_time"][0]                  # [1,50,1440]
        self.cat = []
        for step in range(bd.num_steps):
            tt = 1.0 + step * (-1.0 / bd.num_steps)
            te = np.broadcast_to(sinusoidal_time_embedding(tt)[None, None, :], self.a.shape)
            buf = DevBuf(cat_spec, np.float32)
            host = np.zeros(cat_spec, np.float32)
            host[..., self.a.shape[-1]:] = te
            d.upload(buf, host)
            self.cat.append(buf)
        self.h = d.outputs("time_in")["output"]
        self.silu_out = DevBuf(self.h.shape, np.float32)
        self.silu = ops.get("silu", self.h.shape)
        self.suf = d.outputs("time_out")["output"]
        self.out = d.outputs("decode")["expert_out"]
        self.v = d.outputs("action_out")["output"]
        self.euler = ops.get("axpy", x_spec, float(np.float32(-1.0 / bd.num_steps)))
        self.state_out = DevBuf(d.spec("state_proj")["output"][0], np.float32)
        self.ev = [d.event() for _ in range(4)]
        self.key = None
        self.n_real = None
        self.graph = None
        d.sync()

    def _set_contract(self, lang, n_real: int) -> None:
        """Prompt rows, masks and positions: fixed per (task, camera count)."""
        from .smolvla_split import IMG_TOKENS, make_att_2d_masks

        d, bd = self.d, self.b
        lang_emb, lang_mask = lang
        n_pad = bd.n_cam_slots - n_real
        host = self.prefix.host()
        host[0, bd.n_cam_slots * IMG_TOKENS:bd.prefix_len - 1] = lang_emb[0]
        d.upload_staged(self.prefix)                 # camera/state rows rewritten per call
        for i in range(n_real, bd.n_cam_slots):
            d.d2d(self.slots[i], self.pad_cam, self.pad_cam.nbytes)
        pad = np.concatenate([np.ones((1, IMG_TOKENS), bool)] * n_real
                             + [np.zeros((1, IMG_TOKENS), bool)] * n_pad
                             + [lang_mask, np.ones((1, 1), bool)], axis=1)
        att = np.zeros((1, bd.prefix_len), bool)
        att[0, -1] = True
        d.upload(self.p_mask, make_att_2d_masks(pad, att))
        d.upload(self.p_pos, (np.cumsum(pad, axis=1) - 1).astype(np.int64))
        suffix = np.ones((1, bd.chunk_size), bool)
        d.upload(self.d_mask, np.concatenate(
            [np.broadcast_to(pad[:, None, :], (1, bd.chunk_size, bd.prefix_len)),
             make_att_2d_masks(suffix, suffix)], axis=2))
        d.upload(self.d_pos, (pad.sum(axis=-1, keepdims=True)
                              + np.cumsum(suffix, axis=1) - 1).astype(np.int64))
        if n_real != self.n_real:
            self.n_real = n_real
            d.destroy_graph(self.graph)
            self.graph = None
            self._enqueue()
            d.sync()
            if self.use_graph:
                self.graph = d.capture(self._enqueue)

    def _enqueue(self) -> None:
        d, bd = self.d, self.b
        d.record(self.ev[0])
        for i in range(self.n_real):
            d.enqueue("vision", {"image": self.pix[i], "img_embeds": self.img})
            d.enqueue(self.scale, {"a": self.zeros_img, "b": self.img, "out": self.slots[i]})
        d.record(self.ev[1])
        d.enqueue("state_proj", {"state": self.state, "output": self.state_out})
        d.d2d(self.state_row, self.state_out, self.state_out.nbytes)
        d.enqueue("prefill", {"attention_mask": self.p_mask, "position_ids": self.p_pos,
                              "vlm_embeds": self.prefix, **self.kv})
        past = {k.replace("present_", "past_"): v for k, v in self.kv.items()}
        d.record(self.ev[2])
        half = self.a.shape[-1] * 4
        for step in range(bd.num_steps):
            x_in, x_out = self.x[step % 2], self.x[(step + 1) % 2]
            d.enqueue("action_in", {"action": x_in, "output": self.a})
            d.copy2d(self.cat[step], 2 * half, self.a, half, half, self.a.shape[1])
            d.enqueue("time_in", {"action_time": self.cat[step], "output": self.h})
            d.enqueue(self.silu, {"x": self.h, "out": self.silu_out})
            d.enqueue("time_out", {"hidden": self.silu_out, "output": self.suf})
            d.enqueue("decode", {"attention_mask": self.d_mask, "position_ids": self.d_pos,
                                 "expert_embeds": self.suf, **past, "expert_out": self.out})
            d.enqueue("action_out", {"expert_out": self.out, "output": self.v})
            d.enqueue(self.euler, {"a": x_in, "b": self.v, "out": x_out})
        d.record(self.ev[3])

    @property
    def result(self) -> DevBuf:
        return self.x[self.b.num_steps % 2]

    def canvas_buffer(self, i: int) -> np.ndarray:
        """Pinned uint8 [512,512,3] staging array of camera i, to preprocess into."""
        return self.canvas[i].host()

    def infer(self, pixels: list, lang, state: np.ndarray, noise: np.ndarray,
              key=None, timings: dict | None = None) -> np.ndarray:
        """pixels: per real camera, [1,3,512,512] float in [-1,1] or the uint8
        [512,512,3] canvas (optionally canvas_buffer(i) itself); lang: (embedding rows, mask);
        state [1,32] normalized; key identifies the task (prompt rows are re-uploaded
        when it changes). Omitting key disables prompt caching."""
        d = self.d
        t0 = time.perf_counter()
        if key is None or (key, len(pixels)) != (self.key, self.n_real):
            self._set_contract(lang, len(pixels))
            self.key = key
        for i, p in enumerate(pixels):
            if p.dtype != np.uint8:
                d.upload(self.pix[i], p)
                continue
            if p is self.canvas[i].host():           # filled in place by the caller
                d.upload_staged(self.canvas[i])
            else:
                d.upload(self.canvas[i], p)
            d.enqueue(self.to_pix, {"x": self.canvas[i], "out": self.pix[i]})
        d.upload(self.state, state)
        d.upload(self.x[0], noise)
        if self.graph is not None:
            d.launch(self.graph)
        else:
            self._enqueue()
        d.download(self.result)
        d.sync()
        if timings is not None:
            ev = self.ev
            timings.update(vision=d.elapsed_ms(ev[0], ev[1]), prefill=d.elapsed_ms(ev[1], ev[2]),
                           denoise=d.elapsed_ms(ev[2], ev[3]),
                           total=(time.perf_counter() - t0) * 1e3)
        return self.result.host().copy()


class Groot17Device:
    """GR00T N1.7 as two captured graphs around the frame history.

    Graph A encodes the current frames. Between the graphs, stream-ordered copies file
    the encode into a ring slot and stage the history frame's encode: which slot that is
    depends on capture times alone, so the host picks it without waiting on the GPU.
    Graph B scatters both frames into the prompt and the DeepStack buffers and runs the
    LLM, conditioning and DiT. Same arithmetic as groot17_trt.infer.
    """

    def __init__(self, bundle, engines, lag_s: float, graph: bool = False):
        from .groot17_trt import FrameHistory

        self.b, self.d = bundle, Device(engines)
        d, bd, b = self.d, bundle, bundle.b
        self.history = FrameHistory(lag_s)
        self.vision_in = d.buf_for(bd.vision[0], "pixel_values")
        self.vision_out = [d.outputs(n) for n in bd.vision]
        n_ds = b["deepstack"]
        self.now = [self.vision_out[-1]["vision_tokens"]] + [
            next(o[f"ds_{j}"] for o in self.vision_out if f"ds_{j}" in o) for j in range(n_ds)]
        self.past = [DevBuf(x.shape, x.dtype) for x in self.now]
        self.slots: list[list[DevBuf]] = []
        self.h0 = d.buf_for(bd.llm[0], "h")
        d.upload(self.h0, bd.prompt_embeds)
        self.ds_full = []
        for _ in range(n_ds):
            z = DevBuf(self.h0.shape, self.h0.dtype)
            d.upload(z, np.zeros(z.shape, z.dtype))
            self.ds_full.append(z)
        self.llm_out = [d.outputs(n) for n in bd.llm]
        self.pad_bias = d.buf_for(bd.cond[0], "pad_bias")
        d.upload(self.pad_bias, bd.pad_bias)
        self.state = d.buf_for(bd.cond[0], "state")
        self.cond_out = [d.outputs(n) for n in bd.cond]
        self.text_bias = DevBuf(bd.text_bias.shape, np.float32)
        self.image_bias = DevBuf(bd.image_bias.shape, np.float32)
        d.upload(self.text_bias, bd.text_bias)
        d.upload(self.image_bias, bd.image_bias)
        c_last = self.cond_out[-1] if len(self.cond_out) > 1 else self.cond_out[0]
        self.dit = DitChain(d, bd, bd.consts, {
            "vl": c_last["vl_out" if len(self.cond_out) > 1 else "vl"],
            "state_features": self.cond_out[0]["state_features"],
            "text_bias": self.text_bias, "image_bias": self.image_bias})
        d.sync()
        # One image per run, past frames first: run k is frame k // V, view k % V.
        V, row = b["views"], self.h0.shape[-1] * self.h0.dtype.itemsize
        img_bytes = int(np.prod(self.now[0].shape[1:])) * self.now[0].dtype.itemsize
        runs = _runs(bd.image_positions)
        if len(runs) != V * b["frames"] or any(
                (e - s) * row != img_bytes for s, e in runs):
            raise ValueError("image-token runs do not match frames x views")
        self.copies = [(s * row, k // V, (k % V) * img_bytes, img_bytes)
                       for k, (s, e) in enumerate(runs)]
        self.ev = [d.event() for _ in range(4)]
        self.graphs = None
        self._vision()
        self._rest()                          # every engine runs once before any capture
        d.sync()
        if graph:
            self.graphs = (d.capture(self._vision), d.capture(self._rest))

    def _vision(self) -> None:
        d, bd = self.d, self.b
        d.record(self.ev[0])
        x = self.vision_in
        for name, out in zip(bd.vision, self.vision_out):
            d.enqueue(name, {("pixel_values" if name == bd.vision[0] else "x"): x, **out})
            x = out["vision_tokens"] if name == bd.vision[-1] else out["x_out"]
        d.record(self.ev[1])

    def _rest(self) -> None:
        d, bd = self.d, self.b
        frames = (self.past, self.now)
        for dst, frame, src, n in self.copies:
            for j, target in enumerate([self.h0] + self.ds_full):
                d.d2d(target, frames[frame][j], n, dst, src)
        h = self.h0
        for name, out in zip(bd.llm, self.llm_out):
            feeds = {"h": h}
            feeds.update({k: self.ds_full[int(k[3:])] for k in bd.graph[name]["inputs"]
                          if k.startswith("ds_")})
            d.enqueue(name, {**feeds, **out})
            h = next(iter(out.values()))
        c0, c1 = self.cond_out[0], self.cond_out[1:]
        d.enqueue(bd.cond[0], {"features": h, "pad_bias": self.pad_bias, "state": self.state,
                               **c0})
        vl = c0["vl"]
        for name, out in zip(bd.cond[1:], c1):
            d.enqueue(name, {"vl": vl, "pad_bias": self.pad_bias, **out})
            vl = out["vl_out"]
        d.record(self.ev[2])
        self.dit.enqueue()
        d.record(self.ev[3])

    def _slot(self) -> int:
        """A ring slot no history entry holds (the history prunes what it can never pick)."""
        live = {e for _, e in self.history.entries}
        for i in range(len(self.slots)):
            if i not in live:
                return i
        self.slots.append([DevBuf(x.shape, x.dtype) for x in self.now])
        return len(self.slots) - 1

    @property
    def result(self) -> DevBuf:
        return self.dit.result

    def infer(self, pixel_values, state, noise, t_capture: float,
              timings: dict | None = None) -> tuple[np.ndarray, float]:
        """(actions, age of the history frame in seconds)."""
        d = self.d
        t0 = time.perf_counter()
        d.upload(self.vision_in, pixel_values)
        d.upload(self.state, state)
        d.upload(self.dit.actions[0], noise)
        if self.graphs:
            d.launch(self.graphs[0])
        else:
            self._vision()
        s = self._slot()
        for dst, src in zip(self.slots[s], self.now):
            d.d2d(dst, src, src.nbytes)
        self.history.push(t_capture, s)
        past, age = self.history.past(t_capture)
        for dst, src in zip(self.past, self.slots[past]):
            d.d2d(dst, src, src.nbytes)
        if self.graphs:
            d.launch(self.graphs[1])
        else:
            self._rest()
        d.download(self.result)
        d.sync()
        if timings is not None:
            ev = self.ev
            timings.update(vision=d.elapsed_ms(ev[0], ev[1]),
                           backbone=d.elapsed_ms(ev[1], ev[2]),
                           denoise=d.elapsed_ms(ev[2], ev[3]),
                           total=(time.perf_counter() - t0) * 1e3)
        return self.result.host().copy(), age
