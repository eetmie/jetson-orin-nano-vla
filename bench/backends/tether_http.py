"""Backend: FastCrest Tether (`tether serve`) over its HTTP `/act` endpoint.

Tether is the third-party claim under test. Its hardware table lists SmolVLA on an
8 GB Orin Nano at **~25 ms FP16**, which is roughly an order of magnitude below what
the split TensorRT path measures for a 10-step flow-matching loop on this board
(~110–170 ms projected from per-graph timings). Both numbers cannot describe the same
work, so the point of this backend is to find out which of these is true here:

  * it really is that fast (adaptive step early-exit, CUDA graphs, fused kernels), or
  * the figure is a single forward pass rather than the full denoise loop, or
  * the monolithic export falls back to the CUDA EP — the same ~500 ms regime the
    monolith already measured on this board — or does not build at all in 8 GB.

Every one of those is a publishable result, so the harness records the outcome
instead of treating a failure as a crash.

Two caveats this backend measures around
----------------------------------------
1. **It is a server.** The model lives in another process, so `pids()` returns that
   process (and its children) for the CPU/RSS attribution, and the system-wide
   tegrastats numbers are the ones that count.
2. **The timing includes transport.** `total` is the client's roundtrip: JSON encode,
   base64 image, loopback HTTP, decode. Where the response carries a server-side
   figure it is reported separately as `server`. Compare `server` against the other
   backends' `total`, and use the roundtrip when asking what a control loop would
   actually see. `docs/04-metrics.md` spells this out.

The `/act` request schema is not pinned in Tether's public docs, so the first call
negotiates: several documented-looking payload shapes are tried in order and the
first one the server accepts is kept and recorded in `meta()`. `bench tether-probe`
prints the server's own OpenAPI schema if a shape needs adding.
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from ..obs import Observation
from .base import Backend, InferResult, load_export_info

_ACTION_KEYS = ("actions", "action", "action_chunk", "chunk", "predictions")
_SERVER_MS_KEYS = ("inference_ms", "latency_ms", "infer_ms", "model_ms",
                   "server_ms", "elapsed_ms")


def _png_b64(img: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _jpeg_b64(img: np.ndarray, quality: int = 95) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _candidates(img_png: str, img_jpg: str, task: str, state: list[float]) -> list[tuple[str, dict]]:
    """Payload shapes to try, most likely first."""
    return [
        ("image+instruction+state", {"image": img_png, "instruction": task, "state": state}),
        ("image+prompt+state", {"image": img_png, "prompt": task, "state": state}),
        ("image+task+state", {"image": img_png, "task": task, "state": state}),
        ("images[]+instruction", {"images": [img_png], "instruction": task, "state": state}),
        ("datauri+instruction", {"image": f"data:image/png;base64,{img_png}",
                                 "instruction": task, "state": state}),
        ("jpeg+instruction", {"image": img_jpg, "instruction": task, "state": state}),
        ("observation-nested", {"observation": {"image": img_png, "state": state},
                                "instruction": task}),
    ]


class TetherHttpBackend(Backend):
    name = "tether"
    noise_injected = False          # the server draws its own noise; see bench/parity.py
    preprocess_owned = False        # it takes a raw frame and does its own resize

    def __init__(self, export_dir: Path | None = None, url: str | None = None,
                 port: int = 8000, device: str = "cuda", providers: str | None = None,
                 api_key: str | None = None, extra_args: list[str] | None = None,
                 action_dim: int = 4, startup_timeout_s: float = 900.0,
                 payload_template: str | None = None, log_path: Path | None = None):
        self.export_dir = Path(export_dir) if export_dir else None
        self.url = (url or f"http://127.0.0.1:{port}").rstrip("/")
        self.port = port
        self.device = device
        self.providers = providers
        self.api_key = api_key
        self.extra_args = extra_args or []
        self.action_dim = action_dim
        self.startup_timeout_s = startup_timeout_s
        self.payload_template = payload_template
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self._shape: str | None = None
        self._payload_fn = None
        self._config: dict = {}
        self._startup_s: float | None = None
        self._info = load_export_info(self.export_dir) if self.export_dir else {}

    # -- lifecycle ----------------------------------------------------------
    def load(self) -> None:
        import requests
        self._requests = requests
        if self.export_dir is not None:
            self._spawn()
        self._wait_healthy()
        try:
            r = self._requests.get(f"{self.url}/config", headers=self._headers(), timeout=10)
            self._config = r.json() if r.ok else {"status": r.status_code}
        except Exception as e:
            self._config = {"error": str(e)}

    def _headers(self) -> dict:
        return {"X-Tether-Key": self.api_key} if self.api_key else {}

    def _spawn(self) -> None:
        cmd = ["tether", "serve", str(self.export_dir),
               "--port", str(self.port), "--device", self.device]
        if self.providers:
            cmd += ["--providers", self.providers]
        if self.api_key:
            cmd += ["--api-key", self.api_key]
        cmd += self.extra_args
        log = open(self.log_path, "w") if self.log_path else subprocess.DEVNULL
        self._cmd = cmd
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ), start_new_session=True)
        self._spawn_t0 = t0

    def _wait_healthy(self) -> None:
        t0 = getattr(self, "_spawn_t0", time.perf_counter())
        deadline = t0 + self.startup_timeout_s
        last = None
        while time.perf_counter() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    f"`tether serve` exited with code {self.proc.returncode} before "
                    f"becoming healthy — see {self.log_path}")
            try:
                r = self._requests.get(f"{self.url}/health", timeout=5)
                if r.ok:
                    self._startup_s = time.perf_counter() - t0
                    return
                last = f"HTTP {r.status_code}"
            except Exception as e:
                last = str(e)
            time.sleep(1.0)
        raise TimeoutError(f"{self.url}/health not ready in {self.startup_timeout_s}s "
                           f"(last: {last}). First TRT build can be slow — raise "
                           f"--tether-startup-timeout, or check the serve log.")

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def pids(self) -> list[int]:
        # The model is in the server, not in us. Watch both: ours is the client cost.
        return ([self.proc.pid] if self.proc else []) + [os.getpid()]

    # -- payload negotiation -------------------------------------------------
    def _negotiate(self, obs: Observation) -> None:
        png, jpg = _png_b64(obs.image), _jpeg_b64(obs.image)
        state = [float(x) for x in np.asarray(obs.state).reshape(-1)]
        if self.payload_template:
            tpl = json.loads(Path(self.payload_template).read_text())
            self._shape = f"template:{self.payload_template}"
            self._payload_fn = lambda o: _fill_template(tpl, o)
            return
        errors = []
        for label, body in _candidates(png, jpg, obs.task, state):
            try:
                r = self._requests.post(f"{self.url}/act", json=body,
                                        headers=self._headers(), timeout=300)
            except Exception as e:
                errors.append(f"{label}: {e}")
                continue
            if r.ok and _extract_actions(r.json()) is not None:
                self._shape = label
                self._payload_fn = _payload_builder(label)
                return
            errors.append(f"{label}: HTTP {r.status_code} {r.text[:160]}")
        raise RuntimeError(
            "no /act payload shape was accepted. Run `python -m bench tether-probe "
            f"--url {self.url}` to print the server's own schema, then pass it with "
            "--tether-payload-template. Tried:\n  " + "\n  ".join(errors))

    # -- inference -----------------------------------------------------------
    def meta(self) -> dict:
        return {
            "backend": self.name,
            "url": self.url,
            "export_dir": str(self.export_dir) if self.export_dir else None,
            "spawned": self.proc is not None,
            "serve_cmd": getattr(self, "_cmd", None),
            "startup_s": round(self._startup_s, 1) if self._startup_s else None,
            "payload_shape": self._shape,
            "server_config": self._config,
            "action_dim": self.action_dim,
            "tether_version": _tether_version(),
            "export_info": self._info,
            "note": "total_ms is the client roundtrip and includes PNG encode + HTTP",
        }

    def infer(self, obs: Observation) -> InferResult:
        if self._payload_fn is None:
            self._negotiate(obs)
        t0 = time.perf_counter()
        body = self._payload_fn(obs)
        t_enc = time.perf_counter()
        r = self._requests.post(f"{self.url}/act", json=body,
                                headers=self._headers(), timeout=300)
        r.raise_for_status()
        payload = r.json()
        t1 = time.perf_counter()

        actions = _extract_actions(payload)
        if actions is None:
            raise RuntimeError(f"no action array in /act response: {str(payload)[:200]}")
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim == 3:                       # (1, chunk, dim)
            arr = arr[0]
        if arr.ndim == 1:                       # a single action
            arr = arr[None, :]
        arr = arr[:, :self.action_dim]

        t = {"total": (t1 - t0) * 1000.0,
             "encode_request": (t_enc - t0) * 1000.0,
             "roundtrip": (t1 - t_enc) * 1000.0}
        if (srv := _extract_server_ms(payload)) is not None:
            t["server"] = srv
            t["transport_overhead"] = t["total"] - srv
        return InferResult(arr, t)


def _payload_builder(label: str):
    def build(o: Observation) -> dict:
        png = _png_b64(o.image)
        state = [float(x) for x in np.asarray(o.state).reshape(-1)]
        for lab, body in _candidates(png, _jpeg_b64(o.image), o.task, state):
            if lab == label:
                return body
        raise KeyError(label)
    return build


def _fill_template(tpl: dict, o: Observation) -> dict:
    """Substitute {{image_b64}} / {{instruction}} / {{state}} in a JSON template."""
    png = _png_b64(o.image)
    state = [float(x) for x in np.asarray(o.state).reshape(-1)]

    def walk(v):
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        if v == "{{image_b64}}":
            return png
        if v == "{{instruction}}":
            return o.task
        if v == "{{state}}":
            return state
        return v
    return walk(tpl)


def _extract_actions(payload):
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None
    for k in _ACTION_KEYS:
        if k in payload:
            return payload[k]
    for k in ("result", "data", "output"):
        if isinstance(payload.get(k), (dict, list)):
            got = _extract_actions(payload[k])
            if got is not None:
                return got
    return None


def _extract_server_ms(payload):
    if not isinstance(payload, dict):
        return None
    for k in _SERVER_MS_KEYS:
        v = payload.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    for k in ("timing", "timings", "metrics", "meta"):
        if isinstance(payload.get(k), dict):
            got = _extract_server_ms(payload[k])
            if got is not None:
                return got
    return None


def _tether_version() -> str | None:
    try:
        return subprocess.run(["tether", "--version"], capture_output=True,
                              text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None
