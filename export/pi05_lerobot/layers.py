"""Split π0.5 modules and the refit-template packing shared by the LeRobot exporter.

The wrappers and the ONNX passes are the openpi exporter's (export/pi05/prepare_blocks.py,
prepare_full.py, export_utils.py), so a LeRobot bundle has the same seven layer kinds,
input/output names and FP32-accumulation casts the board's refitted templates expect.
They are copied rather than imported: the openpi stage scripts resolve container paths
at import time.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper, shape_inference
from torch import nn
from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb, eager_attention_forward, rotate_half

MASK_NEG = -2.3819763e38


def array(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy() if t.dtype == torch.bfloat16 else t.detach().cpu().numpy()


class VisionStem(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    def forward(self, image):
        return self.embeddings(image)


class VisionBlock(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden):
        # The pinned LeRobot and OpenPI environments use different Transformers versions.
        result = self.layer(hidden, attention_mask=None)
        return result[0] if isinstance(result, tuple) else result


class VisionTail(nn.Module):
    """Post-norm and projector. `fp32_out` hands the FP16 features to an FP32 residual stream."""

    def __init__(self, norm, projector, fp32_out=False):
        super().__init__()
        self.norm, self.projector, self.fp32_out = norm, projector, fp32_out

    def forward(self, hidden):
        out = self.projector(self.norm(hidden))
        return out.float() if self.fp32_out else out


class GemmaBlock(nn.Module):
    """One Gemma layer; `cached` takes the AdaRMS modulation as inputs instead of `cond`.

    `residual32`: the hidden state in and out is FP32 and both residual adds are FP32;
    the norm's output is rounded to FP16 once, so the projections and MLP stay FP16.
    pi05_base's language residual stream reaches ~6e3, where an FP16 step is 4, and the
    rounding differences between TensorRT and PyTorch then grow through the layers.
    """

    def __init__(self, layer, action=False, cached=False, residual32=False):
        super().__init__()
        self.layer = layer
        self.action, self.cached, self.residual32 = action, cached, residual32

    def norm(self, hidden, norm, cond, modulation):
        if not self.cached:
            x, gate = norm(hidden, cond)
            return (x.half(), gate) if self.residual32 else (x, gate)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        x = hidden.float() * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + norm.eps)
        x = x * (1 + scale.float()) + shift.float()
        return x.to(hidden.dtype), gate.to(hidden.dtype)

    def forward(self, hidden, mask, cos, sin, prefix_k=None, prefix_v=None,
                cond=None, mod_in=None, mod_post=None):
        layer = self.layer
        x, gate = self.norm(hidden, layer.input_layernorm, cond, mod_in)
        shape = (x.shape[0], x.shape[1], -1, 256)
        q = layer.self_attn.q_proj(x).view(shape).transpose(1, 2)
        k = layer.self_attn.k_proj(x).view(shape).transpose(1, 2)
        v = layer.self_attn.v_proj(x).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        new_k, new_v = k, v
        if self.action:
            k, v = torch.cat((prefix_k, k), dim=2), torch.cat((prefix_v, v), dim=2)
        attn, _ = eager_attention_forward(layer.self_attn, q, k, v, mask, layer.self_attn.scaling)
        y = layer.self_attn.o_proj(attn.reshape(hidden.shape[0], hidden.shape[1], 2048))
        if self.residual32:
            y = y.float()
        residual = hidden + y if gate is None else hidden + y * gate
        x, gate = self.norm(residual, layer.post_attention_layernorm, cond, mod_post)
        y = layer.mlp(x)
        if self.residual32:
            y = y.float()
        output = residual + y if gate is None else residual + y * gate
        return (output,) if self.action else (output, new_k, new_v)


class KVBlock(nn.Module):
    """The last language layer as the action expert reads it: only its prefix K and V.

    Its attention output and MLP feed nothing (pi0.5 applies no final norm to the
    prefix), so they are not computed; in FP16 that MLP output also overflows.
    Same norm, projections and RoPE arithmetic as GemmaBlock.
    """

    def __init__(self, layer, residual32=False):
        super().__init__()
        self.layer, self.residual32 = layer, residual32

    def forward(self, hidden, cos, sin):
        layer = self.layer
        x, _ = layer.input_layernorm(hidden, None)
        if self.residual32:
            x = x.half()
        shape = (x.shape[0], x.shape[1], -1, 256)
        k = layer.self_attn.k_proj(x).view(shape).transpose(1, 2)
        v = layer.self_attn.v_proj(x).view(shape).transpose(1, 2)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        return k * cos + rotate_half(k) * sin, v


class ActionInput(nn.Module):
    def __init__(self, projection):
        super().__init__()
        self.projection = projection

    def forward(self, actions):
        return self.projection(actions.half())


class ActionBlock(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.block = GemmaBlock(layer, action=True, cached=True)

    def forward(self, hidden, mask, cos, sin, prefix_k, prefix_v, mod_in, mod_post):
        return self.block(hidden, mask, cos, sin, prefix_k, prefix_v, None, mod_in, mod_post)[0]


class ActionOutput(nn.Module):
    def __init__(self, norm, projection):
        super().__init__()
        self.eps, self.projection = norm.eps, projection

    def forward(self, hidden, mod_final):
        scale, shift, _ = mod_final.chunk(3, dim=-1)
        x = hidden.float() * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + self.eps)
        x = (x * (1 + scale.float()) + shift.float()).half()
        return self.projection(x).float()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def accumulation(graph: onnx.ModelProto) -> int:
    """Every all-FP16 MatMul/Conv computes in FP32 and rounds its result to FP16."""
    inferred = shape_inference.infer_shapes(graph)
    types = {v.name: v.type.tensor_type.elem_type for v in (
        list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info))}
    types.update({v.name: v.data_type for v in graph.graph.initializer})
    nodes, changed = [], 0
    for i, node in enumerate(graph.graph.node):
        if node.op_type in ("MatMul", "Conv") and all(types.get(x) == TensorProto.FLOAT16 for x in node.input):
            operands = []
            for j, operand in enumerate(node.input):
                output = f"fp32_accum_input_{i}_{j}"
                nodes.append(helper.make_node("Cast", [operand], [output], to=TensorProto.FLOAT))
                operands.append(output)
            original_output = node.output[0]
            del node.input[:]
            node.input.extend(operands)
            node.output[0] = f"fp32_accum_output_{i}"
            nodes.append(node)
            nodes.append(helper.make_node("Cast", [node.output[0]], [original_output], to=TensorProto.FLOAT16))
            changed += 1
        else:
            nodes.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    onnx.checker.check_model(graph)
    return changed


def structure_hash(graph: onnx.ModelProto) -> str:
    copy = onnx.ModelProto()
    copy.CopyFrom(graph)
    for tensor in copy.graph.initializer:
        if tensor.data_type in (TensorProto.FLOAT16, TensorProto.FLOAT):
            for field in ("raw_data", "float_data", "double_data", "int32_data", "int64_data"):
                tensor.ClearField(field)
    return hashlib.sha256(copy.SerializeToString()).hexdigest()


def pack(root: Path, graph: onnx.ModelProto, kind: str, label: str, template: bool) -> dict:
    """One component's weights as .npy files; the first of a kind is also its template."""
    directory = root / "weights" / label
    directory.mkdir(parents=True, exist_ok=True)
    descriptors = []
    for i, tensor in enumerate(graph.graph.initializer):
        if tensor.data_type not in (TensorProto.FLOAT16, TensorProto.FLOAT):
            continue
        arr = np.array(numpy_helper.to_array(tensor), copy=True, order="C")
        name = f"{i:03d}.npy"
        np.save(directory / name, arr)
        descriptors.append({"name": tensor.name, "file": name, "dtype": str(arr.dtype),
                            "shape": list(arr.shape), "bytes": arr.nbytes, "sha256": sha(directory / name)})
    record = {"kind": kind, "label": label, "structure_sha256": structure_hash(graph),
              "weights": descriptors, "initializer_bytes": sum(x["bytes"] for x in descriptors)}
    (directory / "index.json").write_text(json.dumps(record, indent=2))
    if template:
        (root / "templates").mkdir(exist_ok=True)
        onnx.save(graph, root / "templates" / f"{kind}.onnx")
        (root / "templates" / f"{kind}.json").write_text(json.dumps(record, indent=2))
    else:
        expected = json.loads((root / "templates" / f"{kind}.json").read_text())
        assert expected["structure_sha256"] == record["structure_sha256"], (
            "different graph structure requires its own template", kind, label)
    return record


def metric(actual: np.ndarray, expected: np.ndarray) -> dict:
    a, b = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        return {"finite": False}
    return {"finite": True,
            "cosine": float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)),
            "max_abs": float(np.max(np.abs(a - b))), "mean_abs": float(np.mean(np.abs(a - b))),
            "range": float(b.max() - b.min())}
