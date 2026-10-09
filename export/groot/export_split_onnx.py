#!/usr/bin/env python3
# VENDORED from spark-projects @ 9e367cd — vla-onnx/groot/export_split_onnx.py
"""Export GR00T N1.6 as split ONNX graphs for the 8 GB Orin, then a mixed-FP16 copy.

Graphs (static shapes, one per TensorRT engine; see groot_split.py for the pipeline):

    vision_{k}  SigLIP2 chunks; the last adds pixel unshuffle + mlp1
    llm_{k}     Qwen3 layer chunks over the padded sequence; the last adds the final norm
    cond        vlln + embodiment-sliced state encoder
    time        the two sinusoidal encodings of t, kept FP32
    dit_{k}     one denoising step in chunks; the last adds the decoder and Euler update

Chunks are packed to --budget-m million params, the per-engine TensorRT build budget
measured on this board for X-VLA (vla-onnx/xvla/notes/split_design.md). The 151680 x 2048
token embedding is NOT a graph: it ships as an FP16 .npy the runtime memory-maps and
gathers from on the CPU, so its 620 MB never becomes resident.

    python export_split_onnx.py --ref work/ref_robocasa_v3.npz --out ~/bundles/groot-n16-split
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import torch

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "common", HERE.parent):    # spark-projects layout, repo export/ layout
    if (_p / "vla_common").is_dir():
        sys.path.insert(0, str(_p))

import re  # noqa: E402

from vla_common.bundle import write_manifest  # noqa: E402
from vla_common.fp16_weights import FP16_SENSITIVE_OPS  # noqa: E402

from groot_split import (MASK_NEG, Cond, Contract, DitChunk, LlmChunk,  # noqa: E402
                         TimeEmb, VisionChunk, layer_plans, load_policy, n_params)

SENSITIVE = tuple(FP16_SENSITIVE_OPS)

# Every node of Qwen3's RMSNorms stays FP32, matched by name, not op type. Layer 2 writes
# a ~16150 "massive activation" into the sink token's residual and every later layer
# carries it: mean(x^2) is then ~1.3e5, and the `+ eps` Add between ReduceMean and Sqrt
# overflows FP16 to inf, rsqrt -> 0, the token is zeroed. Blocking Pow/ReduceMean alone
# left that Add in FP16 (measured: llm_7 cos 0.567, actions cos 0.992). Stock PyTorch
# upcasts the whole norm, so this is what the reference does too.
RMSNORM_NODE = re.compile(r"(input_layernorm|post_attention_layernorm|q_norm|k_norm|^/norm)/")


def strip_fp16_roundtrips(m: onnx.ModelProto) -> int:
    """Remove Cast(FP32->FP16) -> Cast(->FP32) pairs; consumers read the FP32 source.

    ORT's float16 converter inserts this round trip on edges BETWEEN two blocked
    (FP32-kept) nodes, e.g. RMSNorm's Pow -> ReduceMean. With Qwen3's massive activation
    x^2 is ~2.6e8, the FP16 hop makes it inf, and the "FP32" norm returns 0. Dropping
    the pair is strictly more precise and changes no other node.
    """
    typed = onnx.shape_inference.infer_shapes(m)
    types = {v.name: v.type.tensor_type.elem_type
             for v in list(typed.graph.value_info) + list(typed.graph.input) + list(typed.graph.output)}
    types.update({i.name: i.data_type for i in m.graph.initializer})
    prod = {o: n for n in m.graph.node for o in n.output}
    outputs = {o.name for o in m.graph.output}

    def to(n):
        return next(a.i for a in n.attribute if a.name == "to")

    F32, F16 = onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16
    rename = {}
    for n in m.graph.node:
        if n.op_type == "Cast" and to(n) == F32 and n.output[0] not in outputs:
            p = prod.get(n.input[0])
            if p is not None and p.op_type == "Cast" and to(p) == F16 and types.get(p.input[0]) == F32:
                rename[n.output[0]] = p.input[0]
    for n in m.graph.node:
        for i, x in enumerate(n.input):
            if x in rename:
                n.input[i] = rename[x]
    while True:                      # drop nodes nothing reads any more
        used = {x for n in m.graph.node for x in n.input} | outputs
        keep = [n for n in m.graph.node if any(o in used for o in n.output)]
        if len(keep) == len(m.graph.node):
            break
        del m.graph.node[:]
        m.graph.node.extend(keep)
    return len(rename)


def to_mixed_fp16(src: Path, dst: Path) -> None:
    from onnxruntime.transformers import float16
    from onnxruntime.transformers.onnx_model import OnnxModel

    m = onnx.load(str(src))
    nodes = [n.name for n in m.graph.node if RMSNORM_NODE.search(n.name)]
    out = float16.convert_float_to_float16(
        m, keep_io_types=True, node_block_list=nodes,
        op_block_list=list(float16.DEFAULT_OP_BLOCK_LIST) + list(SENSITIVE))
    n = strip_fp16_roundtrips(out)
    if n:
        print(f"  {src.stem}: removed {n} FP32->FP16->FP32 round trips")
    # save_model_to_file topologically sorts (keep_io_types appends input Casts last).
    OnnxModel(out).save_model_to_file(str(dst), use_external_data_format=False)


def dump(module, args, path: Path, inputs, outputs):
    t0 = time.time()
    torch.onnx.export(module, args, str(path), input_names=inputs, output_names=outputs,
                      opset_version=18, dynamo=False, do_constant_folding=True)
    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    empty = [n.name for n in m.graph.node if any(i == "" for i in n.input)
             and n.op_type not in ("Resize", "Clip", "Pad", "Slice")]
    if empty:
        raise SystemExit(f"{path.name}: nodes with empty inputs {empty[:5]}")
    print(f"  {path.name:14s} {n_params(module) / 1e6:6.1f} M  "
          f"{path.stat().st_size / 1e6:6.0f} MB  {time.time() - t0:4.0f} s")


def dedupe_casts(path: Path) -> int:
    """ORT's float16 converter can insert the same `*_cast_to_fp32` Cast twice when one
    tensor feeds two FP32-kept ops (seen in dit_0's attention-mask path): same input,
    same output name, which onnx and ORT reject as non-SSA. Drop the exact repeats."""
    m = onnx.load(str(path))
    seen, keep = {}, []
    for n in m.graph.node:
        key = tuple(n.output)
        if key in seen:
            first = seen[key]
            assert (first.op_type, list(first.input), first.attribute) == \
                (n.op_type, list(n.input), n.attribute), f"{path.name}: {key} defined twice, differently"
            continue
        seen[key] = n
        keep.append(n)
    dropped = len(m.graph.node) - len(keep)
    if dropped:
        del m.graph.node[:]
        m.graph.node.extend(keep)
        onnx.save(m, str(path))
        onnx.checker.check_model(str(path))
    return dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference.py output: fixes the contract")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=320)
    ap.add_argument("--budget-m", type=float, default=105.0)
    ap.add_argument("--keep-fp32", action="store_true", help="also keep the FP32 graphs")
    a = ap.parse_args()

    r = np.load(a.ref)
    meta = json.loads(str(r["meta"]))
    policy = load_policy(meta["checkpoint"], torch.float32)
    eagle = policy.backbone.model
    tok = eagle.config.image_token_index
    ids = r["input_ids"]
    n_real = ids.shape[1]
    assert n_real <= a.seq_len
    c = Contract(meta["embodiment"], int(r["embodiment_id"]), meta["views"],
                 tuple(r["pixel_values"].shape[-2:]), int((ids == tok).sum()) // meta["views"],
                 a.seq_len)
    plans = layer_plans(policy, int(a.budget_m * 1e6))

    fp32 = a.out.parent / (a.out.name + ".fp32")
    if fp32.exists():
        shutil.rmtree(fp32)
    fp32.mkdir(parents=True)
    graphs = []
    S, V = c.seq_len, c.views
    D = eagle.config.text_config.hidden_size
    vd = eagle.config.vision_config.hidden_size
    grid = (c.image_hw[0] // 14) * (c.image_hw[1] // 14)

    def add(name, module, args, inputs, outputs):
        dump(module, args, fp32 / f"{name}.onnx", inputs, outputs)
        graphs.append({"name": name, "file": f"{name}.onnx", "inputs": inputs, "outputs": outputs,
                       "params_m": round(n_params(module) / 1e6, 1)})

    with torch.no_grad():
        for k, (lo, hi) in enumerate(plans["vision"]):
            m = VisionChunk(policy, lo, hi, c.image_hw).eval()
            x = torch.zeros(V, 3, *c.image_hw) if m.first else torch.zeros(V, grid, vd)
            add(f"vision_{k}", m, (x,), ["pixel_values" if m.first else "x"],
                ["vision_tokens" if m.last else "x_out"])
        for k, (lo, hi) in enumerate(plans["llm"]):
            m = LlmChunk(policy, lo, hi, S).eval()
            add(f"llm_{k}", m, (torch.zeros(1, S, D),), ["h"], ["features" if m.last else "h_out"])
        add("cond", Cond(policy, c.embodiment_id).eval(),
            (torch.zeros(1, S, D), torch.zeros(1, 1, c.state_dim)),
            ["features", "state"], ["vl", "state_features"])
        add("time", TimeEmb(policy).eval(), (torch.zeros(1),), ["t"], ["t_proj", "tau"])
        hd = policy.action_head.input_embedding_dim
        dit_args = (torch.zeros(1, 1 + c.action_horizon, hd), torch.zeros(1, hd),
                    torch.zeros(1, S, D), torch.zeros(1, 1, S), torch.zeros(1, 1, S))
        names = ["h", "temb", "vl", "text_bias", "image_bias"]
        for k, (lo, hi) in enumerate(plans["dit"]):
            m = DitChunk(policy, lo, hi, c).eval()
            if m.first:
                args = (torch.zeros(1, c.action_horizon, c.action_dim), torch.zeros(1, 256),
                        torch.zeros(1, 1, hd), torch.zeros(1, 1, hd), *dit_args[2:])
                add(f"dit_{k}", m, args,
                    ["actions", "t_proj", "tau", "state_features", "vl", "text_bias", "image_bias"],
                    ["h_out", "temb"])
            elif m.last:
                add(f"dit_{k}", m, (*dit_args, torch.zeros(1, c.action_horizon, c.action_dim)),
                    names + ["actions"], ["actions_next"])
            else:
                add(f"dit_{k}", m, dit_args, names, ["h_out"])

    emb = eagle.language_model.get_input_embeddings().weight.detach().to(torch.float16).numpy()
    np.save(fp32 / "embed_tokens.npy", emb)

    pad_id = eagle.config.text_config.eos_token_id
    ids_p = np.concatenate([ids[0], np.full(S - n_real, pad_id)]).astype(np.int64)
    bundle = {
        "schema": "groot-n1.6-split/1",
        "model": "nvidia/GR00T-N1.6-3B",
        "checkpoint": meta["checkpoint"],
        "embodiment": c.embodiment, "embodiment_id": c.embodiment_id,
        "views": V, "image_hw": list(c.image_hw), "tokens_per_image": c.tokens_per_image,
        "image_mean": 0.5, "image_std": 0.5,
        "seq_len": S, "prompt_tokens": n_real, "image_token": tok, "pad_token": pad_id,
        "task": meta["task"], "input_ids": ids_p.tolist(),
        "mask_neg": MASK_NEG,
        "action_horizon": c.action_horizon, "action_dim": c.action_dim, "state_dim": c.state_dim,
        "timesteps": c.timesteps(),
        "embed_tokens": "embed_tokens.npy",
        "shortest_image_edge": 256, "crop_fraction": 0.95,
        # Stock-PyTorch FP32 outputs for seeded inputs (reference.py); the runtime runs
        # the engines on the same inputs at load and fails closed on a mismatch.
        "fixture": {"file": "fixture.npz", "source": "stock PyTorch float32"},
        "chunks": {k: [list(p) for p in v] for k, v in plans.items()},
        "graphs": graphs,
    }
    (fp32 / "bundle.json").write_text(json.dumps(bundle, indent=2))

    print(f"\nmixed FP16 -> {a.out}  (FP32: {list(SENSITIVE)} + every RMSNorm node)")
    a.out.mkdir(parents=True, exist_ok=True)
    for g in graphs:
        src, dst = fp32 / g["file"], a.out / g["file"]
        if g["name"] == "time":      # the sinusoids: FP32 end to end (groot_split.TimeEmb)
            shutil.copy2(src, dst)
        else:
            to_mixed_fp16(src, dst)
        g["size_mb"] = round(dst.stat().st_size / 1e6, 1)
    (a.out / "bundle.json").write_text(json.dumps(bundle, indent=2))
    print(f"  {sum(g['size_mb'] for g in graphs):.0f} MB of graphs")
    for g in graphs:
        n = dedupe_casts(a.out / g["file"])
        if n:
            print(f"  {g['name']}: dropped {n} duplicate Cast nodes")
    shutil.copy2(fp32 / "embed_tokens.npy", a.out / "embed_tokens.npy")
    shutil.copy2(a.ref, a.out / "fixture.npz")
    write_manifest(a.out)
    if not a.keep_fp32:
        shutil.rmtree(fp32)


if __name__ == "__main__":
    main()
