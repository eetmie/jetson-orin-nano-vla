# Vendored from the author's fine-tuning pipeline: vla-onnx/smolvla/export_split_onnx.py @ 54be8cd (robot-specific comments trimmed).
"""Split-graph SmolVLA export for the Orin Nano 8 GB (per-component TRT engines).

WHY (see notes/orin-split-findings.md): the monolithic
`sample_actions` export (export_valid_onnx.py) CANNOT TRT-build on the 8 GB Orin
Nano — TRT imports all 450M weights as FP32 working copies at once (~6 GB floor,
node-count-independent), so the build OOMs regardless of FP16 / --num-steps.

The validated fix is to split the model into per-component graphs, each carrying
only its weight slice (so each builds in ≤60 s), and run the flow-matching denoise
loop in Python (prefill ×1 -> decode ×N). This mirrors HF `ainekko/smolvla_base_onnx`
+ github.com/aifoundry-org/ETARS, adapted to our lerobot 0.5.1 weights/config.

Graphs written (9), single RGB image to match the Orin runtime:
  smolvlm_vision      image[1,3,512,512]            -> img_embeds[1,64,960]
  smolvlm_text        tokens[1,T]                   -> lang_embeds[1,T,960]   (dynamic T)
  smolvlm_expert_prefill  (mask,pos,vlm_embeds)     -> 32 KV tensors (fill_kv_cache)
  smolvlm_expert_decode   (mask,pos,expert_embeds,*KV) -> expert_out[1,50,720]
  state_projector     state[1,32]                   -> [1,960]
  action_in_projector action[1,50,32]               -> [1,50,720]
  action_out_projector expert_out[1,50,720]         -> v_t[1,50,32]
  time_in_projector   action_time[1,50,1440]        -> [1,50,H]
  time_out_projector  [1,50,H]                       -> [1,50,720]

The monolithic export stays as the FP32 parity gold (run parity on a big box).
Deploy bundle = these 9 graphs + tokenizer/ + normalization stats; the Orin builds
one FP16 TRT engine per heavy graph and runs the loop in backends/ort.py.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

from vla_common.bundle import write_manifest

import torch
from torch import nn

# Reuse the legacy-ONNX-export monkeypatches the monolith needed (torch.where for
# boolean-mask NonZero, etc.) — the expert/vision graphs hit the same paths.
from export_valid_onnx import patch_smolvla_for_legacy_onnx_export

OPSET = 17


# --- component wrappers (mirror ETARS smolVLA_export.ipynb) -------------------
class VisionWrap(nn.Module):
    def __init__(self, vlme):
        super().__init__()
        self.vlme = vlme

    def forward(self, image):
        return self.vlme.embed_image(image)


class TextWrap(nn.Module):
    def __init__(self, vlme):
        super().__init__()
        self.vlme = vlme

    def forward(self, tokens):
        return self.vlme.embed_language_tokens(tokens)


class _Projected(nn.Module):
    """Stands in for a hoisted k_proj/v_proj: identity, but still answers `.weight.dtype`
    (the expert reads it to pick the cast) without registering the weight."""

    def __init__(self, linear):
        super().__init__()
        self.__dict__["weight"] = linear.weight

    def forward(self, x):
        return x


def cross_layers(vlme) -> list[int]:
    """Layers whose expert cross-attends to the prefix cache (the odd ones in smolvla_base).

    Mirrors the dispatch in SmolVLMWithExpertModel.forward for the decode case."""
    if "cross" not in vlme.attention_mode:
        return []
    n = vlme.self_attn_every_n_layers
    return [i for i in range(vlme.num_vlm_layers) if not (n > 0 and i % n == 0)]


class PrefillWrap(nn.Module):
    """Prefix (image+text+state embeds) -> the VLM KV cache only.

    sample_actions discards the prefill's hidden output and keeps just the cache,
    so we emit the 32 KV tensors (16 layers x key/value) and nothing else.

    hoist_cross_kv: a cross-attention layer's expert projects the fixed prefix K/V
    through its own k_proj/v_proj on every denoise step; emit those projections
    here instead, once per observation. Same shape [1,prefix,kv_heads,head_dim].
    """

    def __init__(self, vlme, hoist_cross_kv: bool = False):
        super().__init__()
        self.vlme = vlme
        self.num_vlm_layers = vlme.num_vlm_layers
        self.cross = set(cross_layers(vlme)) if hoist_cross_kv else set()

    def forward(self, attention_mask, position_ids, vlm_embeds):
        _, new_kv = self.vlme.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[vlm_embeds, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        flat = []
        for i in range(self.num_vlm_layers):
            k, v = new_kv[i]["key_states"], new_kv[i]["value_states"]
            if i in self.cross:
                attn = self.vlme.lm_expert.layers[i].self_attn
                k = attn.k_proj(k.to(attn.k_proj.weight.dtype).reshape(*k.shape[:2], -1)).view(
                    *k.shape[:2], -1, attn.head_dim)
                v = attn.v_proj(v.to(attn.v_proj.weight.dtype).reshape(*v.shape[:2], -1)).view(
                    *v.shape[:2], -1, attn.head_dim)
            flat.append(k)
            flat.append(v)
        return tuple(flat)


class DecodeWrap(nn.Module):
    """One flow-matching step: (suffix embeds + prefill KV) -> expert hidden.

    The denoise loop reuses the prefill KV every step (decode does not update it),
    so we take the cache as input and emit only the expert hidden output.
    """

    def __init__(self, vlme, hoist_cross_kv: bool = False):
        super().__init__()
        self.vlme = vlme
        self.num_vlm_layers = vlme.num_vlm_layers
        self.cross = cross_layers(vlme) if hoist_cross_kv else []

    def forward(self, attention_mask, position_ids, expert_embeds, *past_kv_flat):
        past = {
            i // 2: {
                "key_states": past_kv_flat[i],
                "value_states": past_kv_flat[i + 1],
            }
            for i in range(0, len(past_kv_flat), 2)
        }
        # Hoisted: the cross layers' cache already holds k_proj/v_proj outputs.
        swapped = []
        for i in self.cross:
            attn = self.vlme.lm_expert.layers[i].self_attn
            swapped.append((attn, attn.k_proj, attn.v_proj))
            attn.k_proj, attn.v_proj = _Projected(attn.k_proj), _Projected(attn.v_proj)
        try:
            embeds, _ = self.vlme.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past,
                inputs_embeds=[None, expert_embeds],
                use_cache=True,
                fill_kv_cache=False,
            )
        finally:
            for attn, k, v in swapped:
                attn.k_proj, attn.v_proj = k, v
        return embeds[1]  # expert hidden [1, 50, 720]


# --- vision Gather-indices int64 patch (ETARS) -------------------------------
def patch_gather_indices_int64(onnx_path: str) -> None:
    """Cast non-int Gather indices to int64 in place (TRT/ORT reject float idx)."""
    import onnx
    from onnx import TensorProto, helper, shape_inference

    m = onnx.load(onnx_path)
    try:
        m = shape_inference.infer_shapes(m)
    except Exception:
        pass
    g = m.graph
    int_ok = {TensorProto.INT64, TensorProto.INT32}
    vtype = {}
    for vi in list(g.input) + list(g.value_info) + list(g.output):
        if vi.type.tensor_type and vi.type.tensor_type.elem_type:
            vtype[vi.name] = vi.type.tensor_type.elem_type
    for init in g.initializer:
        vtype[init.name] = init.data_type
    consumers: dict[str, list[int]] = {}
    for i, n in enumerate(g.node):
        for inp in n.input:
            consumers.setdefault(inp, []).append(i)
    casted: dict[str, str] = {}
    changed = 0
    for n in list(g.node):
        if n.op_type in {"Gather", "GatherND", "GatherElements"} and len(n.input) > 1:
            idx = n.input[1]
            if vtype.get(idx) in int_ok:
                continue
            if idx in casted:
                n.input[1] = casted[idx]
                continue
            out = idx + "_idx_i64"
            cast = helper.make_node("Cast", [idx], [out], to=TensorProto.INT64,
                                    name=f"{idx}_to_i64")
            g.node.insert(min(consumers.get(idx, [0])), cast)
            casted[idx] = out
            vtype[out] = TensorProto.INT64
            n.input[1] = out
            changed += 1
    onnx.checker.check_model(m)
    # Re-save with external weights (the torch.onnx export already wrote a sibling
    # .onnx.data; keep the bundle format consistent — small graph + one weights file).
    data_name = Path(onnx_path).name + ".data"
    for stale in (onnx_path, onnx_path + ".data"):
        if Path(stale).exists():
            Path(stale).unlink()
    onnx.save(m, onnx_path, save_as_external_data=True,
              all_tensors_to_one_file=True, location=data_name, size_threshold=1024)
    print(f"  vision Gather int64 patch: cast {changed} indices (re-saved external)")


def _export(model, args_tuple, path, input_names, output_names, dynamic_axes=None):
    model.eval()
    torch.onnx.export(
        model, args_tuple, path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=OPSET, do_constant_folding=False,
    )
    import onnx
    onnx.checker.check_model(path)
    print(f"  wrote {path}  ({Path(path).stat().st_size / 1e6:.1f} MB)")


# --- deploy metadata: stats.json, export_info.json, MANIFEST.sha256 -----------
#
# These three files were hand-assembled per deploy until now, which is why the
# newer bundles silently lost them. A bundle without stats.json still LOADS —
# both the Orin runtime and the benchmark fall back to identity normalization —
# and then drives the machine with unnormalized actions. Writing them here, from
# the checkpoint that was just exported, is the only way the bundle cannot
# disagree with its own weights.


def _resolve_ckpt_dir(model_id: str) -> Path | None:
    """The local checkpoint dir, or the HF snapshot the id resolves to."""
    p = Path(model_id).expanduser()
    if p.is_dir():
        return p
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(model_id))
    except Exception:
        return None


def _read_norm_stats(ckpt_dir: Path) -> dict | None:
    """The checkpoint's own normalizer -> lerobot `stats.json` shape.

    The normalizer safetensors is the authority here, NOT the training dataset's
    meta/stats.json: they are usually identical, but a bundle carrying stats from a
    different run loads fine and produces wrong motion, so the stats have to come
    from the same artefact as the weights. Keys are flat `<feature>.<stat>`;
    stats.json nests them and keeps the tensor shape (image stats stay [3,1,1]).
    """
    hits = sorted(ckpt_dir.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    if not hits:
        return None
    from safetensors.torch import load_file

    stats: dict[str, dict] = {}
    for key, tensor in load_file(str(hits[0])).items():
        feature, stat = key.rsplit(".", 1)
        stats.setdefault(feature, {})[stat] = tensor.float().tolist()
    return stats


def _dataset_meta(ckpt_dir: Path) -> tuple[int | None, list[str]]:
    """(fps, tasks) from the dataset this checkpoint was trained on, if reachable.

    `train_config.json` records the dataset root as an absolute path, so this works
    for a local fine-tune and quietly returns (None, []) for a Hub base model that
    was never trained here.

    EVERY task, not the first one. This used to take `iloc[0]` with the comment "one
    task per dataset here", which stopped being true with the first two-task dataset. Exporting a multi-task checkpoint through that would have shipped
    a bundle claiming to know only "move sand to container": half the fine-tune
    unreachable, the D-pad with nothing to cycle, and no error anywhere -- the runtime
    cannot tell a genuinely single-task bundle from a truncated one.
    """
    cfg_f = ckpt_dir / "train_config.json"
    if not cfg_f.exists():
        return None, []
    try:
        root = json.loads(cfg_f.read_text()).get("dataset", {}).get("root")
    except Exception:
        return None, []
    if not root or not (meta := Path(root) / "meta").is_dir():
        return None, []

    fps = None
    if (info := meta / "info.json").exists():
        try:
            fps = json.loads(info.read_text()).get("fps")
        except Exception:
            pass

    tasks: list[str] = []
    if (tasks_f := meta / "tasks.parquet").exists():
        try:
            import pandas as pd
            col = pd.read_parquet(tasks_f)
            # Order by task_index, which is the integer the dataset's own `task_index`
            # column points at -- and the order the runtime's D-pad cycles in. Row
            # order has matched it in every dataset written so far; sorting states it
            # instead of trusting it. LeRobot v3 puts the string in the INDEX and
            # task_index in a column, but older writers did the reverse, so both.
            if "task_index" in col.columns:
                col = col.sort_values("task_index")
            tasks = ([str(t) for t in col["task"]] if "task" in col.columns
                     else [str(i) for i in col.index])
        except Exception:
            pass
    return fps, tasks


def _feature_dims(stats: dict | None) -> tuple[int | None, int | None, list[str]]:
    """Real (state_dim, action_dim, camera keys) as the checkpoint was trained.

    The graphs are exported at the policy's PADDED width (32/32); the robot's real
    widths live only in the stats. Recording both is what lets a consumer slice the
    padding off without being told the numbers out of band.
    """
    if not stats:
        return None, None, []
    def width(key):
        v = stats.get(key, {}).get("mean")
        return len(v) if isinstance(v, list) else None
    cams = sorted(k for k in stats if k.startswith("observation.images."))
    return width("observation.state"), width("action"), cams


def _git_sha(path: Path) -> str | None:
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="lerobot/smolvla_base")
    ap.add_argument("--out-dir", default="exports-split")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fps", type=int, default=None,
                    help="control rate the actions were authored at. Auto-resolved from "
                         "the training dataset; required if that is not reachable, because "
                         "the Orin runtime otherwise has to guess it.")
    ap.add_argument("--task", action="append", dest="tasks", default=None,
                    help="the instruction string, letter for letter as the dataset "
                         "records it. Auto-resolved from the training dataset, including "
                         "every task of a multi-task one. Repeatable: pass it once per "
                         "instruction to override the resolved list, first one first "
                         "(that is the order the runtime's D-pad cycles in).")
    ap.add_argument("--views", "--cam-slots", type=int, default=1, dest="cam_slots",
                    help="how many camera views the bundle is sized for. (--cam-slots is "
                         "the old name and still works.) The vision graph stays batch-1 "
                         "(the runtime calls it once per real camera), but prefill/decode "
                         "bake in a STATIC prefix length -- 1 view = 113 tokens, 2 = 177 -- "
                         "so a bundle exported at 1 cannot serve a 2-view runtime. The "
                         "public ainekko/smolvla_base_onnx export is 2.")
    ap.add_argument("--hoist-cross-kv", action="store_true",
                    help="emit the cross-attention layers' expert K/V projections from "
                         "prefill (once per observation) instead of recomputing them in "
                         "decode every step. Changes what the present_/past_ tensors of "
                         "those layers hold; a runtime that passes prefill outputs to decode "
                         "inputs by name is unaffected (export.sh also keeps the cache FP16).")
    ap.add_argument("--state-blind", action="store_true",
                    help="camera-only checkpoint: the state input is dead but still wired "
                         "in, and MUST be fed zeros. run_inference reads this flag.")
    args = ap.parse_args()

    patch_smolvla_for_legacy_onnx_export()
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)

    print(f"Loading {args.model_id} on {dev} ...")
    policy = SmolVLAPolicy.from_pretrained(args.model_id).eval().float().to(dev)
    for p in policy.parameters():
        p.requires_grad_(False)
    m = policy.model
    vlme = m.vlm_with_expert
    cfg = policy.config

    L = vlme.num_vlm_layers
    vlm_dim = vlme.config.text_config.hidden_size            # 960
    exp_dim = vlme.expert_hidden_size                        # 720
    chunk = cfg.chunk_size                                   # 50
    act_dim = cfg.max_action_dim                             # 32
    state_dim = cfg.max_state_dim                            # 32
    img_h, img_w = cfg.resize_imgs_with_padding

    with torch.no_grad():
        # derive the real prefix length from a single image + padded lang + state
        n_img = vlme.embed_image(torch.zeros(1, 3, img_h, img_w, device=dev)).shape[1]
        lang_len = cfg.tokenizer_max_length
        # One image-token block per SLOT, not per real camera: an unused slot still
        # occupies its tokens in the prefix, and prefill/decode are traced at a static
        # length. 1 slot -> 64+48+1 = 113; 2 slots -> 128+48+1 = 177, which is what the
        # published ainekko export carries.
        prefix_len = args.cam_slots * n_img + lang_len + 1
        print(f"dims: L={L} vlm_dim={vlm_dim} exp_dim={exp_dim} prefix_len={prefix_len} "
              f"(cam_slots={args.cam_slots} x {n_img} img + {lang_len} lang + 1 state) "
              f"chunk={chunk} act_dim={act_dim}")

        # 1) vision
        _export(VisionWrap(vlme), torch.zeros(1, 3, img_h, img_w, device=dev),
                str(out / "smolvlm_vision.onnx"), ["image"], ["img_embeds"])
        patch_gather_indices_int64(str(out / "smolvlm_vision.onnx"))

        # 2) text (dynamic sequence length)
        _export(TextWrap(vlme), torch.ones(1, lang_len, dtype=torch.long, device=dev),
                str(out / "smolvlm_text.onnx"), ["tokens"], ["lang_embeds"],
                dynamic_axes={"tokens": {1: "T"}, "lang_embeds": {1: "T"}})

        # 3) expert prefill -> KV cache (static prefix_len)
        kv_names = [f"{k}_{i}" for i in range(L) for k in ("present_key", "present_value")]
        pf_mask = torch.ones(1, prefix_len, prefix_len, dtype=torch.bool, device=dev)
        pf_pos = torch.arange(prefix_len, device=dev).unsqueeze(0)
        pf_emb = torch.randn(1, prefix_len, vlm_dim, device=dev)
        hoist = args.hoist_cross_kv
        if hoist:
            assert len(vlme.lm_expert.layers) == L, "hoisting assumes one expert layer per VLM layer"
        _export(PrefillWrap(vlme, hoist), (pf_mask, pf_pos, pf_emb),
                str(out / "smolvlm_expert_prefill.onnx"),
                ["attention_mask", "position_ids", "vlm_embeds"], kv_names)

        # grab the real KV shapes from a prefill run to build the decode dummies
        past_kv = PrefillWrap(vlme, hoist).forward(pf_mask, pf_pos, pf_emb)
        print(f"  KV[0] shape: {tuple(past_kv[0].shape)}  ({len(past_kv)} tensors)")

        # 4) expert decode (suffix attends to prefix KV + itself); total = prefix+chunk
        total = prefix_len + chunk
        dc_mask = torch.ones(1, chunk, total, dtype=torch.bool, device=dev)
        dc_pos = torch.arange(chunk, device=dev).unsqueeze(0)
        dc_emb = torch.randn(1, chunk, exp_dim, device=dev)
        in_names = ["attention_mask", "position_ids", "expert_embeds"] + \
                   [f"{k}_{i}" for i in range(L) for k in ("past_key", "past_value")]
        _export(DecodeWrap(vlme, hoist), (dc_mask, dc_pos, dc_emb, *past_kv),
                str(out / "smolvlm_expert_decode.onnx"), in_names, ["expert_out"])
        if hoist:
            # The split must still be the stock arithmetic: hoisted prefill+decode
            # against the original pair, in FP32, on the export dummies.
            ref = DecodeWrap(vlme).forward(dc_mask, dc_pos, dc_emb,
                                           *PrefillWrap(vlme).forward(pf_mask, pf_pos, pf_emb))
            got = DecodeWrap(vlme, True).forward(dc_mask, dc_pos, dc_emb, *past_kv)
            err = (got - ref).abs().max().item()
            print(f"  hoisted cross K/V: {len(cross_layers(vlme))} layers, decode max |diff| "
                  f"vs per-step projection {err:.3e}")
            assert err < 1e-4 * max(1.0, ref.abs().max().item()), err

        # 5-9) projectors (small, run-once to get shapes)
        def proj(mod, dummy, name, in_name="input"):
            o = mod(dummy)
            _export(mod, dummy, str(out / name), [in_name], ["output"])
            return o

        proj(m.state_proj, torch.zeros(1, state_dim, device=dev),
             "state_projector.onnx", "state")
        proj(m.action_in_proj, torch.zeros(1, chunk, act_dim, device=dev),
             "action_in_projector.onnx", "action")
        proj(m.action_out_proj, torch.zeros(1, chunk, exp_dim, device=dev),
             "action_out_projector.onnx", "expert_out")
        # time MLPs: in takes concat[action_emb, time_emb] = 2*exp_dim
        ti_out = proj(m.action_time_mlp_in, torch.zeros(1, chunk, 2 * exp_dim, device=dev),
                      "time_in_projector.onnx", "action_time")
        proj(m.action_time_mlp_out, torch.zeros(1, chunk, ti_out.shape[-1], device=dev),
             "time_out_projector.onnx", "hidden")

    # deploy bundle: tokenizer + normalization stats + provenance
    try:
        vlme.processor.tokenizer.save_pretrained(str(out / "tokenizer"))
        print(f"Saved tokenizer -> {out / 'tokenizer'}")
    except Exception as e:
        print(f"tokenizer save skipped: {e}")

    ckpt_dir = _resolve_ckpt_dir(args.model_id)
    stats = _read_norm_stats(ckpt_dir) if ckpt_dir else None
    if stats:
        (out / "stats.json").write_text(json.dumps(stats, indent=2))
        # The runtime reads exactly two keys. A base checkpoint carries its pretraining
        # buffer stats instead (`so100.buffer.action` and friends), which look like
        # stats, satisfy `if stats:`, and normalize nothing — so count the keys that
        # actually get used rather than the keys that happen to be present.
        usable = [k for k in ("observation.state", "action") if k in stats]
        print(f"Saved stats.json -> {len(stats)} features, "
              f"{len(usable)}/2 the runtime can use ({', '.join(usable) or 'none'})")
        if len(usable) < 2:
            print("!! stats.json carries NO usable normalization for this robot.\n"
                  "   The runtime falls back to IDENTITY normalization.\n"
                  "   Expected for base weights; never ship this to a robot.")
    else:
        print("!! NO stats.json — the checkpoint carries no normalizer safetensors.\n"
              "   The bundle will load with IDENTITY normalization and drive wrong.\n"
              "   Fine for a base-weight latency benchmark; never ship it to a robot.")

    ds_fps, ds_tasks = _dataset_meta(ckpt_dir) if ckpt_dir else (None, [])
    fps = args.fps or ds_fps
    tasks = args.tasks or ds_tasks
    state_dim_real, action_dim_real, cameras = _feature_dims(stats)

    info = {
        "model_id": str(args.model_id),
        "exported_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "exporter_sha": _git_sha(Path(__file__).resolve().parent),
        "opset": OPSET,
        "torch": torch.__version__,
        # --- what the runtime must not guess -------------------------------------
        "fps": fps,
        # Both keys, deliberately: a multi-task runtime reads the `tasks` list, an older
        # one still finds the single `task` string it knows how to read.
        "task": tasks[0] if tasks else None,
        "tasks": tasks,
        "state_blind": bool(args.state_blind),
        # --- shapes: padded is what the graphs take, real is what the robot has ---
        "chunk_size": int(chunk),
        "num_steps": int(getattr(cfg, "num_steps", 10)),
        "n_action_steps": int(getattr(cfg, "n_action_steps", chunk)),
        "max_state_dim": int(state_dim),
        "max_action_dim": int(act_dim),
        "state_dim": state_dim_real,
        "action_dim": action_dim_real,
        "cameras": cameras,
        # Slots, not real cameras: an unused slot still occupies its image tokens in
        # the prefix, so a consumer has to pad to the same count or compute a
        # different-length sequence. Derived from the prefix this export actually
        # built (1 here; the published ainekko export is 2, prefix 177).
        "n_cam_slots": int(args.cam_slots),
        "image_size": [int(img_h), int(img_w)],
        "lang_len": int(lang_len),
        "prefix_len": int(prefix_len),
        "vlm_layers": int(L),
        "vlm_dim": int(vlm_dim),
        "expert_dim": int(exp_dim),
        "graphs": sorted(f.name for f in out.glob("*.onnx")),
    }
    if args.hoist_cross_kv:
        # A runtime that ignores this would feed raw VLM K/V to layers expecting
        # projected ones and still produce plausible-looking actions.
        # (The cache's dtype is whatever the graphs say: FP32 as exported, FP16 after
        # `fp16_mixed --half-io`, which export.sh applies with this flag.)
        info["kv"] = {"cross_layers": cross_layers(vlme), "cross": "expert-projected"}
    (out / "export_info.json").write_text(json.dumps(info, indent=2))
    print(f"Saved export_info.json -> fps={fps} tasks={tasks!r} "
          f"state_dim={state_dim_real} action_dim={action_dim_real} chunk={chunk}")

    if fps is None:
        print("!! fps is null — run_inference will fall back to a guess. Pass --fps.")
    if not tasks:
        print("!! tasks is empty — run_inference will refuse to start without --task.")
    elif len(tasks) > 1:
        print(f"   multi-task bundle: {len(tasks)} tasks, the first is {tasks[0]!r}.")

    n = write_manifest(out)
    print(f"Saved MANIFEST.sha256 -> {n} files "
          f"(verify on the Orin with: cd <bundle> && sha256sum -c MANIFEST.sha256)")

    print(f"\nDONE — 9 split graphs in {out}/  (deploy bundle; FP32 gold stays the monolith)")


if __name__ == "__main__":
    main()
