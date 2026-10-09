# Vendored from the author's pi0.5 Orin Nano prototype (spark-projects): pi05-spark-inference/prototype/pi05_fp16_full_20261010/prepare_full.py.
# Runs in the openpi container (export/pi05/README.md); paths are that stage layout's.
"""Prepare full FP16 π0.5 LIBERO split inference and deterministic references.

All 27 vision, 18 language and 18 action blocks are retained. Export only
inference weights; embeddings are a mapped sidecar and conditioning is fixed.
"""
import dataclasses
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import onnx
import safetensors.torch as st
import torch
from torch import nn

from export_utils import ROOT, accumulation, pack, sha

INITIAL = ROOT.parent / 'orin_initial_20261010_001'
sys.path.insert(0, str(INITIAL))
from prepare_blocks import GemmaBlock, VisionBlock, array

CHECKPOINT = Path('/workspace/checkpoints/pi05_libero_pytorch')
IMAGE_KEYS = ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']


class VisionStem(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    def forward(self, image):
        return self.embeddings(image)


class VisionTail(nn.Module):
    def __init__(self, norm, projector):
        super().__init__()
        self.norm, self.projector = norm, projector

    def forward(self, hidden):
        return self.projector(self.norm(hidden))


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


def metric(actual, expected):
    a, b = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    return {'finite': finite, 'cosine': float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)) if finite else None,
            'max_abs': float(np.max(np.abs(a - b))) if finite else None,
            'mean_abs': float(np.mean(np.abs(a - b))) if finite else None}


def main():
    from openpi.training import config as cfg
    from openpi.models_pytorch import pi0_pytorch
    from openpi import transforms
    from openpi.shared import normalize

    (ROOT / 'fixtures').mkdir(exist_ok=True)
    train = cfg.get_config('pi05_libero')
    config = dataclasses.replace(train.model, pytorch_compile_mode=None)
    print('Constructing and loading full policy on Spark', flush=True)
    model = pi0_pytorch.PI0Pytorch(config)
    missing, unexpected = st.load_model(model, str(CHECKPOINT / 'model.safetensors'), strict=False)
    assert not unexpected and all('embed_tokens' in k for k in missing)
    core = model.paligemma_with_expert.paligemma.model
    language, vision = core.language_model, core.vision_tower.vision_model
    expert = model.paligemma_with_expert.gemma_expert.model
    rope_frequencies = [m.rotary_emb.inv_freq.clone().float() for m in (language, expert)]
    model.half().eval().cuda()
    for module, frequencies in zip((language, expert), rope_frequencies):
        module.rotary_emb.inv_freq = frequencies.cuda()
        module.rotary_emb.original_inv_freq = frequencies.cuda()
        module.config._attn_implementation = 'eager'
    vision.config._attn_implementation = 'eager'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    data = train.data.create(train.assets_dirs, config)
    stats = normalize.load(CHECKPOINT / 'assets' / data.asset_id)
    output_transform = transforms.compose([*data.model_transforms.outputs,
        transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm), *data.data_transforms.outputs])
    print('Output transforms', data.data_transforms.outputs, data.use_quantile_norm, flush=True)

    stem = VisionStem(vision.embeddings)
    vision_blocks = [VisionBlock(layer) for layer in vision.encoder.layers]
    tail = VisionTail(vision.post_layernorm, core.multi_modal_projector)
    language_blocks = [GemmaBlock(layer) for layer in language.layers]
    action_input = ActionInput(model.action_in_proj)
    action_blocks = [ActionBlock(layer) for layer in expert.layers]
    action_output = ActionOutput(expert.norm, model.action_out_proj)
    styles = torch.from_numpy(np.load(INITIAL / 'conditioning_10step_fp16.npy')).cuda()
    schedule = json.loads((INITIAL / 'conditioning_manifest.json').read_text())
    times = torch.tensor(schedule['timesteps_fp32'], dtype=torch.float32, device='cuda')
    dt = torch.tensor(-0.1, dtype=torch.float32, device='cuda')
    assert len(vision_blocks) == 27 and len(language_blocks) == len(action_blocks) == 18

    with torch.inference_mode():
        positions = torch.arange(-1, 2048, device='cuda')[None]
        dummy = torch.empty((1, len(positions[0]), 2048), dtype=torch.float16, device='cuda')
        rope_cos, rope_sin = language.rotary_emb(dummy, positions)
        np.save(ROOT / 'rope_cos_fp16.npy', array(rope_cos[0]))
        np.save(ROOT / 'rope_sin_fp16.npy', array(rope_sin[0]))
    del dummy
    sample_inputs = {}
    comparisons = []

    def run(fixture, live=False, capture=False):
        prefix_parts, pads, features = [], [], []
        with torch.inference_mode():
            for name in IMAGE_KEYS:
                image = torch.from_numpy(fixture['image_' + name]).cuda().half()
                if image.shape[1] != 3:
                    image = image.permute(0, 3, 1, 2)
                if capture and 'stem' not in sample_inputs:
                    sample_inputs['stem'] = (image,)
                hidden = stem(image)
                for i, block in enumerate(vision_blocks):
                    if capture:
                        sample_inputs[f'vision_{i:02d}'] = (hidden.clone(),)
                    hidden = block(hidden)
                if capture:
                    sample_inputs['vision_tail'] = (hidden.clone(),)
                projected = tail(hidden)
                if capture:
                    direct = core.get_image_features(image)
                    assert torch.equal(projected, direct), 'Split vision differs from full FP16 tower'
                features.append(array(projected))
                prefix_parts.append(projected)
                valid = bool(fixture['image_mask_' + name].item())
                pads.append(torch.full((1, 256), valid, dtype=torch.bool, device='cuda'))
            tokens = torch.from_numpy(fixture['tokens'].astype(np.int64)).cuda()
            embeddings = language.embed_tokens(tokens) * (2048**0.5)
            prefix_parts.append(embeddings)
            pads.append(torch.from_numpy(fixture['token_mask']).cuda())
            hidden, pad = torch.cat(prefix_parts, dim=1), torch.cat(pads, dim=1)
            mask = torch.where(pad[:, None, :, None] & pad[:, None, None, :], 0.0, -2.3819763e38)
            pos = torch.cumsum(pad, dim=-1) - 1
            cos, sin = rope_cos[:, pos[0] + 1], rope_sin[:, pos[0] + 1]
            kv = []
            for i, block in enumerate(language_blocks):
                if capture:
                    sample_inputs[f'language_{i:02d}'] = (hidden.clone(), mask, cos, sin)
                hidden, k, v = block(hidden, mask, cos, sin)
                assert torch.isfinite(hidden).all(), ('Nonfinite FP16 language block', i)
                kv.append((k, v))
            suffix_pos = pad.sum(-1)[:, None] + torch.arange(10, device='cuda')[None]
            suffix_cos, suffix_sin = rope_cos[:, suffix_pos[0] + 1], rope_sin[:, suffix_pos[0] + 1]
            suffix_valid = torch.cat((pad[:, None, :].expand(1, 10, -1),
                                      torch.ones((1, 10, 10), dtype=torch.bool, device='cuda')), dim=-1)
            suffix_mask = torch.where(suffix_valid[:, None], 0.0, -2.3819763e38)
            actions = torch.from_numpy(fixture['noise']).cuda()[None]
            velocity_trace, action_trace = [], [array(actions)]
            for step in range(10):
                if capture and step == 0:
                    sample_inputs['action_input'] = (actions.clone(),)
                hidden = action_input(actions)
                if live:
                    from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
                    import torch.nn.functional as F
                    t_emb = create_sinusoidal_pos_embedding(times[step:step+1], 1024, 4e-3, 4.0, device=times.device).float().half()
                    cond = F.silu(model.time_mlp_out(F.silu(model.time_mlp_in(t_emb))))
                for i, block in enumerate(action_blocks):
                    modulation = styles[step, 2*i:2*i+2]
                    mod_in, mod_post = modulation[0][None, None], modulation[1][None, None]
                    if live:
                        mod_in = expert.layers[i].input_layernorm.dense(cond)[:, None]
                        mod_post = expert.layers[i].post_attention_layernorm.dense(cond)[:, None]
                    args = (hidden, suffix_mask, suffix_cos, suffix_sin, *kv[i], mod_in, mod_post)
                    if capture and step == 0:
                        sample_inputs[f'action_{i:02d}'] = tuple(x.clone() for x in args)
                    hidden = block(*args)
                    assert torch.isfinite(hidden).all(), ('Nonfinite FP16 action block', step, i)
                mod_final = styles[step, 36][None, None]
                if live:
                    mod_final = expert.norm.dense(cond)[:, None]
                if capture and step == 0:
                    sample_inputs['action_output'] = (hidden.clone(), mod_final)
                velocity = action_output(hidden, mod_final)
                assert torch.isfinite(velocity).all(), ('Nonfinite velocity', step)
                actions = actions + dt * velocity
                velocity_trace.append(array(velocity))
                action_trace.append(array(actions))
            output = output_transform({'actions': array(actions)[0], 'state': fixture['state'][0].copy()})
            return {'actions_normalized': array(actions), 'actions_robot': output['actions'],
                    'vision_features': np.stack(features), 'prefix_kv': np.stack([[array(k), array(v)] for k, v in kv]),
                    'velocities': np.stack(velocity_trace), 'action_steps': np.stack(action_trace)}

    for i in range(5):
        with np.load(INITIAL / f'policy_fixture_{i}.npz') as loaded:
            fixture = {k: loaded[k] for k in loaded.files}
        cached = run(fixture, capture=(i == 0))
        live = run(fixture, live=True)
        assert np.array_equal(cached['actions_normalized'], live['actions_normalized']), 'Full cached and live FP16 actions differ'
        comparison = {'fixture': i, 'cached_vs_live_bit_equal': True,
            'fp16_vs_mixed_normalized': metric(cached['actions_normalized'], fixture['actions']),
            'fp16_vs_mixed_robot': metric(cached['actions_robot'], fixture['actions_robot'])}
        comparisons.append(comparison)
        np.savez_compressed(ROOT / 'fixtures' / f'fixture_{i}.npz', **fixture,
                            **{'fp16_' + k: v for k, v in cached.items()})
        print('Full FP16 fixture', json.dumps(comparison), flush=True)

    np.save(ROOT / 'embedding_fp16.npy', array(language.embed_tokens.weight))
    shutil.copyfile(INITIAL / 'conditioning_10step_fp16.npy', ROOT / 'conditioning_10step_fp16.npy')
    shutil.copyfile(INITIAL / 'conditioning_manifest.json', ROOT / 'conditioning_manifest.json')
    shutil.copyfile(CHECKPOINT / 'assets' / data.asset_id / 'norm_stats.json', ROOT / 'norm_stats.json')
    # This policy's output transform is captured explicitly for the lightweight target.
    output_names = [type(t).__name__ for t in data.data_transforms.outputs]
    assert output_names == ['LiberoOutputs'], ('Additional data outputs need implementing', output_names)
    bundle = {'checkpoint_sha256': json.loads((INITIAL / 'manifest.json').read_text())['checkpoint_sha256'],
              'policy': 'pi05_libero', 'image_keys': IMAGE_KEYS, 'valid_cameras': 2, 'image_slots': 3,
              'prefix_tokens': 968, 'prompt_capacity': 200, 'action_horizon': 10, 'action_dim': 32,
              'robot_action_dim': 7, 'num_steps': 10, 'vision_layers': 27, 'language_layers': 18,
              'action_layers': 18, 'dtype': 'FP16 weights and hidden states, FP32 normalization/accumulation/Euler',
              'use_quantiles': data.use_quantile_norm, 'reference_comparisons': comparisons,
              'embedding_sha256': sha(ROOT / 'embedding_fp16.npy'), 'components': []}
    groups = [('stem', stem, 'stem', ['image'], ['hidden']),
              *[(f'vision_{i:02d}', block, 'vision', ['hidden'], ['output']) for i, block in enumerate(vision_blocks)],
              ('vision_tail', tail, 'vision_tail', ['hidden'], ['output']),
              *[(f'language_{i:02d}', block, 'language', ['hidden','mask','cos','sin'], ['output','key','value'])
                for i, block in enumerate(language_blocks)],
              ('action_input', action_input, 'action_input', ['actions'], ['hidden']),
              *[(f'action_{i:02d}', block, 'action', ['hidden','mask','cos','sin','prefix_k','prefix_v','mod_in','mod_post'], ['output'])
                for i, block in enumerate(action_blocks)],
              ('action_output', action_output, 'action_output', ['hidden','mod_final'], ['velocity'])]
    seen = set()
    temporary = ROOT / 'temporary_export.onnx'
    for label, module, kind, names, outputs in groups:
        args = sample_inputs[label]
        with torch.no_grad():
            torch.onnx.export(module, args, temporary, dynamo=False, opset_version=17,
                              input_names=names, output_names=outputs, do_constant_folding=True)
        graph = onnx.load(temporary)
        changed = accumulation(graph)
        record = pack(graph, kind, label, template=kind not in seen)
        seen.add(kind)
        bundle['components'].append({'label': label, 'kind': kind, 'inputs': names, 'outputs': outputs,
                                     'raw_weights_bytes': record['initializer_bytes'], 'accumulation_ops': changed,
                                     'structure_sha256': record['structure_sha256']})
        (ROOT / 'bundle.json').write_text(json.dumps(bundle, indent=2))
        del graph
    temporary.unlink()
    bundle['raw_compute_weights_bytes'] = sum(c['raw_weights_bytes'] for c in bundle['components'])
    bundle['complete'] = True
    (ROOT / 'bundle.json').write_text(json.dumps(bundle, indent=2))
    print('Complete full split export; raw compute GiB', bundle['raw_compute_weights_bytes'] / 1024**3, flush=True)


if __name__ == '__main__':
    main()
