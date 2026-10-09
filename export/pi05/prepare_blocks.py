# Vendored from the author's pi0.5 Orin Nano prototype (spark-projects): pi05-spark-inference/prototype/orin_initial_20261010_001/prepare_blocks.py.
# Runs in the openpi container (export/pi05/README.md); paths are that stage layout's.
"""Freeze LIBERO references and export checkpoint-backed FP16 first-block tests.

This is an isolated compatibility experiment, not a full TensorRT policy export.
Run in the existing pi05-spark container with /workspace mounted.
"""
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx
import safetensors.torch as st
from safetensors import safe_open
import torch
from torch import nn
from transformers.models.gemma.modeling_gemma import (
    GemmaDecoderLayer, apply_rotary_pos_emb, eager_attention_forward,
)
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer

ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path('/workspace/checkpoints/pi05_libero_pytorch')
WEIGHTS = CHECKPOINT / 'model.safetensors'


def array(t):
    return t.detach().cpu().float().numpy() if t.dtype == torch.bfloat16 else t.detach().cpu().numpy()


def capture(kind, records):
    def hook(module, args, kwargs):
        if kind in records:
            return
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        record = {'hidden': array(hidden).astype(np.float16)}
        if kind != 'vision':
            record['mask'] = array(kwargs['attention_mask']).astype(np.float32)
            cos, sin = kwargs['position_embeddings']
            record['cos'], record['sin'] = array(cos).astype(np.float16), array(sin).astype(np.float16)
        if kind == 'action':
            record['cond'] = array(kwargs['adarms_cond']).astype(np.float16)
            k, v = kwargs['past_key_value'][0]
            record['prefix_k'], record['prefix_v'] = array(k).astype(np.float16), array(v).astype(np.float16)
        records[kind] = record
    return hook


class VisionBlock(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden):
        return self.layer(hidden, attention_mask=None)[0]


class GemmaBlock(nn.Module):
    def __init__(self, layer, action=False, cached=False):
        super().__init__()
        self.layer = layer
        self.action, self.cached = action, cached

    def norm(self, hidden, norm, cond, modulation):
        if not self.cached:
            return norm(hidden, cond)
        # Exact FP32 normalization and FP16 rounding boundaries of the local patch.
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
        residual = hidden + y if gate is None else hidden + y * gate
        x, gate = self.norm(residual, layer.post_attention_layernorm, cond, mod_post)
        y = layer.mlp(x)
        output = residual + y if gate is None else residual + y * gate
        return (output,) if self.action else (output, new_k, new_v)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    from openpi.training import config as cfg
    from openpi.models_pytorch import pi0_pytorch
    from openpi.policies import policy as policy_module
    from openpi import transforms
    from openpi.shared import normalize

    torch.manual_seed(3010)
    torch.backends.cuda.matmul.allow_tf32 = False
    train = cfg.get_config('pi05_libero')
    config = dataclasses.replace(train.model, pytorch_compile_mode=None)
    print('Loading existing mixed BF16/FP32 policy on Spark (eager reference)', flush=True)
    model = pi0_pytorch.PI0Pytorch(config)
    missing, unexpected = st.load_model(model, str(WEIGHTS), strict=False)
    assert not unexpected and all('embed_tokens' in k for k in missing), (missing, unexpected)
    model.paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16')
    data = train.data.create(train.assets_dirs, config)
    norm = normalize.load(CHECKPOINT / 'assets' / data.asset_id)
    policy = policy_module.Policy(
        model, transforms=[transforms.InjectDefaultPrompt(None), *data.data_transforms.inputs,
                           transforms.Normalize(norm, use_quantiles=data.use_quantile_norm),
                           *data.model_transforms.inputs],
        output_transforms=[*data.model_transforms.outputs,
                           transforms.Unnormalize(norm, use_quantiles=data.use_quantile_norm),
                           *data.data_transforms.outputs],
        sample_kwargs={'num_steps': 10}, pytorch_device='cuda', is_pytorch=True)
    language = model.paligemma_with_expert.paligemma.language_model
    expert = model.paligemma_with_expert.gemma_expert.model
    vision = model.paligemma_with_expert.paligemma.vision_tower.vision_model
    layer_configs = {'vision': vision.config, 'prefix': language.config, 'action': expert.config}
    records = {}
    handles = [vision.encoder.layers[0].register_forward_pre_hook(capture('vision', records), with_kwargs=True),
               language.layers[0].register_forward_pre_hook(capture('prefix', records), with_kwargs=True),
               expert.layers[0].register_forward_pre_hook(capture('action', records), with_kwargs=True)]
    normalized = {}
    sample_actions = policy._sample_actions
    def sample(device, observation, **kwargs):
        result = sample_actions(device, observation, **kwargs)
        normalized['actions'] = array(result)
        normalized['tokens'] = array(observation.tokenized_prompt)
        normalized['token_mask'] = array(observation.tokenized_prompt_mask)
        normalized['state'] = array(observation.state)
        for key, image in observation.images.items():
            normalized['image_' + key] = array(image)
            normalized['image_mask_' + key] = array(observation.image_masks[key])
        return result
    policy._sample_actions = sample
    prompts = ['pick up the red block', 'put the mug on the plate', 'open the drawer',
               'move the blue bowl to the left', 'place the object inside the basket']
    manifest = {'scope': 'FP16 first-block compatibility, not a full TRT policy',
                'reference': 'existing local openpi mixed BF16/FP32, eager',
                'float32_matmul_precision': torch.get_float32_matmul_precision(),
                'allow_tf32_after_model_constructor': torch.backends.cuda.matmul.allow_tf32,
                'checkpoint_sha256': sha(WEIGHTS), 'torch': torch.__version__,
                'reference_gpu': torch.cuda.get_device_name(0), 'num_steps': 10,
                'action_horizon': 10, 'padded_action_dim': 32, 'robot_action_dim': 7,
                'max_token_len': config.max_token_len, 'discrete_state_input': config.discrete_state_input,
                'fixtures': [], 'components': {}}
    all_records = []
    for i, prompt in enumerate(prompts):
        rng = np.random.default_rng(1000 + i)
        obs = {'observation/image': rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
               'observation/wrist_image': rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
               'observation/state': rng.uniform(-0.2, 0.2, 8).astype(np.float32), 'prompt': prompt}
        noise = rng.standard_normal((10, 32)).astype(np.float32)
        records.clear()
        with torch.inference_mode():
            result = policy.infer(obs, noise=noise)
            repeated = policy.infer(obs, noise=noise)
        assert np.array_equal(result['actions'], repeated['actions']), 'Reference did not reproduce'
        assert np.isfinite(normalized['actions']).all()
        np.savez_compressed(ROOT / f'policy_fixture_{i}.npz', noise=noise, actions_robot=result['actions'],
                            image=obs['observation/image'], wrist=obs['observation/wrist_image'],
                            raw_state=obs['observation/state'], **normalized)
        all_records.append({k: dict(v) for k, v in records.items()})
        manifest['fixtures'].append({'id': i, 'prompt': prompt,
                                    'repeated_actions_bit_equal': True,
                                    'normalized_shape': list(normalized['actions'].shape),
                                    'robot_shape': list(result['actions'].shape),
                                    'block_shapes': {k: list(v['hidden'].shape) for k, v in records.items()}})
        print('Saved reproducible policy fixture', i, manifest['fixtures'][-1], flush=True)
    for handle in handles:
        handle.remove()
    del sample_actions, policy, model, language, expert, vision
    torch.cuda.empty_cache()

    with safe_open(WEIGHTS, framework='pt', device='cpu') as source:
        for kind in ('vision', 'prefix', 'action_live', 'action_cached'):
            action = kind.startswith('action')
            base = 'action' if action else kind
            config = layer_configs[base]
            config._attn_implementation = 'eager'
            layer = SiglipEncoderLayer(config) if base == 'vision' else GemmaDecoderLayer(config, 0)
            weight_prefix = {'vision': 'paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.0.',
                             'prefix': 'paligemma_with_expert.paligemma.model.language_model.layers.0.',
                             'action': 'paligemma_with_expert.gemma_expert.model.layers.0.'}[base]
            weights = {k[len(weight_prefix):]: source.get_tensor(k).to(torch.float16)
                       for k in source.keys() if k.startswith(weight_prefix)}
            layer.half().load_state_dict(weights, strict=True)
            del weights
            layer = layer.eval().cuda()
            wrapper = VisionBlock(layer) if base == 'vision' else GemmaBlock(layer, action, kind == 'action_cached')
            names = ['hidden'] if base == 'vision' else ['hidden', 'mask', 'cos', 'sin']
            if action:
                names += ['prefix_k', 'prefix_v']
                names += ['mod_in', 'mod_post'] if kind == 'action_cached' else ['cond']
            first_args = None
            max_cached_error = 0.0
            for i, records in enumerate(all_records):
                values = dict(records[base])
                if kind == 'action_cached':
                    cond = torch.from_numpy(values.pop('cond')).cuda()
                    with torch.inference_mode():
                        values['mod_in'] = array(layer.input_layernorm.dense(cond).unsqueeze(1))
                        values['mod_post'] = array(layer.post_attention_layernorm.dense(cond).unsqueeze(1))
                args = tuple(torch.from_numpy(values[n]).cuda() for n in names)
                # Explicit positional slots match the wrapper's optional arguments.
                if kind == 'action_cached':
                    invoke_args = (*args[:6], None, *args[6:])
                else:
                    invoke_args = args
                with torch.inference_mode():
                    outputs = wrapper(*invoke_args)
                    if not isinstance(outputs, tuple):
                        outputs = (outputs,)
                    if kind == 'action_cached':
                        live = GemmaBlock(layer, action=True)(*args[:6],
                            torch.from_numpy(records[base]['cond']).cuda())[0]
                        max_cached_error = max(max_cached_error, float((outputs[0] - live).abs().max()))
                output_names = ['output'] if action or base == 'vision' else ['output', 'key', 'value']
                saved = dict(zip(names, (array(a) for a in args)))
                saved.update({'reference_' + n: array(t) for n, t in zip(output_names, outputs)})
                assert all(np.isfinite(saved['reference_' + n]).all() for n in output_names)
                np.savez_compressed(ROOT / f'{kind}_fixture_{i}.npz', **saved)
                if first_args is None:
                    first_args = invoke_args
            path = ROOT / (kind + '.onnx')
            torch.onnx.export(wrapper, first_args, path, dynamo=False, opset_version=17,
                              input_names=names, output_names=output_names, do_constant_folding=True)
            graph = onnx.load(path)
            onnx.checker.check_model(graph)
            initializer_bytes = sum(len(t.raw_data) for t in graph.graph.initializer)
            manifest['components'][kind] = {'onnx_sha256': sha(path), 'onnx_bytes': path.stat().st_size,
                'initializer_bytes': initializer_bytes, 'inputs': names, 'outputs': output_names,
                'cached_vs_live_fp16_max_abs': max_cached_error if kind == 'action_cached' else None}
            print('Exported', kind, manifest['components'][kind], flush=True)
            del layer, wrapper, graph, first_args, args, invoke_args, outputs
            torch.cuda.empty_cache()
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Finished: full mixed-precision reference fixtures and FP16 component exports', flush=True)


if __name__ == '__main__':
    main()
