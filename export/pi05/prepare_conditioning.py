# Vendored from the author's pi0.5 Orin Nano prototype (spark-projects): pi05-spark-inference/prototype/orin_initial_20261010_001/prepare_conditioning.py.
# Runs in the openpi container (export/pi05/README.md); paths are that stage layout's.
"""Prepare all 37 adaptive-normalization sites for the actual FP32 ten-step schedule.

Validates FP16 cache reuse against live FP16 conditioning, not full policy parity.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch
import torch.nn.functional as F
from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding

ROOT = Path(__file__).resolve().parent
WEIGHTS = Path('/workspace/checkpoints/pi05_libero_pytorch/model.safetensors')


def main():
    steps, times = 10, []
    dt = torch.tensor(-1.0 / steps, dtype=torch.float32, device='cuda')
    time = torch.tensor(1.0, dtype=torch.float32, device='cuda')
    while time >= -dt / 2:
        times.append(time.clone())
        time += dt
    assert len(times) == steps
    times = torch.stack(times)
    embeddings = create_sinusoidal_pos_embedding(times, 1024, 4e-3, 4.0, device=times.device).float().half()
    sites = [f'paligemma_with_expert.gemma_expert.model.layers.{i}.{norm}.dense'
             for i in range(18) for norm in ('input_layernorm', 'post_attention_layernorm')]
    sites += ['paligemma_with_expert.gemma_expert.model.norm.dense']
    with safe_open(WEIGHTS, framework='pt', device='cpu') as source, torch.inference_mode():
        def load(name):
            return source.get_tensor(name).half().cuda()
        w_in, b_in = load('time_mlp_in.weight'), load('time_mlp_in.bias')
        w_out, b_out = load('time_mlp_out.weight'), load('time_mlp_out.bias')
        # Compute individually to retain the same GEMM row count as live inference.
        def condition(t):
            return F.silu(F.linear(F.silu(F.linear(t, w_in, b_in)), w_out, b_out))
        conditions = torch.cat([condition(row[None]) for row in embeddings], dim=0)
        cache = torch.empty((10, 37, 3072), dtype=torch.float16, device='cuda')
        equality_checks, max_abs = 0, 0.0
        for site_idx, site in enumerate(sites):
            w, b = load(site + '.weight'), load(site + '.bias')
            for step in range(steps):
                cached = F.linear(conditions[step:step + 1], w, b)
                live = F.linear(condition(embeddings[step:step + 1]), w, b)
                assert torch.equal(cached, live)
                cache[step, site_idx] = cached[0]
                equality_checks += 1
                max_abs = max(max_abs, float((cached - live).abs().max()))
            del w, b
        result = cache.cpu().numpy()
        assert np.isfinite(result).all()
        path = ROOT / 'conditioning_10step_fp16.npy'
        np.save(path, result)
        report = {'num_steps': 10, 'dtype': 'float16', 'shape': list(result.shape),
                  'raw_bytes': result.nbytes, 'file_bytes': path.stat().st_size,
                  'timesteps_fp32': times.cpu().tolist(),
                  'timesteps_fp32_bits': times.cpu().numpy().view(np.uint32).tolist(),
                  'sites': sites, 'cache_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'checkpoint_sha256': json.loads((ROOT / 'manifest.json').read_text())['checkpoint_sha256'],
                  'cache_vs_live_fp16_bit_equal_checks': equality_checks, 'max_abs': max_abs,
                  'scope': 'all timestep conditioning sites only; full FP16 actions still require validation'}
        (ROOT / 'conditioning_manifest.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
