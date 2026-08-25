"""IOBinding for the SmolVLA split denoise loop — measured -24% wall, bit-identical.

The stock runtime re-feeds the KV cache as numpy on every denoising step: 32 tensors,
7.2 MB, ten times = **72 MB of host->device copies per inference** for data that never
changes after prefill. It also lands prefill's KV on the host first, so every tensor
makes a pointless device->host->device round trip.

This module rebinds that. Prefill writes its KV straight to device via IOBinding, those
OrtValues are bound as decode inputs once per inference, and the two constant mask
inputs are bound alongside them. Only `expert_embeds` is rebound per step.

Applied to a live policy instance rather than by editing `bench/vendor/smolvla_split.py`,
because the vendored file has to stay byte-identical to what runs on the robot.

Measured on Orin Nano Super, JetPack 7.2, MAXN_SUPER, projectors already on GPU
(20 cycles, num_steps=10, synthetic obs):

    stock feeds                    prefill 14.9   decode 101.2   wall 181.6 ms
    + IOBinding on decode          prefill 14.9   decode  66.1   wall 142.9 ms
    + prefill KV device-resident   prefill 12.9   decode  65.6   wall 136.2 ms

Output is **bit-identical** to the stock path (max abs diff 0.000e+00 over 8 chunks),
so this is free speed rather than a precision trade.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from ..vendor.smolvla_split import (IMG_TOKENS, MAX_ACTION_DIM, MAX_STATE_DIM, VLM_DIM,
                                    make_att_2d_masks, resize_with_pad_uint8,
                                    sinusoidal_time_embedding)


def enable_iobinding(policy) -> None:
    """Swap `policy.sample_actions` for the device-resident-KV version, in place."""
    policy._io = policy.decode.io_binding()
    policy._pio = policy.prefill.io_binding()
    policy.sample_actions = _sample_actions.__get__(policy, type(policy))


def _sample_actions(self, image_hwc_uint8, instruction, state, noise=None):
    img_emb = self._run_vision(resize_with_pad_uint8(image_hwc_uint8))
    lang_emb, lang_mask = self._embed_language(instruction)

    s = self.norm.normalize_state(np.asarray(state, dtype=np.float32).reshape(-1))
    s_pad = np.zeros((1, MAX_STATE_DIM), dtype=np.float32)
    s_pad[0, :s.shape[0]] = s
    state_emb = self._run_single(self.state_proj, s_pad).reshape(1, 1, VLM_DIM)

    n_pad = self.n_cam_slots - 1
    embs = np.concatenate([img_emb] + [self._pad_cam_emb] * n_pad + [lang_emb, state_emb],
                          axis=1).astype(np.float32)
    pad_masks = np.concatenate(
        [np.ones((1, IMG_TOKENS), dtype=bool)]
        + [np.zeros((1, IMG_TOKENS), dtype=bool)] * n_pad
        + [lang_mask, np.ones((1, 1), dtype=bool)], axis=1)
    att_masks = np.zeros((1, self.prefix_len), dtype=bool)
    att_masks[0, -1] = True

    # prefill: KV outputs bound straight to device, never touching the host
    pio = self._pio
    pio.clear_binding_inputs(); pio.clear_binding_outputs()
    pio.bind_cpu_input("attention_mask",
                       np.ascontiguousarray(make_att_2d_masks(pad_masks, att_masks)))
    pio.bind_cpu_input("position_ids",
                       np.ascontiguousarray((np.cumsum(pad_masks, axis=1) - 1).astype(np.int64)))
    pio.bind_cpu_input("vlm_embeds", np.ascontiguousarray(embs))
    for name in self._prefill_kv_names:
        pio.bind_output(name, "cuda", 0)
    self.prefill.run_with_iobinding(pio)
    kv = pio.get_outputs()                      # OrtValues, already on device

    if noise is None:
        noise = self._rng.standard_normal(
            (1, self.chunk_size, MAX_ACTION_DIM)).astype(np.float32)
    x_t = noise.copy()
    dt = -1.0 / self.num_steps
    t = 1.0

    prefix_pad_2d = np.broadcast_to(pad_masks[:, None, :],
                                    (1, self.chunk_size, self.prefix_len))
    suffix_pad = np.ones((1, self.chunk_size), dtype=bool)
    suffix_att_2d = make_att_2d_masks(suffix_pad, np.ones((1, self.chunk_size), dtype=bool))
    full_att_2d = np.ascontiguousarray(
        np.concatenate([prefix_pad_2d, suffix_att_2d], axis=2))
    pos_ids = np.ascontiguousarray(
        (pad_masks.sum(axis=-1, keepdims=True) + np.cumsum(suffix_pad, axis=1) - 1
         ).astype(np.int64))

    # everything constant across the N steps, bound once
    io = self._io
    io.clear_binding_inputs(); io.clear_binding_outputs()
    for i in range(self.n_layers):
        io.bind_ortvalue_input(f"past_key_{i}", kv[2 * i])
        io.bind_ortvalue_input(f"past_value_{i}", kv[2 * i + 1])
    io.bind_ortvalue_input("attention_mask",
                           ort.OrtValue.ortvalue_from_numpy(full_att_2d, "cuda", 0))
    io.bind_ortvalue_input("position_ids",
                           ort.OrtValue.ortvalue_from_numpy(pos_ids, "cuda", 0))
    out_name = self.decode.get_outputs()[0].name
    io.bind_output(out_name, "cuda", 0)

    while t >= -dt / 2:
        action_emb = self._run_single(self.action_in, x_t)
        time_emb = np.broadcast_to(
            sinusoidal_time_embedding(t)[None, None, :], action_emb.shape)
        ate = self._run_single(
            self.time_in, np.concatenate([action_emb, time_emb], axis=2).astype(np.float32))
        ate = ate * (1.0 / (1.0 + np.exp(-ate)))                  # SiLU
        suffix_embs = np.ascontiguousarray(self._run_single(self.time_out, ate))

        io.bind_ortvalue_input(
            "expert_embeds", ort.OrtValue.ortvalue_from_numpy(suffix_embs, "cuda", 0))
        self.decode.run_with_iobinding(io)
        expert_out = io.get_outputs()[0].numpy()

        x_t += dt * self._run_single(self.action_out, expert_out.astype(np.float32))
        t += dt

    return self.norm.unnormalize_action(x_t[0, :, :self.action_dim])
