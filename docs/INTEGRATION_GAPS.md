# Jetson VLA demo integration gaps

2026-10-26. Later-work memo for `jetson-orin-nano-vla` at `6e2d857ca5e3342fb2e29afd0cb60e295c10cdf5`, compared with LeRobot 0.6.1 (`7e241bd630a3719a56157a497ce5d08f244784f1`). Keep this repository a compact model-export and performance demo that downstream robot projects can build on. Implement robot control in those projects.

The split TensorRT chain, one shared scratch allocation, serialized builds, CUDA graphs, portable ONNX bundles, reference fixtures and cached AOT kernels provide a strong foundation. The gaps below concern extending beyond the measured base/LIBERO profiles. They do not invalidate the retained benchmark measurements against their existing fixture contracts. No implementation changes or new GPU benchmarks accompany this memo.

## Priorities

| Priority | Work | Benefit |
| --- | --- | --- |
| P0 | EVO1 public-policy masks and references | A fine-tune with padded actions reproduces the public policy |
| P0 | X-VLA `auto` action mode and explicit output boundary | Arbitrary-dimension fine-tunes can load correctly |
| P0 | Enforce raw-input preprocessing parity | A PASS also covers the path callers actually feed |
| P1 | Derive runtime dimensions and AOT specializations | Two-view and alternate-horizon checkpoints have defined support |
| P1 | Publish an importable runtime/cache API | Downstream integrations avoid benchmark CLI monkeypatching |
| P1 | Narrow and document GR00T checkpoint compatibility | NVIDIA and LeRobot fine-tunes cannot be confused |
| P2 | GR00T task packs and temporal sampling contract | Downstream callers can use multiple instructions and real history |
| P2 | Regression matrix and capability metadata | Preserve base-model simplicity while documenting generalization |

## EVO1 masks differ from the public policy

Evidence: `export/evo1/reference.py` creates an all-ones action mask and invokes `policy.model.action_head.get_action`. `bench/vendor/evo1_trt.py::infer` and `bench/vendor/trt_device.py::Evo1Device` evolve every padded dimension without a corresponding active-action mask. In LeRobot 0.6.1, `Evo1Policy._prepare_inference_action_mask` enables only the real action dimensions; `flow_matching.py::get_action` masks initial noise, input to the velocity network, predicted velocity and final action.

Consequently, the existing all-active head fixture does not establish parity with `Evo1Policy.predict_action_chunk` when real width is smaller than padded width. Padded values enter a learned action encoder and can affect active channels; merely trimming the final chunk cannot reproduce the masked calculation.

Later fix: export real action dimension and active mask, implement masked integration, and add a reference generated through the public policy and its processors. Keep the current lower-level head fixture separately for continuity of benchmark results. Test a narrow checkpoint, such as four or seven active dimensions inside a padded 24-wide head, with identical injected noise. Check padded channels stay zero at every required boundary. Audit selected embodiment behavior as part of this check.

Acceptance: public-policy and fast-runtime actions agree for the narrow-action case, with normalization/postprocessing checked separately. Trained versus random-head provenance remains visible, but rename or document `deployable` so it means trained-head provenance rather than robot compatibility.

## X-VLA cannot load the arbitrary action mode

Evidence: `bench/vendor/xvla_trt.py` permits only `ee6d`, `agibot_ee6d`, `joint` and `so101_bimanual` through `GRIPPER_BY_MODE`. It raises for `action_mode=auto`, even though the exporter explicitly handles AutoActionSpace and LeRobot 0.6.1 implements it for arbitrary real action widths. The existing Python excavator runtime already consumes the exporter's processor contract.

Later fix: support `auto` with an empty gripper list, preserve padded engine width internally, and expose an explicit postprocessor to trim and unnormalize according to the bundle. Do not apply gripper sigmoid to generic actions. Audit other action spaces against their actual postprocessing: matching gripper indices alone does not prove identical semantics.

Keep two explicit outputs: model-space benchmark output and optional processor-applied physical output. The X-VLA benchmark currently pads supplied state without consuming `processor_contract` normalization and returns model actions without that contract's unnormalization. Document that boundary, or implement the saved processors through a separate small API. Missing stats must not silently become a physical-output claim.

Acceptance: a real `auto` fine-tune loads and matches the checkpoint preprocessor, model and postprocessor; the current base `ee6d` benchmark remains unchanged under its recorded contract.

## Preprocessing reports do not determine fixture PASS

Evidence: X-VLA, EVO1 and GR00T fixture validators calculate an `image_preprocess` report but determine PASS from model actions computed using saved, already processed pixels. X-VLA/EVO1 also gate token IDs, but their image comparisons are still report-only. GR00T N1.7 reports active-dimension parity without making it a separate acceptance condition.

This permits a host image-transform regression to coexist with fixture PASS, because the action comparison bypasses that image-transform path. Narrow active outputs can also merit stronger checking than cosine over all padded dimensions.

Later fix: make raw-image preprocessing a required gate with its own justified numerical tolerance; run a second end-to-end fixture starting from raw images/state/task. Gate active outputs as well as padded model output. Include per-channel absolute error and robust handling of all-zero/constant reference tensors, for which cosine or range-relative errors need care. Avoid calling interpolation bit-exact unless tested against the upstream implementation for the supported shapes.

Acceptance: deliberately perturb resize, normalization or padding and the fixture fails; verify raw-to-processed and raw-to-action paths on at least two image aspect ratios.

## AOT optimizations are specialized to the measured profiles

Confirmed constants:

| Location | Current specialization |
| --- | --- |
| `bench/aot/xvla/plugin.py` | Sequence 262, 16 heads, head dimension 64, 30 action rows |
| Same file, `DWCONV_SHAPES` | Three views at each DaViT stage |
| `bench/aot/evo1/plugin.py` | Output GEMV `N=896`, `K=44800` |
| `bench/aot/groot/plugin.py` | Three-view vision and base-profile attention lengths, including 51/41 action-plus-state queries |

Some EVO1 and GR00T attention variants already derive runtime shapes, but this is not general across all replacements. A checkpoint export accepting arbitrary views/horizon does not imply its AOT build accepts those values.

Later fix: create a specialization descriptor from graph shapes plus validated model metadata. Parameterize the existing kernels and generate unique operation identities for differing specializations. Check support before the first expensive engine build. Unsupported profiles should either fail with an exact explanation or explicitly run baseline TRT and label the result accordingly. Do not silently report a partially optimized cache as the fully optimized reference profile.

Acceptance matrix: one/two/three real views where supported; X-VLA horizons 30 and 50; EVO1 alternate horizon/output-pool width; GR00T N1.7 two-camera custom embodiment. Compare each specialization to baseline with identical checkpoint/noise and retain before/after timings. Keep the published default profiles as regressions.

## Benchmark defaults need a checkpoint-derived runtime contract

Evidence: `bench/cli.py::Resolved` takes state width, noise width, distribution and several defaults from `models.REGISTRY`, while the bundle can describe a custom checkpoint. `bench/run.py::plan` explicitly passes the registry's base camera count. Unknown custom dimensions can therefore fail or receive incorrect synthetic inputs. Missing FPS currently falls back to 30, suitable as a benchmark convention but unsuitable as implied deployment metadata.

Later fix: distinguish a benchmark profile from the loaded checkpoint contract. Resolve required shapes, view count, noise distribution and steps from the actual bundle/engine metadata, and verify agreement. Keep the registry for defaults of known public checkpoints. Return missing training FPS as unknown; synthetic timing can use its own labeled nominal rate. State precisely whether each runtime permits fewer real views or requires an exact count: the documentation is broader than the exact-count checks in several backends.

Acceptance: custom shapes reach the runtime unchanged, incompatible overrides fail before builds, and default published profiles still run with one simple command.

## Runtime reuse currently depends on benchmark wrappers

Evidence: `bench/aot/{xvla,evo1,groot}/run_candidate.py` scripts parse `sys.argv` at module top level, verify a candidate, then monkeypatch `groot_trt.prebuild_engines` so `Backend.load()` does not overwrite the AOT cache with baseline engines. This is effective benchmark glue but an awkward downstream API.

Later fix: expose `load_verified_cache(bundle, cache, mode)`, `infer(processed_observation, noise)` and `close()` without CLI side effects. Give cache loading explicit modes and make preparation a separate operation. Share verification between the CLI and importable loader. Preserve absolute-path checks on legacy caches; future manifests can use content identity for relocation after revalidation. Pin CUDA/TensorRT/SM, graph hashes, engine bytes, precision, constants and AOT specialization in one deployment cache manifest. Existing baseline engine keys already include several of these; consolidate rather than replace them casually.

Acceptance: a ten-line downstream example loads a verified AOT cache and executes fixture inputs without importing `bench.cli`, replacing functions or building engines. Wrong-graph and wrong-runtime caches fail clearly. A C++ fixture-deserialization example would document the intended native reuse boundary.

## GR00T fine-tunes and temporal behavior need explicit scope

The current export shell recognizes NVIDIA `model_type=Gr00tN1d6/Gr00tN1d7`, and `groot17_split.py::load_policy` constructs NVIDIA's `Gr00tN1d7`. LeRobot 0.6.1's `groot` policy supports N1.7 through a wrapper and can save its own config/state dictionary. The demo does not implement a loader/converter for that representation. Its N1.7 `Contract` also defaults to 40 actions and padded width 132; custom values need extraction and validation.

Later fix: state accepted checkpoint representations in the README. Add explicit detection and a conversion/export adapter for LeRobot checkpoints, or reject them with a useful message. Prove weights and full-processor output parity against the source framework. Keep N1.6 labeled as NVIDIA support rather than LeRobot 0.6.1 support. Serialize modality normalization and relative-action decoding for callers requesting physical outputs.

The current GR00T runtime bakes one task and rejects a changed instruction. This is a clear supported limitation. Downstream prompt experiments need arbitrary instruction input, including phrasings not in the training task list. A future task pack must include prompt-dependent masks, rotary positions and KV selections, or expose those as runtime inputs. A task pack should serve as a cache rather than an instruction allowlist; preparation for arbitrary text before a session is also a useful first step. Token-ID replacement alone is not a general solution.

N1.7's history buffer stores inference-time visual features; its lag comes from `video_delta_indices / camera_fps`. This approximates history at inference cadence and falls back to the current frame until history exists. For downstream deployment, record the training sampling timebase, requested lag, selected source timestamps and actual lag. Callers must choose whether startup fallback is acceptable. The demo can retain its lightweight history mode while documenting the distinction from exact dataset-frame history.

History is optional at training time: the custom-embodiment example at the exporter's pinned NVIDIA revision uses two current-frame cameras (`delta_indices=[0]`). Support a one-frame export/runtime/fixture profile explicitly; current code paths that assume `frames[1]` need adaptation. This can simplify initial custom-robot experiments without requiring callers to invent a historical frame. Training with NVIDIA's native code also avoids LeRobot wrapper-checkpoint conversion for a first model, while retaining LeRobot-format datasets.

## RTC constraints are absent from the accelerated sampling interface

Follow-up research on 2026-10-26: NVIDIA's pinned N1.7 action head has an experimental RTC seed/mask primitive, but the demo's `Groot17Device` / `DitChain` only accepts the initial sample and executes unmasked flow steps. `export/groot/groot_split.py::DitChunk.forward` fuses the Euler update into its final output. Supplying a previous chunk as initial noise alone does not implement NVIDIA's frozen prefix and ramp.

Later fix: expose an optional normalized padded action prefix plus a per-element update-strength tensor. Either export the mask into the final step or correct each fused output with `old + strength * (unmasked_next - old)`. Select the original input exactly at strength zero and the original engine output exactly at strength one. Preserve graph-capture-compatible device addresses and verify full-chunk parity for intermediate strengths. Keep action-grid selection, observation age, latency reservation, task changes and actuator handling downstream.

The original Physical Intelligence RTC paper uses a different, derivative-based guidance method. Forward-only OpenPI TensorRT graphs do not supply its clean-trajectory VJP. Do not describe overlap initialization, output blending, or an identity-Jacobian approximation as full paper-equivalent RTC.

Acceptance: disabled RTC leaves the retained fixture output unchanged; every denoising step preserves frozen elements; ramped sampling matches the pinned NVIDIA primitive with identical noise; checkpoint-specific parity and latency are reported. An isolated downstream C++/CUDA helper exists in `/home/joel/kaivuriprokkis-rtc-scaffold`; its analytic CPU/GPU checks are not evidence of trained-model RTC parity.

References: [pinned NVIDIA primitive](https://github.com/NVIDIA/Isaac-GR00T/blob/23ace64f17aa5015259b8609d371eb61a357c776/gr00t/model/gr00t_n1d7/gr00t_n1d7.py), [NVIDIA RTC integration status](https://github.com/NVIDIA/Isaac-GR00T/blob/944fe70c4fd54e3fcb80a7d7258a328f9c061176/getting_started/real_world_deployment.md), [PI reference guidance](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/model.py).

## Suggested later delivery order

1. Correct EVO1 masks and add X-VLA `auto`; strengthen raw-input parity gates.
2. Extract the importable verified-cache API and make checkpoint shapes authoritative.
3. Parameterize AOT shapes while retaining default-profile regression results.
4. Clarify/add LeRobot GR00T checkpoint support, task packs and temporal contracts.
5. Add an optional device-resident RTC sampling interface and per-step parity tests.

Keep the CLI simple, preserve retained result artifacts, and keep robot configuration, safety leases, PWM, camera device access and operator controls in downstream projects. LeRobot pi0.5 remains a demo profile; excluding it from the excavator integration does not require removing it here.

## Primary references

- [Demo runtime and validators](https://github.com/eetmie/jetson-orin-nano-vla/tree/6e2d857ca5e3342fb2e29afd0cb60e295c10cdf5/bench/vendor), [AOT code](https://github.com/eetmie/jetson-orin-nano-vla/tree/6e2d857ca5e3342fb2e29afd0cb60e295c10cdf5/bench/aot), [exporters](https://github.com/eetmie/jetson-orin-nano-vla/tree/6e2d857ca5e3342fb2e29afd0cb60e295c10cdf5/export).
- [LeRobot EVO1 public policy](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/evo1/modeling_evo1.py), [flow matching](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/evo1/flow_matching.py), [processor order](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/evo1/processor_evo1.py).
- [LeRobot X-VLA action spaces](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/xvla/action_hub.py), [GR00T checkpoint loader](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/groot/modeling_groot.py), [GR00T documentation](https://github.com/huggingface/lerobot/blob/v0.6.1/docs/source/groot.mdx).
