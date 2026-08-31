# Audit follow-ups

Audit date: 2026-08-26

Audited commit: `609aef9` (`main`)

Scope: benchmark correctness, reproducibility, tests, performance opportunities, and
whether `ainekko/smolvla_base_onnx` is usable as a parity reference here.

This document records findings and the Spark -> Jetson validation handoff. It does not
claim that the findings have been fixed. Existing result JSONs and `docs/RESULTS.md` are
historical evidence produced by the harness at the recorded commit; they must not be
read as corrected measurements.

## Executive conclusion

The repository has a useful shape: deterministic observations, multiple backends,
per-stage instrumentation, saved action chunks, board monitoring, and generated reports.
The main weakness is not model runtime code. It is that several benchmark claims are
stronger than the evidence currently saved with them.

Before publishing another full matrix:

1. Freeze and hash every model, ONNX graph, tokenizer, stats file, and input fixture.
2. Settle the Ainekko export against its export-era software on the Spark.
3. Measure actual ORT node placement on the Jetson instead of provider preference.
4. Separate observation-source time from backend inference time and report both.
5. Make parity fail closed and require an exact comparison identity.
6. Validate X-VLA's static camera count against the bundle before running it.

The performance work should then happen on the Spark with SSH access to the Jetson. That
is where ONNX export A/Bs, ORT profiles, TensorRT engine builds, Nsight traces, and the
final board measurements can be evaluated without guessing.

## Project map

- `bench/models.py` defines model families, public artifacts, and default shapes.
- `bench/cli.py` resolves model and bundle settings and constructs a backend.
- `bench/obs.py` creates indexed synthetic or frame-backed observations and noise.
- `bench/runner.py` owns load, first inference, warmup, idle, measurement, and JSON output.
- `bench/monitor.py` and `bench/procwatch.py` collect board and process measurements.
- `bench/backends/` contains the supported Torch and split ORT adapters.
- `bench/vendor/` pins the two split runtimes next to the measurements.
- `bench/parity.py` compares saved action chunks; `bench/report.py` renders the report.

There is currently no `tests/` directory and no CI workflow. `python -m unittest discover`
finds zero tests.

## Evidence labels

- **Proven**: follows directly from source control flow or a stable API contract.
- **Measured**: reproducible from the committed result JSONs.
- **Needs Spark**: requires model weights, export environments, or enough memory to export.
- **Needs Jetson**: requires the target ORT/TensorRT/CUDA stack or board instrumentation.
- **Unproven**: plausible, but the required artifact or trace is not committed.

## P0: validity before new results

### A01. Observation work and backend latency use different scopes

Status: **Proven and measured**

Python evaluates `obs[...]` before entering `backend.infer()` at
`bench/runner.py:164-181`. Synthetic observations generate full-resolution float arrays
and Gaussian speckle for every camera at `bench/obs.py:106-121`. That work is inside the
monitor and process windows, but outside each backend's `timings_ms["total"]`.

The stored sustained runs show the size of the gap:

| run | backend mean ms | calls / window s | completed Hz | reported Hz | unreported ms/cycle |
|---|---:|---:|---:|---:|---:|
| `smolvla-base.ort` | 169.38 | 1357 / 300.18 | 4.521 | 5.90 | 51.83 |
| `smolvla-base.torch` | 1179.58 | 244 / 300.63 | 0.812 | 0.85 | 52.51 |
| `xvla-base.ort` | 415.86 | 608 / 300.03 | 2.026 | 2.40 | 77.61 |

Consequences:

- `latency_ms`, `hz_mean`, and control headroom exclude observation work.
- process CPU, whole-board power, thermals, duration, and energy include it.
- frame-backed runs also include lazy decode and cache growth outside reported latency.
- first-inference timing includes `obs[0]`, unlike steady-state backend timing.

Required resolution:

1. Materialize a bounded observation ring before first inference and monitoring.
2. Record independent `observation_ms`, outer `infer_wall_ms`, and backend stage timings.
3. Use completed calls / measured window for achieved throughput.
4. State explicitly whether power represents runtime-only or a deployment pipeline.
5. Re-run board power, CPU, and control-headroom results after the scope is fixed.

Verification: fake-backend tests locally, followed by an A/B on the Jetson using identical
artifacts and observations.

### A02. Provider preference is reported as actual execution placement

Status: **Proven; actual placement needs Jetson**

The ORT backends record `session.get_providers()[0]` at
`bench/backends/ort_split.py:156-159` and
`bench/backends/ort_split_xvla.py:77-86`. `get_providers()` returns configured provider
priority, not the provider assigned to each node or partition. `bench/report.py:54-76`
then labels this value as where graphs "ACTUALLY ran."

The repository already contains a direct contradiction. `docs/RESULTS.md` says the
SmolVLA run used `7 TRT / 2 CPU`, while `docs/06-optimization-backlog.md:267-269` records
that TRT declined all four moved projectors and ORT executed them on CUDA. A graph can
also be partitioned across more than one provider, so one label per session is not
sufficient.

Required resolution:

1. Run one non-headline validation inference with ORT profiling enabled.
2. Save node count and node time by TRT, CUDA, and CPU provider.
3. Rename current metadata to `configured_provider_priority`.
4. Keep profiling disabled for headline latency runs.
5. Fail or prominently mark a run when expected heavy subgraphs do not execute on TRT.

Verification requires the Jetson and both cold and warm engine-cache states.

### A03. Parity identity is incomplete and the command fails open

Status: **Proven**

`bench/parity.py:54-69` truncates both runs to the shorter saved length and decides that
noise is identical from two booleans. `bench/parity.py:154-177` groups runs only by model
family, real views, and camera slots. It does not require the same:

- model/checkpoint or ONNX graph hashes;
- task, observation source, frame identity, seed, or saved observation indices;
- tokenizer, stats, preprocessing, chunk shape, action width, or denoise steps;
- warmup offset or injected noise tensor.

`bench/cli.py:283-288` returns success for errors, no data, or zero comparisons. This was
confirmed with `python -m bench parity does-not-exist`, which reported an error and
returned `0`.

The current SmolVLA result pair happens to save the same seed and indices, so this defect
does not by itself explain its mismatch. It does make automated future verdicts unsafe.

Required resolution:

1. Persist a canonical comparison signature containing all fields above.
2. Align chunks by saved observation index, never by list position.
3. Save or hash exact preprocessed inputs and injected noise.
4. Require one explicit reference per comparison group.
5. Return nonzero for errors, no data, no expected comparison, malformed input, or failure.
6. Handle equal all-zero outputs without producing a false cosine failure.

Verification is a pure CPU test matrix and should not wait for hardware.

### A04. The Ainekko export did not work here, and the reason is unproven

Status: **Mismatch measured; root cause unproven; needs Spark**

The committed arrays reproduce the documented comparison:

| scope | cosine minimum | max absolute difference | mean absolute difference |
|---|---:|---:|---:|
| all 32 emitted dimensions | 0.9393259 | 2.293518 | 0.067420 |
| first 6 declared action dimensions | 0.9393403 | 2.293518 | 0.333292 |

The full comparison is `12.922%` of the reference run's global action range. The error is
therefore not caused by dilution from the 26 padded dimensions. Historical CPU-EP data
also makes TensorRT/FP16 an unlikely explanation for the dominant gap.

What is not established:

- Result JSONs record generic local paths, no Hub revisions, and no graph hashes.
- Both stored runs were made from a dirty worktree at `dfa93a6` without a diff hash.
- The initializer and stage-output investigations are prose, not committed scripts or
  raw tensors.
- Matching 198 vision initializers does not verify every graph, constant, attribute, or
  parameter-to-node mapping.
- The locally re-exported graph reports `1.94%` range error, which still fails the
  repository's own `<= 1%` parity gate at `bench/parity.py:89-94`; its result JSON is not
  committed.

There is also a chronology error in `docs/03-backends.md:302-305`. The
[public Ainekko graphs][ainekko-hf] were uploaded in October 2025. The
[export notebook][etars-export] linked by its model card was committed on 2025-10-10 and
explicitly installs `lerobot==0.3.3`. [LeRobot 0.5.1][lerobot-051] was released on
2026-04-07, so the Ainekko export cannot have been traced under 0.5.1.

Relevant frozen identifiers:

- Ainekko graph upload commit: `f29c684838312c383114f70a433ec367bc95c874`
- public `smolvlm_vision.onnx` SHA-256:
  `a8694e85ddb1edc103046a48f6ad319185722ca31fbb43989e6b528163ad2dcd`
- associated ETARS notebook commit:
  `9ae33a75549a3385170ad968ac3f27878bf8d902`
- current reference environment: LeRobot 0.5.1, Transformers 5.3.0, Torch 2.11.0

Current decision:

> `ainekko/smolvla_base_onnx` did not work for this repository. Scored against our
> LeRobot 0.5.1 reference it disagrees with the checkpoint it is an export of, by
> ~13% of commanded range. Their export notebook pins `lerobot==0.3.3`, so a version
> gap between their trace and our reference is the obvious explanation — but it has
> not been tested against its own 0.3.3-era module, so that is a guess and not a
> finding. **It may well be perfectly correct on the software it was built for.**
> This repository therefore uses a locally traced export, and makes no claim about
> the public one beyond "it did not work here". YMMV.

The Spark experiment that would settle it is specified below. Until it has been run,
nothing in this repository should be read as saying the public export is wrong.

### A05. X-VLA camera count can be ignored or mislabeled

Status: **Proven; current 3/3 result is internally aligned**

`OrtSplitXVLABackend` accepts and stores `valid_views`, but does not pass it to
`XVLASplitPolicy` at `bench/backends/ort_split_xvla.py:47-73`. The policy instead reads
the value from `bundle.json` and slices inputs at
`bench/vendor/xvla_split_ort.py:275-286,349-354`.

`scripts/run_all.sh:70-79` sweeps view counts against one fixed bundle even though the
vision graph batch dimension is export-time static. A request larger than the bundle can
generate and discard cameras while recording the requested count. A smaller request can
fail shape validation.

Required resolution:

1. Require requested views to equal the bundle's `valid_views` before loading engines.
2. Use a separate bundle and engine cache for each exported camera count.
3. Record requested, bundle-declared, and actually processed views separately.

## P1: correctness and reliability

### A06. CLI and artifact validation gaps

Status: **Proven**

- SmolVLA Torch parses `--num-steps` but never passes it to the backend
  (`bench/cli.py:155-181`).
- Negative SmolVLA ORT steps make `dt` positive and can leave the denoise loop running
  forever (`bench/vendor/smolvla_split.py:397-424`).
- Negative X-VLA steps execute no denoise iterations and can return a plausible success.
- `--iters 0` and `--duration-s 0` still perform one measured inference.
- family/model/bundle conflicts and checkpoint/export identity are not cross-validated.
- a Hugging Face tokenizer ID is converted to `Path` and then ignored unless it exists
  locally (`bench/cli.py:175-180`, `bench/backends/torch_smolvla.py:110-116`).
- missing stats silently become identity normalization.

All counts, dimensions, durations, view counts, and step counts need explicit range
validation. Comparable runs should fail when required tokenizer/stats provenance is
missing rather than silently substituting a fallback.

### A07. Unsupported Tether backend removed

Status: **Resolved by removal**

The Tether backend, CLI commands, environment, dependencies, and runbook entries
were removed because it does not provide an 8 GB Orin-local deployment and its remote
fallback is outside this repository's scope. No Tether credentials or unsupported speed
claims are persisted by this repo.

### A08. Successful status does not validate outputs or instruments

Status: **Proven**

The runner does not require finite actions, stable chunk shapes, expected dimensions,
positive timings, monitor samples, process samples, or expected TensorRT execution. A
backend can return NaNs and still reach `status: ok`; Python's JSON encoder can then emit
non-standard `NaN`. Monitor-thread failures also do not necessarily fail a run.

Add a result contract before setting `status: ok`, and distinguish model execution,
parity, placement, and instrumentation status rather than collapsing them into one flag.

### A09. Process and cache tracking are incomplete

Status: **Proven**

- `ProcWatch` follows direct children only (`bench/procwatch.py:84-89`), not a recursive
  model-server process tree.
- child creation and exit can distort cumulative CPU deltas.
- summed RSS can double-count shared pages; PSS is preferable for multiprocess runs.
- SmolVLA prebuild skips work when any three `*.engine` files exist in a shared cache,
  without checking graph or artifact identity (`bench/vendor/smolvla_split.py:155-170`).
- X-VLA repeats prebuild subprocess work on every load; the committed load took 366.35 s.

Engine caches need a manifest keyed by graph hash, precision, device, ORT/TRT/CUDA
versions, workspace, and optimization level.

### A10. Statistics and result ingestion have edge-case errors

Status: **Proven**

- ordered quartiles drop the final `n % 4` samples (`bench/runner.py:50-72`); for
  `[1, 1, 1, 1, 100]` the reported drift is zero.
- percentile indexing is non-standard and especially visible for small samples.
- result writes are non-atomic.
- malformed JSON files are silently skipped, while valid non-object JSON can crash.
- all-zero equal chunks produce a NaN cosine and fail parity.
- `runtime.steps` is rendered with an `ms` suffix.

These are deterministic CPU fixes, but expected definitions should first be frozen in
golden tests.

### A11. Automation and portability are absent

Status: **Proven**

There are no unit tests or CI workflows. Pure commands such as `models`, `parity`, and
`report` also eagerly import Linux-only `ProcWatch`; on Windows, module import fails at
`os.sysconf` before the command runs (`bench/cli.py:23`, `bench/procwatch.py:22-23`).

Hardware-independent code should be covered on ordinary Linux CI. Hardware jobs should
be a separate manual or self-hosted Jetson workflow.

## Performance opportunities after validity

These are priorities for profiling, not promised speedups.

### P01. Keep X-VLA split chains on device

`bench/vendor/xvla_split_ort.py:349-430` uses `session.run()` for every split graph.
Outputs return as NumPy and are fed into the next GPU session, forcing synchronization
and host/device traffic at every boundary. The denoise chain repeats four sessions ten
times, while constant conditioning is returned to and re-fed from the host.

The committed run attributes 393.693 ms of its 415.86 ms backend wall time to session
calls. IOBinding, preallocated OrtValues, device-resident conditioning, and a device-side
interpolation/update are the highest-value X-VLA profile target. Use ORT profiling and
Nsight Systems to count copies and synchronization before changing graph boundaries.

### P02. Finish the SmolVLA device-resident denoise path

Existing IOBinding correctly keeps the KV cache on device, but each step still runs four
projectors through ordinary sessions, creates a CUDA OrtValue, copies decode output to
NumPy, and performs Euler integration on the host
(`bench/backends/split_iobind.py:109-137`).

The current two-camera result exposes 12.348 ms in projectors and 15.946 ms of
unattributed host wall. Fusing the projectors, SiLU, decode, action projection, and
possibly the Euler update removes 40 small projector calls per inference. The likely
gain must be measured after A01 is corrected.

### P03. Make startup and cache behavior measurable

Separate artifact/tokenizer load, cache validation, engine build, session creation, and
first execution. Measure both cold and warm caches. Persist effective builder settings
and the manifest that selected each cached engine.

### P04. Remove smaller repeated work only after profiling

Candidates include preallocated masks and OrtValues, cached timestep embeddings, cached
X-VLA token IDs, `astype(copy=False)`, output buffers, and ORT thread-pool tuning. These
are likely secondary to split-chain device residency and should be isolated A/Bs.

## Spark -> Jetson validation plan

### Phase 1: freeze artifacts and inputs on the Spark

1. Check out the exact benchmark commit and record a clean full Git SHA.
2. Pin Hub revisions for the PyTorch checkpoint, Ainekko export, tokenizer, and any local
   replacement export.
3. Write a manifest containing SHA-256, byte size, graph input/output signatures, and
   source revision for every file.
4. Save one canonical fixture containing raw uint8 images, resized model pixels, token
   IDs and masks, raw and normalized state, camera masks, and injected noise.
5. Give the fixture and manifest stable IDs that are copied into every result JSON.

Do not use a mutable Hub `main` revision as evidence.

### Phase 2: settle the Ainekko question on the Spark

1. Recreate the associated LeRobot 0.3.3 environment and recover the exact Transformers
   version used by the exporter. Record the resolved lock, not only requested versions.
2. Run the frozen public vision graph on CPU FP32 against the export-era PyTorch vision
   module using the exact same preprocessed tensor.
3. Save outputs at the vision boundary and final action boundary, not only summary
   cosines.
4. Repeat the same fixture under the repository's LeRobot 0.5.1 / Transformers 5.3.0
   reference.
5. Export one frozen module/checkpoint with the export-era exporter, Torch 2.11 legacy
   export, and Torch 2.11 Dynamo export; compare each to the same in-memory module.
6. If needed, run hybrid final-action tests with public vision + local remaining graphs
   and local vision + public remaining graphs.

Decision rule:

| result | conclusion |
|---|---|
| public ONNX matches export-era PyTorch, not 0.5.1 | version compatibility problem |
| public ONNX fails against its exact export-era module on CPU FP32 | export inconsistent with its own era; worth raising upstream |
| CPU FP32 matches but Jetson TRT does not | runtime or precision problem |
| local vision substitution fixes final actions | vision graph is causal |

### Phase 3: validate the harness and placement on the Jetson

1. Verify power mode and clocks from readable system state; do not accept a permission
   error as proof that clocks are pinned.
2. Build engines from an empty, artifact-specific cache and retain build logs.
3. Run ORT profiling for one validation inference and save actual node placement.
4. Run the old and corrected timing scopes against the same pre-materialized fixture.
5. Validate each X-VLA bundle at exactly its exported view count.
6. Capture Nsight Systems traces for X-VLA chaining and SmolVLA denoise IOBinding.
7. Run the full performance/power matrix only after parity, placement, and instrumentation
   gates pass.

### Phase 4: commit evidence and close findings

1. Commit result JSONs, artifact manifests, commands, and concise profile summaries.
2. Keep superseded results labeled as historical instead of silently overwriting them.
3. Update `docs/03-backends.md`, `docs/04-metrics.md`, `docs/05-runbook.md`, and
   `docs/RESULTS.md` from the collected evidence.
4. Mark each audit ID resolved, accepted, or still open in this document.

## Acceptance gates for published results

- **Provenance**: every artifact, tokenizer, stats file, fixture, and code state is hashed.
- **Input identity**: compared runs have the same comparison signature and saved indices.
- **Output validity**: actions and timings are finite and shapes match the artifact.
- **Parity**: numerical comparisons meet the stated threshold; failures remain visible in
  speed tables.
- **Placement**: expected heavy nodes are confirmed on TRT from a validation profile.
- **Instrumentation**: monitor and process windows contain valid samples.
- **Board state**: power mode, clocks, temperatures, and software versions are recorded.
- **Repeatability**: headline numbers include multiple fresh-process runs, not one process.
- **Security**: persisted metadata contains no API keys or other credentials.

## Test backlog

1. Fake `ObsSource` and backend tests for timing scope, exact iteration count, output
   validation, monitor failures, cleanup, and atomic result writing.
2. Parity tests for signatures, seeds, indices, warmups, zero vectors, non-finite values,
   shape mismatches, missing files, malformed JSON, and exit codes.
3. CLI tests for positive ranges, config precedence, family conflicts, tokenizer IDs,
   stats requirements, denoise-step propagation, and X-VLA view/bundle matching.
4. Tiny ONNX tests proving that configured provider order is not actual node placement.
5. Golden monitor/parser, statistics, and Markdown-report tests.
6. Linux CI for CPU tests and shell lint; a separate manual self-hosted Jetson job for
   engines, profiling, power, and long-duration measurements.

## Non-goals

The base checkpoints were not fine-tuned for a robot in this repository. Synthetic
actions can establish runtime parity but cannot establish task success or whether a
changed denoise budget is safe on hardware. Camera capture, the control stack, and robot
validation remain separate deployment measurements.


## X-VLA initial audit addendum

Audit date: 2026-08-26

Scope: the X-VLA benchmark adapters in this repository, the split runtime/exporter in
`spark-projects/orin-nano/xvla-runtime`, the excavator fine-tune project in
`spark-projects/xvla-spark-finetune`, the saved base-model evidence, and the X-VLA
bundles currently present on the Orin. This pass records findings only; no runtime,
exporter, training, engine-cache, or robot-control fix was applied.

Code and artifact snapshot:

- benchmark repository: clean `fe98810` after the audit-contract patch;
- Spark projects: tracked files at `cd197a6` (an unrelated editor swap file in the
  sibling `scene-reconstruction` project was left untouched);
- Orin benchmark checkout: `75c6658` with the already staged audit patch, not yet
  synchronized to `fe98810`;
- Orin base bundles: `xvla-base-split`, `xvla-base-split-fp16`, and
  `xvla-base-split-3cam`; all three pass their complete `MANIFEST.sha256` checks;
- exporter recorded by those bundles: `851695a`, LeRobot 0.6.1, Torch 2.11.0+cu130,
  Transformers 5.5.4, opset 17;
- excavator checkpoint: only `outputs/digging/ir/checkpoints/000250` exists.

Implementation update (2026-08-28):

- LeRobot 0.6.1 / Transformers 5.5.4 is the matching checkpoint/export stack; the
  earlier Torch failure was the raw config's stale 1024-token length, while the saved
  processor and exporter both require 50 tokens.
- X01 and X02 are closed mechanically by schema-v2 bundles: exact 3-D/4-D physical
  features and MEAN_STD tensors, 20-D model padding, AutoActionSpace trimming, local
  offline tokenizer tree, processor hashes, checkpoint tree hash, graph-to-checkpoint
  binding, and pre-engine manifest verification.
- Full 10-step local CPU FP32 parity passes at conditioning, padded 20-D model action,
  normalized four-axis action, and physical four-axis action (all cosine 1.000000).
- On the Orin, the fine-tuned one-camera FP16 TensorRT run is 348.81 ms p50 / 353.74 ms
  p95 (2.87 Hz), returns 50 x 4 physical actions, and places all 12 graphs on TRT.
  Against Jetson Torch FP32 at 2090.15 ms p50, eight exact seeded observations pass
  with cosine_min 0.9999994 and 0.083% maximum error as a fraction of action range.
- A fresh-process warm-cache repeat is bit-identical, with 355.87 ms p50. Evidence is
  saved under `results/xvla-digging-contract/`.
- Device-resident conditioning and denoiser split chaining are bit-identical. In the
  paired 40-inference A/B they improve p50 from 355.15 to 351.70 ms (1.0%) and p95 from
  363.23 to 354.57 ms (2.4%), so the path remains an explicit opt-in.
- The identity-bound engine-cache manifest reduces fresh-process warm load from 62.53
  to 25.64 seconds. It binds the cache to bundle, precision, builder options,
  ORT/TensorRT/CUDA/L4T, hardware, and exact engine/timing-file inventory.
- One non-headline validation inference profiled every ORT session. All 12 expected
  split graphs emitted TensorRT node events, with no CUDA/CPU fallback node events;
  compressed raw traces and the parsed summary are saved with the result evidence.
- The canonical real-IR fixture contains aligned raw state and frames from held-out
  episodes 5/15/25/35/45/55/65/75. Torch FP32, CPU ORT FP32, and two fresh Jetson
  TensorRT FP16 processes share the exact comparison signature, input hashes, and
  seeded noise hashes. CPU ORT reaches cosine 1.000000 / 0.021% maximum range error;
  TensorRT reaches cosine 0.9999763 / 0.394%, and its repeats are bit-identical.
- X06's full device-loop experiment is also complete. Fused interpolation keeps x1,
  action, conditioning, and split intermediates on CUDA and passes parity/placement,
  but its paired p50 is 0.4% slower, so it remains experimental.
- X03 remains open. X04 and X05 are closed mechanically. X07 remains open: model-only TRT
  RSS was about 6.22 GB, with only about 567 MB system RAM free and existing swap use
  during the original baseline run.

The per-finding problem statements below preserve what was observed on 2026-08-26.
Their status lines and the implementation update above describe the current code.

The earlier X-VLA runtime work is useful prior evidence, but is not a current publishable
benchmark result. It reported action cosine `1.000000` and maximum absolute error
`6.5e-4` for the FP32-graph bundle, and cosine `1.000000` / `7.4e-4` for the FP16-weight
bundle, against one seeded PyTorch fixture. The committed benchmark result
`results/xvla-base.ort.json` instead has no artifact record, validity contract,
comparison signature, explicit PyTorch reference, or parity verdict.

### X01. The fine-tuned action/state contract is not deployable

Priority: **P0**

Status: **Closed mechanically; real robot validation remains required**

The 250-step excavator checkpoint requires a processor contract that is materially
different from the base `ee6d` benchmark:

- raw state is 3-D `[lift, tilt, scoop]` and uses `MEAN_STD` normalization;
- real action is 4-D `[slew, lift, tilt, scoop]` and uses `MEAN_STD` normalization;
- the X-VLA model still pads both tensors to width 20 internally;
- `action_mode` is `auto`;
- the policy postprocessor truncates to the real four axes and unnormalizes them.

The split exporter writes `max_state_dim` and `action_mode`, but not the real state/action
dimensions, normalization tensors, processor identities, or output feature names. The
split runtime has a fixed gripper-index table for `ee6d`, `agibot_ee6d`, `joint`, and
`so101_bimanual`; it rejects `action_mode="auto"` during load. Even if that guard were
removed, `sample_actions()` currently copies raw state directly into the padded model
input and returns the full padded model output without the checkpoint's state normalizer,
four-axis slice, or action unnormalizer.

Required resolution:

1. Make the export bundle carry the real input/output feature contract and exact
   preprocessor/postprocessor state.
2. Normalize the real 3-D state before padding it to the graph's 20-D input.
3. Define `auto` postprocessing from bundle metadata rather than a guessed gripper map.
4. Slice the real four action axes and unnormalize them before returning a robot chunk.
5. Save parity at both the normalized 20-D model boundary and the physical four-axis
   controller boundary.

This finding was the blocker that schema-v2 bundles and the physical-boundary parity
gate now close mechanically; real robot validation remains separate.

### X02. Bundle checksums do not freeze model or tokenizer identity

Priority: **P0 for parity claims; P1 for base-model experimentation**

Status: **Closed for schema-v2 deployable bundles**

The three current Orin bundles are internally intact: their manifests verify every
listed file. Their provenance identifies the checkpoint only as the relative path
`models/xvla-base`; it does not record an immutable Hub revision or checkpoint tree
hash. The tokenizer is not included in the bundle and the runtime falls back to the
mutable identifier `facebook/bart-large`. When that identifier is used, the benchmark
records no tokenizer hash. The older Spark-local one-view and two-view bundles predate
both provenance and manifest generation entirely.

Because token IDs feed the exported text encoder, a graph manifest alone is not enough
to establish input identity.

Required resolution:

1. Record the exact `lerobot/xvla-base` Hub commit and checkpoint tree hash, or the full
   tree hash of a local fine-tuned checkpoint.
2. Save the tokenizer into each deployable bundle, include it in the manifest, and make
   offline local loading mandatory on the robot.
3. Include pre/postprocessor tensors and their hashes for fine-tuned bundles.
4. Reject incomplete or identity-ambiguous bundles before engine construction.

### X03. The excavator training target is a smoke test, not a candidate

Priority: **P0 before starting a long training run**

Status: **Measured from the saved train configuration and checkpoint directory**

Only checkpoint `000250` exists. It was deliberately produced as a throughput and
configuration smoke test. The saved run used:

- `local/masi_digging_ir`, the older 82-episode dataset family, with 74 train episodes;
- chunk and action horizon 50;
- one real IR feature but a three-slot model buffer;
- 10 denoising steps, batch 32, seed 1000;
- 1.823 seconds per optimizer step at the end of the probe.

A 20,000-step run at that measured rate is about 10.1 hours. Since the proven SmolVLA
robot candidate now uses the cleaned 181-episode IR dataset and a 12-action chunk, the
old X-VLA recipe would not be an apples-to-apples model comparison or necessarily the
right deployment target.

Before spending that compute, explicitly settle and record:

1. cleaned dataset and held-out episode identity;
2. one-camera versus two-camera target;
3. chunk 12, 30, or 50 and the corresponding evaluation horizon;
4. task string, state/action feature order, training FPS, normalization, and seed;
5. checkpoint/evaluation cadence so an earlier useful model is not lost behind only a
   final checkpoint.

Chunk 12 is a candidate because it matches the currently successful robot policy, not a
pre-approved X-VLA choice. X-VLA's sequence cost and task quality must be measured.

### X04. Current X-VLA parity evidence is not attached to the hardened harness

Priority: **P0 before publishing or deploying an export**

Status: **Closed mechanically on real IR; task-quality validation remains separate**

The standalone X-VLA parity script retains one seeded reference and reports excellent
base-model parity. It does not provide the multi-observation, real-input, exact-identity
evidence now required by this repository. The historical `xvla-base.ort` result was made
before observation materialization, strict output validation, atomic results, exact
comparison signatures, and explicit reference selection were added.

Completed rerun:

1. Build a canonical fixture from at least eight real excavator IR observations with
   aligned raw state, exact resized pixels, tokens, masks, domain ID, and seeded `x1`.
2. Emit the FP32 PyTorch gold from the exact checkpoint. **Done on the Jetson:** eight
   observations, 2079.09 ms p50, exact checkpoint and processor identity.
3. Validate CPU ORT against that gold to isolate export correctness. **Done on the
   Spark:** cosine minimum 1.000000 and 0.021% maximum range error.
4. Validate Orin FP16 TRT against both, including per-observation cosine, maximum error
   as a percentage of reference range, and the real controller axes. **Done:** cosine
   minimum 0.9999763 and 0.394% maximum physical-action range error.
5. Repeat the warmed Orin run in a fresh process and require deterministic output.
   **Done:** all eight saved 50 x 4 chunks are bit-identical.
6. Save an ORT profile proving actual node placement for every heavy graph. **Done:**
   12/12 graphs have TensorRT node events and no CUDA/CPU fallback node events.

The fixture is a replay of recorded observations, not a live camera/control test. The
checkpoint remains the X03 smoke target, so these results certify export/runtime
mechanics and numerical behavior only.

### X05. Direct runtime range and cache validation remain fail-open

Priority: **P1**

Status: **Closed**

The hardened benchmark adapter and standalone policy now reject non-positive denoising
steps. `run_pipeline.py` also rejects non-positive duration and negative report
intervals before loading artifacts.

`prebuild_engines()` now writes and verifies an identity-bound cache manifest. A valid
hit skips all twelve subprocesses; mismatched identity or missing, resized, or mixed
engine/timing files fail closed. On the Orin, the validation phase fell from 36.80 to
5.04 seconds and total fresh-process load fell from 62.53 to 25.64 seconds. The remaining
five seconds are dominated by full bundle-integrity verification.

The result metadata records cache validation/prebuild and session-load time separately.
CPU tests cover invalid direct-runner values plus stale and mixed cache rejection.

### X06. Device residency is the first semantics-preserving performance target

Priority: **P1 after parity is established**

Status: **Full device loop implemented and measured; no promotion on current result**

The historical 3-camera benchmark reports 415.86 ms mean backend wall time, of which
393.693 ms is inside session calls and 22.163 ms is Python/NumPy. The denoise family
accounts for 295.512 ms and executes four split sessions per step for 10 steps: 40
session boundaries per inference. Every ordinary `session.run()` returns an intermediate
to NumPy before the next GPU session consumes it; `x_t` interpolation also runs on the
host.

The implemented IOBinding path keeps conditioning and three split intermediates on CUDA.
It is bit-identical across eight exact signed observations. The paired 40-inference run
improves p50 by 1.0% and p95 by 2.4%; session-call time falls, but IOBinding bookkeeping
returns part of that saving to Python. It remains opt-in.

The fused candidate changes the first denoiser interface from `x_t` to fixed `x1` plus
the previous action and computes `x_t = x1*t + action*(1-t)` inside `denoise_0`. With
IOBinding, those tensors and every split intermediate stay on CUDA across all ten steps;
only the final action returns to the host. CPU ORT differs from the ordinary graph by at
most 4.172e-7. Jetson TensorRT reaches cosine minimum 0.9999791 and 0.422% maximum range
error versus Torch, repeats bit-identically, and all 12 profiled graphs have TensorRT
node events with no fallback.

It is not faster in the paired 40-call real-IR A/B:

| path | p50 ms | p95 ms | achieved Hz | Python/numpy ms |
|---|---:|---:|---:|---:|
| partial IOBinding; host interpolation | 339.28 | 350.10 | 2.936 | 19.180 |
| full IOBinding; fused interpolation | 340.55 | 349.77 | 2.922 | 17.577 |

The 1.6 ms reduction in Python/numpy time is offset by roughly 3.2 ms more measured graph
time. The candidate remains useful as a proven mechanism, but is not promoted. A future
latency attempt should reduce graph launches or denoiser compute and needs trace evidence
before another implementation A/B.

Reducing denoising steps is a separate, behavior-changing experiment. Ten to five steps
could remove roughly half of the historical 296 ms denoise cost, but it cannot be
certified with base-model synthetic actions. It needs a trained checkpoint, recorded
episodes, and eventual guarded robot validation.

The SmolVLA vision optimization does not transfer directly: an operator inventory of
both local X-VLA bundles found zero `IsNaN` nodes in all vision and denoise graphs and no
`IsNaN -> Where` attention guards to remove.

### X07. Memory headroom needs an integrated deployment measurement

Priority: **P1 before robot integration**

Status: **Fresh model-only headroom is tight; integrated measurement remains open**

The fine-tuned model-only run used about 6.22 GB process RSS and left only about 567 MB
of system RAM free while swap was already in use. That proves the one-camera split can
run, but not that the same process can safely host RealSense capture, preprocessing, the
controller, logging, and transient allocations without swap or latency spikes.

Run the next measurement with the real camera/control stack in dry-run mode. Record
PSS/RSS, system available memory, swap, power, thermals, completed throughput, and
p50/p95/p99 latency.

### Remaining fix and validation order

1. Preserve and synchronize the independently dirty Orin checkout; current work ran from
   an isolated package under `/tmp`.
2. Freeze the cleaned training target described by X03 before starting a long run.
3. Reuse the frozen real-IR fixture when X03 produces a trained candidate checkpoint.
4. Run X07 with the real camera/control stack in hardware-disconnected dry-run mode.
5. Treat fewer denoising steps and any live robot test as separate guarded decisions.

[ainekko-hf]: https://huggingface.co/ainekko/smolvla_base_onnx
[etars-export]: https://github.com/aifoundry-org/ETARS/blob/9ae33a75549a3385170ad968ac3f27878bf8d902/notebooks/smolVLA_export.ipynb
[lerobot-051]: https://github.com/huggingface/lerobot/releases/tag/v0.5.1
