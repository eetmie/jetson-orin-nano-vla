# Audit follow-ups

Audit date: 2026-08-26

Audited commit: `609aef9` (`main`)

Scope: benchmark correctness, reproducibility, tests, performance opportunities, and
the `ainekko/smolvla_base_onnx` compatibility claim.

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
- `bench/backends/` contains Torch, split ORT, and monolithic ORT adapters.
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

### A04. The Ainekko result proves incompatibility, not export causality

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

> Treat `ainekko/smolvla_base_onnx` as incompatible and unvalidated for the repository's
> LeRobot 0.5.1 reference. Do not describe it as intrinsically defective until it has
> been tested against its exact export-era module and tensors.

The Spark experiment that settles this is specified below.

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
were removed because its monolithic path does not build on the 8 GB Orin Nano.
No Tether credentials or unsupported speed claims are persisted by this repo.

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
| public ONNX fails against its exact export-era module on CPU FP32 | defective export |
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

[ainekko-hf]: https://huggingface.co/ainekko/smolvla_base_onnx
[etars-export]: https://github.com/aifoundry-org/ETARS/blob/9ae33a75549a3385170ad968ac3f27878bf8d902/notebooks/smolVLA_export.ipynb
[lerobot-051]: https://github.com/huggingface/lerobot/releases/tag/v0.5.1
