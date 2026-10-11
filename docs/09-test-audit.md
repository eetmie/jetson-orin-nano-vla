# Test audit — 2026-10-10/11

Validation uses a fresh exporter clone on an aarch64 DGX Spark (Python 3.12.3)
and a fresh Git
clone at `~/vla-fresh-clone-20261010` on the Orin Nano Super 8 GB. The board's
original checkout and uncommitted experiments are preserved. This exercises the
existing JetPack 7.2.1 / L4T R39.2.1 system; it does not flash a new OS.

## Automated regression suite

Run from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

| File | Checks | Coverage and limits |
|---|---:|---|
| `test_core.py` | 15 | Percentiles, parity identity, finite actions, observation order, bounded benchmark work and atomic JSON writes. Backend substitutes do not test CUDA. |
| `test_evo1.py` | 6 | Trained/bootstrap model validity, bundle metadata, fixture shape and deterministic noise. No engine execution. |
| `test_trt_device.py` | 16 | Capture cleanup/failure, graph retirement, prompt/mask updates, camera changes, alternating chain buffers and real NumPy/OpenCV preprocessing. CUDA calls are mocked. |
| `test_boundary_rule.py` | 1 | FP16 engine boundaries: a consumer-side cast proves exactness, a producer-side cast alone does not. Synthetic ONNX. |
| `test_env_memory.py` | 1 | Framework versions are recorded without importing unused frameworks. |
| `test_smolvla_candidate_cache.py` | 4 | Changed engine/source, unsafe manifest names and legacy vision-manifest compatibility. Synthetic files. |
| `test_usage_setup.py` | 12 | Missing or changed weights, incomplete manifest coverage, unsafe paths, unsupported bundles, overwrite refusal, interpreter symlinks, isolated environment creation and pinned export command planning. Hub downloads are stubbed. |
| `test_model_setup.py` | 10 | Selected cache chains for all six models, retained failure logs, manifest coverage, trained EVO1/GR00T identity, pinned fetch and matching Transformers download/cache revisions. No GPU inference. |
| `test_groot_candidate_cache.py` | 7 | Changed graph/engine/timestep constants, metadata/embeddings/external weights, missing or foreign engines, bundle identity, stale native build keys and accumulation mismatch. Synthetic files. |
| `test_pi05_artifacts.py` | 14 | Trained-content identity, generation/size/SHA enforcement, strict HTTP ranges, transient retries, verified resume, overwrite refusal and model-flag conversion selection. Synthetic artifacts and network substitutes. |
| `test_http_download.py` | 3 | Interrupted range recovery, chunk hash rechecks and repair, pin/size identity and unsafe state refusal. Synthetic HTTP responses. |
| `test_hub_download.py` | 8 | Full SHA enforcement, Hub metadata identity, LFS/Git-blob checksums, credential boundaries, standard cached blob links, safe paths and preservation of changed files. Network substitutes. |

The original 41 checks are retained; the expanded suite has **97 checks**.
Fast regression tests do not prove installer completeness or GPU compatibility.
The hardware/export checks below supply that evidence.

The completed cold-download follow-up uses a separate empty download/cache
directory on the export host. Verification reports and negative attempts are retained in
[`results/cold-downloads-20261011/`](../results/cold-downloads-20261011/).
The six-model GPU evidence below remains the earlier completed audit.

## Fresh exports and installs

Four new exporter venvs install independently with complete dependency locks:
SmolVLA, X-VLA/EVO1, GR00T N1.6 and GR00T N1.7. Every installed module version
matches its constraints; both GR00T sparse source checkouts match their full commit
pins. pi0.5 uses a freshly built container pinned by its base-image digest, exact
additional dependency closure and openpi commit. Image-owned NVIDIA modules are
fixed by the digest and recorded in the evidence.

All six portable bundles were regenerated from stock checkpoints, including their
reference fixtures. In that first audit, raw model weights were reused after checksum/revision checks
to avoid repeated large downloads. Auxiliary Hub model loads use the pinned lock;
the fresh-cache export exposed and fixed a mismatch between downloading a SHA
snapshot and Transformers subsequently looking up `refs/main`. X-VLA
additionally needed BART's model config in its tokenizer download to retain special
token defaults. The corrected exporter checks a saved-tokenizer round trip; an
empty-cache check reproduces the stock fixture's padded token IDs exactly.

pi0.5's 16 original GCS checkpoint files are pinned by object generation, size and
SHA-256. In the first audit, the public download path was checked on small objects
and the start of a large object; full WAN downloads were reserved for the follow-up.
Its official JAX-to-PyTorch conversion was executed anew. All trained tensor bytes
match the earlier conversion; the unused expert language-output head can differ
and is excluded from the trained-content digest. Actual checkpoint file hashes
are recorded. Full padded/compact exports and five stock-policy fixture checks
ran in the new container. A variable-shadowing failure in compact preparation was
fixed and the complete export rerun successfully. The public Docker wrapper also
completed its full export and fixture gates from the verified converted checkpoint;
weights copy across bind mounts when hard links are unavailable.

On the Orin, the new runtime and builder venvs use independent pip caches and link
only JetPack's TensorRT Python package. Unrelated system Python modules cannot
satisfy missing imports. Both pass `pip check` and import/CUDA doctor probes. The
builder compiles an explicit SM87 AOT PTX probe without Torch. Native model
subprocesses disable Hub access; the board's new Hub cache remains empty.

## GPU checks

The setup builds fresh ordinary engines and selected Triton replacements from the
new exports. No historical TensorRT plans are copied. Each runtime checks the
stock-policy full action chunk, device/host chain parity and image conversion
before benchmarking. The parity and exact-scale gates are retained.

The fresh exports exposed two SmolVLA graph forms absent from the retained bundle:
a static `sqrt` attention scale and reshape dimensions assembled from constants.
The builder evaluates those expressions, including TensorRT's separate constant
`Unsqueeze` nodes, and still verifies their exact scale and index order. A static
check covers all 48 attention reshape expressions in the fresh ONNX graph. The
expert graph also retains nested constant-weight transposes; the FFN builder now
evaluates their actual matrix layout before packing the gate/up weights.
EVO1/X-VLA export scratch `_meta_*.json` files follow the shipping
manifest's existing exclusion rule; metadata used by the runtime, including
`bundle.json`, remains checksum-covered. pi0.5's standalone template builder now
adds the project root before importing the shared compiler adapter.

**All six models pass** their final fresh-export GPU checks. SmolVLA's final
expert/benchmark retry uses the verified ordinary/vision caches built in the same
fresh checkout; the evidence records those inherited stages and retains each
failed attempt. Final GPU results are recorded in the
[audit evidence](../results/usage-scaffolding-20261010/).
The short runs validate setup and parity; the README's existing 300-second
speed/RAM measurements remain the performance reference.

## Cold WAN follow-up — 2026-10-11

A separate empty directory and empty Hub/Xet caches test all primary policy
checkpoints and required auxiliary files at their pinned revisions. Cosmos uses
its gated config/processor/tokenizer files; BART uses tokenizer/config files;
unrelated BART/Cosmos weights and SmolVLM ONNX variants are unnecessary.

All 16 original pi0.5 GCS objects and its tokenizer have now been downloaded
anew and checked against generation/size/SHA locks. The cold conversion passes
the trained-tensor identity gate. All five full-policy fixture checks pass;
822 portable artifacts match the earlier Orin-tested bundle byte for byte.
The two changed metadata files differ only in the actual converted-file hash.
The complete public wrapper also passes download-resume, conversion, export and
fixture checks after the selector fix, reusing only verified files from this cold
audit. Two seeded conversions produce the same complete converted-file hash.

Cold transfers uncovered transient GCS disconnects and repeated Xet body-decoding
failures. Shared bounded HTTP transfers now validate range responses, retry
transient failures, and verify every full file before publication. pi0.5 resume
rechecks completed files and saved range hashes, then fetches missing or changed
ranges. Legacy partial files without range metadata restart. The pinned upstream
converter incorrectly selected adaptive norms by checkpoint path name; the
wrapper now uses the explicit model flag and fails if the pinned selector changes.

BART's newly downloaded tokenizer reproduces the stock X-VLA fixture IDs offline.
Cosmos config/processor loading also passes offline from a pinned local snapshot.
Using a remote repo ID caused the pinned tokenizer to request model metadata even
in offline mode; N1.7's exporter now passes the local snapshot path.
**The complete cold-download check passes:** all nine Hub snapshots (107 files),
16 GCS checkpoint objects and the tokenizer, totaling 37.42 GB of unique inputs.
An independent final rehash verifies every file again. The eight earlier
GPU-export input weight files match exactly; the ninth SmolVLM weight file also
matches its earlier auxiliary cache. No warm model bytes populated the cold run.

GR00T N1.7's updated public wrapper also completes a full export in another fresh
source clone using the cold checkpoint and cached Cosmos files. Its fixture
inputs are byte-identical to the earlier GPU-tested export. Float32 action
differences are at most 1.20e-6; metadata changes only the input paths. The
already freshly installed, pinned exporter venv was reused for this replay.

High connection counts exhausted some range retries; the recorded retries
retain verified chunks and use lower concurrency. N1.6's final large transfer
passes through the public downloader at its default 32-worker setting.
Logs and verification reports live in
[the cold evidence directory](../results/cold-downloads-20261011/).

## GR00T AOT follow-up — 2026-10-11

A separate checkout at `~/vla-groot-triton-20261011` uses the checksum-verified
fresh GR00T bundles and the isolated runtime/compiler environments from the
previous audit. Ordinary and custom vision/DiT engines are rebuilt on the same
Orin with matching FP32 accumulation, optimization level 2 and 512 MB workspace.
The runtime has neither Torch nor Triton; Triton 3.6.0 compiles serialized SM87
kernels in the builder environment. Both GR00T build doctors pass actual import,
CUDA and AOT compilation probes.

Vision-only, DiT-only and combined candidates pass the stock fixture, but the
vision replacements fail a wider 12-case action comparison and are excluded from
the selected recipe. The varied images, state and noise use fixed capture times
so N1.7 compares identical history frames. DiT-only passes all cases, including
N1.7's 14 used action dimensions. These comparisons measure drift from a matching
native TensorRT control; the bundle fixture separately checks the stock FP32
policy. They do not measure robot-task success.

Both public setup commands pass from new engine work directories (26 ordinary
engines and eight selected AOT engines each). Each fresh selected cache repeats
all 12 stress cases successfully; N1.7 used-dimension drift stays within 0.103%.
All 95 regression checks pass in the isolated Orin runtime as well as on the
desktop. The pinned venvs and verified exports from the earlier audit are reused.

The [GR00T evidence](../results/groot-triton-20261011/) retains both successful and
rejected candidates, operator sweeps and paired speed/RAM measurements. Candidate
wrappers verify graph, engine and timestep-constant hashes before loading;
builders refuse stale native source/stack keys and accumulation mismatches.

## GR00T follow-up — 2026-10-12

The vision replacements of the first round failed because their kernels rounded scores
to FP16 as the exported graph does, which is less accurate than TensorRT's own fused
attention. With FP32 scores, vision, causal LLM and DiT replacements together beat the
native engines against seven stock FP32 references per model (bundle fixture plus
`reference*.py --seed 1..6`, scored on the embodiment's used action dims). That stock
comparison, not drift from a native control, is now the acceptance check for GR00T
kernels ([evidence](../results/groot-triton-20261012/)). 97 regression checks pass.

## Which experiment tests are active

The setup directly uses the selected family builders and `run_candidate.py`
wrappers, with explicit bundle/cache paths. `experiments/` also contains optional
operator microbenchmarks, tokenizer comparisons, profiling and stress probes.
These are standalone tools and are not discovered by unittest; some historical
probes default to home-directory bundles and need explicit paths. Archived
`results/**/sources*` files preserve measured code and are not active tests.

The first retained-bundle audit and its failed/corrected builds remain archived.
Their older 52-test and SmolVLA-only claims describe that earlier run, not the
expanded all-model validation.

## Limits

A second physical Jetson is not required to find Python imports and engine-build
failures: isolated venvs, a new clone, empty Hub cache and fresh engine caches on
this board exercise those paths. Static/internet audits alone cannot establish
AOT compatibility or action parity. A freshly flashed matching JetPack image
would additionally test inherited native system libraries; that OS-level test
has not been performed. The optional PyTorch/ORT comparison environments have
captured dependency locks but have not been freshly installed in this round.
Robot-task success, live camera capture and robot control are outside these
synthetic benchmarking checks.
