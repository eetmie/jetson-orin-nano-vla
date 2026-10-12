# 5. Benchmark runbook

Run one model process at a time because CPU and GPU share the same 8 GB memory pool.
The default observations are deterministic and in memory; no camera or robot is
needed. These measurements test inference cost and action parity, not robot-task
success.

## 1. Prepare and verify the board

On a fresh matching JetPack image, with SSH and Tailscale connected:

```bash
scripts/01_initial_setup.sh --inspect
sudo scripts/01_initial_setup.sh --apply --disable-docker --reboot-if-needed
# Reconnect after reboot, then check the actual state:
sudo scripts/01_initial_setup.sh --verify --disable-docker
```

The [host recipe](01-host-setup.md) checks headless boot, MAXN_SUPER GPU power
masks, all CPU/GPU/EMC clocks, 4 GiB swap and required services. Console auto-login
is unnecessary. A USB phone tether may need re-enabling after reboot.

## 2. Create the native environments

```bash
scripts/12_env_trt.sh
scripts/14_env_triton.sh
.venv-trt/bin/python -m bench selftest --seconds 4
```

pip remains the default installer. For the pinned uv experiment, install the
documented uv version and prefix both setup commands with `VLA_ENV_MANAGER=uv`;
see [environments](02-environments.md). Environment directories must be new.
The inference environment excludes Torch and Triton; the separate compiler
environment serializes Triton kernels into TensorRT engines.

## 3. Export and copy complete bundles

Run the [public export recipes](10-all-model-setup.md) on the export machine.
Copy each entire bundle, including external weights, tokenizers, fixtures and
`MANIFEST.sha256`, to a stable directory on the board. Use model-key subdirectory
names for the all-model command. TensorRT engines are built on the board rather
than copied from another machine.

EVO1 LIBERO's public recipe uses its trained action head. The older
`evo1-bootstrap` experiment is a separate nondeployable policy with random action
weights and must not be substituted for the trained checkpoint.

## 4. Build, validate and benchmark

```bash
python3 -m bench.tools.vla_setup --model smolvla-base \
    --bundle ~/vla-bundles/smolvla-base --work-dir ~/vla-runs/smolvla-new

# All six README workflow models, run serially:
python3 -m bench.tools.vla_setup --model all --bundles-dir ~/vla-bundles \
    --work-dir ~/vla-runs/all-new --keep-going

# The optional seventh model:
python3 -m bench.tools.vla_setup --model pi05-base \
    --bundle ~/vla-bundles/pi05-base --work-dir ~/vla-runs/pi05-base-new
```

Work directories must be new. Setup checks bundle checksums, GPU access and
stock-policy action fixtures, builds fresh native/AOT engines and runs each
final benchmark for 300 seconds. `--duration-s 10` is a functional smoke test.
Retain `setup.json`, stage logs, doctor reports, parity and benchmark JSONs;
`--model all` also writes `matrix.json`. A failed stage is recorded and causes
nonzero exit even when `--keep-going` tests remaining models.

If a heavy cold build fails from GPU allocation pressure, preserve the failed
logs, reboot to an idle headless board, verify the host and retry in a new work
directory. Four GiB swap is the tested configuration; see the measured
[build-memory limits](01-host-setup.md#swap--4-gb-and-build-on-a-freshly-booted-headless-board).

## Optional comparison runtimes

`scripts/run_all.sh` retains the PyTorch and ORT comparison recipes. Those use
`scripts/10_env_torch.sh`, `scripts/13_env_torch_xvla.sh` and
`scripts/11_env_ort.sh` where applicable, plus their matching checkpoints.
See [comparison runtimes](10-all-model-setup.md#comparison-runtimes).
For explicit native runs, use `.venv-trt/bin/python -m bench trt-split` with the
exported bundle; the selected setup recipes above also apply their AOT kernels.
Keep the stock model's denoising step count so the embedded action fixture remains
comparable.

## Preserve measurements

Use unique output paths. Each benchmark records validity, latency, memory, power,
thermals, clocks, stack versions and representative actions. To render a results
directory:

```bash
python3 -m bench report results --out docs/RESULTS.md
```

For the existing reference-board deployment:

```bash
python3 -m bench.tools.archive_remote_results
```

Its default source is `joel@192.168.8.230:~/jetson-orin-nano-vla/results/`.
Each invocation creates a UTC-stamped local snapshot with sizes, SHA-256 hashes
and recorded run times. Failed transfers are marked `INCOMPLETE`. For an isolated
test on another board, archive that test's explicit run directory instead.
Engines and bundles remain separate artifacts; snapshotting measurements cannot
recover an earlier overwritten file.
