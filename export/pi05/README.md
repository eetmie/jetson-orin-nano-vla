# pi0.5 LIBERO export

`export/export.sh pi05-libero OUT` builds the pinned NGC container, downloads the
pinned public GCS checkpoint, converts it to PyTorch, and exports a complete
compact FP16 bundle for `bench.tools.vla_setup --model pi05-libero`.
Run on a Linux NVIDIA GPU export machine; build native engines on the Orin.

```bash
export/export.sh pi05-libero "$HOME/vla-bundles/pi05-libero"
# Or use existing converted weights (trained tensor digest must match):
export/pi05/export.sh "$HOME/vla-bundles/pi05-libero" /path/to/pi05_libero_pytorch
```

`OUT` and `OUT.work` must be new. Logs, the container image ID and intermediate
reference stages are retained in `OUT.work`. No historical experiment directory,
manual weight copy, or additional `third_party/openpi` checkout is required.
Interrupted remote downloads before conversion can resume with:

```bash
export/pi05/export.sh --resume-download "$HOME/vla-bundles/pi05-libero"
```

Resume rechecks completed files against their pinned SHA-256 and verifies saved
range hashes before fetching missing or changed chunks. Legacy partial files
without range metadata restart. Large files use bounded HTTP ranges with transient retries;
only complete checksum-verified files are published. A changed completed file
is rejected and preserved. Completed bundles are never overwritten.

Pins are in `source-lock.json`, `checkpoint-lock.json`, `tokenizer-lock.json` and
`constraints.txt`: upstream commit, immutable container digest, checkpoint object
generations and SHA-256, tokenizer generation and SHA-256, and all Python modules.
The raw checkpoint downloads use the [GCS generation selector](https://cloud.google.com/storage/docs/json_api/v1/objects/get).
The converter selects adaptive normalization from the explicit model flag,
so the local directory name does not affect the converted tensors.
The seeded conversion also fixes initialization of the unused expert language
output head. That head is absent from the trained JAX checkpoint and is never
used for actions. The converted-weight gate checks every other named tensor's
dtype, shape and bytes, plus tied-weight metadata, against the pinned reference.
The actual complete converted-file checksum is recorded in the exported bundle.

The pipeline creates five seeded stock mixed BF16/FP32 policy fixtures, checks
cached versus live FP16 conditioning, exports all model layers into a padded
baseline, then recomputes the compact reference with masked-view skipping and
cleared padding. A separately recomputed gathered reference checks those changes
bit for bit. The compact-versus-padded diagnostic retains its cosine and absolute
error gates. It writes the task token table and hashes every portable artifact.

The final bundle contains seven layer-template kinds, 67 sets of component
weights, memory-mapped FP16 embeddings, normalization/conditioning/RoPE tables,
five full-policy fixtures and the task token table. All 27 vision, 18 language
and 18 action layers remain. The 521-token prefix fits two cameras plus the
fixture prompts; longer prompts require a larger prefix and new templates.

See [all-model setup](../../docs/10-all-model-setup.md) for board commands and
[test audit](../../docs/09-test-audit.md) for validation evidence. The model is
Physical Intelligence's `pi05_libero`; consult its upstream license before
redistributing weights.
