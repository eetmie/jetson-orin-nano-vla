# π0.5 LIBERO (openpi)

For your own OpenPI checkpoint, use `./export.sh CHECKPOINT` with its observation recipe
and training configuration; see [custom OpenPI export](../README.md#openpi-checkpoints).
That path shares the refittable template exporter with LeRobot and records your own
weights and stock-policy fixtures. The command below retains the locked public recipe.

`./export.sh pi05-libero [converted-checkpoint] [out-dir]` builds openpi's pinned NGC
container, downloads the pinned public GCS checkpoint, converts it to PyTorch and
exports the compact FP16 bundle. Needs Docker with the NVIDIA container toolkit.

`out-dir` and `out-dir.work` must be new; the work directory keeps logs and the
reference stages. An interrupted download resumes with
`export/pi05/export.sh --resume-download <out-dir>`.

Pins: `source-lock.json` (openpi commit, container digest), `checkpoint-lock.json`
(object generations and SHA-256), `tokenizer-lock.json`, `constraints.txt`. An existing
converted checkpoint must match the pinned tensor digests.

The bundle holds seven refittable layer templates with each layer's FP16 weights, five
stock-policy fixtures and the task token table. Its 521-token prefix fits two cameras and
the fixture prompts; longer prompts need a larger prefix and new templates.
