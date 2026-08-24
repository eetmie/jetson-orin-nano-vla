"""What can be benchmarked: model families, their public artefacts, and their shapes.

The repo is about *what a VLA costs on this board*, not about any one policy. A model
is therefore a small spec — where its PyTorch weights live, where its split ONNX export
lives (if one exists), which tokenizer it needs, and the handful of shapes a benchmark
has to know before it can build an input. Everything else is read off the artefact
itself at load time, because the artefact is the authority: chunk length and camera
slot count are baked into the exported graphs and differ between exports of the same
model.

Two families are wired up.

`smolvla`  450 M. Vision + text + a Gemma-ish expert, prefilled once into a KV cache,
           then a flow-matching decode loop of `num_steps` Euler updates.
`xvla`     880 M. DaViT + BART, then a bidirectional policy transformer re-run in full
           on every denoising step — **no KV cache is possible**, because the
           conditioning tokens attend to the action tokens and change each step. Its
           loop is also not Euler integration: it re-forms `x_t` by interpolating a
           fixed noise draw against the current action estimate. Porting SmolVLA's
           update here yields plausible-looking garbage, which is exactly why the two
           families get separate runtimes rather than a shared "denoise loop".

Base weights produce **meaningless actions** — they were never fine-tuned on any robot
here. That is fine for latency, memory, CPU and power, which is what the base entries
are for, and it is why parity is always measured against another runtime of the *same*
weights rather than against any notion of task success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelSpec:
    key: str
    family: str                       # "smolvla" | "xvla"
    label: str
    params_m: float

    torch_repo: str | None = None     # HF id or local dir for the PyTorch checkpoint
    split_repo: str | None = None     # HF id for a ready-made split ONNX export
    split_dir: str | None = None      # local split export (wins over split_repo)
    tokenizer: str | None = None      # HF id; "bundle" = take it from the split export

    task: str = "pick up the object and place it in the box"
    chunk_size: int | None = None     # None -> read from the artefact
    num_steps: int | None = None
    state_dim: int = 6
    action_dim: int | None = None     # None -> the full padded width
    image_views: int = 1              # real cameras to synthesize
    #: Camera SLOTS the published export was built with. Not the same as real cameras:
    #: an unused slot still occupies its image tokens in the prefix, so PyTorch has to
    #: pad to the same count or the two runtimes compute different-length sequences.
    cam_slots: int | None = None

    notes: str = ""
    extras: dict = field(default_factory=dict)

    def resolved_split_dir(self) -> Path | None:
        return Path(self.split_dir).expanduser() if self.split_dir else None


REGISTRY: dict[str, ModelSpec] = {
    "smolvla-base": ModelSpec(
        key="smolvla-base",
        family="smolvla",
        label="SmolVLA 450M (base)",
        params_m=450.0,
        torch_repo="lerobot/smolvla_base",
        # The only public split ONNX of SmolVLA: the same nine graphs this repo's
        # split backend expects. Base weights, two camera slots (prefix 177).
        split_repo="ainekko/smolvla_base_onnx",
        # lerobot/smolvla_base ships NO tokenizer files. The vocab-exact one is the
        # VLM's own — a mismatched tokenizer means different token ids, a different
        # language embedding, and a policy conditioned on something it never saw.
        tokenizer="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        task="pick up the cube and place it in the box",
        chunk_size=50,
        num_steps=10,
        state_dim=6,
        image_views=1,
        cam_slots=2,
        notes="Reference split export has 2 camera slots; the runtime pads the unused "
              "one with the all -1 image, matching lerobot's convention.",
    ),
    "xvla-base": ModelSpec(
        key="xvla-base",
        family="xvla",
        label="X-VLA 0.9B (base)",
        params_m=879.7,
        torch_repo="lerobot/xvla-base",
        # No public ONNX export exists. Produce one with the split exporter — see
        # docs/03-backends.md. Apache 2.0, so the export is redistributable.
        split_repo=None,
        tokenizer="facebook/bart-large",
        task="pick up the cube and place it in the box",
        chunk_size=30,
        num_steps=10,
        state_dim=8,
        action_dim=20,
        image_views=1,
        notes="ee6d 20-dim action space, 3 declared image views. Feed only the real "
              "cameras: padded views are zeroed by the runtime and never need a "
              "forward pass, so one camera means a batch-1 vision engine.",
        extras={"action_mode": "ee6d", "num_image_views": 3, "requires_lerobot": "0.6.1"},
    ),
}


def get(key: str) -> ModelSpec:
    if key in REGISTRY:
        return REGISTRY[key]
    raise KeyError(f"unknown model {key!r}. Known: {', '.join(sorted(REGISTRY))}. "
                   f"For anything else pass --checkpoint / --bundle directly and set "
                   f"--family.")


def describe() -> str:
    rows = ["| key | family | params | torch | split ONNX |",
            "|---|---|---|---|---|"]
    for s in REGISTRY.values():
        rows.append(f"| `{s.key}` | {s.family} | {s.params_m:.0f} M | "
                    f"`{s.torch_repo or '—'}` | `{s.split_repo or '— (export it)'}` |")
    return "\n".join(rows)


def fetch(repo: str, subdir: str | None = None, cache: Path | None = None) -> Path:
    """Download an HF repo and return the local path.

    Large safetensors have been observed to stall here: the process stays alive, the
    file stops growing, no exception is raised, so `snapshot_download`'s own retry
    never fires. If that happens, `scripts/fetch_models.sh` uses the `hf` CLI, whose
    resume behaviour is better, and a curl fallback is documented there.
    """
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=repo,
        local_dir=str((cache or Path.home() / "bundles") / (subdir or repo.replace("/", "__"))),
        max_workers=4,
    )
    return Path(local)
