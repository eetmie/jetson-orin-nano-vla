"""What can be benchmarked: model families, their base artefacts, and their shapes.

The repo is about *what a VLA costs on this board*, not about any one policy. A model
is therefore a small spec — where its PyTorch weights live, where its split ONNX export
lives (if one exists), which tokenizer it needs, and the handful of shapes a benchmark
has to know before it can build an input. Everything else is read off the artefact
itself at load time, because the artefact is the authority: chunk length and camera
slot count are baked into the exported graphs and differ between exports of the same
model.

Two deployable base families and one EVO1 bootstrap profile are wired up.

`smolvla`  450 M. Vision + text + a Gemma-ish expert, prefilled once into a KV cache,
           then a flow-matching decode loop of `num_steps` Euler updates.
`xvla`     880 M. DaViT + BART, then a bidirectional policy transformer re-run in full
           on every denoising step — **no KV cache is possible**, because the
           conditioning tokens attend to the action tokens and change each step. Its
           loop is also not Euler integration: it re-forms `x_t` by interpolating a
           fixed noise draw against the current action estimate. Porting SmolVLA's
           update here yields plausible-looking garbage, which is exactly why the two
           families get separate runtimes rather than a shared "denoise loop".
`evo1`     775 M in the current bootstrap export. InternVL3 vision/language stages feed
           a cached action context and a 32-step Euler flow loop. The present action
           head is deterministic random initialization, so it is an infrastructure
           benchmark only and is explicitly rejected for robot control.

Base weights are not task-specific robot policies. They are used here only for latency,
memory, CPU, power, and same-weight runtime parity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelSpec:
    key: str
    family: str                       # "smolvla" | "xvla" | "evo1"
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
    noise_distribution: str = "normal"

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
        # Verified base export used for the retained Orin Nano result: nine graphs,
        # two camera slots, 177-token prefix.
        split_repo="eetmie/smolvla-base-onnx",
        # lerobot/smolvla_base ships NO tokenizer files. The vocab-exact one is the
        # VLM's own — a mismatched tokenizer means different token ids, a different
        # language embedding, and a policy conditioned on something it never saw.
        tokenizer="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        task="pick up the cube and place it in the box",
        chunk_size=50,
        num_steps=10,
        state_dim=6,
        # Native = fill the slots the published export was BUILT with. This repo
        # measures models as shipped, not as a particular robot would tune them; a
        # padded slot costs no vision pass, so feeding fewer cameras than the export
        # declares measures a configuration nobody published. Drop to --views 1 to see
        # what a camera costs -- that is a finding, not the default.
        image_views=2,
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
        # Verified base export used for the retained Orin Nano result: twelve graphs,
        # three image views and the full ten-step denoising loop.
        split_repo="eetmie/xvla-base-onnx",
        tokenizer="facebook/bart-large",
        task="pick up the cube and place it in the box",
        chunk_size=30,
        num_steps=10,
        state_dim=8,
        action_dim=20,
        # Native: the checkpoint declares num_image_views=3. See the note on
        # smolvla-base above -- --views 1 is the interesting variation, not the default.
        image_views=3,
        notes="ee6d 20-dim action space, 3 declared image views. Feed only the real "
              "cameras: padded views are zeroed by the runtime and never need a "
              "forward pass, so one camera means a batch-1 vision engine.",
        extras={"action_mode": "ee6d", "num_image_views": 3, "lang_len": 50,
                "requires_lerobot": "0.6.1"},
    ),
    "evo1-bootstrap": ModelSpec(
        key="evo1-bootstrap",
        family="evo1",
        label="EVO1 775M (nondeployable bootstrap)",
        params_m=775.2,
        torch_repo=None,
        split_repo=None,
        tokenizer="bundle",
        task="move sand to the container",
        chunk_size=50,
        num_steps=32,
        state_dim=24,
        action_dim=24,
        image_views=1,
        cam_slots=1,
        noise_distribution="uniform",
        notes="Infrastructure benchmark only: deterministic random action head; never "
              "use its actions to control a robot.",
        extras={
            "deployable": False,
            "requires_lerobot": "0.6.1",
            "vlm_base": "OpenGVLab/InternVL3-1B-hf",
            "vlm_revision": "014c0583a0d4bedf29fbe2dbff4f865eb998e171",
        },
    ),
    "evo1-libero": ModelSpec(
        key="evo1-libero",
        family="evo1",
        label="EVO1 775M (LIBERO)",
        params_m=775.2,
        # The LeRobot-format LIBERO checkpoint, and the reason this entry exists: it
        # replaces evo1-bootstrap's randomly initialized action head with trained
        # weights, so an EVO1 number here finally means something about a policy
        # rather than only about the plumbing.
        #
        # NOT MINT-SJTU/Evo1_LIBERO. That repository ships the author's DeepSpeed
        # checkpoint -- mp_rank_00_model_states.pt plus norm_stats.json, no
        # model.safetensors and no processor configs -- so it is not loadable as a
        # LeRobot PreTrainedPolicy and cannot be exported by this pipeline without a
        # conversion step that does not exist yet. Same for every MINT-SJTU Evo1_* and
        # EVO-Depth-* artifact.
        torch_repo="zuoxingdong/evo1_libero",
        split_repo=None,
        tokenizer="bundle",
        task="pick up the black bowl and place it on the plate",
        chunk_size=50,
        num_steps=32,
        # The checkpoint declares max_state_dim/max_action_dim 24 and pads into them:
        # LIBERO itself is 8-dim state and 7-dim action. The padded width is what both
        # runtimes must agree on, so it is what is recorded here.
        state_dim=24,
        action_dim=24,
        # config.json input_features names two cameras, observation.images.image and
        # .image2, against an architectural max_views of 3. Two is what was trained.
        image_views=2,
        cam_slots=2,
        noise_distribution="uniform",
        notes="Trained LIBERO policy, unlike evo1-bootstrap. Actions are meaningful "
              "for LIBERO's own embodiment only.",
        extras={
            "deployable": True,
            "requires_lerobot": "0.6.1",
            "vlm_base": "OpenGVLab/InternVL3-1B-hf",
            "vlm_revision": "014c0583a0d4bedf29fbe2dbff4f865eb998e171",
            "trained_on": "lerobot/libero",
        },
    ),
    "evo-depth-libero": ModelSpec(
        key="evo-depth-libero",
        family="evo1",
        label="EVO-Depth (LIBERO) — no LeRobot-format weights published",
        params_m=775.2,
        # Registered so it is a known target rather than a thing to rediscover, but it
        # CANNOT BE RUN TODAY and no amount of pipeline work changes that on its own.
        # Every published EVO-Depth artifact is the author's DeepSpeed format:
        #   MINT-SJTU/EVO-Depth-LIBERO     four sub-checkpoints (libero_10, _goal,
        #                                  _object, _spatial), each mp_rank_00_model_
        #                                  states.pt + norm_stats.json
        #   MINT-SJTU/EVO-Depth-MetaWorld  same shape
        #   MINT-SJTU/EVO-Depth-Arena      same shape
        #   liujiting/Evo-depth            README only, no weights at all
        # There is no model.safetensors and no policy_preprocessor.json anywhere, so
        # Evo1Policy.from_pretrained has nothing to load. Unblocking it needs a
        # DeepSpeed -> LeRobot conversion, and a depth observation contract that this
        # repository's synthetic RGB observations do not currently produce.
        torch_repo=None,
        split_repo=None,
        tokenizer="bundle",
        task="pick up the black bowl and place it on the plate",
        chunk_size=50,
        num_steps=32,
        state_dim=24,
        action_dim=24,
        image_views=2,
        cam_slots=2,
        noise_distribution="uniform",
        notes="BLOCKED: no LeRobot-format checkpoint exists. Needs a DeepSpeed "
              "conversion and a depth observation channel before it can be measured.",
        extras={
            "deployable": False,
            "blocked": "no LeRobot-format weights published; DeepSpeed author format "
                       "only, and depth observations are not implemented",
            "requires_lerobot": "0.6.1",
            "candidate_repos": [
                "MINT-SJTU/EVO-Depth-LIBERO",
                "MINT-SJTU/EVO-Depth-MetaWorld",
                "MINT-SJTU/EVO-Depth-Arena",
            ],
        },
    ),
}


def get(key: str) -> ModelSpec:
    if key in REGISTRY:
        return REGISTRY[key]
    raise KeyError(
        f"unknown model {key!r}. Known: {', '.join(sorted(REGISTRY))}.")


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
