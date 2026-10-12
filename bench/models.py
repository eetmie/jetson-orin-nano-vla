"""The models this repository runs, and the shapes a benchmark needs before loading one.

Chunk length, camera slots and the prompt are baked into each exported bundle and read
from it at load time; the values here are the base checkpoints' defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelSpec:
    key: str
    family: str                       # smolvla | xvla | evo1 | groot | pi05
    label: str
    checkpoint: str                   # what ./export.sh <key> exports by default
    tokenizer: str = "bundle"         # HF id, or "bundle": shipped in the export
    task: str = "pick up the object and place it in the box"
    chunk_size: int | None = None
    num_steps: int | None = None
    state_dim: int = 6
    action_dim: int | None = None     # None: the full padded width
    image_views: int = 1
    #: Camera slots the export was built with. An unused slot still occupies its image
    #: tokens in the prefix, so this can exceed the cameras actually fed.
    cam_slots: int | None = None
    noise_distribution: str = "normal"
    extras: dict = field(default_factory=dict)


REGISTRY: dict[str, ModelSpec] = {s.key: s for s in (
    ModelSpec(
        key="smolvla-base", family="smolvla", label="SmolVLA 450M (base)",
        checkpoint="lerobot/smolvla_base",
        # lerobot/smolvla_base ships no tokenizer; the VLM's own is the vocab-exact one.
        tokenizer="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        task="pick up the cube and place it in the box",
        chunk_size=50, num_steps=10, state_dim=6, image_views=2, cam_slots=2,
    ),
    ModelSpec(
        key="xvla-base", family="xvla", label="X-VLA 0.9B (base)",
        checkpoint="lerobot/xvla-base", tokenizer="facebook/bart-large",
        task="pick up the cube and place it in the box",
        chunk_size=30, num_steps=10, state_dim=8, action_dim=20, image_views=3,
        extras={"action_mode": "ee6d", "num_image_views": 3, "lang_len": 50},
    ),
    ModelSpec(
        key="evo1-libero", family="evo1", label="EVO1 775M (LIBERO)",
        checkpoint="zuoxingdong/evo1_libero",
        task="pick up the black bowl and place it on the plate",
        # Padded widths: LIBERO itself is 8-dim state, 7-dim action.
        chunk_size=50, num_steps=32, state_dim=24, action_dim=24,
        image_views=2, cam_slots=2, noise_distribution="uniform",
    ),
    ModelSpec(
        key="groot-n16-base", family="groot", label="GR00T N1.6 3B (base)",
        checkpoint="nvidia/GR00T-N1.6-3B",
        task="pick up the red cube and place it in the bowl",
        # Padded widths (max_state_dim / max_action_dim); embodiment robocasa_panda_omron.
        chunk_size=50, num_steps=4, state_dim=128, action_dim=128, image_views=3,
        extras={"license": "NVIDIA One-Way Noncommercial License"},
    ),
    ModelSpec(
        key="groot-n17-base", family="groot", label="GR00T N1.7 3B (base)",
        checkpoint="nvidia/GR00T-N1.7-3B",
        task="pick up the red cube and place it in the bowl",
        # Embodiment xdof_relative_eef_relative_joint: each camera now and 30 frames earlier.
        chunk_size=40, num_steps=4, state_dim=132, action_dim=132, image_views=3,
        extras={"license": "NVIDIA License (non-commercial use limitation)"},
    ),
    ModelSpec(
        key="pi05-libero", family="pi05", label="π0.5 LIBERO (openpi)",
        checkpoint="pi05-libero",
        task="pick up the red block",
        # pi05_libero has discrete_state_input=False: the state never reaches the model.
        chunk_size=10, num_steps=10, state_dim=8, action_dim=32, image_views=2,
    ),
    ModelSpec(
        key="pi05-base", family="pi05", label="π0.5 base (LeRobot)",
        checkpoint="lerobot/pi05_base",
        task="pick up the red block",
        chunk_size=50, num_steps=10, state_dim=32, action_dim=32, image_views=3,
    ),
)}


def get(key: str) -> ModelSpec:
    if key in REGISTRY:
        return REGISTRY[key]
    raise KeyError(f"unknown model {key!r}. Known: {', '.join(REGISTRY)}.")
