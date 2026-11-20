from pathlib import Path
import numpy as np
import torch
from safetensors.numpy import load_file
from bench.obs import SyntheticObs
from lerobot.policies.cosmos3.processor_cosmos3 import (
    Cosmos3VideoCanvasProcessorStep as Canvas,
)

src = SyntheticObs(
    task="Pick up the cube.",
    chunk_size=16,
    state_dim=8,
    max_action_dim=64,
    h=256,
    w=256,
    n_views=2,
)
cam = Canvas(["a", "b"], (256, 256), 20, 16)
canvases = []
for i in range(16):
    x = torch.from_numpy(np.concatenate(src[i].images, axis=1)).permute(2, 0, 1)[
        None, None
    ]
    canvases.append(cam.fit_canvas(x)[0].permute(0, 2, 1, 3, 4).numpy())
stats = load_file("ckpt/policy_postprocessor_step_0_unnormalizer_processor.safetensors")
Path("pack").mkdir(exist_ok=True)


# Factor the fixed antialiased bicubic resize into its separable tap tables.
def taps(size, target):
    basis = torch.eye(size)[:, None, None, :].expand(-1, 1, 2, -1)
    weights = torch.nn.functional.interpolate(
        basis, (2, target), mode="bicubic", align_corners=False, antialias=True
    )[:, 0, 0, :].T
    rows = [row.nonzero().flatten() for row in weights]
    n = max(len(row) for row in rows)
    idx = np.zeros((target, n), np.int32)
    vals = np.zeros_like(idx, np.float32)
    for i, row in enumerate(rows):
        idx[i, : len(row)] = row.numpy()
        vals[i, : len(row)] = weights[i, row].numpy()
    return idx, vals


from rgb import prepare_canvas

xi, xw = taps(512, 320)
yi, yw = taps(256, 160)
tbl = dict(xi=xi, xw=xw, yi=yi, yw=yw, canvas_hw=np.array([192, 320]))
errors = []
for i, reference in enumerate(canvases):
    test = prepare_canvas(src[i].images, tbl)
    errors.append(np.max(abs(np.rint((test + 1) * 127.5) - reference.astype("f4"))))
print("largest pixel delta in 16 observations", max(errors))
assert max(errors) <= 1
np.savez("pack/observations.npz", **tbl, lo=stats["action.q01"], hi=stats["action.q99"])
