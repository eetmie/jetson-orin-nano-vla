import numpy as np


def prepare_canvas(images, table):
    frame = np.concatenate(images, axis=1)
    x = frame[:, table["xi"], :].astype(np.float32)
    x = (x * table["xw"][None, :, :, None]).sum(axis=2)
    y = (x[table["yi"]] * table["yw"][:, :, None, None]).sum(axis=1)
    y = np.rint(y).clip(0, 255)
    h, w = table["canvas_hw"]
    y = np.pad(y, ((0, h - y.shape[0]), (0, w - y.shape[1]), (0, 0)), mode="reflect")
    return (y.transpose(2, 0, 1)[None, :, None] / 127.5 - 1).astype(np.float32)
