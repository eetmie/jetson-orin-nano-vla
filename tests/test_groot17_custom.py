"""CPU regressions for custom GR00T N1.7 bundles: one-frame history, AOT shape checks,
labels, camera count, zero RTC inputs and the device-chain repeat gate. Engine numerics
still need the board."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from bench.cli import Resolved
from bench.model_contract import aot_unavailable, display_name
from bench.vendor.groot17_trt import Bundle
from bench.vendor.trt_device import Groot17Device

# The published base bundles' values (vla-clean-install-20261012/bundles).
BASE17 = {"schema": "groot-n1.7-split/2", "model": "nvidia/GR00T-N1.7-3B", "views": 3,
          "frames": 2, "image_hw": [256, 256], "action_horizon": 40, "seq_len": 448,
          "kv_rows": {"text": 28, "image": 384}, "task": "pick up the red cube",
          "checkpoint": "/x/export/.checkpoints/nvidia--GR00T-N1.7-3B--dcd885bb6820f298"}
BASE16 = {"schema": "groot-n1.6-split/2", "model": "nvidia/GR00T-N1.6-3B", "views": 3,
          "image_hw": [252, 252], "action_horizon": 50, "seq_len": 320,
          "kv_rows": {"text": 48, "image": 243},
          "checkpoint": "/x/export/.checkpoints/nvidia--GR00T-N1.6-3B--40a3ac89bf96d463"}
# An excavator fine-tune: two cameras, one frame, prompt K/V not gathered.
CUSTOM17 = {**BASE17, "views": 2, "frames": 1, "seq_len": 256, "kv_rows": None,
            "checkpoint": "/home/u/groot/runs/digging640_rtc12"}


class Groot17FrameTests(unittest.TestCase):
    def device(self, frames: int, views: int = 2):
        dev = Groot17Device.__new__(Groot17Device)
        dev.one_frame = frames == 1
        dev.now, dev.past = ["now"], ([] if frames == 1 else ["past"])
        dev.h0, dev.ds_full = "h0", []
        dev.copies = [(0, k * 8, k // views, 0, 8) for k in range(views * frames)]
        dev.b = SimpleNamespace(llm=[], cond=["cond_0"], graph={})
        dev.llm_out, dev.cond_out = [], [{"vl": "vl"}]
        dev.state, dev.pad_bias, dev.ev = "state", "pad", [None] * 4
        dev.dit = Mock()
        dev.d = Mock()
        return dev

    def sources(self, dev):
        dev._rest()
        return [c.args[1] for c in dev.d.d2d.call_args_list]

    def test_one_frame_reads_the_current_encode(self):
        self.assertEqual(self.sources(self.device(frames=1)), ["now", "now"])

    def test_two_frames_read_history_first(self):
        self.assertEqual(self.sources(self.device(frames=2)), ["past", "past", "now", "now"])

    def test_one_frame_never_touches_the_history(self):
        dev = self.device(frames=1)
        dev.history, dev.graphs = Mock(), None
        dev._vision, dev._rest = Mock(), Mock()
        dev.vision_in = "pixels"
        dev.dit.actions, dev.dit.result.host.return_value = ["noise"], np.zeros(3)
        _, age = dev.infer(np.zeros(1), np.zeros(1), np.zeros(1), t_capture=5.0)
        self.assertEqual(age, 0.0)
        dev.history.push.assert_not_called()
        dev.d.d2d.assert_not_called()


class HostChainFrameTests(unittest.TestCase):
    def frames_passed(self, frames: int):
        from bench.backends.trt_split_groot import TrtSplitGrootBackend
        from bench.vendor.groot17_trt import FrameHistory

        be = TrtSplitGrootBackend.__new__(TrtSplitGrootBackend)
        be.bundle = SimpleNamespace(b={"frames": frames, "state_dim": 4},
                                    preprocess=lambda im: np.zeros((3, 2, 2), np.float32))
        be.engines, be.device = SimpleNamespace(run=None), None
        be.history, be.history_ages_ms = FrameHistory(0.0), []
        obs = SimpleNamespace(images=[np.zeros((2, 2, 3), np.uint8)], state=np.zeros(4),
                              noise=np.zeros((1, 2, 4), np.float32))
        timings = {"backbone": 1.0, "denoise": 1.0}
        with patch("bench.vendor.groot17_trt.encode_frames", return_value="enc"), \
             patch("bench.vendor.groot17_trt.infer",
                   side_effect=lambda b, r, f, s, n, timings: timings.update(
                       backbone=1.0, denoise=1.0) or np.zeros((1, 2, 4))) as infer:
            be._infer17(obs)
        return infer.call_args.args[2]

    def test_one_frame_bundle_gets_one_frame(self):
        self.assertEqual(self.frames_passed(1), ["enc"])

    def test_two_frame_bundle_gets_history_and_current(self):
        self.assertEqual(self.frames_passed(2), ["enc", "enc"])


class RepeatabilityGateTests(unittest.TestCase):
    def test_leaking_device_chain_is_refused(self):
        from bench.backends.trt_split_groot import _check_device

        parity = {"threshold": 0.9999, "max_pct_range_threshold": 0.5}
        ok = {"cosine": 1.0, "max_pct_range": 0.1}
        _check_device(parity, {"repeatable": True}, ok)
        with self.assertRaisesRegex(ValueError, "unrelated observation"):
            _check_device(parity, {"repeatable": False}, ok)
        with self.assertRaisesRegex(ValueError, "fails the fixture"):
            _check_device(parity, {"repeatable": True}, {"cosine": 0.9, "max_pct_range": 0.1})


class AotShapeTests(unittest.TestCase):
    def test_base_bundles_keep_their_kernels(self):
        self.assertIsNone(aot_unavailable(None, BASE17))
        self.assertIsNone(aot_unavailable(None, BASE16))

    def test_camera_count_and_prompt_length_follow_the_bundle(self):
        self.assertIsNone(aot_unavailable(None, CUSTOM17))      # board: rtc12 bundle, parity PASS

    def test_other_shapes_fall_back_with_a_reason(self):
        self.assertIn("action_horizon 40",
                      aot_unavailable(None, {**BASE17, "action_horizon": 16}))
        self.assertIn("image_hw [256, 256]",
                      aot_unavailable(None, {**CUSTOM17, "image_hw": [224, 224]}))

    def test_other_legacy_bundles_are_untouched(self):
        self.assertIsNone(aot_unavailable(None, {"valid_views": 3}))


class LabelTests(unittest.TestCase):
    def test_base_checkpoints_keep_the_profile_name(self):
        self.assertEqual(display_name("groot-n17-base", BASE17), "groot-n17-base")
        self.assertEqual(display_name("groot-n16-base", BASE16), "groot-n16-base")

    def test_custom_checkpoint_is_named_after_its_directory(self):
        self.assertEqual(display_name("groot-n17-base", CUSTOM17), "groot-n17:digging640_rtc12")

    def test_saved_specification_and_other_families_win(self):
        self.assertEqual(display_name("groot-n17-base", CUSTOM17, {"name": "mine"}), "mine")
        self.assertEqual(display_name("xvla-base", {"checkpoint": "/a/b"}), "xvla-base")


class BundleFileTests(unittest.TestCase):
    def bundle(self, info: dict, dit_inputs=("actions",)) -> Path:
        root = Path(tempfile.mkdtemp())
        np.save(root / "embed.npy", np.zeros((8, 4), np.float16))
        (root / "bundle.json").write_text(json.dumps({
            **info, "embed_tokens": "embed.npy", "input_ids": [1, 7, 7, 2, 0],
            "prompt_tokens": 4, "image_token": 7, "mask_neg": -1e4, "action_dim": 132,
            "graphs": [{"name": "dit_0", "inputs": list(dit_inputs), "outputs": []}]}))
        return root

    def test_groot_bundle_sets_the_camera_count(self):
        root = self.bundle(CUSTOM17)
        args = SimpleNamespace(model=None, task=None, views=None)
        with patch("bench.run.detect_model", return_value="groot-n17-base"):
            r = Resolved(args, root)
        self.assertEqual(r.views, 2)
        self.assertEqual(r.model_name, "groot-n17:digging640_rtc12")

    def test_rtc_inputs_are_fed_as_zeros_only_when_the_graphs_take_them(self):
        rtc = Bundle(self.bundle(CUSTOM17, ("actions", "rtc_mask", "prefix")))
        self.assertEqual({k: v.shape for k, v in rtc.rtc_off.items()},
                         {"rtc_mask": (1, 40, 1), "prefix": (1, 40, 132)})
        self.assertFalse(any(v.any() for v in rtc.rtc_off.values()))
        self.assertEqual(Bundle(self.bundle(CUSTOM17)).rtc_off, {})


if __name__ == "__main__":
    unittest.main()
