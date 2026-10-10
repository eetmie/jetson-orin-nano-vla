"""CPU regressions for CUDA resource ownership and persistent prompt buffers.

These exercise the runtime's control flow with a fake CUDA API/device. Numerical
TensorRT parity and performance still need the Orin Nano and its engines.
"""

from __future__ import annotations

import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from bench.vendor.evo1_trt import causal_mask
from bench.vendor.trt_device import Device, Evo1Device, SmolVLADevice, XVLADevice


class FakeCuda:
    def __init__(self, end_rc=0, instantiate_rc=0):
        self.end_rc, self.instantiate_rc = end_rc, instantiate_rc
        self.source_graphs = set()
        self.executables = set()
        self.capturing = False

    def cudaStreamBeginCapture(self, stream, mode):
        self.capturing = True
        return 0

    def cudaStreamEndCapture(self, stream, output):
        self.capturing = False
        if self.end_rc == 0:
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 10
            self.source_graphs.add(10)
        return self.end_rc

    def cudaGraphInstantiate(self, output, graph, flags):
        if self.instantiate_rc == 0:
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 20
            self.executables.add(20)
        return self.instantiate_rc

    def cudaGraphDestroy(self, graph):
        self.source_graphs.remove(graph.value)
        return 0

    def cudaGraphExecDestroy(self, graph):
        self.executables.remove(graph.value)
        return 0


class GraphLifetimeTests(unittest.TestCase):
    def device(self, **kwargs):
        d = Device.__new__(Device)
        d.cu = FakeCuda(**kwargs)
        d.stream = ctypes.c_void_p(1)
        d.capturing = False
        return d

    def test_executable_survives_source_cleanup_then_can_be_retired(self):
        d = self.device()
        graph = d.capture(lambda: self.assertTrue(d.capturing))
        self.assertFalse(d.capturing)
        self.assertEqual(d.cu.source_graphs, set())
        self.assertEqual(d.cu.executables, {graph.value})
        d.destroy_graph(graph)
        d.destroy_graph(None)
        self.assertEqual(d.cu.executables, set())

    def test_instantiation_failure_releases_source(self):
        d = self.device(instantiate_rc=2)
        with self.assertRaisesRegex(RuntimeError, "cudaGraphInstantiate"):
            d.capture(lambda: None)
        self.assertEqual(d.cu.source_graphs, set())
        self.assertEqual(d.cu.executables, set())
        self.assertFalse(d.capturing)

    def test_enqueue_failure_ends_capture_and_releases_source(self):
        for end_rc in (0, 901):
            with self.subTest(end_rc=end_rc):
                d = self.device(end_rc=end_rc)
                enqueue = Mock(side_effect=ValueError("bad engine binding"))
                with self.assertRaisesRegex(ValueError, "bad engine binding"):
                    d.capture(enqueue)
                self.assertFalse(d.capturing)
                self.assertFalse(d.cu.capturing)
                self.assertEqual(d.cu.source_graphs, set())
                self.assertEqual(d.cu.executables, set())

    def test_invalidated_capture_is_reported_without_instantiation(self):
        d = self.device(end_rc=901)
        with self.assertRaisesRegex(RuntimeError, "cudaStreamEndCapture"):
            d.capture(lambda: None)
        self.assertEqual(d.cu.executables, set())
        self.assertFalse(d.capturing)


class Buffer:
    def __init__(self, shape=(1, 2, 2), dtype=np.float32):
        self.data = np.zeros(shape, dtype)
        self.dtype = self.data.dtype

    def host(self):
        return self.data


class FakeDevice:
    def __init__(self):
        self.uploads = []
        self.destroy_graph = Mock()
        self.capture = Mock(return_value=object())
        self.launch = Mock()
        self.d2d = Mock()
        self.enqueue = Mock()

    def upload(self, dst, arr):
        dst.data = np.array(arr, copy=True)
        self.uploads.append(dst)

    def download(self, src):
        pass

    def upload_staged(self, dst):
        self.uploads.append(dst)

    def sync(self):
        pass


class PromptTests(unittest.TestCase):
    def evo1(self):
        model = Evo1Device.__new__(Evo1Device)
        model.d = FakeDevice()
        model.b = SimpleNamespace(embed=np.arange(20).reshape(10, 2),
                                  hidden=2, image_token=7)
        model.vision_in, model.h0 = Buffer(), Buffer()
        model.mask, model.cmask, model.state = Buffer(), Buffer(), Buffer()
        model.actions = [Buffer(), Buffer()]
        model.time_index = [None, None]
        model.task_ids = model.task_mask = model.copies = None
        model.graph = None
        model.use_graph = True
        model._enqueue = Mock()
        return model

    def test_evo1_mask_change_refreshes_both_masks_without_recapture(self):
        model = self.evo1()
        ids = np.array([[7, 1, 0]], np.int64)
        mask = np.array([[True, True, False]])
        args = (np.zeros(1), ids, mask, np.zeros(1), np.zeros((1, 2, 2)))
        model.infer(*args)
        graph = model.graph
        count = model.d.uploads.count(model.h0)
        model.infer(*args)
        self.assertEqual(model.d.uploads.count(model.h0), count)
        # Mutation of the caller's existing array must invalidate the cached mask.
        mask[0, 1] = False
        model.infer(*args)
        np.testing.assert_array_equal(model.cmask.data, mask)
        np.testing.assert_array_equal(model.mask.data, causal_mask(mask))
        self.assertEqual(model.d.uploads.count(model.h0), count + 1)
        self.assertIs(model.graph, graph)
        self.assertEqual(model.d.capture.call_count, 1)

    def test_evo1_layout_change_retires_only_the_previous_executable(self):
        model = self.evo1()
        mask = np.ones((1, 3), bool)
        model._set_prompt(np.array([[7, 1, 0]]), mask)
        previous = model.graph
        model.d.destroy_graph.reset_mock()
        model._set_prompt(np.array([[1, 7, 0]]), mask)
        model.d.destroy_graph.assert_called_once_with(previous)
        self.assertEqual(model.d.capture.call_count, 2)

    def test_xvla_reuses_ids_but_detects_in_place_prompt_changes(self):
        model = XVLADevice.__new__(XVLADevice)
        model.d = FakeDevice()
        model.b = SimpleNamespace(gripper=0)
        model.vision_in, model.ids, model.proprio = Buffer(), Buffer(), Buffer()
        model.x1, model.action = Buffer(), Buffer()
        model.task_ids = None
        model.graph = object()
        ids = np.array([[1, 2, 0]], np.int64)
        args = (np.zeros(1), ids, np.zeros(1), np.zeros(1))
        model.infer(*args)
        model.infer(*args)
        self.assertEqual(model.d.uploads.count(model.ids), 1)
        ids[0, 1] = 3
        model.infer(*args)
        np.testing.assert_array_equal(model.ids.data, ids)
        self.assertEqual(model.d.uploads.count(model.ids), 2)
        self.assertEqual(model.d.uploads.count(model.vision_in), 3)

    def test_smolvla_no_key_refreshes_prompt_while_explicit_key_caches(self):
        model = SmolVLADevice.__new__(SmolVLADevice)
        model.d = FakeDevice()
        model.pix, model.state = [Buffer()], Buffer()
        model.x = [Buffer(), Buffer()]
        model.b = SimpleNamespace(num_steps=2)
        model.key, model.n_real = None, 1
        model.graph = object()
        model._set_contract = Mock()
        first = (np.zeros((1, 1, 2)), np.ones((1, 1), bool))
        second = (np.ones((1, 1, 2)), np.ones((1, 1), bool))
        args = ([np.zeros(1)], first, np.zeros(1), np.zeros((1, 2, 2)))
        model.infer(*args)
        model.infer(args[0], second, args[2], args[3])
        self.assertEqual(model._set_contract.call_count, 2)
        self.assertIs(model._set_contract.call_args.args[0], second)
        model.infer(*args, key="pick")
        model.infer(*args, key="pick")
        self.assertEqual(model._set_contract.call_count, 3)
        model.infer(*args, key="place")
        self.assertEqual(model._set_contract.call_count, 4)

    def smolvla_uint8(self):
        model = SmolVLADevice.__new__(SmolVLADevice)
        model.d = FakeDevice()
        model.pix = [Buffer((1, 3, 4, 4)), Buffer((1, 3, 4, 4))]
        model.canvas = [Buffer((4, 4, 3), np.uint8), Buffer((4, 4, 3), np.uint8)]
        model.to_pix = "op_siglip_u8"
        model.state, model.x = Buffer(), [Buffer(), Buffer()]
        model.b = SimpleNamespace(num_steps=2)
        model.key, model.n_real, model.graph = "pick", 2, object()
        model._set_contract = Mock()
        return model

    def test_smolvla_uint8_canvas_converts_on_gpu_and_float_does_not(self):
        model = self.smolvla_uint8()
        lang = (np.zeros((1, 1, 2)), np.ones((1, 1), bool))
        staged = model.canvas_buffer(0)
        staged[...] = 7
        given = np.full((4, 4, 3), 9, np.uint8)
        model.infer([staged, given], lang, np.zeros(1), np.zeros((1, 2, 2)), key="pick")
        # Camera 0 was written in place (no copy), camera 1 copied into its canvas.
        self.assertIs(model.d.uploads[0], model.canvas[0])
        np.testing.assert_array_equal(model.canvas[1].data, given)
        calls = [c.args for c in model.d.enqueue.call_args_list]
        self.assertEqual(calls, [("op_siglip_u8", {"x": model.canvas[0], "out": model.pix[0]}),
                                 ("op_siglip_u8", {"x": model.canvas[1], "out": model.pix[1]})])
        model.d.enqueue.reset_mock()
        model.infer([np.zeros((1, 3, 4, 4), np.float32)] * 2, lang, np.zeros(1),
                    np.zeros((1, 2, 2)), key="pick")
        model.d.enqueue.assert_not_called()
        self.assertIs(model.d.uploads[-4], model.pix[0])

    def test_siglip_lut_matches_host_conversion(self):
        from bench.vendor.smolvla_split import resize_pad_canvas, siglip_lut, siglip_normalize
        img = np.random.default_rng(3).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        canvas = resize_pad_canvas(img, 32)
        np.testing.assert_array_equal(siglip_lut()[canvas].transpose(2, 0, 1)[None],
                                      siglip_normalize(canvas))
        reused = np.full((32, 32, 3), 5, np.uint8)
        resize_pad_canvas(img, 32, out=reused)
        np.testing.assert_array_equal(reused, canvas)

    def test_smolvla_camera_change_retires_graph_but_prompt_change_reuses_it(self):
        model = SmolVLADevice.__new__(SmolVLADevice)
        model.d = FakeDevice()
        model.b = SimpleNamespace(n_cam_slots=2, prefix_len=130, chunk_size=2)
        model.prefix = Buffer((1, 130, 2))
        model.slots = [Buffer(), Buffer()]
        model.pad_cam = SimpleNamespace(nbytes=512)
        model.p_mask, model.p_pos = Buffer(), Buffer()
        model.d_mask, model.d_pos = Buffer(), Buffer()
        model.n_real = model.graph = None
        model.use_graph = True
        model._enqueue = Mock()
        lang = (np.ones((1, 1, 2)), np.ones((1, 1), bool))
        model._set_contract(lang, 1)
        previous = model.graph
        model.d.destroy_graph.reset_mock()
        model._set_contract((lang[0] * 2, lang[1]), 1)
        model.d.destroy_graph.assert_not_called()
        self.assertIs(model.graph, previous)
        model._set_contract(lang, 2)
        model.d.destroy_graph.assert_called_once_with(previous)
        self.assertEqual(model.d.capture.call_count, 2)


if __name__ == "__main__":
    unittest.main()
