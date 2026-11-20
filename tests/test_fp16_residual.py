"""fp16_mixed --fp32-residual: the residual Adds are found, the bias Adds are not."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))

try:
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper
except ImportError:                      # the board's runtime venv has no onnx
    helper = None


@unittest.skipIf(helper is None, "onnx not installed")
class ResidualAdds(unittest.TestCase):
    def test_stream_through_casts(self):
        from vla_common.fp16_mixed import residual_adds

        w = numpy_helper.from_array(np.eye(4, dtype=np.float32), "w")
        b = numpy_helper.from_array(np.zeros(4, np.float32), "b")
        nodes = [
            # x -> norm -> linear(+bias) -> residual add -> (Cast) -> norm -> linear -> residual add
            helper.make_node("LayerNormalization", ["x", "b", "b"], ["n0"], name="ln0"),
            helper.make_node("MatMul", ["n0", "w"], ["m0"], name="mm0"),
            helper.make_node("Add", ["m0", "b"], ["bias0"], name="bias0"),
            helper.make_node("Add", ["x", "bias0"], ["r0"], name="res0"),
            helper.make_node("Cast", ["r0"], ["c0"], to=TensorProto.FLOAT, name="cast0"),
            helper.make_node("Pow", ["c0", "b"], ["p0"], name="pow0"),
            helper.make_node("MatMul", ["p0", "w"], ["m1"], name="mm1"),
            helper.make_node("Add", ["c0", "m1"], ["y"], name="res1"),
        ]
        vi = lambda n: helper.make_tensor_value_info(n, TensorProto.FLOAT, [2, 4])
        g = helper.make_graph(nodes, "g", [vi("x")], [vi("y")], [w, b])
        self.assertEqual(sorted(residual_adds(helper.make_model(g))), ["res0", "res1"])


if __name__ == "__main__":
    unittest.main()
