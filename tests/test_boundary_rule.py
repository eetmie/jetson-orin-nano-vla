"""FP16 engine boundaries: only a consumer-side proof counts (see boundary_report)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))

try:
    import onnx
    from onnx import TensorProto, helper
except ImportError:                      # the board's runtime venv has no onnx
    onnx = None


def _graph(path: Path, name: str, ins, outs, nodes) -> None:
    g = helper.make_graph(nodes, name, ins, outs)
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)]), str(path))


@unittest.skipIf(onnx is None, "onnx not installed")
class BoundaryRuleTests(unittest.TestCase):
    def bundle(self, consumer_casts: bool) -> Path:
        d = Path(tempfile.mkdtemp())
        f32 = lambda n: helper.make_tensor_value_info(n, TensorProto.FLOAT, [2])
        f16 = TensorProto.FLOAT16
        # producer: FP16 value cast up, the pattern TensorRT may fuse into FP32 epilogues
        _graph(d / "a.onnx", "a", [f32("x")], [f32("h_out")], [
            helper.make_node("Cast", ["x"], ["x16"], to=f16),
            helper.make_node("Cast", ["x16"], ["h_out"], to=TensorProto.FLOAT)])
        reader = [helper.make_node("Cast", ["h"], ["h16"], to=f16),
                  helper.make_node("Cast", ["h16"], ["y"], to=TensorProto.FLOAT)] \
            if consumer_casts else [helper.make_node("Relu", ["h"], ["y"])]
        _graph(d / "b.onnx", "b", [f32("h")], [f32("y")], reader)
        return d

    def test_consumer_side_cast_is_exact_producer_side_is_not(self):
        from vla_common.boundary_report import boundaries, exact

        for casts, want in ((True, "reads"), (False, None)):
            p, c = boundaries(self.bundle(casts))
            self.assertEqual(exact("h", p, c), want)


if __name__ == "__main__":
    unittest.main()
