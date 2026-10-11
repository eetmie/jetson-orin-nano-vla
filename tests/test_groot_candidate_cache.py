"""Prevent GR00T from running engines or timestep constants changed after validation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path=Path(__file__).resolve().parents[1]/'experiments/groot_triton/candidate.py'
spec=importlib.util.spec_from_file_location('groot_candidate_cache',path)
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)


class GrootCandidateCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.bundle=self.root/'bundle';self.cache=self.root/'cache'
        self.bundle.mkdir();self.cache.mkdir()
        (self.bundle/'dit_0.onnx').write_bytes(b'pinned graph')
        (self.cache/'dit_0.engine').write_bytes(b'pinned engine')
        (self.cache/'step_constants.npz').write_bytes(b'pinned modulation')
        (self.bundle/'bundle.json').write_text('{}')
        (self.bundle/'embedding_fp16.npy').write_bytes(b'pinned embedding')
        self.manifest=dict(bundle=str(self.bundle),accumulate='fp32',**candidate.pin(self.cache,self.bundle))
        (self.cache/'candidate.json').write_text(json.dumps(self.manifest))

    def test_validated_cache_is_accepted(self):
        self.assertEqual(candidate.verify(self.cache,self.bundle),self.manifest)

    def test_changed_graph_engine_and_modulation_are_rejected(self):
        for path,key in [(self.bundle/'dit_0.onnx','source_sha256'),
                         (self.cache/'dit_0.engine','all_engine_sha256'),
                         (self.cache/'step_constants.npz','constants_sha256')]:
            with self.subTest(path=path.name):
                original=path.read_bytes();path.write_bytes(b'changed content')
                with self.assertRaisesRegex(ValueError,key):candidate.verify(self.cache,self.bundle)
                path.write_bytes(original)

    def test_changed_metadata_embeddings_and_external_weights_are_rejected(self):
        for path in [self.bundle/'bundle.json', self.bundle/'embedding_fp16.npy',
                     self.bundle/'dit_0.onnx.data']:
            with self.subTest(path=path.name):
                original = path.read_bytes() if path.exists() else None
                path.write_bytes(b'changed auxiliary input')
                with self.assertRaisesRegex(ValueError, 'bundle_artifact_sha256'):
                    candidate.verify(self.cache, self.bundle)
                if original is None: path.unlink()
                else: path.write_bytes(original)

    def test_missing_or_added_engines_are_rejected(self):
        engine=self.cache/'dit_0.engine';original=engine.read_bytes();engine.unlink()
        with self.assertRaisesRegex(ValueError,'all_engine_sha256'):candidate.verify(self.cache,self.bundle)
        engine.write_bytes(original)
        (self.cache/'vision_0.engine').write_bytes(b'foreign engine')
        with self.assertRaisesRegex(ValueError,'all_engine_sha256'):candidate.verify(self.cache,self.bundle)

    def test_requested_precision_cannot_misreport_the_pinned_engines(self):
        candidate.validate_accumulate(self.manifest, ['--accumulate', 'fp32'])
        with self.assertRaisesRegex(ValueError, 'accumulation differs'):
            candidate.validate_accumulate(self.manifest, ['--accumulate', 'auto'])

    def test_stale_native_engine_sources_and_settings_are_rejected(self):
        (self.cache / 'candidate.json').unlink()
        (self.bundle / 'bundle.json').write_text(json.dumps({'graphs': [{'name': 'dit_0'}]}))
        stack = 'tensorrt=10.16 cudart=13020 opt_level=2 workspace_mb=512 strongly_typed accumulate=fp32'
        key = self.cache / 'dit_0.onnx.sha256'
        key.write_text(f"{candidate.sha256(self.bundle / 'dit_0.onnx')} {stack}\n")
        candidate.validate_base(self.cache, self.bundle, 'fp32', stack)
        for wrong in [stack.replace('fp32', 'auto'), stack.replace('13020', '13010')]:
            with self.assertRaisesRegex(ValueError, 'stale'):
                candidate.validate_base(self.cache, self.bundle, 'auto', wrong)
        (self.bundle / 'dit_0.onnx').write_bytes(b're-exported graph')
        with self.assertRaisesRegex(ValueError, 'stale'):
            candidate.validate_base(self.cache, self.bundle, 'fp32', stack)

    def test_same_bytes_from_another_bundle_are_rejected(self):
        other=self.root/'other';other.mkdir();(other/'dit_0.onnx').write_bytes(b'pinned graph')
        with self.assertRaisesRegex(ValueError,'belongs to'):candidate.verify(self.cache,other)


if __name__=='__main__':unittest.main()
