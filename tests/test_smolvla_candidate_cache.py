"""Experimental prebuild bypass must reject changed engines and source graphs."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.smolvla_triton.candidate_cache import sha256, verify_candidate

class CandidateCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.bundle, self.cache = root/'bundle', root/'cache'
        self.bundle.mkdir()
        self.cache.mkdir()
        for name in ['smolvlm_vision.onnx','smolvlm_expert_decode.onnx']:
            (self.bundle/name).write_bytes(name.encode())
        for name in ['vision.engine','decode.engine']:
            (self.cache/name).write_bytes(name.encode())
        self.manifest = dict(source_onnx_sha256=sha256(self.bundle/'smolvlm_vision.onnx'),
            engine_sha256=sha256(self.cache/'vision.engine'),
            all_engine_sha256={p.name:sha256(p) for p in self.cache.glob('*.engine')},
            additional_source_sha256={'smolvlm_expert_decode.onnx':sha256(self.bundle/'smolvlm_expert_decode.onnx')})
        self.write_manifest()

    def write_manifest(self):
        (self.cache/'candidate.json').write_text(json.dumps(self.manifest))

    def test_rejects_changed_expert_engine(self):
        verify_candidate(self.cache,self.bundle)
        (self.cache/'decode.engine').write_bytes(b'changed engine')
        with self.assertRaisesRegex(ValueError,'engine digest changed: decode.engine'):
            verify_candidate(self.cache,self.bundle)

    def test_rejects_changed_expert_source(self):
        (self.bundle/'smolvlm_expert_decode.onnx').write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError,'ONNX digest changed: smolvlm_expert_decode.onnx'):
            verify_candidate(self.cache,self.bundle)

    def test_rejects_paths_outside_cache(self):
        self.manifest['all_engine_sha256']['../outside.engine'] = 'unused'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError,'invalid engine filename'):
            verify_candidate(self.cache,self.bundle)

    def test_previous_vision_manifest_is_supported(self):
        self.manifest.pop('all_engine_sha256')
        self.manifest.pop('additional_source_sha256')
        self.write_manifest()
        self.assertEqual(verify_candidate(self.cache,self.bundle),self.manifest)

if __name__ == '__main__':
    unittest.main()
