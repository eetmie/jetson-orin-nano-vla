"""Immutable Hub identity, credential boundaries and publication integrity."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch
from urllib.request import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'export'))
from vla_common import hub_download as downloads

REV = 'a' * 40


class HubDownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'snapshot'
        self.data = b'pinned file bytes'
        self.file = SimpleNamespace(rfilename='model.safetensors', size=len(self.data),
                                    blob_id='b' * 40, lfs=SimpleNamespace(sha256=hashlib.sha256(self.data).hexdigest()))
        self.info = SimpleNamespace(sha=REV, siblings=[self.file])
        self.metadata = SimpleNamespace(commit_hash=REV, size=len(self.data),
                                        location='https://cdn.example/model?secret=signature')
        self.requests = []
        self.hub = ModuleType('huggingface_hub')
        self.hub.HfApi = lambda **kw: SimpleNamespace(model_info=lambda *a, **k: self.info)
        self.hub.get_token = lambda: 'private-token'
        self.hub.hf_hub_url = lambda *a, **k: 'https://huggingface.co/repo/resolve/pin/file'
        self.hub.get_hf_file_metadata = lambda *a, **k: self.metadata
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {'huggingface_hub': self.hub}).start()
        patch.object(downloads, 'urlopen', side_effect=self.response).start()

    def response(self, request, **kw):
        self.requests.append(request)
        return io.BytesIO(self.data)

    def run_download(self):
        return downloads.download('owner/model', REV, self.root)

    def test_publishes_sha_verified_file_without_cdn_credentials_or_manifest_secrets(self):
        report = self.run_download()
        self.assertEqual((self.root / self.file.rfilename).read_bytes(), self.data)
        self.assertEqual(report['status'], 'pass')
        self.assertIsNone(self.requests[0].get_header('Authorization'))
        manifest = (self.root / '.vla-hub-snapshot.json').read_text()
        self.assertNotIn('private-token', manifest)
        self.assertNotIn('signature', manifest)
        self.assertNotIn('https://', manifest)
        self.run_download()
        self.assertEqual(len(self.requests), 1)

    def test_git_blob_integrity_and_hub_auth_for_gated_small_files(self):
        self.file.lfs = None
        self.file.blob_id = hashlib.sha1(f'blob {len(self.data)}\0'.encode() + self.data).hexdigest()
        self.metadata.location = 'https://huggingface.co/repo/resolve/pin/config.json'
        report = self.run_download()
        self.assertEqual(report['files'][0]['digest_kind'], 'git-blob-sha1')
        self.assertEqual(self.requests[0].get_header('Authorization'), 'Bearer private-token')

    def test_redirects_never_forward_hub_credentials_to_cdn(self):
        request = Request('https://huggingface.co/file', headers={'Authorization': 'Bearer private-token'})
        handler = downloads.HubRedirects()
        cdn = handler.redirect_request(request, None, 302, 'Found', {}, 'https://cdn.example/file')
        self.assertIsNone(cdn.get_header('Authorization'))
        hub = handler.redirect_request(request, None, 302, 'Found', {}, 'https://huggingface.co/next')
        self.assertEqual(hub.get_header('Authorization'), 'Bearer private-token')

    def test_moving_revision_and_snapshot_mismatch_fail_before_mutation(self):
        with self.assertRaisesRegex(ValueError, 'commit SHA'):
            downloads.download('owner/model', 'main', self.root)
        self.info.sha = 'b' * 40
        with self.assertRaisesRegex(ValueError, 'different snapshot'):
            self.run_download()
        self.assertFalse(self.root.exists())

    def test_metadata_mismatch_and_corrupt_body_never_publish(self):
        self.metadata.commit_hash = 'b' * 40
        with self.assertRaisesRegex(ValueError, 'metadata'):
            self.run_download()
        self.assertFalse((self.root / self.file.rfilename).exists())
        self.metadata.commit_hash = REV
        self.data = b'changed bytes'
        with self.assertRaisesRegex(ValueError, 'integrity'):
            self.run_download()
        self.assertFalse((self.root / self.file.rfilename).exists())
        self.assertFalse((self.root / '.vla-hub-snapshot.json').exists())

    def test_existing_corrupt_file_and_snapshot_identity_are_preserved(self):
        self.run_download()
        target = self.root / self.file.rfilename
        target.write_bytes(b'user data')
        with self.assertRaisesRegex(ValueError, 'Existing Hub file'):
            self.run_download()
        self.assertEqual(target.read_bytes(), b'user data')
        marker = self.root / '.vla-hub-snapshot.json'
        report = json.loads(marker.read_text()); report['repo'] = 'different/repo'
        marker.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'different Hub snapshot'):
            self.run_download()

    def test_unsafe_paths_and_symlink_partial_are_rejected(self):
        self.file.rfilename = '../outside'
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            self.run_download()
        self.assertFalse(self.root.exists())
        self.file.rfilename = 'model.safetensors'
        self.root.mkdir()
        outside = self.root.parent / 'outside'
        outside.write_bytes(b'user data')
        (self.root / 'model.safetensors.vla-partial').symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            self.run_download()
        self.assertEqual(outside.read_bytes(), b'user data')

    def test_standard_cache_blob_links_are_verified_and_preserved(self):
        blobs = self.root.parent / 'blobs'
        blobs.mkdir()
        blob = blobs / self.file.lfs.sha256
        blob.write_bytes(self.data)
        self.root.mkdir()
        target = self.root / self.file.rfilename
        target.symlink_to(blob)
        report = downloads.download('owner/model', REV, self.root, cache_blobs=blobs)
        self.assertEqual(report['status'], 'pass')
        self.assertTrue(target.is_symlink())
        self.assertEqual(self.requests, [])
        blob.write_bytes(b'changed cache')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            downloads.download('owner/model', REV, self.root, cache_blobs=blobs)
        self.assertEqual(blob.read_bytes(), b'changed cache')
        target.unlink()
        outside = self.root.parent / 'outside'
        outside.write_bytes(self.data)
        target.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            downloads.download('owner/model', REV, self.root, cache_blobs=blobs)


if __name__ == '__main__':
    unittest.main()
