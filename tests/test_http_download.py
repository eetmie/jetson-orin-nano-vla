"""Recover interrupted model ranges without accepting changed cached bytes."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'export'))
from vla_common import http_download as downloads


class HTTPRangeResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.partial = Path(self.tmp.name) / 'model.partial'
        self.data = b'0123456789abcdef'
        self.item = {'size': len(self.data), 'sha256': hashlib.sha256(self.data).hexdigest()}
        self.requests = []
        self.fail_at = None

    def response(self, request, **kwargs):
        start, end = map(int, request.get_header('Range').removeprefix('bytes=').split('-'))
        self.requests.append(start)
        data = self.data[start:end + 1]
        if start == self.fail_at:
            data = data[:-1]
        stream = io.BytesIO(data)
        stream.status = 206
        stream.headers = {'Content-Range': f'bytes {start}-{end}/{len(self.data)}'}
        return stream

    def transfer(self, item=None):
        with patch.object(downloads.time, 'sleep'):
            downloads.transfer('https://example/file?secret-query', item or self.item, self.partial,
                               workers=2, range_bytes=4, open_url=self.response, resume=True)

    def state(self):
        return self.partial.with_name(self.partial.name + '.ranges.json')

    def test_failed_transfer_retains_valid_ranges_and_resume_only_fetches_missing_ranges(self):
        self.fail_at = 4
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            self.transfer()
        state = json.loads(self.state().read_text())
        completed = set(map(int, state['completed']))
        self.assertTrue(completed)
        self.assertNotIn(4, completed)
        self.assertNotIn('secret-query', self.state().read_text())
        self.requests.clear(); self.fail_at = None
        self.transfer()
        self.assertTrue(downloads.verified(self.partial, self.item))
        self.assertEqual(set(self.requests), {0, 4, 8, 12} - completed)

    def test_resume_rechecks_completed_range_bytes_and_repairs_changed_data(self):
        self.transfer()
        with self.partial.open('r+b') as stream:
            stream.write(b'xxxx')
        self.requests.clear()
        self.transfer()
        self.assertEqual(self.requests, [0])
        self.assertTrue(downloads.verified(self.partial, self.item))

    def test_range_identity_size_and_symlink_mismatches_preserve_existing_files(self):
        self.transfer()
        before = self.state().read_bytes()
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.transfer({**self.item, 'sha256': '0' * 64})
        self.assertEqual(self.partial.read_bytes(), self.data)
        self.assertEqual(self.state().read_bytes(), before)
        self.partial.write_bytes(b'user data')
        with self.assertRaisesRegex(ValueError, 'size'):
            self.transfer()
        self.assertEqual(self.partial.read_bytes(), b'user data')
        self.state().unlink()
        outside = self.partial.parent / 'outside'
        outside.write_bytes(before)
        self.state().symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            self.transfer()
        self.assertEqual(outside.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
