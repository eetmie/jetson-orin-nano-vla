"""Pin an experimental X-VLA engine cache to the bundle graphs and engine bytes it was built from."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def pin(cache, bundle):
    cache, bundle = Path(cache), Path(bundle)
    return dict(source_sha256={p.name: sha256(p) for p in sorted(bundle.glob('*.onnx'))},
                all_engine_sha256={p.name: sha256(p) for p in sorted(cache.glob('*.engine'))
                                   if not p.name.startswith('op_')})


def verify(cache, bundle):
    cache, bundle = Path(cache).expanduser().resolve(), Path(bundle).expanduser().resolve()
    manifest = json.loads((cache/'candidate.json').read_text())
    if manifest['bundle'] != str(bundle):
        raise ValueError(f"candidate cache belongs to {manifest['bundle']}")
    now = pin(cache, bundle)
    for key in ('source_sha256', 'all_engine_sha256'):
        if now[key] != manifest[key]:
            changed = sorted(set(now[key].items()) ^ set(manifest[key].items()))
            raise ValueError(f'{key} changed: {changed[:4]}')
    return manifest
