"""Verify experimental engine manifests before bypassing production prebuild."""
import hashlib
import json
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1<<20), b''):
            digest.update(block)
    return digest.hexdigest()

def verify_candidate(cache, bundle=None):
    cache = Path(cache).expanduser().resolve()
    bundle = Path(bundle or Path.home()/'bundles/smolvla-base-split').expanduser().resolve()
    manifest = json.loads((cache/'candidate.json').read_text())
    if sha256(bundle/'smolvlm_vision.onnx') != manifest['source_onnx_sha256']:
        raise ValueError('vision ONNX digest changed')
    if sha256(cache/'vision.engine') != manifest['engine_sha256']:
        raise ValueError('vision engine digest changed')
    for name, digest in manifest.get('all_engine_sha256', {}).items():
        if Path(name).name != name or not name.endswith('.engine'):
            raise ValueError(f'invalid engine filename: {name}')
        if sha256(cache/name) != digest:
            raise ValueError(f'engine digest changed: {name}')
    for name, digest in manifest.get('additional_source_sha256', {}).items():
        if Path(name).name != name or not name.endswith('.onnx'):
            raise ValueError(f'invalid source filename: {name}')
        if sha256(bundle/name) != digest:
            raise ValueError(f'ONNX digest changed: {name}')
    return manifest
