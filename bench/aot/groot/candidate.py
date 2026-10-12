"""Pin an experimental GR00T engine cache to the bundle graphs and engine bytes it was built from."""
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
                                   if not p.name.startswith('op_')},
                constants_sha256={p.name: sha256(p) for p in sorted(cache.glob('*.npz'))},
                bundle_artifact_sha256={p.relative_to(bundle).as_posix(): sha256(p)
                    for p in sorted(bundle.rglob('*')) if p.is_file() and p.suffix != '.onnx'
                    and not (p.name.startswith('_meta_') and p.suffix == '.json')})


def verify(cache, bundle):
    cache, bundle = Path(cache).expanduser().resolve(), Path(bundle).expanduser().resolve()
    manifest = json.loads((cache/'candidate.json').read_text())
    if manifest['bundle'] != str(bundle):
        raise ValueError(f"candidate cache belongs to {manifest['bundle']}")
    now = pin(cache, bundle)
    keys = ['source_sha256', 'all_engine_sha256', 'constants_sha256']
    # Initial sweep caches predate auxiliary-input pins; new builds pin the whole bundle.
    if 'bundle_artifact_sha256' in manifest:
        keys.append('bundle_artifact_sha256')
    for key in keys:
        if now[key] != manifest[key]:
            changed = sorted(set(now[key].items()) ^ set(manifest[key].items()))
            raise ValueError(f'{key} changed: {changed[:4]}')
    return manifest


def validate_base(cache, bundle, accumulate, stack):
    """Refuse to copy engines built from other graphs, precision or runtime stacks."""
    cache, bundle = Path(cache), Path(bundle)
    if (cache / 'candidate.json').exists():
        manifest = verify(cache, bundle)
        if manifest['accumulate'] != accumulate or manifest['tensorrt'] != stack.split()[0].split('=')[1]:
            raise ValueError('base candidate precision or TensorRT version differs')
    info = json.loads((bundle / 'bundle.json').read_text())
    for graph in info['graphs']:
        name = graph['name']
        expected = f"{sha256(bundle / (name + '.onnx'))} {stack}"
        key = cache / (name + '.onnx.sha256')
        if not (cache / (name + '.engine')).is_file() or not key.is_file() or key.read_text().strip() != expected:
            raise ValueError(f'base engine {name} is missing or stale; rebuild the native cache')


def validate_accumulate(manifest, args):
    if '--accumulate' in args and args[args.index('--accumulate') + 1] != manifest['accumulate']:
        raise ValueError('requested accumulation differs from the pinned engine cache')
