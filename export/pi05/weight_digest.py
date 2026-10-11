"""Hash named tensor contents independently of safetensors header ordering."""
import hashlib
import json
from pathlib import Path
import struct


def tensor_digest(path, excluded=()):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length > 16 * 1024**2:
            raise ValueError("Unexpected safetensors header size")
        header = json.loads(stream.read(length))
        h.update(json.dumps(header.pop("__metadata__", {}), sort_keys=True, separators=(",", ":")).encode())
        for name, entry in sorted(header.items()):
            if name in excluded:
                continue
            h.update(json.dumps([name, entry["dtype"], entry["shape"]], separators=(",", ":")).encode())
            start, end = entry["data_offsets"]
            if not 0 <= start <= end <= Path(path).stat().st_size - 8 - length:
                raise ValueError("Invalid safetensors data offsets")
            stream.seek(8 + length + start)
            remaining = end - start
            while remaining:
                block = stream.read(min(8 * 1024**2, remaining))
                if not block:
                    raise ValueError("Truncated safetensors tensor")
                h.update(block)
                remaining -= len(block)
    return h.hexdigest()
