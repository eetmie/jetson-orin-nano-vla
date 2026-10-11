"""Bounded HTTP ranges, transient retries and checksums for pinned model files."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from http.client import IncompleteRead
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RANGE_BYTES = 32 * 1024**2


def retry(operation):
    for attempt in range(4):
        try:
            return operation()
        except (URLError, ConnectionError, TimeoutError, IncompleteRead) as exc:
            if isinstance(exc, HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 3:
                raise
            print(f"  retrying transfer after {type(exc).__name__} (attempt {attempt + 2}/4)", flush=True)
            time.sleep(2**attempt)


def verified(path, item):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            h.update(block)
    return path.stat().st_size == item["size"] and h.hexdigest() == item["sha256"]


def transfer(url, item, partial, workers=32, *, open_url=urlopen, range_bytes=RANGE_BYTES, resume=False):
    state_path = partial.with_name(partial.name + '.ranges.json')
    if partial.is_symlink() or state_path.is_symlink():
        raise ValueError('Unsafe partial range checkpoint')
    if (workers == 1 and not (resume and state_path.exists())) or item["size"] < 2 * range_bytes:
        with partial.open("xb") as stream:
            def copy():
                stream.seek(0)
                stream.truncate()
                with open_url(url, timeout=120) as source:
                    for block in iter(lambda: source.read(1024**2), b""):
                        stream.write(block)
            retry(copy)
        return
    # Disjoint positional writes bound explicit payload buffers to one MiB per
    # worker (HTTP/TLS and thread overhead are additional). A complete
    # file is published only after its final size and SHA-256 checks below.
    state = {'identity': item, 'range_bytes': range_bytes, 'completed': {}}
    if resume and state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get('identity') != item or previous.get('range_bytes') != range_bytes:
            raise ValueError('Partial range checkpoint identity mismatch')
        if not partial.is_file() or partial.stat().st_size != item['size']:
            raise ValueError('Partial range checkpoint size mismatch')
        with partial.open('rb') as source:
            for key, expected in previous.get('completed', {}).items():
                offset = int(key)
                if offset < 0 or offset >= item['size'] or offset % range_bytes:
                    raise ValueError('Invalid completed range checkpoint')
                source.seek(offset)
                remaining = min(range_bytes, item['size'] - offset)
                h = hashlib.sha256()
                while remaining:
                    block = source.read(min(1024**2, remaining))
                    if not block:
                        break
                    h.update(block)
                    remaining -= len(block)
                if not remaining and h.hexdigest() == expected:
                    state['completed'][key] = expected
        mode = 'r+b'
    else:
        mode = 'xb'
    temporary_state = state_path.with_name(state_path.name + '.tmp')
    if temporary_state.is_symlink():
        raise ValueError('Unsafe partial range checkpoint temporary file')

    def save_state():
        temporary_state.write_text(json.dumps(state, sort_keys=True) + '\n')
        os.replace(temporary_state, state_path)

    with partial.open(mode) as stream:
        fd = stream.fileno()
        if mode == 'xb':
            os.ftruncate(fd, item['size'])
        save_state()
        completed = sum(min(range_bytes, item['size'] - int(key)) for key in state['completed'])
        if completed:
            print(f"  resumed {completed}/{item['size']} checksum-verified range bytes", flush=True)

        def copy_range_once(start):
            end = min(start + range_bytes, item['size']) - 1
            request = Request(url, headers={'Range': f'bytes={start}-{end}'})
            h = hashlib.sha256()
            with open_url(request, timeout=120) as source:
                expected = f"bytes {start}-{end}/{item['size']}"
                if source.status != 206 or source.headers.get('Content-Range') != expected:
                    raise ValueError(f'Invalid checkpoint range response: {expected}')
                generation = source.headers.get('x-goog-generation')
                if generation is not None and item.get('generation') is not None and generation != str(item['generation']):
                    raise ValueError('Checkpoint generation mismatch')
                offset = start
                while offset <= end:
                    block = source.read(min(1024**2, end + 1 - offset))
                    if not block:
                        raise IncompleteRead(b'', end + 1 - offset)
                    h.update(block)
                    remaining = memoryview(block)
                    while remaining:
                        written = os.pwrite(fd, remaining, offset)
                        if written <= 0:
                            raise OSError('Checkpoint positional write made no progress')
                        remaining = remaining[written:]
                        offset += written
                    remaining.release()
                    del remaining, block
                if source.read(1):
                    raise ValueError(f'Oversized checkpoint range: {expected}')
            return start, end + 1 - start, h.hexdigest()

        def copy_range(start):
            try:
                return retry(lambda: copy_range_once(start))
            except IncompleteRead as exc:
                raise ValueError(f'Truncated checkpoint range {start} after retries') from exc

        error = None
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(copy_range, start)
                       for start in range(0, item['size'], range_bytes)
                       if str(start) not in state['completed']]
            try:
                for future in as_completed(futures):
                    start, size, sha = future.result()
                    state['completed'][str(start)] = sha
                    completed += size
                    save_state()
                    print(f"  received {completed}/{item['size']} bytes", flush=True)
            except BaseException as exc:
                error = exc
                for future in futures:
                    future.cancel()
        # Active ranges may finish while cancellation waits; retain those too.
        for future in futures:
            if not future.cancelled() and future.exception() is None:
                start, _, sha = future.result()
                state['completed'][str(start)] = sha
        save_state()
        if error:
            raise error
