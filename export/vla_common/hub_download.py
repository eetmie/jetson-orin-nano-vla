"""Download immutable Hub snapshots through bounded, checksum-verified HTTP."""
import argparse
from fnmatch import fnmatch
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .http_download import transfer


class HubRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urlsplit(newurl).hostname != 'huggingface.co':
            redirected.remove_header('Authorization')
        return redirected


urlopen = build_opener(HubRedirects()).open


def digest(path, size, lfs):
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            sha.update(block)
            blob.update(block)
    return sha.hexdigest(), sha.hexdigest() if lfs else blob.hexdigest()


def safe_target(root, name, allowed_blob=None):
    parts = PurePosixPath(name).parts
    if not parts or PurePosixPath(name).is_absolute() or '..' in parts or '\\' in name:
        raise ValueError('Unsafe Hub filename')
    target = root.joinpath(*parts)
    for parent in (root, *target.parents, target):
        if parent.is_symlink():
            if parent == target and allowed_blob is not None:
                if (target.resolve() == allowed_blob.resolve() and not allowed_blob.is_symlink()
                        and not any(p.is_symlink() for p in allowed_blob.parents)):
                    continue
            raise ValueError('Symlink in Hub download destination')
    return target


def download(repo, revision, destination, *, include=(), exclude=(), workers=32, cache_blobs=None):
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Hub revision must be a full commit SHA')
    from huggingface_hub import HfApi, get_token, get_hf_file_metadata, hf_hub_url
    token = get_token()
    info = HfApi(token=token).model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError('Hub returned a different snapshot revision')
    root = Path(destination)
    marker = safe_target(root, '.vla-hub-snapshot.json')
    if marker.exists():
        previous = json.loads(marker.read_text())
        if (previous['repo'], previous['revision']) != (repo, revision):
            raise ValueError('Destination belongs to a different Hub snapshot')
    selected = [f for f in info.siblings
                if (not include or any(fnmatch(f.rfilename, p) for p in include))
                and not any(fnmatch(f.rfilename, p) for p in exclude)]
    if not selected:
        raise ValueError('No Hub files matched the requested patterns')
    def file_target(item):
        expected = item.lfs.sha256 if item.lfs else item.blob_id
        blob = Path(cache_blobs) / expected if cache_blobs is not None else None
        return safe_target(root, item.rfilename, blob)
    # Validate every path before creating any files.
    for item in selected:
        if item.size is None or not (item.lfs or item.blob_id):
            raise ValueError('Hub omitted file integrity metadata')
        file_target(item)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = safe_target(root, '.vla-download.lock')
    with lock_path.open('a+b') as lock:
        # Parallel family exports can share a pinned auxiliary cache safely.
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        report = {'repo': repo, 'revision': revision, 'status': 'running', 'files': []}
        for item in selected:
            target = file_target(item)
            target.parent.mkdir(parents=True, exist_ok=True)
            lfs = item.lfs
            expected = lfs.sha256 if lfs else item.blob_id
            if target.exists():
                sha, actual = digest(target, item.size, lfs)
                if target.stat().st_size != item.size or actual != expected:
                    raise ValueError(f'Existing Hub file failed integrity: {item.rfilename}')
                print(f'verified existing {repo}/{item.rfilename}', flush=True)
            else:
                meta = get_hf_file_metadata(hf_hub_url(repo, item.rfilename, revision=revision), token=token)
                if meta.commit_hash != revision or meta.size != item.size:
                    raise ValueError('Hub file metadata disagrees with pinned snapshot')
                partial = safe_target(root, item.rfilename + '.vla-partial')
                range_state = safe_target(root, item.rfilename + '.vla-partial.ranges.json')
                if partial.exists() and not range_state.exists():
                    partial.unlink()
                # Send Hub credentials only to the Hub, never to signed CDN URLs.
                # Small Git files can retain a Hub URL (including gated metadata).
                def open_file(url_or_request, **kwargs):
                    request = url_or_request if isinstance(url_or_request, Request) else Request(url_or_request)
                    if token and request.host == 'huggingface.co':
                        request.add_header('Authorization', f'Bearer {token}')
                    return urlopen(request, **kwargs)
                print(f'downloading {repo}/{item.rfilename} ({item.size} bytes)', flush=True)
                identity = {'size': item.size, 'repo': repo, 'revision': revision, 'file': item.rfilename,
                            'sha256' if lfs else 'git_blob_sha1': expected}
                transfer(meta.location, identity, partial, workers,
                         open_url=open_file, range_bytes=4 * 1024**2, resume=True)
                sha, actual = digest(partial, item.size, lfs)
                if partial.stat().st_size != item.size or actual != expected:
                    raise ValueError(f'Hub file failed integrity: {item.rfilename}')
                os.replace(partial, target)
                range_state.unlink(missing_ok=True)
            report['files'].append({'file': item.rfilename, 'size': item.size, 'sha256': sha,
                                    'expected_digest': expected, 'digest_kind': 'sha256' if lfs else 'git-blob-sha1'})
        report['status'] = 'pass'
        temporary = safe_target(root, '.vla-hub-snapshot.json.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        os.replace(temporary, marker)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo')
    parser.add_argument('--revision', required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument('--local-dir')
    destination.add_argument('--cache-dir')
    parser.add_argument('--include', action='append', default=[])
    parser.add_argument('--exclude', action='append', default=[])
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        parser.error('--workers must be between 1 and 64')
    root = args.local_dir or str(Path(args.cache_dir) / ('models--' + args.repo.replace('/', '--')) / 'snapshots' / args.revision)
    blobs = (Path(args.cache_dir) / ('models--' + args.repo.replace('/', '--')) / 'blobs'
             if args.cache_dir else None)
    download(args.repo, args.revision, root, include=args.include, exclude=args.exclude,
             workers=args.workers, cache_blobs=blobs)


if __name__ == '__main__':
    main()
