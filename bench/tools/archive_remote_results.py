"""Copy remote measurement files into a new, uniquely named local snapshot.

Never overwrites earlier snapshots. Run before reusing a remote output filename;
files overwritten before the first snapshot cannot be reconstructed by this tool.
Requires SSH and rsync, and copies results only (not bundles or engine caches).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='joel@192.168.8.230')
    parser.add_argument('--remote-results', default='~/jetson-orin-nano-vla/results')
    parser.add_argument('--out', type=Path, default=Path('results/remote-snapshots'))
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    dest = args.out.expanduser().resolve() / f'{stamp}-{uuid.uuid4().hex[:8]}'
    dest.mkdir(parents=True, exist_ok=False)
    data = dest / 'files'
    command = ['rsync', '-a', '--exclude=/remote-snapshots/', '-e', 'ssh -o BatchMode=yes', '--',
               f'{args.host}:{args.remote_results.rstrip("/")}/', str(data) + '/']
    completed = subprocess.run(command, check=False)
    entries = []
    for file in sorted(data.rglob('*')):
        if file.is_symlink():
            entries.append(dict(path=str(file.relative_to(data)), symlink=str(file.readlink())))
        elif file.is_file():
            payload = file.read_bytes()
            entry = dict(path=str(file.relative_to(data)), bytes=len(payload),
                         sha256=hashlib.sha256(payload).hexdigest())
            if file.suffix == '.json':
                try:
                    result = json.loads(payload)
                    if isinstance(result, dict) and result.get('label'):
                        entry.update(label=result['label'], status=result.get('status'),
                                     run_utc=result.get('env', {}).get('utc'))
                except (ValueError, UnicodeDecodeError):
                    entry['json_parse_failed'] = True
            entries.append(entry)
    manifest = dict(status='COMPLETE' if completed.returncode == 0 else 'INCOMPLETE',
                    archived_utc=stamp, host=args.host, remote_results=args.remote_results,
                    rsync_exit=completed.returncode, files=entries)
    (dest/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'{manifest["status"]}: {len(entries)} files -> {dest}', flush=True)
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
