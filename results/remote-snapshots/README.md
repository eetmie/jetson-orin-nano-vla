# Local copies of Jetson measurements

Each timestamped folder is a separate snapshot of the remote `results/` directory.
`files/` contains the original measurements, saved actions, logs and other result
files; `manifest.json` records SHA-256 checksums, sizes and benchmark run labels.
The first snapshot retains 66 files, including 32 labeled benchmark runs.

Create a new snapshot from the local repository with
`python3 -m bench.tools.archive_remote_results`. Existing snapshots are never
overwritten. Snapshot before reusing remote filenames and after each testing
session. New runs should also have unique remote filenames. Files overwritten
before a snapshot cannot be reconstructed from the latest measurement.

Keep this folder local when syncing work to the board. The archive command also
excludes remote `remote-snapshots/` directories to avoid copying nested archives.

Engine binaries, bundles and raw Nsight traces outside remote `results/` are not
copied by this command. Experiment manifests and retained sources are also saved
under `results/smolvla-triton-20261010/`.
