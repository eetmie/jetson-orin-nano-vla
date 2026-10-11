"""Run an exporter with pinned Hub reads, including nested processor/model loads."""
from functools import wraps
import json
from pathlib import Path
import re
import runpy
import sys


def revision_for(repo_id, requested, pins):
    if requested and re.fullmatch(r"[0-9a-f]{40}", str(requested)):
        return requested
    if requested not in (None, "main"):
        raise ValueError(f"Floating Hub revision {requested!r}: use a full commit SHA")
    if repo_id not in pins:
        raise ValueError(f"Unpinned auxiliary Hub repo {repo_id!r}: add its commit to export/hub-lock.json")
    return pins[repo_id]


def install():
    import huggingface_hub as hub
    pins = json.loads((Path(__file__).resolve().parents[1] / "hub-lock.json").read_text())
    for name in ("snapshot_download", "hf_hub_download"):
        original = getattr(hub, name)

        def guarded(*args, _original=original, **kwargs):
            repo = kwargs.get("repo_id") or args[0]
            kwargs["revision"] = revision_for(repo, kwargs.get("revision"), pins)
            return _original(*args, **kwargs)

        setattr(hub, name, wraps(original)(guarded))

    # Transformers performs a second cache lookup after downloading. Pin that
    # lookup too: otherwise it searches refs/main after fetching a SHA snapshot
    # and can return an empty config on a genuinely fresh cache.
    import transformers.utils.hub as transformers_hub
    for name in ("cached_file", "cached_files"):
        original = getattr(transformers_hub, name)

        def cached(*args, _original=original, **kwargs):
            repo = kwargs.get("path_or_repo_id") or args[0]
            if not Path(str(repo)).is_dir():
                kwargs["revision"] = revision_for(str(repo), kwargs.get("revision"), pins)
            return _original(*args, **kwargs)

        setattr(transformers_hub, name, wraps(original)(cached))


def main():
    script = Path(sys.argv[1]).resolve()
    sys.argv = sys.argv[1:]
    sys.path.insert(0, str(script.parent))
    install()
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
