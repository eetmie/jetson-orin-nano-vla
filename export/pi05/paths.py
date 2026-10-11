"""Explicit stage inputs; no dependency on a historical experiment directory."""
import os
from pathlib import Path


def required(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is required; use export/pi05/export.py")
    return Path(value).resolve()


def stage_root():
    # openpi otherwise downloads a moving GCS tokenizer object. Route its read to
    # the generation-pinned file materialized by the orchestrator.
    from openpi.shared import download
    tokenizer = required("PI05_TOKENIZER")
    original = download.maybe_download

    def pinned(url, *args, **kwargs):
        if str(url) == "gs://big_vision/paligemma_tokenizer.model":
            return tokenizer
        if str(url).startswith(("gs://", "http://", "https://")):
            raise ValueError(f"Unpinned openpi download: {url}")
        return original(url, *args, **kwargs)

    download.maybe_download = pinned
    return required("PI05_STAGE_ROOT")


def checkpoint():
    return required("PI05_CHECKPOINT")


def initial_root():
    return required("PI05_INITIAL_ROOT")


def baseline_root():
    return required("PI05_BASELINE_ROOT")
