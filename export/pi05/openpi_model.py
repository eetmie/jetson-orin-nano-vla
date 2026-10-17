"""Resolve an explicitly selected OpenPI training configuration inside the host container."""
import importlib.util
import os
from pathlib import Path
import sys


def register_configs():
    file = os.environ.get("PI05_CONFIG_FILE")
    if file:
        path = Path(file)
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("_vla_openpi_user_config", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not callable(getattr(module, "register", None)):
            raise ValueError("--openpi-config-file must define register() to register its TrainConfig")
        module.register()


def configure_downloads():
    from openpi.shared import download
    original = download.maybe_download
    tokenizer = Path(os.environ["PI05_TOKENIZER"])
    def pinned(url, *args, **kwargs):
        if str(url) == "gs://big_vision/paligemma_tokenizer.model":
            return tokenizer
        if str(url).startswith(("gs://", "http://", "https://")):
            raise ValueError(f"unresolved auxiliary OpenPI download: {url}")
        return original(url, *args, **kwargs)
    download.maybe_download = pinned


def train_config():
    from openpi.training import config
    register_configs()
    return config.get_config(os.environ["PI05_CONFIG"])
