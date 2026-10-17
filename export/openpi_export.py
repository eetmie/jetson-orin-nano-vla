"""Host orchestration for a custom OpenPI checkpoint, using the pinned container."""
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "export"


def export_openpi(checkpoint: Path, out: Path, spec: dict, config_file: Path | None):
    work = out.with_name(out.name + ".work")
    if work.exists():
        raise ValueError(f"work directory must be new: {work}")
    if config_file is not None and not config_file.is_file():
        raise ValueError(f"OpenPI configuration module does not exist: {config_file}")
    out.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir()
    (work / "model_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    from export.pi05.download import download
    download(json.loads((HERE / "pi05/tokenizer-lock.json").read_text()), work / "tokenizer")
    subprocess.run(["docker", "build", "--iidfile", str(work / "image-id"),
                    "-f", str(HERE / "pi05/Dockerfile"), str(ROOT)], check=True)
    image = (work / "image-id").read_text().strip()
    command = ["docker", "run", "--rm", "--gpus", "all", "--ipc=host", "--user", f"{os.getuid()}:{os.getgid()}",
               "-e", "OPENPI_DATA_HOME=/work/cache", "-e", "HF_HOME=/work/hf-cache",
               "-e", "XDG_CACHE_HOME=/work/cache", "-e", "PI05_TOKENIZER=/work/tokenizer/paligemma_tokenizer.model",
               "-e", f"PI05_CONFIG={spec['openpi']['openpi_config']}",
               "-v", f"{ROOT}:/repo:ro", "-v", f"{checkpoint}:/checkpoint:ro",
               "-v", f"{work}:/work", "-v", f"{out.parent}:/output"]
    if config_file:
        config_file = config_file.resolve()
        command += ["-v", f"{config_file.parent}:/user-config:ro", "-e", f"PI05_CONFIG_FILE=/user-config/{config_file.name}"]
    if (checkpoint / "model.safetensors").is_file():
        converted = "/checkpoint"
    else:
        if not (checkpoint / "params").is_dir():
            raise ValueError("OpenPI checkpoint needs model.safetensors or a JAX params/ directory")
        subprocess.run([*command, image, "python", "/repo/export/pi05/convert.py",
                        "--config_name", spec["openpi"]["openpi_config"], "--checkpoint_dir", "/checkpoint",
                        "--output_path", "/work/checkpoint"], check=True)
        if (checkpoint / "assets").is_dir():
            shutil.copytree(checkpoint / "assets", work / "checkpoint/assets", dirs_exist_ok=True)
        converted = "/work/checkpoint"
    subprocess.run([*command, image, "python", "/repo/export/pi05_lerobot/export_split.py",
                    "--checkpoint", converted, "--out", f"/output/{out.name}",
                    "--openpi-spec", "/work/model_spec.json"], check=True)
