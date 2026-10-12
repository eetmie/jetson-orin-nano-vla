"""Create an isolated venv exposing only JetPack's TensorRT Python package."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import venv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("venv", type=Path)
    parser.add_argument("--without-pip", action="store_true",
                        help="omit pip when packages will be managed externally with uv")
    args = parser.parse_args(argv)
    root = Path(os.path.abspath(args.venv.expanduser()))
    if root.exists():
        parser.error(f"{root} exists; choose a new environment directory")
    spec = importlib.util.find_spec("tensorrt")
    if spec is None or not spec.origin:
        parser.error("system Python cannot find TensorRT; install JetPack's Python bindings first")
    package = Path(spec.origin).resolve().parent
    if not (package / "tensorrt.so").is_file():
        parser.error(f"expected JetPack TensorRT bindings at {package}; this recipe targets the JetPack layout")
    venv.EnvBuilder(with_pip=not args.without_pip, system_site_packages=False).create(root)
    purelib = Path(subprocess.check_output([str(root / "bin/python"), "-c",
        "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
    (purelib / "tensorrt").symlink_to(package, target_is_directory=True)
    print(f"{root}: linked JetPack TensorRT from {package}; other system Python packages excluded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
