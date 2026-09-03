#!/usr/bin/env python3
"""Validate the EVO1 split runtime against its embedded native LeRobot fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.evo1_parity import validate_fixture
from bench.vendor.evo1_split_ort import (
    Evo1SplitPolicy,
    prebuild_engines,
    verify_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--skip-prebuild", action="store_true")
    args = parser.parse_args()

    bundle = verify_bundle(args.bundle)
    if not args.skip_prebuild:
        prebuild_engines(args.bundle, args.cache_dir, args.precision)
    policy = Evo1SplitPolicy(
        args.bundle,
        args.cache_dir,
        args.precision,
        num_steps=args.num_steps,
        allow_bootstrap=True,
    )
    parity = validate_fixture(policy, args.bundle, threshold=args.threshold)
    document = {
        **parity,
        "warning": bundle["warning"],
        "num_steps": policy.steps,
        "providers": {
            name: session.get_providers()
            for name, session in policy.sessions.items()
        },
    }
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0 if parity["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
