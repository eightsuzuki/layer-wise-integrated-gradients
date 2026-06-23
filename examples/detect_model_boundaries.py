#!/usr/bin/env python3
"""Print auto-detected LIG boundaries (z / u / z_next) for Hugging Face models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="*",
        default=["bert-base-uncased", "gpt2", "distilbert-base-uncased"],
        help="Hugging Face model ids (default: bert-base-uncased gpt2 distilbert-base-uncased)",
    )
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="Download weights and run module introspection (slower, exact)",
    )
    parser.add_argument("--device", default=None, help="cpu | cuda (with --load-weights)")
    args = parser.parse_args()

    from lig import describe_boundaries

    for model_name in args.models:
        info = describe_boundaries(
            model_name,
            load_weights=args.load_weights,
            device=args.device,
        )
        print(json.dumps(info, indent=2, ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()
