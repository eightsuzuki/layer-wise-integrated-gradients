#!/usr/bin/env python3
"""Build paper-demo JSON from a PTB layer_ig cache sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PAPER_TEXT = (
    "The firm's drop in net reflected weaker revenue in transactions for its own "
    "account -- a decline of 19% to $314.6 million on reduced revenue from "
    "trading fixed-income securities ."
)


def cache_to_lig_payload(
    cache: dict,
    *,
    baseline: str,
    source_id: str | None = None,
    cache_subdir: str | None = None,
    cache_parent: str | None = None,
    demo_source: str | None = None,
) -> dict:
    tokens = cache["tokens"]
    z2z = cache["z2z"]
    text = " ".join(
        t for t in tokens if t not in ("[CLS]", "[SEP]")
    )
    layers: dict = {}
    for layer_idx, matrix in enumerate(z2z):
        layers[str(layer_idx)] = {
            "layer_idx": layer_idx,
            "z2z": {
                "baseline": baseline,
                "description": "Layer-whole IG (z -> z), L2-scalarized",
                "shape": [len(matrix), len(matrix[0]) if matrix else 0],
                "matrix": matrix,
            },
        }
    payload = {
        "text": text,
        "tokens": tokens,
        "model": "bert-base-uncased",
        "model_type": "bert",
        "demo_source": demo_source or "ptb_dev_sample_00410",
        "config": {
            "num_steps": 32,
            "granularity": ["layer"],
            "baseline_layer": baseline,
        },
        "z2z": z2z,
        "layers": layers,
    }
    if source_id:
        payload["cache_source"] = {
            "id": source_id,
            "parent": cache_parent,
            "subdir": cache_subdir,
        }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--baseline", default="zero")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cache = json.loads(args.source_cache.read_text(encoding="utf-8"))
    payload = cache_to_lig_payload(cache, baseline=args.baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {args.out} ({len(payload['tokens'])} tokens, {len(payload['z2z'])} layers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
