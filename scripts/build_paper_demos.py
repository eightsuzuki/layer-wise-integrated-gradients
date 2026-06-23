#!/usr/bin/env python3
"""Export PTB z2z cache samples listed in examples/paper_demo/manifest.json."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "examples/paper_demo/manifest.json"
DATA_DIR = ROOT / "examples/paper_demo/json"


def resolve_cache_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    env = os.environ.get("PTB_CACHE_ROOT")
    if env:
        return Path(env)
    monorepo = ROOT.parent / "cache" / "ptb_ig_analysis"
    return monorepo if monorepo.is_dir() else Path("cache/ptb_ig_analysis")


def export_name(sample_id: str, source_id: str) -> str:
    return f"{sample_id}__{source_id}.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--sample", help="Export only this sample id")
    parser.add_argument("--source", help="Export only this cache source id")
    args = parser.parse_args()

    cache_root = resolve_cache_root(args.cache_root)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sources = manifest.get("sources") or []
    if not sources:
        raise SystemExit("manifest.json must define at least one entry in 'sources'")

    if args.source:
        sources = [s for s in sources if s["id"] == args.source]
        if not sources:
            print(f"Unknown source: {args.source}", file=sys.stderr)
            return 1

    samples = manifest["samples"]
    if args.sample:
        samples = [s for s in samples if s["id"] == args.sample]
        if not samples:
            print(f"Unknown sample: {args.sample}", file=sys.stderr)
            return 1

    sys.path.insert(0, str(ROOT))
    import runpy

    mod = runpy.run_path(str(ROOT / "scripts/build_paper_demo.py"))
    cache_to_lig_payload = mod["cache_to_lig_payload"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    default_out: Path | None = None

    for source in sources:
        cache_base = cache_root / source["cache_parent"] / source["cache_subdir"]
        if not cache_base.is_dir():
            print(f"Skip missing cache dir: {cache_base}", file=sys.stderr)
            continue

        baseline = source.get("baseline", "zero")
        for entry in samples:
            sample_id = entry["id"]
            index = entry["index"]
            cache_file = cache_base / f"sample_{index:05d}.json"
            if not cache_file.exists():
                print(f"Skip missing cache file: {cache_file}", file=sys.stderr)
                continue

            cache = json.loads(cache_file.read_text(encoding="utf-8"))
            payload = cache_to_lig_payload(
                cache,
                baseline=baseline,
                source_id=source["id"],
                cache_parent=source["cache_parent"],
                cache_subdir=source["cache_subdir"],
                demo_source=f"ptb_dev_{sample_id}",
            )
            out = args.out_dir / export_name(sample_id, source["id"])
            out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Wrote {out} ({len(payload['tokens'])} tokens, source={source['id']})")
            if source.get("default"):
                default_out = out

    if default_out is not None:
        legacy = ROOT / "examples/paper_demo/lig_z2z_zero.json"
        legacy.write_text(default_out.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Wrote {legacy} (default source)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
