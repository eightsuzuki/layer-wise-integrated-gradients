#!/usr/bin/env python3
"""Replace <!-- PROFILE_RESULTS --> in docs/LIG_COMPUTATION.md from profile JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARKER = "<!-- PROFILE_RESULTS -->"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("json_path", type=Path)
    p.add_argument("--gpu", default="")
    p.add_argument(
        "--doc",
        type=Path,
        default=ROOT / "docs" / "LIG_COMPUTATION.md",
    )
    args = p.parse_args()
    data = json.loads(args.json_path.read_text(encoding="utf-8"))
    lt = data.get("layer_times_s") or []
    rows = [
        f"- sample: dev #{data.get('sample_idx', '?')}, baseline: `{data.get('baseline', '?')}`",
        f"- device: `{data.get('device', '?')}`"
        + (f" (CUDA_VISIBLE_DEVICES={args.gpu})" if args.gpu else ""),
        f"- seq_len: {data.get('seq_len', '?')}, layers profiled: {len(lt)}",
        "",
        "| metric | seconds |",
        "|--------|---------|",
        f"| model_load | {float(data.get('model_load_s', 0)):.3f} |",
        f"| full_forward | {float(data.get('full_forward_s', 0)):.3f} |",
        f"| layer_ig_total | {float(data.get('layer_ig_total_s', 0)):.3f} |",
        f"| layer_ig_mean | {float(data.get('layer_ig_mean_s', 0)):.3f} |",
        f"| est_layer_forwards | {data.get('est_layer_forwards', '?')} |",
        "",
        "Per-layer IG time (s): " + ", ".join(f"{float(t):.3f}" for t in lt),
        "",
        f"_Profile JSON: `{args.json_path}`_",
    ]
    table = "\n".join(rows)
    doc = args.doc
    text = doc.read_text(encoding="utf-8")
    if MARKER in text:
        doc.write_text(text.replace(MARKER, table), encoding="utf-8")
    else:
        start = "## 7. プロファイル結果\n\n"
        end = "\n## 8."
        i = text.find(start)
        j = text.find(end, i if i >= 0 else 0)
        if i < 0 or j < 0:
            print(f"cannot locate section 7 in {doc}", file=sys.stderr)
            return 1
        doc.write_text(text[: i + len(start)] + table + text[j:], encoding="utf-8")
    print(f"Updated {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
