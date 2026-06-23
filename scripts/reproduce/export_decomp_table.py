#!/usr/bin/env python3
"""Export top-N L2 rows per layer_ig_baseline_group for tab_decomp_top."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    rows: list[dict] = []
    with args.summary_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("l2_mean") or r.get("n_pairs", "0") in ("0", ""):
                continue
            rows.append({**r, "l2_mean_f": float(r["l2_mean"])})

    by_group: dict[str, list[dict]] = {}
    for r in rows:
        by_group.setdefault(r.get("layer_ig_baseline_group", ""), []).append(r)

    out_rows: list[dict] = []
    for group in sorted(by_group):
        for rank, r in enumerate(
            sorted(by_group[group], key=lambda x: x["l2_mean_f"])[: args.top],
            start=1,
        ):
            out_rows.append(
                {
                    "rank_in_group": rank,
                    "layer_ig_baseline_group": group,
                    "layer_ig_baseline": r.get("layer_ig_baseline", ""),
                    "att_ig_baseline": r.get("att_ig_baseline", ""),
                    "mlp_ig_baseline": r.get("mlp_ig_baseline", ""),
                    "l2_mean": r["l2_mean"],
                    "n_pairs": r.get("n_pairs", ""),
                    "pair_name": r.get("pair_name", ""),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(out_rows[0].keys()) if out_rows else [
        "rank_in_group",
        "layer_ig_baseline_group",
        "layer_ig_baseline",
        "att_ig_baseline",
        "mlp_ig_baseline",
        "l2_mean",
        "n_pairs",
        "pair_name",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
