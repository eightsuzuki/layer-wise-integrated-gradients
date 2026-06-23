#!/usr/bin/env python3
"""Compare layer-direct z2z vs ATT+MLP composed z2z (Experiment A, public release scope)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.reproduce.ptb_loader import ptb_cache_root

_BASE = "steps32_bert-base-uncased_maxlen128_z_to_z"

_LAYER_SPECS = [
    {
        "lig_key": "lig_zero",
        "layer_ig_baseline_group": "zero",
        "layer_ig_baseline": "Zero",
        "lig_pair_label": "Zero",
        "layer_ig_suffix": f"{_BASE}_layer_ig_baseline_zero",
        "apply_layer_ig_zero_suffix": True,
    },
    {
        "lig_key": "lig_itb",
        "layer_ig_baseline_group": "itb",
        "layer_ig_baseline": "ITB (input token)",
        "lig_pair_label": "ITB",
        "layer_ig_suffix": f"{_BASE}_layer_ig_baseline_self_input_token",
        "apply_layer_ig_zero_suffix": False,
    },
    {
        "lig_key": "lig_itb_zero_ratio",
        "layer_ig_baseline_group": "itb_zero_ratio",
        "layer_ig_baseline": "LAYER-ITB-zeroRatio",
        "lig_pair_label": "LAYER-ITB-zeroRatio",
        "layer_ig_suffix": (
            f"{_BASE}_layer_ig_baseline_self_input_token_self_contrib_zero_base_ratio"
        ),
        "apply_layer_ig_zero_suffix": False,
    },
]

_COMPOSED_VARIANTS = [
    {
        "composed_suffix": f"{_BASE}_ATT_zero_MLP_zero",
        "pair_key_att": "att0",
        "pair_key_mlp": "mlp0",
        "att_name": "zero",
        "mlp_name": "zero",
        "att_ig_baseline": "zero",
        "mlp_ig_baseline": "zero",
    },
    {
        "composed_suffix": f"{_BASE}_ATT_zero_MLP_ATTITBa0",
        "pair_key_att": "att0",
        "pair_key_mlp": "mlp_attitba0",
        "att_name": "zero",
        "mlp_name": "ATTITBa=0",
        "att_ig_baseline": "zero",
        "mlp_ig_baseline": "ATTITBa=0",
    },
    {
        "composed_suffix": f"{_BASE}_ATT_ITB_raw_MLP_zero",
        "pair_key_att": "att_itb",
        "pair_key_mlp": "mlp0",
        "att_name": "ITB",
        "mlp_name": "zero",
        "att_ig_baseline": "ITB raw",
        "mlp_ig_baseline": "zero",
    },
    {
        "composed_suffix": f"{_BASE}_ATT_ITB_raw_MLP_ATTITBa0",
        "pair_key_att": "att_itb",
        "pair_key_mlp": "mlp_attitba0",
        "att_name": "ITB",
        "mlp_name": "ATTITBa=0",
        "att_ig_baseline": "ITB raw",
        "mlp_ig_baseline": "ATTITBa=0",
    },
    {
        "composed_suffix": f"{_BASE}_ATT_ITB_map_MLP_zero",
        "pair_key_att": "att_itbmap",
        "pair_key_mlp": "mlp0",
        "att_name": "ITB-mapRatio",
        "mlp_name": "zero",
        "att_ig_baseline": "ITB-mapRatio",
        "mlp_ig_baseline": "zero",
    },
    {
        "composed_suffix": f"{_BASE}_ATT_ITB_map_MLP_ATTITBa0",
        "pair_key_att": "att_itbmap",
        "pair_key_mlp": "mlp_attitba0",
        "att_name": "ITB-mapRatio",
        "mlp_name": "ATTITBa=0",
        "att_ig_baseline": "ITB-mapRatio",
        "mlp_ig_baseline": "ATTITBa=0",
    },
    {
        "composed_suffix": f"{_BASE}_ATT_ITB_zero_base_ratio_MLP_zero",
        "pair_key_att": "att_itb_zbr",
        "pair_key_mlp": "mlp0",
        "att_name": "ITB-zeroRatio",
        "mlp_name": "zero",
        "att_ig_baseline": "ITB-zeroRatio",
        "mlp_ig_baseline": "zero",
    },
    {
        "composed_suffix": f"{_BASE}_ATT_ITB_zero_base_ratio_MLP_ATTITBa0",
        "pair_key_att": "att_itb_zbr",
        "pair_key_mlp": "mlp_attitba0",
        "att_name": "ITB-zeroRatio",
        "mlp_name": "ATTITBa=0",
        "att_ig_baseline": "ITB-zeroRatio",
        "mlp_ig_baseline": "ATTITBa=0",
    },
]

COMPOSED_OLD_SUFFIX = {
    f"{_BASE}_ATT_zero_MLP_zero": f"{_BASE}_baseline_zero",
    f"{_BASE}_ATT_ITB_raw_MLP_zero": f"{_BASE}_baseline_self_input_token",
}


def build_baseline_pairs() -> list[dict]:
    out: list[dict] = []
    for ls in _LAYER_SPECS:
        for cv in _COMPOSED_VARIANTS:
            pair_key = f"{ls['lig_key']}_{cv['pair_key_att']}_{cv['pair_key_mlp']}"
            pair_name = (
                f"LIG({ls['lig_pair_label']}) | "
                f"ATT({cv['att_name']})+MLP({cv['mlp_name']})"
            )
            out.append(
                {
                    "pair_key": pair_key,
                    "pair_name": pair_name,
                    "apply_layer_ig_zero_suffix": ls["apply_layer_ig_zero_suffix"],
                    "layer_ig_suffix": ls["layer_ig_suffix"],
                    "composed_suffix": cv["composed_suffix"],
                    "layer_ig_baseline_group": ls["layer_ig_baseline_group"],
                    "layer_ig_baseline": ls["layer_ig_baseline"],
                    "att_ig_baseline": cv["att_ig_baseline"],
                    "mlp_ig_baseline": cv["mlp_ig_baseline"],
                }
            )
    return out


BASELINE_PAIRS = build_baseline_pairs()


def resolve_composed_dir(composed_base: Path, suffix: str) -> Path | None:
    candidates = [composed_base / suffix]
    if suffix in COMPOSED_OLD_SUFFIX:
        candidates.append(composed_base / COMPOSED_OLD_SUFFIX[suffix])
    extra = composed_base.parent.parent.parent / "results/uas_layer_ig_vs_composed/composed_z2z"
    candidates.append(extra / suffix)
    for p in candidates:
        if p.exists():
            return p
    return None


def load_z2z(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    z2z = data.get("z2z")
    if z2z is None:
        raise KeyError(f"no 'z2z' in {path}")
    return np.asarray(z2z, dtype=np.float64)


def compare_vectors(c_direct: np.ndarray, c_comp: np.ndarray) -> dict:
    n = min(len(c_direct), len(c_comp))
    if n == 0:
        return {"corr": np.nan, "l2_dist": np.nan}
    a = np.asarray(c_direct[:n], dtype=np.float64)
    b = np.asarray(c_comp[:n], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(a, b)[0, 1]
    if np.isnan(corr):
        corr = 1.0 if np.allclose(a, b) else 0.0
    l2_dist = float(np.linalg.norm(a - b))
    return {"corr": float(corr), "l2_dist": l2_dist}


def run_baseline(
    baseline_name: str,
    layer_ig_dir: Path,
    composed_dir: Path,
    sample_indices: list[int],
) -> dict:
    results = []
    for idx in sample_indices:
        layer_file = layer_ig_dir / f"sample_{idx:05d}.json"
        comp_file = composed_dir / f"sample_{idx:05d}.json"
        if not layer_file.exists() or not comp_file.exists():
            continue
        try:
            z_d = load_z2z(layer_file)
            z_c = load_z2z(comp_file)
        except Exception as exc:
            print(f"  sample {idx}: {exc}", file=sys.stderr)
            continue
        if np.all(z_d == 0):
            continue
        layers = min(z_d.shape[0], z_c.shape[0])
        tokens = min(z_d.shape[1], z_c.shape[1], z_d.shape[2], z_c.shape[2])
        for layer in range(layers):
            for j in range(tokens):
                c_d = z_d[layer, :tokens, j]
                c_c = z_c[layer, :tokens, j]
                if np.all(c_d == 0):
                    continue
                row = compare_vectors(c_d, c_c)
                row["sample"] = idx
                row["layer"] = layer
                row["token_j"] = j
                results.append(row)
    if not results:
        return {"baseline": baseline_name, "n_pairs": 0, "n_samples": 0}
    l2_vals = [r["l2_dist"] for r in results]
    corr_vals = [r["corr"] for r in results]
    return {
        "baseline": baseline_name,
        "n_pairs": len(results),
        "n_samples": len({r["sample"] for r in results}),
        "l2_dist_mean": float(np.mean(l2_vals)),
        "l2_dist_std": float(np.std(l2_vals)),
        "corr_mean": float(np.mean(corr_vals)),
        "corr_std": float(np.std(corr_vals)),
    }


def _empty_stats_row(cfg: dict, layer_ig_suffix: str, composed_suffix: str) -> dict:
    return {
        "baseline": cfg["pair_name"],
        "pair_key": cfg["pair_key"],
        "n_pairs": 0,
        "n_samples": 0,
        "layer_ig_baseline_group": cfg["layer_ig_baseline_group"],
        "layer_ig_baseline": cfg["layer_ig_baseline"],
        "att_ig_baseline": cfg["att_ig_baseline"],
        "mlp_ig_baseline": cfg["mlp_ig_baseline"],
        "layer_ig_dir_suffix": layer_ig_suffix,
        "composed_dir_suffix": composed_suffix,
    }


def write_csv(rows: list[dict], path: Path, *, split: str, start: int, end: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_name",
        "pair_key",
        "layer_ig_baseline_group",
        "layer_ig_baseline",
        "att_ig_baseline",
        "mlp_ig_baseline",
        "split",
        "sample_start",
        "sample_end",
        "layer_ig_dir_suffix",
        "composed_dir_suffix",
        "n_pairs",
        "n_samples",
        "l2_mean",
        "l2_std",
        "corr_mean",
        "corr_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in rows:
            w.writerow(
                {
                    "pair_name": s.get("baseline", s.get("pair_name", "")),
                    "pair_key": s.get("pair_key", ""),
                    "layer_ig_baseline_group": s.get("layer_ig_baseline_group", ""),
                    "layer_ig_baseline": s.get("layer_ig_baseline", ""),
                    "att_ig_baseline": s.get("att_ig_baseline", ""),
                    "mlp_ig_baseline": s.get("mlp_ig_baseline", ""),
                    "split": split,
                    "sample_start": start,
                    "sample_end": end,
                    "layer_ig_dir_suffix": s.get("layer_ig_dir_suffix", ""),
                    "composed_dir_suffix": s.get("composed_dir_suffix", ""),
                    "n_pairs": s.get("n_pairs", 0),
                    "n_samples": s.get("n_samples", 0),
                    "l2_mean": s.get("l2_dist_mean", ""),
                    "l2_std": s.get("l2_dist_std", ""),
                    "corr_mean": s.get("corr_mean", ""),
                    "corr_std": s.get("corr_std", ""),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="LAYER z2z vs composed z2z L2 comparison")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=99)
    parser.add_argument("--layer-ig-zero-suffix", default="")
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args()

    cache = ptb_cache_root()
    base = cache / "samples" / args.split
    layer_ig_base = base / "z2z/layer_ig"
    composed_base = base / "z2z/composed"
    sample_indices = list(range(args.start, args.end + 1))

    all_stats: list[dict] = []
    for cfg in BASELINE_PAIRS:
        layer_suffix = cfg["layer_ig_suffix"]
        if cfg.get("apply_layer_ig_zero_suffix") and args.layer_ig_zero_suffix:
            layer_suffix += args.layer_ig_zero_suffix
        layer_dir = layer_ig_base / layer_suffix
        composed_dir = resolve_composed_dir(composed_base, cfg["composed_suffix"])
        if not layer_dir.exists() or composed_dir is None:
            all_stats.append(_empty_stats_row(cfg, layer_suffix, cfg["composed_suffix"]))
            continue
        stats = run_baseline(cfg["pair_name"], layer_dir, composed_dir, sample_indices)
        stats.update(
            {
                "pair_key": cfg["pair_key"],
                "layer_ig_baseline_group": cfg["layer_ig_baseline_group"],
                "layer_ig_baseline": cfg["layer_ig_baseline"],
                "att_ig_baseline": cfg["att_ig_baseline"],
                "mlp_ig_baseline": cfg["mlp_ig_baseline"],
                "layer_ig_dir_suffix": layer_suffix,
                "composed_dir_suffix": cfg["composed_suffix"],
            }
        )
        all_stats.append(stats)
        if stats["n_pairs"]:
            print(
                f"{cfg['pair_name']}: L2={stats['l2_dist_mean']:.4f} "
                f"(n_pairs={stats['n_pairs']})"
            )

    if args.csv_out:
        write_csv(all_stats, args.csv_out, split=args.split, start=args.start, end=args.end)
        print(f"Wrote {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
