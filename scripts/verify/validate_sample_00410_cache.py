#!/usr/bin/env python3
"""Validate sample_00410 layer-direct and composed z2z caches."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reproduce.compose_z2z import COMBINATIONS
from utils.calculations.ig.z2z.compose_att_mlp import compute_z2z_from_att_mlp
from utils.calculations.ig.z2z.layer_itb_zero_ratio import apply_layer_z2z_itb_zero_base_ratio
from utils.reproduce.ptb_loader import ptb_cache_root

_BASE = "steps32_bert-base-uncased_maxlen128_z_to_z"
SAMPLE_IDX = 410
SAMPLE_NAME = f"sample_{SAMPLE_IDX:05d}.json"

LAYER_SPECS = [
    {
        "name": "zero",
        "baseline_method": "zero",
        "cache_suffix": f"{_BASE}_layer_ig_baseline_zero",
    },
    {
        "name": "itb",
        "baseline_method": "self_input_token",
        "cache_suffix": f"{_BASE}_layer_ig_baseline_self_input_token",
    },
    {
        "name": "itb_zero_ratio",
        "baseline_method": None,
        "cache_suffix": (
            f"{_BASE}_layer_ig_baseline_self_input_token_self_contrib_zero_base_ratio"
        ),
    },
]


def load_z2z(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    z2z = data.get("z2z")
    if z2z is None:
        raise KeyError(f"no 'z2z' in {path}")
    return np.asarray(z2z, dtype=np.float64)


def compare_arrays(
    cached: np.ndarray,
    fresh: np.ndarray,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> dict:
    if cached.shape != fresh.shape:
        return {
            "match": False,
            "reason": f"shape mismatch {cached.shape} vs {fresh.shape}",
        }
    diff = np.abs(cached - fresh)
    max_abs = float(diff.max())
    rel = diff / (np.abs(fresh) + 1e-12)
    max_rel = float(rel.max())
    ok = np.allclose(cached, fresh, rtol=rtol, atol=atol)
    return {
        "match": bool(ok),
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "rtol": rtol,
        "atol": atol,
    }


def recompute_layer_direct(
    baseline_method: str,
    split: str = "dev",
    sample_idx: int = SAMPLE_IDX,
    num_steps: int = 32,
) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer

    from scripts.reproduce.run_layer_direct_ig import (
        compute_direct_z2z_for_sample,
        load_sample_from_ptb,
    )
    from utils.common.unified_bert_model import load_unified_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = load_unified_model("bert-base-uncased", use_lightning_trainer=False)
    model.eval()
    model.to(device)

    sample = load_sample_from_ptb(split, sample_idx)
    if not sample:
        raise RuntimeError(f"PTB sample {sample_idx} not found")

    result = compute_direct_z2z_for_sample(
        unified_model=model,
        tokenizer=tokenizer,
        sample_data=sample,
        num_steps=num_steps,
        baseline_method=baseline_method,
    )
    if result is None:
        raise RuntimeError(f"recompute failed for baseline={baseline_method}")
    z2z_list, _ = result
    return np.asarray(z2z_list, dtype=np.float64)


def validate_layer_direct(
    cache_root: Path,
    split: str,
    recompute: bool,
    rtol: float,
    atol: float,
) -> list[dict]:
    layer_base = cache_root / "samples" / split / "z2z" / "layer_ig"
    rows: list[dict] = []

    zero_path = layer_base / LAYER_SPECS[0]["cache_suffix"] / SAMPLE_NAME
    itb_path = layer_base / LAYER_SPECS[1]["cache_suffix"] / SAMPLE_NAME
    ratio_path = layer_base / LAYER_SPECS[2]["cache_suffix"] / SAMPLE_NAME

    for spec in LAYER_SPECS:
        path = layer_base / spec["cache_suffix"] / SAMPLE_NAME
        row: dict = {"method": spec["name"], "cache_path": str(path), "exists": path.exists()}
        if not path.exists():
            row["status"] = "MISSING"
            rows.append(row)
            continue

        cached = load_z2z(path)
        row["shape"] = list(cached.shape)

        if spec["name"] == "itb_zero_ratio":
            if not zero_path.exists() or not itb_path.exists():
                row["status"] = "SKIP_DERIVE"
                row["reason"] = "need zero+itb caches for derivation"
            else:
                derived = apply_layer_z2z_itb_zero_base_ratio(
                    load_z2z(itb_path), load_z2z(zero_path)
                )
                cmp = compare_arrays(cached, derived, rtol=rtol, atol=atol)
                row.update(cmp)
                row["check"] = "derived_from_itb_zero"
                row["status"] = "PASS" if cmp["match"] else "FAIL"
            rows.append(row)
            continue

        if recompute:
            t0 = time.perf_counter()
            fresh = recompute_layer_direct(spec["baseline_method"], split=split)
            row["recompute_s"] = round(time.perf_counter() - t0, 2)
            cmp = compare_arrays(cached, fresh, rtol=rtol, atol=atol)
            row.update(cmp)
            row["check"] = "recompute"
            row["status"] = "PASS" if cmp["match"] else "FAIL"
        else:
            row["check"] = "exists_only"
            row["status"] = "OK"
        rows.append(row)

    return rows


def validate_composed(cache_root: Path, split: str, rtol: float, atol: float) -> list[dict]:
    base = cache_root / "samples" / split
    att_base = base / "att"
    mlp_base = base / "mlp"
    composed_base = base / "z2z" / "composed"
    rows: list[dict] = []

    for att_dir, mlp_dir, out_dir in COMBINATIONS:
        att_path = att_base / att_dir / SAMPLE_NAME
        mlp_path = mlp_base / mlp_dir / SAMPLE_NAME
        out_path = composed_base / out_dir / SAMPLE_NAME
        row: dict = {
            "composed_suffix": out_dir,
            "cache_path": str(out_path),
            "exists": out_path.exists(),
            "att_exists": att_path.exists(),
            "mlp_exists": mlp_path.exists(),
        }
        if not out_path.exists():
            row["status"] = "MISSING"
            rows.append(row)
            continue

        cached = load_z2z(out_path)
        row["shape"] = list(cached.shape)

        if not att_path.exists() or not mlp_path.exists():
            row["status"] = "OK_NO_RECOMPOSE"
            rows.append(row)
            continue

        att_data = json.loads(att_path.read_text(encoding="utf-8"))
        mlp_data = json.loads(mlp_path.read_text(encoding="utf-8"))
        z2z = compute_z2z_from_att_mlp(att_data.get("attns"), mlp_data.get("mlp"))
        if not z2z:
            row["status"] = "RECOMPOSE_EMPTY"
            rows.append(row)
            continue

        fresh = np.asarray(z2z, dtype=np.float64)
        cmp = compare_arrays(cached, fresh, rtol=rtol, atol=atol)
        row.update(cmp)
        row["check"] = "recompose_att_mlp"
        row["status"] = "PASS" if cmp["match"] else "FAIL"
        rows.append(row)

    return rows


def write_report(
    out_path: Path,
    layer_rows: list[dict],
    composed_rows: list[dict],
    args: argparse.Namespace,
    cache_root: Path,
) -> None:
    lines = [
        "# sample_00410 cache validation",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Sample: PTB dev #{SAMPLE_IDX}",
        f"- Cache root: `{cache_root}`",
        f"- Recompute layer IG: `{args.recompute_layer_ig}`",
        "",
        "## Layer-direct z2z (3 methods)",
        "",
        "| method | status | check | shape | max_abs_diff | max_rel_diff | recompute_s |",
        "|--------|--------|-------|-------|--------------|--------------|-------------|",
    ]
    for r in layer_rows:
        shape = r.get("shape", "")
        lines.append(
            f"| {r['method']} | {r.get('status', '')} | {r.get('check', '')} | "
            f"{shape} | {r.get('max_abs_diff', '')} | {r.get('max_rel_diff', '')} | "
            f"{r.get('recompute_s', '')} |"
        )

    lines.extend(
        [
            "",
            "## Composed z2z (8 combinations)",
            "",
            "| suffix | status | check | shape | max_abs_diff | max_rel_diff |",
            "|--------|--------|-------|-------|--------------|--------------|",
        ]
    )
    for r in composed_rows:
        suffix = r["composed_suffix"].replace(f"{_BASE}_", "")
        lines.append(
            f"| {suffix} | {r.get('status', '')} | {r.get('check', '')} | "
            f"{r.get('shape', '')} | {r.get('max_abs_diff', '')} | "
            f"{r.get('max_rel_diff', '')} |"
        )

    layer_fail = sum(1 for r in layer_rows if r.get("status") in ("FAIL", "MISSING"))
    comp_fail = sum(1 for r in composed_rows if r.get("status") in ("FAIL", "MISSING"))
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Layer-direct failures/missing: {layer_fail} / {len(layer_rows)}",
            f"- Composed failures/missing: {comp_fail} / {len(composed_rows)}",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--recompute-layer-ig", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "scripts/verify/reports/sample_00410_cache_validation.md",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    cache_root = args.cache_root or ptb_cache_root()
    layer_rows = validate_layer_direct(
        cache_root, args.split, args.recompute_layer_ig, args.rtol, args.atol
    )
    composed_rows = validate_composed(cache_root, args.split, args.rtol, args.atol)

    write_report(args.report, layer_rows, composed_rows, args, cache_root)

    payload = {
        "sample_idx": SAMPLE_IDX,
        "layer_direct": layer_rows,
        "composed": composed_rows,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    failures = [
        r for r in layer_rows + composed_rows if r.get("status") in ("FAIL", "MISSING")
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
