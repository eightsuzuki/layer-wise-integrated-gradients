#!/usr/bin/env python3
"""Compose z2z from ATT + MLP PTB caches (public release combinations).

Defaults reproduce the paper: affine composition (Eq. layer-decomp-hat) from
the head-space MLP caches. --mlp-suffix selects which MLP cache to read; the
paper uses the head-space ones written by scripts/reproduce/run_mlp_head_space_ig.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.calculations.ig.z2z.compose_att_mlp import compute_z2z_from_att_mlp
from utils.reproduce.ptb_loader import ptb_cache_root

PREFIX_ATT = "steps32_bert-base-uncased_maxlen128_z_to_u"
PREFIX_MLP = "steps32_bert-base-uncased_maxlen128_u_to_z"
PREFIX_Z2Z = "steps32_bert-base-uncased_maxlen128_z_to_z"

COMBINATIONS = [
    (
        f"{PREFIX_ATT}_baseline_zero",
        f"{PREFIX_MLP}_baseline_zero_mlp_residual_on",
        f"{PREFIX_Z2Z}_ATT_zero_MLP_zero",
    ),
    (
        f"{PREFIX_ATT}_baseline_zero",
        "steps32_bert-base-uncased_maxlen128_u_to_z_baseline_att_itb_attitba0_mlp_residual_on",
        f"{PREFIX_Z2Z}_ATT_zero_MLP_ATTITBa0",
    ),
    (
        f"{PREFIX_ATT}_baseline_self_input_token_direct_zero",
        f"{PREFIX_MLP}_baseline_zero_mlp_residual_on",
        f"{PREFIX_Z2Z}_ATT_ITB_raw_MLP_zero",
    ),
    (
        f"{PREFIX_ATT}_baseline_self_input_token_direct_zero",
        "steps32_bert-base-uncased_maxlen128_u_to_z_baseline_att_itb_attitba0_mlp_residual_on",
        f"{PREFIX_Z2Z}_ATT_ITB_raw_MLP_ATTITBa0",
    ),
    (
        f"{PREFIX_ATT}_baseline_self_input_token_self_contrib_att_map_ratio",
        f"{PREFIX_MLP}_baseline_zero_mlp_residual_on",
        f"{PREFIX_Z2Z}_ATT_ITB_map_MLP_zero",
    ),
    (
        f"{PREFIX_ATT}_baseline_self_input_token_self_contrib_att_map_ratio",
        "steps32_bert-base-uncased_maxlen128_u_to_z_baseline_att_itb_attitba0_mlp_residual_on",
        f"{PREFIX_Z2Z}_ATT_ITB_map_MLP_ATTITBa0",
    ),
    (
        f"{PREFIX_ATT}_baseline_self_input_token_self_contrib_zero_base_ratio",
        f"{PREFIX_MLP}_baseline_zero_mlp_residual_on",
        f"{PREFIX_Z2Z}_ATT_ITB_zero_base_ratio_MLP_zero",
    ),
    (
        f"{PREFIX_ATT}_baseline_self_input_token_self_contrib_zero_base_ratio",
        "steps32_bert-base-uncased_maxlen128_u_to_z_baseline_att_itb_attitba0_mlp_residual_on",
        f"{PREFIX_Z2Z}_ATT_ITB_zero_base_ratio_MLP_ATTITBa0",
    ),
]


def process_sample(att_path: Path, mlp_path: Path, out_path: Path, mode: str) -> bool:
    if out_path.exists():
        return True
    if not att_path.exists() or not mlp_path.exists():
        return False
    att_data = json.loads(att_path.read_text(encoding="utf-8"))
    mlp_data = json.loads(mlp_path.read_text(encoding="utf-8"))
    z2z = compute_z2z_from_att_mlp(att_data.get("attns"), mlp_data.get("mlp"), mode=mode)
    if not z2z:
        return False
    payload = {
        "tokens": att_data.get("tokens", mlp_data.get("tokens", [])),
        "z2z": z2z,
        "_metadata": {
            "composed_from": [str(att_path), str(mlp_path)],
            "composition_mode": mode,
            "source_att_dir": att_path.parent.name,
            "source_mlp_dir": mlp_path.parent.name,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=99)
    parser.add_argument(
        "--mode", choices=["affine", "prod"], default="affine",
        help="affine reproduces the paper; prod is the older unnormalized chaining",
    )
    parser.add_argument(
        "--mlp-suffix", default="__headspacefix",
        help="suffix of the MLP cache directories to read. The paper uses the "
             "head-space caches (per-head split taken before the output "
             "projection, matching Eq. mlp-o); pass '' for the older caches.",
    )
    parser.add_argument(
        "--out-suffix", default="",
        help="suffix for the composed output directories (keeps runs separate)",
    )
    args = parser.parse_args()

    cache = ptb_cache_root()
    base = cache / "samples" / args.split
    att_base = base / "att"
    mlp_base = base / "mlp"
    out_base = base / ("z2z/composed_affine" if args.mode == "affine" else "z2z/composed")

    for att_dir, mlp_dir, out_dir in COMBINATIONS:
        att_path = att_base / att_dir
        mlp_path = mlp_base / (mlp_dir + args.mlp_suffix)
        dest = out_base / (out_dir + args.out_suffix)
        if not mlp_path.is_dir():
            print(
                f"missing MLP cache {mlp_path}\n"
                f"  build it with scripts/reproduce/run_mlp_head_space_ig.py, "
                f"or pass --mlp-suffix '' to use the older non-head-space caches "
                f"(which do not reproduce the published numbers).",
                file=sys.stderr,
            )
            return 1
        ok = skipped = 0
        for idx in range(args.start, args.end + 1):
            name = f"sample_{idx:05d}.json"
            if process_sample(att_path / name, mlp_path / name, dest / name, args.mode):
                ok += 1
            else:
                skipped += 1
        print(f"{out_dir}: wrote/skipped {ok}/{skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
