#!/usr/bin/env python3
"""Build the MLP (u -> z) IG caches the paper composes from.

Attribution is taken in head space -- with respect to ``u^{(l,h)}``, the head
output *before* the output projection ``W_o``, which is what Eq. (mlp-o)
differentiates. Slicing the post-projection residual coordinates into
``head_dim`` blocks instead does not give per-head quantities, because ``W_o``
has already mixed them.

Completeness is already tight at ``--num-steps 32`` here (relative error on the
order of 1e-7), so unlike the ATT boundary this side does not need a larger
step count.

Output goes to::

    $PTB_CACHE_ROOT/samples/<split>/mlp/
        steps32_bert-base-uncased_maxlen128_u_to_z_baseline_zero_mlp_residual_on__headspacefix
        steps32_bert-base-uncased_maxlen128_u_to_z_baseline_att_itb_attitba0_mlp_residual_on__headspacefix

which is what ``compose_z2z.py`` reads by default.

Requires PTB (LDC99T42) -- see docs/REPRODUCTION.md. Nothing is downloaded.

Sharing a GPU: pass ``--device cuda:1`` to pick a specific card, or split the
corpus across cards by running several shards::

    python scripts/reproduce/run_mlp_head_space_ig.py --start 0    --end 849  --device cuda:0 &
    python scripts/reproduce/run_mlp_head_space_ig.py --start 850  --end 1699 --device cuda:1 &

Shards write disjoint files and skip work that is already on disk, so a run can
be interrupted and resumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import BertModel, BertTokenizerFast

from utils.calculations.ig.mlp.bert_mlp_head_space_ig import (
    aggregate_subword_mlp_to_words,
    compute_mlp_head_space_ig,
)
from utils.reproduce.device import resolve as resolve_device
from utils.reproduce.ptb_loader import (
    load_ptb_dataset,
    ptb_cache_root,
    require_ptb_depparse_dir,
)

BASELINE_DIR_NAME = {
    "zero": "baseline_zero_mlp_residual_on",
    "attitba0": "baseline_att_itb_attitba0_mlp_residual_on",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="dev")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=1699, help="inclusive")
    p.add_argument("--num-samples", type=int, default=1700)
    p.add_argument("--num-steps", type=int, default=32)
    p.add_argument("--mlp-baselines", default="zero,attitba0")
    p.add_argument("--max-sequence-length", type=int, default=128)
    p.add_argument("--out-suffix", default="__headspacefix")
    p.add_argument(
        "--device",
        default="auto",
        help="'auto' picks the CUDA card with the most free memory; or cuda:N / cpu",
    )
    p.add_argument(
        "--completeness-warn",
        type=float,
        default=0.05,
        help="print a warning when an IG column's completeness error exceeds this",
    )
    args = p.parse_args()

    baselines = [b.strip() for b in args.mlp_baselines.split(",") if b.strip()]
    unknown = [b for b in baselines if b not in BASELINE_DIR_NAME]
    if unknown:
        raise SystemExit(f"unknown MLP baseline(s): {unknown} (choose from {list(BASELINE_DIR_NAME)})")

    depparse_dir = require_ptb_depparse_dir()
    device = resolve_device(args.device)
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = (
        BertModel.from_pretrained("bert-base-uncased", attn_implementation="eager")
        .eval()
        .to(device)
    )
    for param in model.parameters():
        param.requires_grad_(False)

    samples = load_ptb_dataset(args.split, num_samples=args.num_samples, base_dir=depparse_dir)

    cache_root = ptb_cache_root()
    out_dirs = {}
    for b in baselines:
        name = (
            f"steps{args.num_steps}_bert-base-uncased_maxlen{args.max_sequence_length}"
            f"_u_to_z_{BASELINE_DIR_NAME[b]}{args.out_suffix}"
        )
        d = cache_root / "samples" / args.split / "mlp" / name
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[b] = d
        print(f"{b} -> {d}", file=sys.stderr)

    num_layers = model.config.num_hidden_layers
    t0 = time.time()
    done = 0
    worst_rel_err = 0.0

    for idx in range(args.start, min(args.end + 1, len(samples))):
        out_files = {b: out_dirs[b] / f"sample_{idx:05d}.json" for b in baselines}
        pending = [b for b in baselines if not out_files[b].exists()]
        if not pending:
            continue
        words = samples[idx].get("words", [])
        if not words:
            continue

        enc = tok(
            " ".join(words),
            return_tensors="pt",
            max_length=args.max_sequence_length,
            truncation=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hidden_states = model(**enc, output_hidden_states=True).hidden_states
        tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
        seq_len = len(tokens)

        per_baseline = {b: [] for b in pending}
        for layer_idx in range(num_layers):
            z = hidden_states[layer_idx]
            layer_rows = {b: [] for b in pending}
            for j in range(seq_len):
                for b in pending:
                    per_head, _total, verification = compute_mlp_head_space_ig(
                        model,
                        layer_idx,
                        j,
                        z,
                        enc["attention_mask"],
                        mlp_baseline=b,
                        num_steps=args.num_steps,
                    )
                    rel_err = float(verification["relative_error"])
                    worst_rel_err = max(worst_rel_err, rel_err)
                    if rel_err > args.completeness_warn:
                        print(
                            f"[warn] completeness rel_err={rel_err:.4f} "
                            f"sample={idx} layer={layer_idx} token={j} baseline={b}",
                            file=sys.stderr,
                            flush=True,
                        )
                    layer_rows[b].append(per_head.tolist())
            for b in pending:
                per_baseline[b].append(layer_rows[b])

        for b, mlp_results in per_baseline.items():
            # The ATT cache is in word space, so aggregate here too; leaving the
            # MLP side in subword space would make the composition pair
            # different tokens on the two sides.
            mlp_words = aggregate_subword_mlp_to_words(
                tok, words, mlp_results, max_sequence_length=args.max_sequence_length
            )
            payload = {
                "tokens": tokens,
                "words": words,
                "mlp": mlp_words,
                "_metadata": {
                    "mlp_baseline_method": b,
                    "mlp_residual_mode": "mlp_residual_on",
                    "per_head_boundary": "pre_proj",
                    "token_space": "word (aggregated from subwords, sum)",
                    "num_steps": args.num_steps,
                    "worst_completeness_rel_err": worst_rel_err,
                },
            }
            tmp = out_files[b].with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.rename(out_files[b])  # atomic, so a killed shard leaves no half file

        done += 1
        if done % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"[{idx}] {done} sentences / {elapsed / 60:.1f} min "
                f"({elapsed / done:.2f} s per sentence)",
                file=sys.stderr,
                flush=True,
            )

    print(
        f"done: {done} sentences written, worst completeness rel_err {worst_rel_err:.2e}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
