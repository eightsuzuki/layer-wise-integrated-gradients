#!/usr/bin/env python3
"""Profile layer-direct IG for one PTB sample (default: sample_00410)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-idx", type=int, default=410)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--baseline", default="zero", choices=["zero", "self_input_token"])
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--layers", type=int, default=None, help="limit layers (default: all)")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer
    from utils.common.unified_bert_model import load_unified_model
    from utils.calculations.ig.z2z.layer_direct_ig import compute_layer_direct_ig_all_targets
    from scripts.reproduce.run_layer_direct_ig import load_sample_from_ptb

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: dict = {"device": device, "sample_idx": args.sample_idx, "baseline": args.baseline}

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = load_unified_model("bert-base-uncased", use_lightning_trainer=False)
    model.eval()
    model.to(device)
    sync(device)
    results["model_load_s"] = time.perf_counter() - t0

    sample = load_sample_from_ptb(args.split, args.sample_idx)
    if not sample:
        print(f"Failed to load PTB sample {args.sample_idx}", file=sys.stderr)
        return 1
    text = sample.get("text") or " ".join(sample.get("words", []))

    t1 = time.perf_counter()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states
    sync(device)
    results["full_forward_s"] = time.perf_counter() - t1
    results["seq_len"] = int(inputs["input_ids"].shape[1])
    results["num_layers"] = model.config.num_hidden_layers

    num_layers = args.layers if args.layers is not None else model.config.num_hidden_layers
    layer_times = []
    attention_mask = inputs["attention_mask"].float()

    for layer_idx in range(num_layers):
        z_layer = hidden_states[layer_idx]
        t_layer = time.perf_counter()
        compute_layer_direct_ig_all_targets(
            bert_model=model,
            z_layer=z_layer,
            attention_mask=attention_mask,
            layer_idx=layer_idx,
            num_steps=args.num_steps,
            baseline_method=args.baseline,
        )
        sync(device)
        layer_times.append(time.perf_counter() - t_layer)

    results["layer_times_s"] = layer_times
    results["layer_ig_total_s"] = sum(layer_times)
    results["layer_ig_mean_s"] = sum(layer_times) / len(layer_times) if layer_times else 0.0
    T, L, S = results["seq_len"], len(layer_times), args.num_steps
    results["est_layer_forwards"] = L * T * (1 + S) if args.baseline == "zero" else L * T * (T + S)

    print(json.dumps(results, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
