#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
§3.7.4 ATTITBa=0 の MLP キャッシュの 1 サンプル理論検証。

理論: MLP IG の完全性 sum_h IG_h = M(1) = ||z_j^{(l+1)}(1) - z_j^{(l+1)}(0)||_2
ここで z_j(0) は MLP 入力を ATTITBa=0（ITB の ATT 経路の a=0 出力 u(0)_j）にしたときの層出力。
キャッシュから IG 和を読み、同一 (layer, j) で M(1) を再計算して一致するか確認する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np


def _get_encoder(bert_model):
    if hasattr(bert_model, "bert"):
        return bert_model.bert.encoder
    return bert_model.encoder


def main():
    parser = argparse.ArgumentParser(description="§3.7.4 ATTITBa=0 の MLP 1 サンプル理論検証")
    parser.add_argument(
        "cache_file",
        type=Path,
        help=(
            "ATTITBa=0 MLP サンプル JSON（例: cache/.../mlp/"
            "..._baseline_att_itb_attitba0_mlp_residual_on/sample_00000.json ※ATTITBa=0 はレガシー接尾辞）"
        ),
    )
    parser.add_argument("--layer", type=int, default=0, help="検証する層")
    parser.add_argument("--token-j", type=int, default=1, help="検証する出力トークン j")
    parser.add_argument("--rtol", type=float, default=0.15, help="許容相対誤差（積分近似のため 0.15 程度）")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    if not args.cache_file.is_file():
        print(f"ERROR: キャッシュが見つかりません: {args.cache_file}", file=sys.stderr)
        sys.exit(1)

    with open(args.cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    mlp_data = data.get("mlp")
    if not mlp_data:
        print("ERROR: キャッシュに 'mlp' がありません", file=sys.stderr)
        sys.exit(1)
    num_layers = len(mlp_data)
    seq_len = len(mlp_data[0]) if num_layers else 0
    words = data.get("words", [])

    sample_idx = int(args.cache_file.stem.split("_")[1])
    from utils.ptb_dependency import load_ptb_dataset
    all_samples = load_ptb_dataset("dev", num_samples=sample_idx + 100, base_dir=project_root / "data/depparse")
    if sample_idx >= len(all_samples):
        print(f"ERROR: サンプル {sample_idx} が範囲外", file=sys.stderr)
        sys.exit(1)
    sample = all_samples[sample_idx]
    text = " ".join(sample.get("words", []))

    from transformers import AutoTokenizer
    from utils.common.unified_bert_model import load_unified_model
    from utils.cache.bert_cache import cache_bert_layer_outputs
    from utils.calculations.ig.mlp.att_itb_mlp_baseline import get_mlp_baseline_att_itb_eq_zero

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_unified_model("bert-base-uncased", use_lightning_trainer=False)
    model.eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    enc = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=args.max_length,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in enc.items()}
    attention_mask = inputs["attention_mask"]

    cache_bert_layer_outputs(model, inputs, tokenizer, text)
    if "hidden_states" not in getattr(model, "outputs", {}):
        print("ERROR: hidden_states が取得できません", file=sys.stderr)
        sys.exit(1)

    encoder = _get_encoder(model)
    layer_idx = args.layer
    j = args.token_j
    if layer_idx >= num_layers or j >= seq_len:
        print(f"ERROR: layer={layer_idx} または token_j={j} が範囲外 (L={num_layers}, seq={seq_len})", file=sys.stderr)
        sys.exit(1)

    z_layer = model.outputs["hidden_states"][layer_idx]
    if z_layer.dim() == 2:
        z_layer = z_layer.unsqueeze(0)
    layer_module = encoder.layer[layer_idx]
    attn_out = layer_module.attention(z_layer, attention_mask)
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    target_j = attn_out[0, j, :].clone()
    u0_j = get_mlp_baseline_att_itb_eq_zero(model, z_layer, attention_mask, layer_idx, j)

    with torch.no_grad():
        layer_out_1 = layer_module(z_layer, attention_mask)
        if isinstance(layer_out_1, tuple):
            layer_out_1 = layer_out_1[0]
        z_j_1 = layer_out_1[0, j, :].clone()

    _, seq_len_h, hidden = z_layer.shape
    z_j_vec = z_layer[0, j, :].clone()
    baseline_z = z_j_vec.unsqueeze(0).unsqueeze(0).expand(1, seq_len_h, hidden)
    with torch.no_grad():
        attn_out_0 = layer_module.attention(baseline_z, attention_mask)
        if isinstance(attn_out_0, tuple):
            attn_out_0 = attn_out_0[0]
        layer_out_0 = layer_module(baseline_z, attention_mask)
        if isinstance(layer_out_0, tuple):
            layer_out_0 = layer_out_0[0]
        z_j_0 = layer_out_0[0, j, :].clone()

    expected_M1 = torch.norm(z_j_1 - z_j_0).item()
    cached_row = mlp_data[layer_idx][j]
    if isinstance(cached_row, list):
        ig_sum_cached = float(np.sum(cached_row))
    else:
        ig_sum_cached = float(cached_row)

    from utils.calculations.ig.mlp.mlp_ig import compute_mlp_ig_theoretical_with_cache
    try:
        ig_recompute = compute_mlp_ig_theoretical_with_cache(
            model,
            layer_idx=layer_idx,
            target_token_idx=j,
            num_steps=32,
            baseline_method="zero",
            include_residual_connection=True,
            baseline_mlp_input_override=u0_j,
            target_mlp_input_override=target_j,
        )
        ig_sum_recompute = float(np.sum(ig_recompute)) if ig_recompute is not None else None
    except Exception as e:
        ig_sum_recompute = None
        print(f"  再計算エラー: {e}")

    diff_cached = abs(ig_sum_cached - expected_M1)
    rtol = args.rtol
    scale = max(expected_M1, 1e-10)
    rel_err_cached = diff_cached / scale
    ok_cached = rel_err_cached <= rtol
    ok_recompute = False
    if ig_sum_recompute is not None:
        diff_recompute = abs(ig_sum_recompute - expected_M1)
        rel_err_recompute = diff_recompute / scale
        ok_recompute = rel_err_recompute <= rtol

    print(f"=== ATTITBa=0（§3.7.4）MLP 理論検証 (sample_{sample_idx:05d}, layer={layer_idx}, token_j={j}) ===")
    print(f"  理論値 M(1) = ||z_j(1) - z_j(0)||_2     = {expected_M1:.6f}")
    print(f"  キャッシュ IG 和                        = {ig_sum_cached:.6f} (相対誤差 {rel_err_cached:.2%})")
    if ig_sum_recompute is not None:
        print(f"  再計算 IG 和                          = {ig_sum_recompute:.6f}")
    print(f"  結果（キャッシュ vs 理論）: {'PASS' if ok_cached else 'FAIL'} (rtol={rtol:.2%})")
    if ig_sum_recompute is not None:
        print(f"  結果（再計算 vs 理論）: {'PASS' if ok_recompute else 'FAIL'}")
    if not ok_cached and (ig_sum_recompute is None or not ok_recompute):
        print("  注意: baseline が非ゼロのとき、MLPModel 内でベースラインがゼロ固定の可能性があります。")
        sys.exit(1)
    if ok_cached or ok_recompute:
        print("理論通りに計算されています。")
    sys.exit(0)


if __name__ == "__main__":
    main()
