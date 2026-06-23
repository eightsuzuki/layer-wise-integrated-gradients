#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer IG キャッシュの完全性検証。

理論: sum_i IG_{i,j}^{Layer} ≈ L_j(1) - L_j(0)。zero ベースラインでは L_j(0)=0 なので
sum_i IG_{i,j} ≈ L_j(1) = ||z_j^{(l+1)}(1) - z_j^{(l+1)}(0)||_2。

キャッシュから z2z[layer][:, j] の和を取り、同じ入力で LayerDirectIGWrapper による
L_j(1) を再計算して rtol 内で一致するか確認する。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_sample_from_ptb(split: str, sample_idx: int, base_dir: Path) -> dict | None:
    from utils.ptb_dependency.common.dependency_syntax import load_ptb_dataset

    try:
        all_samples = load_ptb_dataset(
            split=split,
            num_samples=sample_idx + 1,
            base_dir=base_dir,
            data_format="txt",
        )
        if sample_idx >= len(all_samples):
            return None
        sample = all_samples[sample_idx]
        if "words" in sample and "text" not in sample:
            sample["text"] = " ".join(sample["words"])
        return sample
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Layer IG キャッシュの完全性: sum_i IG_{i,j} ≈ L_j(1) を検証"
    )
    parser.add_argument(
        "cache_file",
        type=Path,
        help="Layer IG サンプル JSON のパス（例: cache/.../layer_ig/..._baseline_zero/sample_00000.json）",
    )
    parser.add_argument(
        "--split",
        default="dev",
        help="PTB の split（パスから推論できない場合に使用）",
    )
    parser.add_argument(
        "--ptb-data-dir",
        type=Path,
        default=Path("data/depparse"),
        help="PTB データディレクトリ",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="検証する層インデックス",
    )
    parser.add_argument(
        "--token-j",
        type=int,
        default=1,
        help="検証するターゲットトークン j（0 は [CLS] のため 1 を推奨）",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.40,
        help="許容相対誤差（積分近似のため 0.40 程度）",
    )
    parser.add_argument(
        "--model",
        default="bert-base-uncased",
        help="モデル名",
    )
    parser.add_argument(
        "--maxlen",
        type=int,
        default=128,
        help="最大系列長",
    )
    args = parser.parse_args()

    cache_path = args.cache_file
    if not cache_path.is_file():
        print(f"ERROR: キャッシュが見つかりません: {cache_path}", file=sys.stderr)
        sys.exit(1)

    # パスから split を推論（.../samples/dev/... -> dev）
    parts = cache_path.parts
    if "samples" in parts:
        idx = parts.index("samples")
        if idx + 1 < len(parts):
            args.split = parts[idx + 1]
    sample_idx = int(cache_path.stem.split("_")[1])  # sample_00000 -> 0

    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    z2z = data.get("z2z")
    if not z2z:
        print("ERROR: キャッシュに 'z2z' がありません", file=sys.stderr)
        sys.exit(1)
    z2z_arr = np.asarray(z2z, dtype=np.float64)
    if z2z_arr.ndim != 3:
        print(f"ERROR: z2z は [L,T,T] である必要があります。shape={z2z_arr.shape}", file=sys.stderr)
        sys.exit(1)

    num_layers, seq_len, _ = z2z_arr.shape
    if args.layer < 0 or args.layer >= num_layers:
        print(f"ERROR: --layer は 0..{num_layers - 1} の範囲で指定してください", file=sys.stderr)
        sys.exit(1)
    if args.token_j < 0 or args.token_j >= seq_len:
        print(f"ERROR: --token-j は 0..{seq_len - 1} の範囲で指定してください", file=sys.stderr)
        sys.exit(1)

    ig_sum_cache = float(np.sum(z2z_arr[args.layer, :, args.token_j]))
    baseline_method = data.get("_metadata", {}).get("z2z_baseline_method", "zero")

    sample_data = load_sample_from_ptb(args.split, sample_idx, args.ptb_data_dir)
    if not sample_data:
        print(f"ERROR: PTB サンプル {sample_idx} を読み込めません", file=sys.stderr)
        sys.exit(1)
    text = sample_data.get("text") or " ".join(sample_data.get("words", []))

    import torch
    from transformers import AutoTokenizer

    from utils.common.unified_bert_model import load_unified_model
    from utils.calculations.ig.z2z.layer_direct_ig import (
        LayerDirectIGWrapper,
        _compute_baseline_z,
    )
    from utils.calculations.shared.device_utils import ensure_model_on_device

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_unified_model(args.model, use_lightning_trainer=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.maxlen,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states
    if not hidden_states:
        print("ERROR: hidden_states が取得できません", file=sys.stderr)
        sys.exit(1)

    z_layer = hidden_states[args.layer]
    attention_mask = inputs["attention_mask"]
    ensure_model_on_device(model)

    baseline_z = _compute_baseline_z(
        baseline_method=baseline_method,
        z_layer=z_layer,
        bert_model=model,
        layer_idx=args.layer,
        target_token_idx=args.token_j,
        attention_mask=attention_mask,
    )
    wrapper = LayerDirectIGWrapper(
        bert_model=model,
        layer_idx=args.layer,
        target_token_idx=args.token_j,
        attention_mask=attention_mask,
    )
    wrapper.to(device)
    wrapper.eval()
    wrapper.set_baseline_output(baseline_z)
    with torch.no_grad():
        L_j_at_input = wrapper(z_layer).squeeze().cpu().item()

    diff = abs(ig_sum_cache - L_j_at_input)
    rtol_val = args.rtol * max(abs(L_j_at_input), 1e-10)
    passed = diff <= rtol_val
    status = "PASS" if passed else "FAIL"
    print(
        f"{status} layer={args.layer} j={args.token_j} "
        f"ig_sum_cache={ig_sum_cache:.6f} L_j(1)={L_j_at_input:.6f} "
        f"diff={diff:.6f} rtol*L_j={rtol_val:.6f}"
    )
    if not passed:
        print(
            f"  完全性: sum_i IG_{{i,j}} ≈ L_j(1) が rtol={args.rtol} で満たされていません。",
            file=sys.stderr,
        )
        sys.exit(1)
    print("  完全性 OK（理論通り）")
    sys.exit(0)


if __name__ == "__main__":
    main()
