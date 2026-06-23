#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PTB 用 Layer（一気通貫）直接 z2z IG 実行スクリプト

各サンプルで BERT を output_hidden_states=True で 1 回 forward し、
hidden_states[layer_idx] を z^{(l)} として全層・全ターゲットで
compute_layer_direct_ig_all_targets を呼び、[num_layers, num_tokens, num_tokens]
を合成 z2z と同じ形式で JSON に保存する。
"""

import argparse
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
import torch
from transformers import AutoTokenizer

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from utils.common.logging_setup import get_logger, setup_unified_logging
from utils.common.unified_bert_model import UnifiedBertModel, load_unified_model
from utils.calculations.ig.z2z.layer_direct_ig import compute_layer_direct_ig_all_targets
from utils.reproduce.ptb_loader import load_ptb_dataset, require_ptb_depparse_dir, ptb_cache_root

logger = get_logger(__name__)


def load_sample_from_ptb(
    split: str,
    sample_idx: int,
    base_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """PTB から 1 サンプルを読み込む。"""
    if base_dir is None:
        base_dir = require_ptb_depparse_dir()
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
    except Exception as e:
        logger.error("PTB 読み込みエラー (sample %s): %s", sample_idx, e)
        return None


def save_json_with_z2z(sample_file: Path, data: Dict[str, Any]) -> bool:
    """z2z を含む JSON を保存する。"""
    try:
        temp_file = sample_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        temp_file.rename(sample_file)
        return True
    except Exception as e:
        logger.error("JSON 保存エラー (%s): %s", sample_file, e)
        return False


def compute_direct_z2z_for_sample(
    unified_model: UnifiedBertModel,
    tokenizer: Any,
    sample_data: Dict[str, Any],
    num_steps: int,
    baseline_method: str,
    max_sequence_length: int = 128,
) -> Optional[tuple]:
    """
    1 サンプルについて Layer 直接 z2z を計算する。

    Returns:
        (z2z_list, tokens_with_special) または None
        z2z_list: List of [seq_len, seq_len], length = num_layers
    """
    text = sample_data.get("text", "")
    if not text:
        words = sample_data.get("words", [])
        text = " ".join(words) if words else ""
    if not text:
        logger.warning("テキストが取得できません")
        return None

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_sequence_length,
    )
    device = next(unified_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = unified_model(**inputs)

    hidden_states = outputs.hidden_states
    if not hidden_states:
        logger.warning("hidden_states が取得できません")
        return None

    num_layers = unified_model.config.num_hidden_layers
    # hidden_states[0] = embedding, hidden_states[layer_idx] = 層 layer_idx の入力 z^{(layer_idx)}
    attention_mask = inputs["attention_mask"]

    z2z_list: List[List[List[float]]] = []
    for layer_idx in range(num_layers):
        z_layer = hidden_states[layer_idx]  # [1, seq_len, hidden]
        mat = compute_layer_direct_ig_all_targets(
            bert_model=unified_model,
            z_layer=z_layer,
            attention_mask=attention_mask,
            layer_idx=layer_idx,
            num_steps=num_steps,
            baseline_method=baseline_method,
        )
        z2z_list.append(mat.tolist())

    # トークン列（[CLS] ... [SEP]）
    token_ids = inputs["input_ids"][0]
    tokens_with_special = tokenizer.convert_ids_to_tokens(token_ids.tolist())
    return z2z_list, tokens_with_special


def run(
    split: str = "dev",
    start_sample: int = 0,
    end_sample: Optional[int] = None,
    num_samples: int = 100,
    baseline_method: str = "zero",
    ig_num_steps: int = 32,
    model_name: str = "bert-base-uncased",
    max_sequence_length: int = 128,
    ptb_data_dir: Optional[Path] = None,
    no_cache: bool = False,
    output_suffix: Optional[str] = None,
) -> Dict[str, Any]:
    """PTB サンプル範囲で Layer 直接 z2z を計算し、JSON で保存する。"""
    ptb_data_dir = ptb_data_dir or require_ptb_depparse_dir()
    safe_model_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    # Layer 全体で IG（一気通貫）。ATT/MLP を分けず層 z^(l)→z^(l+1) を 1 本で積分
    output_dir_name = (
        f"steps{ig_num_steps}_{safe_model_name}_maxlen{max_sequence_length}"
        f"_z_to_z_layer_ig_baseline_{baseline_method}"
    )
    if output_suffix:
        output_dir_name = f"{output_dir_name}_{output_suffix}"
    sample_cache_dir = (
        ptb_cache_root()
        / f"samples/{split}/z2z/layer_ig/{output_dir_name}"
    )
    sample_cache_dir.mkdir(parents=True, exist_ok=True)

    if end_sample is None:
        end_sample = num_samples - 1
    actual_end = min(end_sample, num_samples - 1)
    sample_range = list(range(start_sample, actual_end + 1))

    logger.info("Layer 直接 z2z 計算開始")
    logger.info("分割: %s, サンプル: %s〜%s", split, start_sample, actual_end)
    logger.info("ベースライン: %s, ステップ数: %s", baseline_method, ig_num_steps)
    logger.info("出力: %s", sample_cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    unified_model = load_unified_model(model_name, use_lightning_trainer=False)
    unified_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unified_model.to(device)
    logger.info("device: %s", device)
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA が利用できません。CPU で実行しています。GPU を使う場合は UV の .venv で実行してください: bash scripts/run_phase_a_uv.sh"
        )

    stats = {"processed": 0, "skipped": 0, "errors": 0}

    for sample_idx in sample_range:
        sample_file = sample_cache_dir / f"sample_{sample_idx:05d}.json"
        if not no_cache and sample_file.exists():
            try:
                with open(sample_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if existing.get("z2z") and existing.get("_metadata", {}).get("z2z_computation_method") in ("direct", "layer_ig"):
                    stats["skipped"] += 1
                    continue
            except Exception:
                pass

        sample_data = load_sample_from_ptb(split=split, sample_idx=sample_idx, base_dir=ptb_data_dir)
        if sample_data is None:
            stats["errors"] += 1
            continue

        try:
            result = compute_direct_z2z_for_sample(
                unified_model=unified_model,
                tokenizer=tokenizer,
                sample_data=sample_data,
                num_steps=ig_num_steps,
                baseline_method=baseline_method,
                max_sequence_length=max_sequence_length,
            )
        except Exception as e:
            logger.exception("サンプル %s の計算エラー: %s", sample_idx, e)
            stats["errors"] += 1
            continue

        if result is None:
            stats["errors"] += 1
            continue

        z2z_list, tokens_with_special = result
        tokens_no_special = sample_data.get("tokens", []) or sample_data.get("words", [])
        if tokens_no_special and tokens_no_special[0] == "[CLS]" and tokens_no_special[-1] == "[SEP]":
            tokens_no_special = tokens_no_special[1:-1]
        words = tokens_no_special.copy()

        output_data = {
            "tokens": tokens_with_special,
            "tokens_without_special": tokens_no_special,
            "words": words,
            "words_with_special": tokens_with_special,
            "relns": sample_data.get("relns", []),
            "heads": sample_data.get("heads", []),
            "z2z": z2z_list,
            "_metadata": {
                "z2z_computation_method": "layer_ig",
                "z2z_baseline_method": baseline_method,
                "z2z_computed_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
                "ig_num_steps": ig_num_steps,
                "has_cls_sep_tokens": True,
            },
        }

        if save_json_with_z2z(sample_file, output_data):
            stats["processed"] += 1
        else:
            stats["errors"] += 1

    logger.info("完了: 処理=%s, スキップ=%s, エラー=%s", stats["processed"], stats["skipped"], stats["errors"])
    return {"stats": stats, "output_dir": str(sample_cache_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="PTB Layer 直接 z2z IG 実行")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--start-sample", type=int, default=0)
    parser.add_argument("--end-sample", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--baseline-method", type=str, default="zero", choices=["zero", "self_input_token"])
    parser.add_argument("--ig-num-steps", type=int, default=32)
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--ptb-data-dir", type=Path, default=Path("data/depparse"))
    parser.add_argument("--no-cache", action="store_true", help="既存ファイルを上書き")
    parser.add_argument("--output-suffix", type=str, default=None, help="出力ディレクトリ名に付与する接尾辞（再計算時に既存を上書きしない場合に e.g. OLD を指定）")
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    if args.log_file:
        setup_unified_logging(log_file_path=args.log_file, log_level=logging.INFO, enable_console=True, enable_file=True)

    run(
        split=args.split,
        start_sample=args.start_sample,
        end_sample=args.end_sample,
        num_samples=args.num_samples,
        baseline_method=args.baseline_method,
        ig_num_steps=args.ig_num_steps,
        model_name=args.model_name,
        max_sequence_length=args.max_sequence_length,
        ptb_data_dir=args.ptb_data_dir,
        no_cache=args.no_cache,
        output_suffix=args.output_suffix,
    )


if __name__ == "__main__":
    main()
