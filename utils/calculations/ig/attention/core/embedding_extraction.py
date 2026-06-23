"""
埋め込み抽出

軽量版埋め込み抽出（キャッシュシステムを使わない、または事前計算済みhidden statesを使用）
"""

import logging
from typing import Dict, Optional, Tuple

import lightning as L
import torch

logger = logging.getLogger(__name__)


def extract_embeddings_fast(
    bert_model: L.LightningModule,
    inputs: Dict[str, torch.Tensor],
    layer_idx: int,
    debug: bool = False,
    cached_hidden_states: Optional[Tuple] = None,  # 事前計算済みhidden states
) -> tuple:
    """
    軽量版埋め込み抽出（キャッシュシステムを使わない、または事前計算済みhidden statesを使用）

    Args:
        bert_model: BERTモデル
        inputs: 入力テンソル
        layer_idx: 層インデックス
        debug: デバッグフラグ
        cached_hidden_states: 事前計算済みhidden states（Tuple[torch.Tensor, ...]）が提供される場合はBERT推論をスキップ

    Returns:
        tuple: (input_embeddings, attention_mask, token_type_ids)
    """
    if debug:
        logger.debug("⚡ 軽量版埋め込み抽出開始")

    # キャッシュされたhidden statesが提供されている場合はそれを使用
    if cached_hidden_states is not None:
        # hidden_states[0] = embeddings後, hidden_states[layer_idx] が層layer_idxの入力
        # Layer 0の場合は hidden_states[0] (Embedding層の出力) を使用
        if layer_idx < 0 or layer_idx >= len(cached_hidden_states):
            logger.error(
                f"Layer {layer_idx}のインデックスが範囲外: "
                f"cached_hidden_states length={len(cached_hidden_states)}"
            )
            raise IndexError(
                f"Layer {layer_idx} is out of range for cached_hidden_states (length={len(cached_hidden_states)})"
            )
        input_embeddings = cached_hidden_states[layer_idx]

        # Layer 0での形状確認（デバッグ用）
        if layer_idx == 0 and debug:
            logger.debug(
                f"Layer 0: input_embeddings shape={input_embeddings.shape}, "
                f"device={input_embeddings.device}, dtype={input_embeddings.dtype}"
            )

        # デバイス確認（cached_hidden_statesを使用する前に）
        model_device = next(bert_model.parameters()).device

        # デバイスが異なる場合は移動（通常は同じデバイス）
        if input_embeddings.device != model_device:
            input_embeddings = input_embeddings.to(model_device)

        # attention_maskをcached_hidden_statesのシーケンス長に合わせて生成
        # 注意: cached_hidden_statesはバッチ処理時の長さで保存されているため、
        # inputsのattention_maskと長さが一致しない場合がある（正常な動作）
        cached_seq_len = input_embeddings.shape[1]
        batch_size = input_embeddings.shape[0]

        # cached_hidden_statesの長さに合わせてattention_maskを生成
        # 直接GPUで作成（CPU→GPU移動を削除して高速化）
        attention_mask = torch.ones(
            (batch_size, cached_seq_len), device=model_device, dtype=torch.long
        )
        token_type_ids = torch.zeros(
            (batch_size, cached_seq_len), device=model_device, dtype=torch.long
        )

        if debug:
            logger.debug(
                f"⚡ キャッシュされたhidden statesを使用: {input_embeddings.shape} (device: {input_embeddings.device})"
            )
            logger.debug(
                f"⚡ マスク形状: {attention_mask.shape} (device: {attention_mask.device})"
            )

        return input_embeddings, attention_mask, token_type_ids

    # モデルの種類を判定して適切にアクセス
    if hasattr(bert_model, "bert"):
        # Lightning包装されたBERTモデル
        embeddings_layer = bert_model.bert.embeddings
        if debug:
            logger.debug("⚡ Lightning包装されたBERTモデルを検出")
    else:
        # 直接のBERTモデル (BertWithHooks等)
        embeddings_layer = bert_model.embeddings
        if debug:
            logger.debug("⚡ 直接のBERTモデル (BertWithHooks) を検出")

    # 層lの入力隠れ状態 z^{(l)} を取得
    with torch.no_grad():
        outputs = (
            bert_model(**inputs, output_hidden_states=True, output_attentions=False)
            if hasattr(bert_model, "__call__")
            else bert_model.bert(
                **inputs, output_hidden_states=True, output_attentions=False
            )
        )
        hidden_states = outputs.hidden_states  # tuple(len=L+1)
        # hidden_states[0] = embeddings後, hidden_states[l] が層lの入力, hidden_states[l+1] が層lの出力
        input_embeddings = hidden_states[layer_idx]
        attention_mask = inputs.get(
            "attention_mask", torch.ones_like(inputs["input_ids"])
        )
        token_type_ids = inputs.get(
            "token_type_ids", torch.zeros_like(inputs["input_ids"])
        )

    if debug:
        logger.debug(f"⚡ 埋め込み形状: {input_embeddings.shape}")
        logger.debug(f"⚡ マスク形状: {attention_mask.shape}")

    return input_embeddings, attention_mask, token_type_ids

