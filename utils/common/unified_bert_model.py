"""
軽量フォールバック実装。

本来の unified_bert_model が存在しない環境でも、最低限のロード関数を提供する。
"""

from __future__ import annotations

from transformers import AutoModel


def load_unified_model(model_name: str = "bert-base-uncased"):
    return AutoModel.from_pretrained(
        model_name,
        output_attentions=True,
        output_hidden_states=True,
    )

