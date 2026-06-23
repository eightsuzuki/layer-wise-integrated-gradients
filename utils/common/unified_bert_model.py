"""
軽量フォールバック実装。

本来の unified_bert_model が存在しない環境でも、最低限のロード関数を提供する。
"""

from __future__ import annotations

from transformers import AutoModel, PreTrainedModel

UnifiedBertModel = PreTrainedModel


def load_unified_model(
    model_name: str = "bert-base-uncased",
    *,
    use_lightning_trainer: bool = False,
    **kwargs,
):
    if use_lightning_trainer:
        raise NotImplementedError(
            "use_lightning_trainer=True is not supported in the fallback unified_bert_model"
        )
    return AutoModel.from_pretrained(
        model_name,
        output_attentions=True,
        output_hidden_states=True,
        **kwargs,
    )
