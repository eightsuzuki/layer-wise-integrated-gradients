"""
BERT層出力キャッシュシステム

BERTモデルの各層の出力をキャッシュし、IG計算などの分析で再利用します。
"""

import torch
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class BertCache:
    """BERT層出力キャッシュ"""

    def __init__(self):
        # MLP入力キャッシュ（ATT出力=MLP入力、LayerNorm前）
        self.mlp_input_cache: Dict[int, torch.Tensor] = {}
        # 後方互換性のためのbeta_cache
        self.beta_cache: Dict[int, torch.Tensor] = {}
        # Attention出力キャッシュ
        self.alpha_cache: Dict[int, torch.Tensor] = {}
        # その他のキャッシュ
        self.hidden_states_cache: Dict[int, torch.Tensor] = {}
        self.attn_weights_cache: Dict[int, torch.Tensor] = {}
        self.qkv_cache: Dict[str, Dict[int, torch.Tensor]] = {
            "q": {},
            "k": {},
            "v": {},
        }

    def clear(self):
        """すべてのキャッシュをクリア"""
        self.mlp_input_cache.clear()
        self.beta_cache.clear()
        self.alpha_cache.clear()
        self.hidden_states_cache.clear()
        self.attn_weights_cache.clear()
        for qkv_type in self.qkv_cache:
            self.qkv_cache[qkv_type].clear()

    def get_info(self) -> Dict:
        """キャッシュ情報を取得"""
        return {
            "mlp_input_cache_size": len(self.mlp_input_cache),
            "beta_cache_size": len(self.beta_cache),
            "alpha_cache_size": len(self.alpha_cache),
            "hidden_states_cache_size": len(self.hidden_states_cache),
            "attn_weights_cache_size": len(self.attn_weights_cache),
            "cached_layers": sorted(self.mlp_input_cache.keys()),
        }


# グローバルキャッシュインスタンス
bert_cache = BertCache()


def cache_bert_layer_outputs(
    model,
    inputs: Dict[str, torch.Tensor],
    tokenizer,
    text: str,
    max_layers: Optional[int] = None,
) -> None:
    """
    BERT層出力をキャッシュに保存

    Args:
        model: UnifiedBertModelまたはBertLightningModule
        inputs: トークナイザーからの入力辞書
        tokenizer: トークナイザー（使用されないが、API互換性のため保持）
        text: 入力テキスト（使用されないが、API互換性のため保持）
        max_layers: キャッシュする最大層数（Noneの場合は全層）
    """
    # キャッシュをクリア
    bert_cache.clear()

    # モデルを評価モードに設定
    model.eval()

    # デバイスを確認
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # フックを登録してMLP入力をキャッシュ
    hooks = []

    def build_mlp_input_cache_hook(layer_idx):
        def hook(module, input, output):
            # ATTの出力（LayerNorm前）を取得
            if len(input) > 0 and isinstance(input[0], torch.Tensor):
                att_output = input[0].detach()
                # キャッシュに保存
                bert_cache.mlp_input_cache[layer_idx] = att_output
                bert_cache.beta_cache[layer_idx] = att_output  # 後方互換性

        return hook

    # 各層のLayerNormにフックを登録（MLP入力キャッシュ用）
    num_layers = max_layers if max_layers is not None else model.config.num_hidden_layers
    for layer_idx in range(num_layers):
        if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
            layer = model.bert.encoder.layer[layer_idx]
            hook = layer.output.LayerNorm.register_forward_hook(
                build_mlp_input_cache_hook(layer_idx)
            )
            hooks.append(hook)
        elif hasattr(model, "encoder"):
            layer = model.encoder.layer[layer_idx]
            hook = layer.output.LayerNorm.register_forward_hook(
                build_mlp_input_cache_hook(layer_idx)
            )
            hooks.append(hook)

    try:
        # モデルを実行してキャッシュを構築
        with torch.no_grad():
            outputs = model(**inputs)

            # unified_modelのoutputsからキャッシュを取得
            if hasattr(model, "outputs"):
                # Attention出力をキャッシュ
                for layer_idx, attn_output in model.outputs.get("attn", {}).items():
                    if layer_idx < num_layers:
                        bert_cache.alpha_cache[layer_idx] = attn_output.detach()

                # Hidden statesをキャッシュ
                for layer_idx, hidden_state in model.outputs.get("hidden_states", {}).items():
                    if layer_idx < num_layers:
                        bert_cache.hidden_states_cache[layer_idx] = hidden_state.detach()

                # Attention weightsをキャッシュ
                for layer_idx, attn_weights in model.outputs.get("attn_weights", {}).items():
                    if layer_idx < num_layers:
                        bert_cache.attn_weights_cache[layer_idx] = attn_weights.detach()

                # QKVをキャッシュ
                qkv_outputs = model.outputs.get("qkv", {})
                for qkv_type in ["q", "k", "v"]:
                    if qkv_type in qkv_outputs:
                        for layer_idx, qkv_value in qkv_outputs[qkv_type].items():
                            if layer_idx < num_layers:
                                bert_cache.qkv_cache[qkv_type][layer_idx] = qkv_value.detach()

        logger.debug(f"BERT層出力キャッシュ完了: {len(bert_cache.mlp_input_cache)}層")

    finally:
        # フックを削除
        for hook in hooks:
            hook.remove()


def clear_bert_cache() -> None:
    """BERTキャッシュをクリア"""
    bert_cache.clear()
    logger.debug("BERTキャッシュをクリアしました")


def get_cache_info() -> Dict:
    """キャッシュ情報を取得"""
    return bert_cache.get_info()


def get_separated_beta_for_head(
    layer_idx: int,
    head_idx: int,
    num_heads: int = 12,
    head_dim: int = 64,
) -> Optional[torch.Tensor]:
    """
    ヘッドごとに分離されたbeta（MLP入力）を取得

    Args:
        layer_idx: 層インデックス
        head_idx: ヘッドインデックス
        num_heads: ヘッド数
        head_dim: ヘッド次元

    Returns:
        ヘッドごとのbetaテンソル [seq_len, head_dim]
    """
    # beta_cacheまたはmlp_input_cacheから取得
    beta = bert_cache.beta_cache.get(layer_idx) or bert_cache.mlp_input_cache.get(layer_idx)
    if beta is None:
        return None

    # ヘッドごとに分割
    # betaの形状: [batch_size, seq_len, hidden_size]
    batch_size, seq_len, hidden_size = beta.shape
    if hidden_size != num_heads * head_dim:
        # 自動検出を試みる
        if hidden_size % num_heads == 0:
            head_dim = hidden_size // num_heads
        else:
            logger.warning(
                f"hidden_size ({hidden_size}) が num_heads * head_dim と一致しません"
            )
            return None

    # 形状を変更: [batch_size, seq_len, num_heads, head_dim]
    beta_reshaped = beta.view(batch_size, seq_len, num_heads, head_dim)
    # 指定されたヘッドを取得: [batch_size, seq_len, head_dim]
    beta_head = beta_reshaped[:, :, head_idx, :]

    return beta_head.squeeze(0)  # [seq_len, head_dim]を返す（batch_size=1の場合）

