# attention_models.py
"""
Attention Integrated Gradients用モデルクラス
理論文書に基づいた正確な実装
"""

from typing import Dict, List, Optional, Tuple

import lightning as L
import torch
import torch.nn as nn

from utils.calculations.shared.device_utils import (
    ensure_model_on_device,
    ensure_tensors_on_device,
)


class AttentionModel(nn.Module):
    """
    Attention IG計算用モデル
    理論文書に基づいた評価関数の実装

    理論文書の定義:
    - z^(l): ATTへの入力 (ATT_INPUT)
    - u^(l,h): ATTの出力 = MLPへの入力 (ATT_OUTPUT = MLP_INPUT)
    - 入力: {z_i^{(l)}}_i （層lのヘッドhの入力）
    - ベースライン: {z_i^{base}}_i = 0
    - 出力: u_i'^{(l,h)}(a) = ATT_i'^{(l,h)}({z_i^{(l)}}_i:a)
    - IGで評価する関数: A_i'(a) = ||u_i'^{(l,h)}(a) - u_i'^{(l,h)}(0)||_2
    """

    def __init__(
        self,
        bert_model: L.LightningModule,
        layer_idx: int,
        target_token_idx: int,
        target_head_idx: Optional[int] = None,
        debug: bool = False,
    ):
        super().__init__()
        self.bert_model = bert_model
        self.layer_idx = layer_idx
        self.target_token_idx = target_token_idx
        self.target_head_idx = target_head_idx
        self.debug = debug

        # デバイス設定
        self.device = ensure_model_on_device(bert_model)

        # 設定情報
        self.config = bert_model.config
        self.num_heads = self.config.num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        if self.debug:
            print(f"AttentionModel初期化:")
            print(f"  Layer: {layer_idx}")
            print(f"  Target token: {target_token_idx}")
            print(f"  Target head: {target_head_idx}")
            print(f"  Hidden size: {self.hidden_size}")
            print(f"  Num heads: {self.num_heads}")
            print(f"  Head dim: {self.head_dim}")
            print(f"  Device: {self.device}")

    def forward(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        compute_baseline: bool = False,
    ) -> torch.Tensor:
        """
        評価関数の計算

        理論文書の定義に基づく:
        A_i'(a) = ||u_i'^{(l,h)}(a) - u_i'^{(l,h)}(0)||_2

        Args:
            input_embeddings: 入力埋め込み z^{(l)} [batch_size, seq_len, hidden_size]
            attention_mask: アテンションマスク
            token_type_ids: トークンタイプID
            compute_baseline: ベースライン計算フラグ

        Returns:
            torch.Tensor: 評価関数値（スカラー）
        """
        if compute_baseline:
            # ベースライン計算: 埋め込みをゼロベクトルに（z^{base} = 0）
            baseline_embeddings = torch.zeros_like(input_embeddings)
            u_baseline = self._compute_attention_output(
                baseline_embeddings, attention_mask, token_type_ids
            )
            return u_baseline
        else:
            # 実際の値計算
            u_actual = self._compute_attention_output(
                input_embeddings, attention_mask, token_type_ids
            )
            return u_actual

    def _compute_attention_output(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attention出力の計算（後方互換性のため、ノルムを返す）

        Args:
            input_embeddings: 入力埋め込み z^{(l)}
            attention_mask: アテンションマスク
            token_type_ids: トークンタイプID

        Returns:
            torch.Tensor: u_i'^{(l,h)}のL2ノルム（スカラー）
        """
        # ベクトル出力を取得してノルムを計算
        output_vector = self._compute_attention_output_vector(
            input_embeddings, attention_mask, token_type_ids
        )
        return torch.norm(output_vector)

    def _compute_attention_output_vector(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attention出力ベクトルの計算（理論式用）

        理論文書の定義に基づく:
        u_i'^{(l,h)}(a) = ATT_i'^{(l,h)}({z_i^{(l)}}_i:a)

        Args:
            input_embeddings: 入力埋め込み z^{(l)}
            attention_mask: アテンションマスク
            token_type_ids: トークンタイプID

        Returns:
            torch.Tensor: u_i'^{(l,h)}のベクトル（ノルムではなくベクトルそのもの）
        """
        # 勾配計算のためにno_gradを使用しない
        # BERTモデルの種類に応じて適切にアクセス
        if hasattr(self.bert_model, "bert"):
            # Lightning包装されたBERTモデル
            embeddings_layer = self.bert_model.bert.embeddings
            encoder_layers = self.bert_model.bert.encoder.layer
        else:
            # 直接のBERTモデル
            embeddings_layer = self.bert_model.embeddings
            # UnifiedBertModel対応
        if hasattr(self.bert_model, "bert"):
            encoder_layers = self.bert_model.bert.encoder.layer
        else:
            encoder_layers = self.bert_model.encoder.layer

        # attention_maskの型を修正
        if attention_mask.dtype != torch.float32:
            attention_mask = attention_mask.float()

        # 位置エンコーディングとトークンタイプ埋め込みを追加
        embeddings = self._add_positional_embeddings(
            input_embeddings, token_type_ids, embeddings_layer
        )

        # 指定された層まで順次計算
        hidden_states = embeddings
        for layer_idx in range(self.layer_idx + 1):
            layer = encoder_layers[layer_idx]

            if layer_idx == self.layer_idx:
                # 対象層のAttention出力のみを取得
                attention_output = layer.attention.self(
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )

                if isinstance(attention_output, tuple):
                    attention_weights = attention_output[1]  # attention weights
                    attention_output = attention_output[0]  # attention output
                else:
                    attention_weights = None

                # 対象トークンのAttention出力を取得（ベクトル）
                target_output = self._extract_target_output(
                    attention_output, attention_weights
                )

                if self.debug:
                    print(f"  Target output vector shape: {target_output.shape}")
                    print(
                        f"  Target output vector norm: {torch.norm(target_output).item():.6f}"
                    )

                return target_output
            else:
                # 他の層は通常通り計算
                layer_output = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                )
                if isinstance(layer_output, tuple):
                    hidden_states = layer_output[0]
                else:
                    hidden_states = layer_output

        # ここには到達しないはず
        return torch.zeros(
            self.head_dim if self.target_head_idx is not None else self.hidden_size,
            device=self.device,
        )

    def _add_positional_embeddings(
        self,
        input_embeddings: torch.Tensor,
        token_type_ids: torch.Tensor,
        embeddings_layer,
    ) -> torch.Tensor:
        """
        位置エンコーディングとトークンタイプ埋め込みを追加

        Args:
            input_embeddings: 入力埋め込み
            token_type_ids: トークンタイプID
            embeddings_layer: 埋め込み層

        Returns:
            torch.Tensor: 位置エンコーディング付き埋め込み
        """
        seq_length = input_embeddings.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_type_ids)

        # 位置埋め込みとトークンタイプ埋め込みを追加（MPNet 等は token_type なし）
        position_embeddings = embeddings_layer.position_embeddings(position_ids)
        if hasattr(embeddings_layer, "token_type_embeddings"):
            token_type_embeddings = embeddings_layer.token_type_embeddings(token_type_ids)
            embeddings = input_embeddings + position_embeddings + token_type_embeddings
        else:
            embeddings = input_embeddings + position_embeddings
        embeddings = embeddings_layer.LayerNorm(embeddings)
        embeddings = embeddings_layer.dropout(embeddings)

        return embeddings

    def _extract_target_output(
        self,
        attention_output: torch.Tensor,
        attention_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        対象トークンのAttention出力を抽出

        Args:
            attention_output: Attention出力 [batch_size, seq_len, hidden_size]
            attention_weights: Attention重み [batch_size, num_heads, seq_len, seq_len]

        Returns:
            torch.Tensor: 対象トークンの出力
        """
        if self.target_head_idx is not None:
            # 特定ヘッドの出力を抽出
            # attention_outputを各ヘッドに分割
            batch_size, seq_len, hidden_size = attention_output.shape
            attention_output = attention_output.view(
                batch_size, seq_len, self.num_heads, self.head_dim
            )

            # 対象ヘッドの対象トークンの出力
            target_output = attention_output[
                0, self.target_token_idx, self.target_head_idx, :
            ]
        else:
            # 全ヘッドの出力（対象トークンの全次元）
            target_output = attention_output[0, self.target_token_idx, :]

        return target_output

    def compute_theoretical_verification(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        ig_values: List[float],
        description: str = "",
    ) -> Dict[str, float]:
        """
        理論的検証の実行

        Args:
            input_embeddings: 入力埋め込み
            attention_mask: アテンションマスク
            token_type_ids: トークンタイプID
            ig_values: IG値のリスト
            description: 説明文

        Returns:
            Dict[str, float]: 検証結果
        """
        print(f"\n=== Attention理論的検証 {description} ===")

        # 理論的差分値の計算: ||u_i'^{(l,h)}(1) - u_i'^{(l,h)}(0)||_2
        actual_value = self.forward(
            input_embeddings, attention_mask, token_type_ids, compute_baseline=False
        )
        baseline_value = self.forward(
            input_embeddings, attention_mask, token_type_ids, compute_baseline=True
        )

        theoretical_diff = (actual_value - baseline_value).item()
        print(f"実際の値: {actual_value.item():.6f}")
        print(f"ベースライン値: {baseline_value.item():.6f}")
        print(f"理論的差分値: {theoretical_diff:.6f}")

        # IG値の総和
        ig_sum = sum(ig_values)
        print(f"IG値総和: {ig_sum:.6f}")

        # 相対誤差の計算
        if abs(theoretical_diff) > 1e-10:
            relative_error = abs(theoretical_diff - ig_sum) / abs(theoretical_diff)
            print(f"相対誤差: {relative_error:.6f} ({relative_error*100:.4f}%)")
        else:
            relative_error = float("inf")
            print(f"相対誤差: 無限大 (理論値がゼロに近い)")

        # 判定
        is_valid = relative_error < 0.01  # 1%以下
        status = "✅ 理論と一致" if is_valid else "❌ 理論と乖離"
        print(f"判定: {status}")

        return {
            "theoretical_diff": theoretical_diff,
            "ig_sum": ig_sum,
            "relative_error": relative_error,
            "is_valid": is_valid,
        }


def create_attention_model(
    bert_model: L.LightningModule,
    layer_idx: int,
    target_token_idx: int,
    target_head_idx: Optional[int] = None,
    debug: bool = False,
) -> AttentionModel:
    """
    AttentionModelの作成

    Args:
        bert_model: BERTモデル
        layer_idx: 対象レイヤー
        target_token_idx: 対象トークンインデックス
        target_head_idx: 対象ヘッドインデックス (None=全ヘッド)
        debug: デバッグフラグ

    Returns:
        AttentionModel: 作成されたモデル
    """
    return AttentionModel(
        bert_model=bert_model,
        layer_idx=layer_idx,
        target_token_idx=target_token_idx,
        target_head_idx=target_head_idx,
        debug=debug,
    )
