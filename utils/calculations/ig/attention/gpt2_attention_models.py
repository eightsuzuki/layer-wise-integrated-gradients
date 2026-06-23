# gpt2_attention_models.py
"""
GPT-2用 Attention Integrated Gradients モデルクラス
理論文書に基づいた正確な実装（Pre-LN構造対応）

BERTとの主な違い:
- Pre-LN構造 (LayerNorm -> Attention -> Add)
- 因果マスク (下三角行列)
- token_type_ids なし
- 位置埋め込み: wte + wpe (LayerNormなし)
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import GPT2Model

from utils.calculations.shared.device_utils import (
    ensure_model_on_device,
    ensure_tensors_on_device,
)


class GPT2AttentionModel(nn.Module):
    """
    GPT-2用 Attention IG計算モデル (Pre-LN構造対応)

    理論文書の定義:
    - z^(l): LayerNorm前の残差ストリーム (ATTへの入力)
    - u^(l,h): ATTの出力 = MLPへの入力 (ATT_OUTPUT = MLP_INPUT)
    - 入力: {z_i^{(l)}}_i （層lのヘッドhの入力）
    - ベースライン: {z_i^{base}}_i = 0
    - 出力: u_i'^{(l,h)}(a) = ATT_i'^{(l,h)}({z_i^{(l)}}_i:a)
    - IGで評価する関数: A_i'(a) = ||u_i'^{(l,h)}(a) - u_i'^{(l,h)}(0)||_2

    GPT-2 Pre-LN構造:
    z^(l) -> LayerNorm -> Attention -> u_attn -> Add -> z^(l+1)
    """

    def __init__(
        self,
        gpt2_model: GPT2Model,
        layer_idx: int,
        target_token_idx: int,
        target_head_idx: Optional[int] = None,
        use_last_token: bool = True,
        debug: bool = False,
    ):
        """
        Args:
            gpt2_model: GPT2Model インスタンス
            layer_idx: 対象レイヤー
            target_token_idx: 対象トークンインデックス
            target_head_idx: 対象ヘッドインデックス (None=全ヘッド)
            use_last_token: Trueの場合、target_token_idxを無視して最後のトークンを使用
            debug: デバッグモード
        """
        super().__init__()
        self.gpt2_model = gpt2_model
        self.layer_idx = layer_idx
        self.target_token_idx = target_token_idx
        self.target_head_idx = target_head_idx
        self.use_last_token = use_last_token
        self.debug = debug

        # デバイス設定
        self.device = ensure_model_on_device(gpt2_model)

        # 設定情報
        self.config = gpt2_model.config
        self.num_heads = self.config.n_head
        self.hidden_size = self.config.n_embd
        self.head_dim = self.hidden_size // self.num_heads

        # ベースライン出力のキャッシュ
        self.u_baseline_target = None

        if self.debug:
            print(f"GPT2AttentionModel初期化:")
            print(f"  Layer: {layer_idx}")
            print(f"  Target token: {target_token_idx}")
            print(f"  Target head: {target_head_idx}")
            print(f"  Use last token: {use_last_token}")
            print(f"  Hidden size: {self.hidden_size}")
            print(f"  Num heads: {self.num_heads}")
            print(f"  Head dim: {self.head_dim}")
            print(f"  Device: {self.device}")

    def forward(
        self,
        input_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        評価関数の計算（Captum IntegratedGradients互換: 入力テンソル1つのみ）

        理論文書の定義に基づく:
        A_i'(a) = ||u_i'^{(l,h)}(a) - u_i'^{(l,h)}(0)||_2

        Args:
            input_embeddings: 入力埋め込み z^{(l)} [batch_size, seq_len, hidden_size]

        Returns:
            torch.Tensor: 評価関数値（スカラー）
        """
        batch_size, seq_len, hidden_size = input_embeddings.shape

        # use_last_tokenの場合、target_token_idxを上書き
        actual_target_idx = seq_len - 1 if self.use_last_token else self.target_token_idx

        # 因果マスクを内部で生成
        attention_mask = self._create_causal_mask(seq_len, input_embeddings.device)

        # 実際の値計算
        u_actual = self._compute_attention_output(
            input_embeddings, attention_mask, actual_target_idx
        )
        return u_actual

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        因果マスクの生成（GPT-2用下三角マスク）

        Args:
            seq_len: シーケンス長
            device: デバイス

        Returns:
            torch.Tensor: 因果マスク [1, 1, seq_len, seq_len]
        """
        # 下三角行列 (i >= j のとき 1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        # [1, 1, seq_len, seq_len] に reshape
        mask = mask.view(1, 1, seq_len, seq_len)
        # 0の位置に大きな負の値を設定（softmax後に0になる）
        mask = (1.0 - mask) * -10000.0
        return mask

    def _compute_attention_output(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        target_token_idx: int,
    ) -> torch.Tensor:
        """
        Attention出力の計算（後方互換性のため、ノルムを返す）

        Args:
            input_embeddings: 入力埋め込み z^{(l)}
            attention_mask: アテンションマスク
            target_token_idx: 対象トークンインデックス

        Returns:
            torch.Tensor: u_i'^{(l,h)}のL2ノルム（スカラー）
        """
        # ベクトル出力を取得してノルムを計算（バッチ次元を保持）
        output_vector = self._compute_attention_output_vector(
            input_embeddings, attention_mask, target_token_idx
        )
        return torch.norm(output_vector, dim=-1)

    def _compute_attention_output_vector(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        target_token_idx: int,
    ) -> torch.Tensor:
        """
        Attention出力ベクトルの計算（理論式用、Pre-LN構造）

        理論文書の定義に基づく:
        u_i'^{(l,h)}(a) = ATT_i'^{(l,h)}({z_i^{(l)}}_i:a)

        GPT-2 Pre-LN構造:
        z^(l) -> LayerNorm -> Attention -> u_attn

        Args:
            input_embeddings: 入力埋め込み z^{(l)}
            attention_mask: アテンションマスク
            target_token_idx: 対象トークンインデックス

        Returns:
            torch.Tensor: u_i'^{(l,h)}のベクトル（ノルムではなくベクトルそのもの）
        """
        # GPT-2のブロック構造にアクセス
        blocks = self.gpt2_model.h

        # 位置エンコーディングを追加
        embeddings = self._add_positional_embeddings(input_embeddings)

        # 指定された層まで順次計算
        hidden_states = embeddings
        for block_idx in range(self.layer_idx + 1):
            block = blocks[block_idx]

            if block_idx == self.layer_idx:
                # 対象層のAttention出力のみを取得（Pre-LN構造）
                # z^(l) -> LayerNorm -> Attention
                z_normalized = block.ln_1(hidden_states)

                # Attention出力を取得
                attn_output = block.attn(
                    z_normalized,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )

                if isinstance(attn_output, tuple):
                    attention_weights = attn_output[1]  # attention weights
                    attention_output = attn_output[0]  # attention output
                else:
                    attention_weights = None
                    attention_output = attn_output

                # 対象トークンのAttention出力を取得（ベクトル）
                target_output = self._extract_target_output(
                    attention_output, attention_weights, target_token_idx
                )

                if self.debug:
                    print(f"  Target output vector shape: {target_output.shape}")
                    print(
                        f"  Target output vector norm: {torch.norm(target_output).item():.6f}"
                    )

                return target_output
            else:
                # 他の層は通常通り計算（Pre-LN: LN -> Attn -> Add -> LN -> MLP -> Add）
                # Attention部分
                attn_output = block.attn(
                    block.ln_1(hidden_states),
                    attention_mask=attention_mask,
                )
                if isinstance(attn_output, tuple):
                    attn_output = attn_output[0]

                # 残差接続
                hidden_states = hidden_states + attn_output

                # MLP部分
                mlp_output = block.mlp(block.ln_2(hidden_states))
                hidden_states = hidden_states + mlp_output

        # ここには到達しないはず
        batch_size = input_embeddings.shape[0]
        out_dim = self.head_dim if self.target_head_idx is not None else self.hidden_size
        return torch.zeros(batch_size, out_dim, device=self.device)

    def _add_positional_embeddings(
        self,
        input_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        位置エンコーディングを追加（GPT-2用: wte + wpe）

        Args:
            input_embeddings: 入力埋め込み

        Returns:
            torch.Tensor: 位置エンコーディング付き埋め込み
        """
        seq_length = input_embeddings.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        # GPT-2の位置埋め込みを追加
        position_embeddings = self.gpt2_model.wpe(position_ids)

        # input_embeddings + position_embeddings (token_type_embeddingsはなし)
        embeddings = input_embeddings + position_embeddings

        return embeddings

    def _extract_target_output(
        self,
        attention_output: torch.Tensor,
        attention_weights: Optional[torch.Tensor],
        target_token_idx: int,
    ) -> torch.Tensor:
        """
        対象トークンのAttention出力を抽出（バッチ次元を保持）

        Args:
            attention_output: Attention出力 [batch_size, seq_len, hidden_size]
            attention_weights: Attention重み [batch_size, num_heads, seq_len, seq_len]
            target_token_idx: 対象トークンインデックス

        Returns:
            torch.Tensor: 対象トークンの出力 [batch_size, head_dim] or [batch_size, hidden_size]
        """
        if self.target_head_idx is not None:
            batch_size, seq_len, hidden_size = attention_output.shape
            attention_output = attention_output.view(
                batch_size, seq_len, self.num_heads, self.head_dim
            )
            target_output = attention_output[
                :, target_token_idx, self.target_head_idx, :
            ]
        else:
            target_output = attention_output[:, target_token_idx, :]

        return target_output

    def compute_theoretical_verification(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        ig_values: List[float],
        description: str = "",
    ) -> Dict[str, float]:
        """
        理論的検証の実行

        Args:
            input_embeddings: 入力埋め込み
            attention_mask: アテンションマスク
            ig_values: IG値のリスト
            description: 説明文

        Returns:
            Dict[str, float]: 検証結果
        """
        print(f"\n=== GPT-2 Attention理論的検証 {description} ===")

        # 理論的差分値の計算: ||u_i'^{(l,h)}(1) - u_i'^{(l,h)}(0)||_2
        actual_value = self.forward(input_embeddings)
        baseline_embeddings = torch.zeros_like(input_embeddings)
        baseline_value = self.forward(baseline_embeddings)

        theoretical_diff = (actual_value - baseline_value).squeeze().item()
        print(f"実際の値: {actual_value.squeeze().item():.6f}")
        print(f"ベースライン値: {baseline_value.squeeze().item():.6f}")
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


def create_gpt2_attention_model(
    gpt2_model: GPT2Model,
    layer_idx: int,
    target_token_idx: int,
    target_head_idx: Optional[int] = None,
    use_last_token: bool = True,
    debug: bool = False,
) -> GPT2AttentionModel:
    """
    GPT2AttentionModelの作成

    Args:
        gpt2_model: GPT2Model
        layer_idx: 対象レイヤー
        target_token_idx: 対象トークンインデックス
        target_head_idx: 対象ヘッドインデックス (None=全ヘッド)
        use_last_token: Trueの場合、最後のトークンを使用
        debug: デバッグフラグ

    Returns:
        GPT2AttentionModel: 作成されたモデル
    """
    return GPT2AttentionModel(
        gpt2_model=gpt2_model,
        layer_idx=layer_idx,
        target_token_idx=target_token_idx,
        target_head_idx=target_head_idx,
        use_last_token=use_last_token,
        debug=debug,
    )
