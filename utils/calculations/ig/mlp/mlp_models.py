# mlp_models.py
"""
MLP IG計算用のモデルクラス
最終層・中間層を統一的に処理するMLPModelクラス

理論文書に基づく命名:
- z^(l): ATTへの入力 (ATT_INPUT)
- u^(l,h): ATTの出力 = MLPへの入力 (ATT_OUTPUT = MLP_INPUT)
- z^(l+1): MLPの出力 (MLP_OUTPUT)
"""

from typing import Optional

import torch
import torch.nn as nn


class MLPModel(nn.Module):
    """
    MLPのIG計算用モデル（最終層・中間層を共通化）
    is_final_layer=True で最終層、Falseで中間層として動作

    理論文書の定義に基づく実装:
    - 最終層: G^{(L)}({u_i^{(L,h)}}_h) = Output_i
    - 中間層: G^{(l)}({u_i^{(l,h)}}_h) = z_i^{(l+1)}

    ここで:
    - u^(l,h): ATTの出力 = MLPへの入力
    - z^(l+1): MLPの出力
    """

    def __init__(
        self,
        model_mlp,
        layer_idx,
        target_token_idx,
        is_final_layer,
        target_head_idx=None,
        include_residual_connection: bool = True,
        baseline_mlp_input: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.model_mlp = model_mlp
        self.layer_idx = layer_idx
        self.target_token_idx = target_token_idx
        self.is_final_layer = is_final_layer
        self.target_head_idx = target_head_idx
        self.include_residual_connection = include_residual_connection
        # §3.7.4 ATTITBa=0 等: 非ゼロベースラインのとき評価を ||z(u) - z(baseline)|| にする
        self.baseline_mlp_input = baseline_mlp_input

    def _build_layernorm_input(self, mlp_input_expanded, mlp_output):
        """LayerNormへの入力を構築（残差接続の有無を切り替え可能）。"""
        if self.include_residual_connection:
            return mlp_input_expanded + mlp_output
        return mlp_output

    def forward(self, mlp_input):
        """
        理論文書に基づくMLP処理

        重要: mlp_input (u) は既に LayerNorm(z^{(l)} + W_o^{(l)} * ATT_OUTPUT) の結果
        つまり、MLP層への直接入力として使用する

        理論文書に基づく命名:
        - mlp_input = u^(l,h): ATTの出力 = MLPへの入力

        最終層: {u_i^{(L,h)}}_h → Output_i
        中間層: {u_i^{(l,h)}}_h → z_i^{(l+1)}
        """
        from utils.cache.bert_cache import bert_cache

        # モデルのdtypeを取得して統一（float16/float32の不一致を回避）
        model_dtype = next(self.model_mlp.parameters()).dtype
        mlp_input = mlp_input.to(dtype=model_dtype)

        # u（MLP入力）を[batch, seq_len, 768]に拡張
        if len(mlp_input.shape) == 2:
            # 対象トークンのみの場合、シーケンス長を取得してパディング
            # キャッシュからシーケンス長を取得
            # 新しい命名を優先、なければ旧命名を使用
            mlp_input_cache = (
                bert_cache.mlp_input_cache
                if bert_cache.mlp_input_cache
                else bert_cache.beta_cache
            )
            if len(mlp_input_cache) > 0:
                sample_mlp_input = next(iter(mlp_input_cache.values()))
                seq_len = sample_mlp_input.shape[1]
            else:
                seq_len = 1  # デフォルト値
            mlp_input_expanded = mlp_input.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            mlp_input_expanded = mlp_input
            seq_len = mlp_input_expanded.shape[1]

        # 理論文書の定義に基づくMLP処理
        # mlp_input_expanded (u) は既に LayerNorm(z^{(l)} + W_o^{(l)} * ATT_OUTPUT) の結果

        # Step 1: MLP処理
        # W_1 * u + b_1
        # UnifiedBertModel対応: model.bert.encoder.layer を使用
        if hasattr(self.model_mlp, "bert"):
            # UnifiedBertModelの場合
            encoder_layer = self.model_mlp.bert.encoder.layer[self.layer_idx]
        else:
            # 従来のモデルの場合
            encoder_layer = self.model_mlp.encoder.layer[self.layer_idx]

        intermediate_output = encoder_layer.intermediate.dense(mlp_input_expanded)

        # GELU activation
        intermediate_output = torch.nn.functional.gelu(intermediate_output)

        # W_2 * GELU(...) + b_2
        mlp_output = encoder_layer.output.dense(intermediate_output)

        # Step 2: 残差接続 + LayerNorm
        # LayerNorm(u + MLP(u))
        # ここで u = mlp_input_expanded (ATTの出力 = MLPへの入力)
        layernorm_input = self._build_layernorm_input(mlp_input_expanded, mlp_output)
        final_output = encoder_layer.output.LayerNorm(layernorm_input)

        if self.is_final_layer:
            return self._process_final_layer(final_output, mlp_input, seq_len)
        else:
            return self._process_intermediate_layer(final_output, mlp_input, seq_len)

    def _process_final_layer(self, final_output, mlp_input, seq_len):
        """最終層の処理"""
        # --- 最終層: G^{(L)}({u_i^{(L,h)}}_h) = Output_i ---
        # ここで u^(L,h) はATTの出力 = MLPへの入力
        # 対象トークンの最終出力を取得
        target_output = final_output[:, self.target_token_idx, :]

        # ベースライン（u=0）での処理
        zero_mlp_input = torch.zeros_like(mlp_input)
        if len(zero_mlp_input.shape) == 2:
            zero_mlp_input_expanded = zero_mlp_input.unsqueeze(1).expand(
                -1, seq_len, -1
            )
        else:
            zero_mlp_input_expanded = zero_mlp_input

        # ベースライン処理
        # UnifiedBertModel対応
        if hasattr(self.model_mlp, "bert"):
            baseline_encoder_layer = self.model_mlp.bert.encoder.layer[self.layer_idx]
        else:
            baseline_encoder_layer = self.model_mlp.encoder.layer[self.layer_idx]

        baseline_intermediate = baseline_encoder_layer.intermediate.dense(
            zero_mlp_input_expanded
        )
        baseline_intermediate = torch.nn.functional.gelu(baseline_intermediate)
        baseline_mlp_output = baseline_encoder_layer.output.dense(baseline_intermediate)
        baseline_layernorm_input = self._build_layernorm_input(
            zero_mlp_input_expanded, baseline_mlp_output
        )
        baseline_final_output = baseline_encoder_layer.output.LayerNorm(
            baseline_layernorm_input
        )
        baseline_output = baseline_final_output[:, self.target_token_idx, :]

        # 評価関数: ||Output_i(u) - Output_i(0)||_2
        # ここで u はMLPへの入力（ATTの出力）
        output = torch.norm(target_output - baseline_output, dim=-1)  # [batch]
        return output.unsqueeze(0) if output.dim() == 0 else output

    def _process_intermediate_layer(self, final_output, mlp_input, seq_len):
        """中間層の処理"""
        # --- 中間層: G^{(l)}({u_i^{(l,h)}}_h) = z_i^{(l+1)}（統合出力） ---
        # ベースライン: 指定があればそれを使い、なければゼロ（§3.7.4 ATTITBa=0 対応）
        if self.baseline_mlp_input is not None:
            b = self.baseline_mlp_input
            if b.dim() == 1:
                b = b.unsqueeze(0)
            baseline_mlp_input_expanded = b.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            zero_mlp_input = torch.zeros_like(mlp_input)
            if len(zero_mlp_input.shape) == 2:
                baseline_mlp_input_expanded = zero_mlp_input.unsqueeze(1).expand(
                    -1, seq_len, -1
                )
            else:
                baseline_mlp_input_expanded = zero_mlp_input

        # ベースライン処理: MLP(baseline) の結果
        # UnifiedBertModel対応
        if hasattr(self.model_mlp, "bert"):
            baseline_encoder_layer = self.model_mlp.bert.encoder.layer[self.layer_idx]
        else:
            baseline_encoder_layer = self.model_mlp.encoder.layer[self.layer_idx]

        baseline_intermediate = baseline_encoder_layer.intermediate.dense(
            baseline_mlp_input_expanded
        )
        baseline_intermediate = torch.nn.functional.gelu(baseline_intermediate)
        baseline_mlp_output = baseline_encoder_layer.output.dense(baseline_intermediate)
        baseline_layernorm_input = self._build_layernorm_input(
            baseline_mlp_input_expanded, baseline_mlp_output
        )
        baseline_final_output = baseline_encoder_layer.output.LayerNorm(
            baseline_layernorm_input
        )
        # デバッグ情報: ベースライン計算の確認
        self._debug_baseline_calculation(
            baseline_mlp_input_expanded,
            baseline_intermediate,
            baseline_mlp_output,
            baseline_final_output,
            None,
        )

        # 統合出力 z^{(l+1)} の対象トークンベクトルを取得
        # [batch, hidden]
        target_z_next = final_output[:, self.target_token_idx, :]
        # [batch, hidden]
        baseline_z_next = baseline_final_output[:, self.target_token_idx, :]

        # 評価関数: ||z_i^{(l+1)}(u) - z_i^{(l+1)}(0)||_2
        # ここで u はMLPへの入力（ATTの出力）
        diff = target_z_next - baseline_z_next
        output = torch.norm(diff, dim=-1)  # [batch]

        return output.unsqueeze(0) if output.dim() == 0 else output

    def _debug_baseline_calculation(
        self,
        zero_mlp_input_expanded,
        baseline_intermediate,
        baseline_mlp_output,
        baseline_final_output,
        baseline_att_output_heads,
    ):
        """ベースライン計算のデバッグ情報出力"""
        try:
            import streamlit as st

            # 中間層ベースライン計算デバッグ出力をコメントアウト（保守性のため残しておく）
            # st.write(f"中間層ベースライン計算デバッグ:")
            # st.write(f"  zero_mlp_input_expanded (u=0) shape: {zero_mlp_input_expanded.shape}")
            # st.write(
            #     f"  zero_mlp_input_expanded norm: {torch.norm(zero_mlp_input_expanded).item():.6f}"
            # )
            # st.write(f"  baseline_intermediate shape: {baseline_intermediate.shape}")
            # st.write(
            #     f"  baseline_intermediate norm: {torch.norm(baseline_intermediate).item():.6f}"
            # )
            # st.write(f"  baseline_mlp_output shape: {baseline_mlp_output.shape}")
            # st.write(
            #     f"  baseline_mlp_output norm: {torch.norm(baseline_mlp_output).item():.6f}"
            # )
            # st.write(f"  baseline_final_output shape: {baseline_final_output.shape}")
            # st.write(
            #     f"  baseline_final_output norm: {torch.norm(baseline_final_output).item():.6f}"
            # )
            # st.write(f"  baseline_att_output_heads (u) shape: {baseline_att_output_heads.shape}")
            # st.write(
            #     f"  baseline_att_output_heads norm: {torch.norm(baseline_att_output_heads).item():.6f}"
            # )
        except ImportError:
            pass

    def _process_specific_head(self, att_output_heads, baseline_att_output_heads):
        """特定ヘッドの処理"""
        # 特定ヘッドのu（ATT出力）
        # 理論文書に基づく命名: u^(l+1,h) = 次層のATTへの入力（この関数では使用されないが、命名を統一）
        target_att_output_head = att_output_heads[
            :, self.target_token_idx, self.target_head_idx, :
        ]
        baseline_att_output_head = baseline_att_output_heads[
            :, self.target_token_idx, self.target_head_idx, :
        ]

        # 理論文書の正しい評価関数: ||u_i^{(l+1,h)}(u) - u_i^{(l+1,h)}(0)||_2
        # ここで u はMLPへの入力（ATTの出力）
        diff = target_att_output_head - baseline_att_output_head
        target_att_output_norm = torch.norm(diff, dim=-1)  # [batch]
        # バッチの各要素に対して個別に評価関数を計算
        output = target_att_output_norm

        # デバッグ情報
        self._debug_specific_head(
            target_att_output_head,
            baseline_att_output_head,
            diff,
            target_att_output_norm,
            output,
        )

        return output.unsqueeze(0) if output.dim() == 0 else output

    def _debug_specific_head(
        self,
        target_att_output_head,
        baseline_att_output_head,
        diff,
        target_att_output_norm,
        output,
    ):
        """特定ヘッドのデバッグ情報出力"""
        try:
            import streamlit as st

            # 中間層特定ヘッドデバッグ出力をコメントアウト（保守性のため残しておく）
            # st.write(f"中間層デバッグ（特定ヘッド {self.target_head_idx}）:")
            # st.write(f"  target_att_output_head (u) shape: {target_att_output_head.shape}")
            # target_norm = torch.norm(target_att_output_head, dim=-1).mean()
            # st.write(f"  target_att_output_head norm: {target_norm.item():.6f}")
            # st.write(f"  baseline_att_output_head (u=0) shape: {baseline_att_output_head.shape}")
            # baseline_norm = torch.norm(baseline_att_output_head, dim=-1).mean()
            # st.write(f"  baseline_att_output_head norm: {baseline_norm.item():.6f}")
            # st.write(f"  diff shape: {diff.shape}")
            # diff_norm = torch.norm(diff, dim=-1).mean()
            # st.write(f"  diff norm: {diff_norm.item():.6f}")
            # # バッチサイズが1の場合のみitem()を使用
            # if target_att_output_norm.numel() == 1:
            #     st.write(f"  target_att_output_norm: {target_att_output_norm.item():.6f}")
            #     st.write(f"  output: {output.item():.6f}")
            # else:
            #     st.write(
            #         f"  target_att_output_norm mean: {target_att_output_norm.mean().item():.6f}"
            #     )
            #     st.write(f"  output shape: {output.shape}")
            #     st.write(f"  output mean: {output.mean().item():.6f}")
            # st.write(f"  target_head_idx: {self.target_head_idx}")
        except ImportError:
            pass

    def _process_all_heads(self, att_output_heads, baseline_att_output_heads):
        """全ヘッドの処理"""
        # 全ヘッドのu（ATT出力）（target_head_idx=Noneの場合のみ）
        # 理論文書に基づく命名: u^(l+1,h) = 次層のATTへの入力
        target_att_output_heads = att_output_heads[:, self.target_token_idx, :, :]
        baseline_att_output_heads_token = baseline_att_output_heads[
            :, self.target_token_idx, :, :
        ]

        # 理論文書の正しい評価関数: ||{u_i^{(l+1,h)}(u) - u_i^{(l+1,h)}(0)}_h||_2
        # ここで u はMLPへの入力（ATTの出力）
        diff = target_att_output_heads - baseline_att_output_heads_token
        target_att_output_norms = torch.norm(diff, dim=-1)  # [batch, heads]
        all_heads_norm = torch.norm(target_att_output_norms, dim=-1)  # [batch]
        output = all_heads_norm.mean()

        # デバッグ情報
        self._debug_all_heads(
            target_att_output_heads,
            baseline_att_output_heads_token,
            diff,
            target_att_output_norms,
            all_heads_norm,
            output,
        )

        return output.unsqueeze(0) if output.dim() == 0 else output

    def _debug_all_heads(
        self,
        target_att_output_heads,
        baseline_att_output_heads_token,
        diff,
        target_att_output_norms,
        all_heads_norm,
        output,
    ):
        """全ヘッドのデバッグ情報出力"""
        try:
            import streamlit as st

            # 中間層全ヘッドデバッグ出力をコメントアウト（保守性のため残しておく）
            # st.write(f"中間層デバッグ（全ヘッド）:")
            # st.write(f"  target_att_output_heads (u) shape: {target_att_output_heads.shape}")
            # st.write(
            #     f"  baseline_att_output_heads_token (u=0) shape: {baseline_att_output_heads_token.shape}"
            # )
            # st.write(f"  diff shape: {diff.shape}")
            # st.write(f"  target_att_output_norms shape: {target_att_output_norms.shape}")
            # st.write(f"  all_heads_norm: {all_heads_norm.item():.6f}")
            # st.write(f"  output: {output.item():.6f}")
            # st.write(f"  target_head_idx: {self.target_head_idx} (None = 全ヘッド)")
        except ImportError:
            pass
