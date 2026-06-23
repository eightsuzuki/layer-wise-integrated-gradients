# mlp_ig.py
"""
MLP Integrated Gradients計算
理論文書に基づいたMLP入力u（ATT出力=MLP入力）の貢献度分析

理論文書に基づく命名:
- z_i^(l): トークンiに対するATTへの入力 (ATT_INPUT)（文脈上iが明らかな場合はz^(l)と略記）
- u_i^(l,h): トークンiに対するATTの出力 = MLPへの入力 (ATT_OUTPUT = MLP_INPUT)（文脈上iが明らかな場合はu^(l,h)と略記）
- z_i^(l+1): MLPの出力 (MLP_OUTPUT)（文脈上iが明らかな場合はz^(l+1)と略記）
"""

from typing import Dict, List, Optional

import lightning as L
import numpy as np
import torch
from captum.attr import IntegratedGradients

from utils.calculations.shared.device_utils import ensure_model_on_device

from .mlp_models import MLPModel


def _apply_layernorm_with_optional_residual(
    encoder_layer,
    mlp_input: torch.Tensor,
    mlp_output: torch.Tensor,
    include_residual_connection: bool,
) -> torch.Tensor:
    """MLP出力に対して、残差接続の有無を切り替えてLayerNormを適用する。"""
    if include_residual_connection:
        layernorm_input = mlp_input + mlp_output
    else:
        layernorm_input = mlp_output
    return encoder_layer.output.LayerNorm(layernorm_input)


def compute_mlp_ig_theoretical_with_cache(
    model_mlp: L.LightningModule,
    layer_idx: int,
    target_token_idx: int,
    target_head_idx: Optional[int] = None,
    num_steps: int = 32,
    baseline_inputs: Optional[dict] = None,
    is_global_analysis: bool = False,
    baseline_method: str = "zero",
    include_residual_connection: bool = True,
    baseline_mlp_input_override: Optional["torch.Tensor"] = None,
    target_mlp_input_override: Optional["torch.Tensor"] = None,
) -> Optional[np.ndarray]:
    """
    Captumを使用したMLP入力u（ATT出力=MLP入力）のIG計算（キャッシュシステム版）

    理論文書に基づく実装:
    - 最終層MLP: {u_i^(L,h)}_h → Output_i（特殊な扱い）
    - 中間層MLP: {u_i^(l,h)}_h → z_i^(l+1)（標準的な扱い）

    ここで:
    - u_i^(l,h): トークンiに対するATTの出力 = MLPへの入力（文脈上iが明らかな場合はu^(l,h)と略記）
    - z_i^(l+1): MLPの出力（文脈上iが明らかな場合はz^(l+1)と略記）

    Args:
        model_mlp: MLPフック付きBERTモデル
        layer_idx: 対象レイヤー
        target_token_idx: 対象トークンインデックス
        target_head_idx: 対象ヘッドインデックス（中間層の場合、Noneの場合は全ヘッド考慮）
        num_steps: 積分分割数
        baseline_inputs: ベースライン入力（Noneの場合はゼロベクトル、非推奨）
        is_global_analysis: 全体分析かどうか（Trueの場合は検証をスキップ）
        baseline_method: ベースライン選択方法 ("zero" のみ; 公開版では att_itb_a0 は lig.mlp_lig_ig 経由)
        include_residual_connection: MLP残差接続 u + MLP(u) を評価関数に含めるか
        baseline_mlp_input_override: 指定時はこれを MLP のベースライン入力として使用（§3.7.4 ATTITBa=0：ATT の a=0 出力受け渡し）
        target_mlp_input_override: 指定時はキャッシュの代わりにこれを対象トークンの MLP 入力として使用

    Returns:
        Optional[np.ndarray]: 各ヘッドのIG値 [num_heads]
    """
    try:
        # PyTorchは自動的にメモリ管理を行うため、明示的なクリーンアップは不要
        # メモリ不足エラーが発生した場合のみ、PyTorchの自動管理に任せる

        # キャッシュの確認
        from utils.cache.bert_cache import bert_cache

        # 新しい命名を優先、なければ旧命名を使用
        mlp_input_cache = (
            bert_cache.mlp_input_cache
            if bert_cache.mlp_input_cache
            else bert_cache.beta_cache
        )
        if len(mlp_input_cache) == 0:
            raise ValueError(
                "キャッシュが準備されていません。先にキャッシュ処理を実行してください。"
            )

        # 最終層かどうかを判定
        is_final_layer = layer_idx == model_mlp.config.num_hidden_layers - 1

        # デバッグ出力（streamlitが利用可能な場合のみ）
        _debug_layer_info(layer_idx, is_final_layer)

        # キャッシュされたu（ATT出力=MLP入力）を取得
        cached_mlp_input = mlp_input_cache.get(layer_idx)
        if cached_mlp_input is None:
            raise ValueError(
                f"Layer {layer_idx} のu（ATT出力=MLP入力）がキャッシュされていません"
            )

        _debug_cache_info(cached_mlp_input, target_token_idx, model_mlp.config)

        # 対象トークンのu（MLP入力）を取得
        if target_mlp_input_override is not None:
            target_mlp_input = target_mlp_input_override.to(cached_mlp_input.device)
            if target_mlp_input.dim() == 1:
                target_mlp_input = target_mlp_input.unsqueeze(0)
        else:
            target_mlp_input = cached_mlp_input[
                :, target_token_idx, :
            ]  # [batch, hidden_dim]

        # デバイスを統一
        device = ensure_model_on_device(model_mlp)
        target_mlp_input = target_mlp_input.to(device)

        # モデルのdtypeを取得して統一（float16/float32の不一致を回避）
        model_dtype = next(model_mlp.parameters()).dtype
        target_mlp_input = target_mlp_input.to(dtype=model_dtype)

        # 入力検証: target_mlp_inputが0ベクトルかどうかをチェック
        target_mlp_input_norm = torch.norm(target_mlp_input).item()
        if target_mlp_input_norm < 1e-6:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Layer {layer_idx}, Token {target_token_idx}: "
                f"target_mlp_inputがほぼ0ベクトルです (norm={target_mlp_input_norm:.6e})。"
                f"有効なトークンでu_i^(l,h) = 0になるのは異常です。"
            )

        # ベースライン計算（理論文書「4.IGの経路の定義について.md」に基づく）
        if baseline_mlp_input_override is not None:
            baseline_mlp_input = baseline_mlp_input_override.to(device).to(dtype=model_dtype)
            if baseline_mlp_input.dim() == 1:
                baseline_mlp_input = baseline_mlp_input.unsqueeze(0)
        elif baseline_method == "zero":
            # 方法1: ゼロベースライン（5.1.1節）
            # u_i^(l,h)(0) = 0 （ゼロベクトル）
            baseline_mlp_input = torch.zeros_like(target_mlp_input)
        # elif baseline_method == "self_input_token":
        #     # 方法2: 自己入力トークンベースライン（5.節）
        #     # u_i^(l,h)(0) = u_i^(l,h) （各トークン位置iに対して、そのトークン自身のATT出力をベースラインに設定）
        #     # 理論: 自己トークンiの寄与度は定義により0になる
        #     # 各トークン位置iに対して、そのトークン自身のATT出力u_iをベースラインとして使用
        #     # 
        #     # 【理論的制約】MLPのIG計算において、Self Input Token Baselineは理論的に実装不可能です。
        #     # 各トークンiに対して、そのトークン自身のu_iをベースラインとして使用すると、
        #     # u_i - u_i = 0となり、すべてのIG値が必ず0になってしまいます。
        #     # 詳細は理論文書「4.IGの経路の定義について.md」の5.1.2.4節を参照。
        #     reference_token_idx = target_token_idx  # 各トークン自身のATT出力をベースラインに設定
        #     baseline_mlp_input = cached_mlp_input[
        #         :, reference_token_idx, :
        #     ].clone().to(device).to(dtype=model_dtype)
        else:
            raise ValueError(
                f"未知のベースライン方法: {baseline_method}。"
                f"サポートされている方法: 'zero'"
            )

        _debug_ig_start(is_final_layer, target_mlp_input, baseline_mlp_input, num_steps)

        # MLPModelクラスを使用（§3.7.4 ATTITBa=0 のときは baseline を渡して完全性を満たす）
        mlp_model = MLPModel(
            model_mlp,
            layer_idx,
            target_token_idx,
            is_final_layer,
            target_head_idx,
            include_residual_connection=include_residual_connection,
            baseline_mlp_input=baseline_mlp_input,
        )

        # CaptumのIntegratedGradientsを初期化
        ig = IntegratedGradients(mlp_model)

        # 理論的検証（全体分析の場合はスキップ）
        if not is_global_analysis:
            theoretical_diff = _perform_theoretical_validation(
                mlp_model, target_mlp_input, baseline_mlp_input, is_final_layer
            )
        else:
            theoretical_diff = 0.0  # 全体分析の場合はダミー値

        # IG計算実行
        ig_attributions = _execute_ig_calculation(
            ig, target_mlp_input, baseline_mlp_input, num_steps
        )

        # 結果処理
        ig_values = _process_ig_results(
            ig_attributions,
            model_mlp.config.num_attention_heads,
            model_mlp.config.hidden_size,
        )

        # Self Input Token Baselineの場合、自己トークンの寄与度を0に設定（理論的には0になるべき）
        # 理論文書「4.IGの経路の定義について.md」5.1.2.4節参照
        # 【理論的制約】MLPのIG計算において、Self Input Token Baselineは理論的に実装不可能です。
        # 各トークンiに対して、そのトークン自身のu_iをベースラインとして使用すると、
        # u_i - u_i = 0となり、すべてのIG値が必ず0になってしまいます。
        # そのため、この処理はコメントアウトされています。
        # if baseline_method == "self_input_token":
        #     # 自己トークンの寄与度は定義により0になる
        #     # IG_{i',i'}^{MLP,self} = 0 （u_{i'} - u_{i'}(0) = 0 のため）
        #     # ヘッドごとのIG値は各ヘッドの寄与度を表すため、理論的には0になるべきだが、
        #     # 数値計算の誤差により0でない場合があるため、明示的に0に設定
        #     # 注意: MLPの場合は入力が{u_i^(l,h)}_hで、これは全ヘッドを結合したもの（768次元）
        #     # したがって、ヘッドごとの個別の寄与度を0に設定することはできない
        #     # 代わりに、IG値全体が理論的に0に近くなることを期待する
        #     # （実際の実装では、ヘッドごとのIG値の合計が0に近くなることを確認）
        #     pass  # MLPの場合はヘッドごとの個別設定は困難なため、理論的検証に任せる

        # 理論的検証（全体分析の場合はスキップ）
        if not is_global_analysis:
            _validate_ig_results(ig_values, theoretical_diff, is_final_layer)
            
            # 完全性理論の検証: sum_{i,h} IG_{i,i'}^{MLP,zero} = ||z_{i'}^{(l+1)}||_2
            # Zero Baselineのみ（override 時は theoretical_diff が既に正しいベースラインで検証済み）
            if baseline_method == "zero" and not is_final_layer and baseline_mlp_input_override is None:
                # 実際のz_{i'}^{(l+1)}を取得
                # キャッシュされた全トークンのu（MLP入力）を取得
                cached_mlp_input_full = cached_mlp_input.to(device).to(dtype=model_dtype)
                
                # レイヤー構造を取得
                if hasattr(model_mlp, "bert"):
                    encoder_layer = model_mlp.bert.encoder.layer[layer_idx]
                else:
                    encoder_layer = model_mlp.encoder.layer[layer_idx]
                
                # MLP処理を実行してz_{i'}^{(l+1)}を取得
                with torch.no_grad():
                    # MLPの中間層
                    mlp_intermediate = encoder_layer.intermediate.dense(cached_mlp_input_full)
                    mlp_intermediate = torch.nn.functional.gelu(mlp_intermediate)
                    # MLPの出力層
                    mlp_output = encoder_layer.output.dense(mlp_intermediate)
                    # LayerNormを適用して次レイヤーの入力埋め込みz_{i'}^{(l+1)}を取得
                    next_layer_input = _apply_layernorm_with_optional_residual(
                        encoder_layer=encoder_layer,
                        mlp_input=cached_mlp_input_full,
                        mlp_output=mlp_output,
                        include_residual_connection=include_residual_connection,
                    )
                    # 対象トークンの次レイヤー入力埋め込みを取得
                    z_i_prime_l_plus_1 = next_layer_input[:, target_token_idx, :]  # [batch, hidden]
                
                # ||z_{i'}^{(l+1)}||_2を計算
                z_i_prime_l_plus_1_norm = torch.norm(z_i_prime_l_plus_1).item()
                
                # IG値の合計
                ig_sum = ig_values.sum().item()
                
                # 完全性理論の検証
                if z_i_prime_l_plus_1_norm > 1e-6:  # ゼロでない場合のみ検証
                    relative_error = abs(ig_sum - z_i_prime_l_plus_1_norm) / z_i_prime_l_plus_1_norm
                    if relative_error > 0.1:  # 10%以上の誤差
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"Layer {layer_idx}, Token {target_token_idx}: "
                            f"完全性理論の違反: IG合計={ig_sum:.6f}, 期待値={z_i_prime_l_plus_1_norm:.6f}, "
                            f"相対誤差={relative_error:.2%}"
                        )

        # CPUに移動してからnumpyに変換
        result = ig_values.cpu().numpy()

        # メモリクリーンアップ（頻度を下げるため削除 - 呼び出し側で管理）
        del ig_values, ig_attributions

        return result

    except Exception as e:
        # メモリ不足エラーの場合のみ、PyTorchに自動クリーンアップを任せる
        # 明示的なクリーンアップは不要（PyTorchが自動処理）
        raise Exception(f"MLP IG計算エラー: {e}")


def compute_mlp_ig_batch(
    model_mlp: L.LightningModule,
    layer_idx: int,
    target_token_indices: List[int],
    target_head_indices: Optional[List[int]] = None,
    num_steps: int = 32,
    is_global_analysis: bool = False,
    baseline_method: str = "zero",
    include_residual_connection: bool = True,
) -> List[Optional[np.ndarray]]:
    """
    バッチ処理版MLP IG計算（GPU最適化版）

    Args:
        model_mlp: MLPフック付きBERTモデル
        layer_idx: 対象レイヤー
        target_token_indices: 対象トークンインデックスリスト
        target_head_indices: 対象ヘッドインデックスリスト（Noneの場合は全ヘッド考慮）
        num_steps: 積分分割数
        is_global_analysis: 全体分析かどうか

    Returns:
        List[Optional[np.ndarray]]: 各タスクのIG値リスト
    """
    try:
        # キャッシュの確認
        from utils.cache.bert_cache import bert_cache

        if len(bert_cache.beta_cache) == 0:
            raise ValueError(
                "キャッシュが準備されていません。先にキャッシュ処理を実行してください。"
            )

        # 最終層かどうかを判定
        is_final_layer = layer_idx == model_mlp.config.num_hidden_layers - 1

        # キャッシュされたu（MLP入力）を取得
        cached_beta = bert_cache.beta_cache.get(layer_idx)
        if cached_beta is None:
            raise ValueError(
                f"Layer {layer_idx} のu（MLP入力）がキャッシュされていません"
            )

        # デバイスを統一
        device = ensure_model_on_device(model_mlp)

        # モデルのdtypeを取得（float16/float32の不一致を回避）
        model_dtype = next(model_mlp.parameters()).dtype

        # バッチ処理用の結果リスト
        results = []

        # 各タスクを並列処理
        for i, token_idx in enumerate(target_token_indices):
            head_idx = target_head_indices[i] if target_head_indices else None

            # 対象トークンのu（MLP入力）を取得
            target_mlp_input = cached_beta[:, token_idx, :].to(device)
            # モデルのdtypeを取得して統一（float16/float32の不一致を回避）
            model_dtype = next(model_mlp.parameters()).dtype
            target_mlp_input = target_mlp_input.to(dtype=model_dtype)
            
            # ベースライン計算
            if baseline_method == "zero":
                baseline_mlp_input = torch.zeros_like(target_mlp_input)
            # elif baseline_method == "self_input_token":
            #     # 自己入力トークンベースライン: u_i^(l,h)(0) = u_{i'}^(l,h)
            #     # 固定された参照トークン（最初のトークン）のATT出力をベースラインに設定
            #     # 
            #     # 【理論的制約】MLPのIG計算において、Self Input Token Baselineは理論的に実装不可能です。
            #     # 各トークンiに対して、そのトークン自身のu_iをベースラインとして使用すると、
            #     # u_i - u_i = 0となり、すべてのIG値が必ず0になってしまいます。
            #     # 詳細は理論文書「4.IGの経路の定義について.md」の5.1.2.4節を参照。
            #     reference_token_idx = 0
            #     baseline_mlp_input = cached_beta[:, reference_token_idx, :].clone().to(device).to(dtype=model_dtype)
            else:
                raise ValueError(f"未知のベースライン方法: {baseline_method}")

            # MLPModelクラスを使用
            mlp_model = MLPModel(
                model_mlp,
                layer_idx,
                token_idx,
                is_final_layer,
                head_idx,
                include_residual_connection=include_residual_connection,
            )

            # CaptumのIntegratedGradientsを初期化
            ig = IntegratedGradients(mlp_model)

            # IG計算実行
            ig_attributions = _execute_ig_calculation(
                ig, target_mlp_input, baseline_mlp_input, num_steps
            )

            # 結果処理
            ig_values = _process_ig_results(
                ig_attributions,
                model_mlp.config.num_attention_heads,
                model_mlp.config.hidden_size,
            )

            result_numpy = ig_values.cpu().numpy()

            # 中間テンソルを削除（PyTorchの自動メモリ管理に任せる）
            del ig_values, ig_attributions, target_mlp_input, baseline_mlp_input

            results.append(result_numpy)

        # PyTorchの自動メモリ管理に任せる（明示的なクリーンアップは不要）
        return results

    except Exception as e:
        # メモリ不足エラーの場合のみ、PyTorchに自動クリーンアップを任せる
        raise Exception(f"バッチMLP IG計算エラー: {e}")


def compute_mlp_ig_optimized_batch(
    model_mlp: L.LightningModule,
    layer_idx: int,
    tasks: List[Dict],
    num_steps: int = 32,
    is_global_analysis: bool = False,
    baseline_method: str = "zero",
    include_residual_connection: bool = True,
) -> List[Dict]:
    """
    最適化されたバッチ処理版MLP IG計算（GPUメモリ効率版）

    Args:
        model_mlp: MLPフック付きBERTモデル
        layer_idx: 対象レイヤー
        tasks: タスクリスト [{"token_idx": int, "head_idx": Optional[int], ...}]
        num_steps: 積分分割数
        is_global_analysis: 全体分析かどうか

    Returns:
        List[Dict]: 各タスクの結果リスト
    """
    try:
        # キャッシュの確認
        from utils.cache.bert_cache import bert_cache

        if len(bert_cache.beta_cache) == 0:
            raise ValueError(
                "キャッシュが準備されていません。先にキャッシュ処理を実行してください。"
            )

        # 最終層かどうかを判定
        is_final_layer = layer_idx == model_mlp.config.num_hidden_layers - 1

        # キャッシュされたu（MLP入力）を取得
        cached_beta = bert_cache.beta_cache.get(layer_idx)
        if cached_beta is None:
            raise ValueError(
                f"Layer {layer_idx} のu（MLP入力）がキャッシュされていません"
            )

        # デバイスを統一
        device = ensure_model_on_device(model_mlp)

        # モデルのdtypeを取得（float16/float32の不一致を回避）
        model_dtype = next(model_mlp.parameters()).dtype

        # 結果リスト
        results = []

        # ベクトル化（全トークン一括）: 中間層のみ（最終層は1結果のため従来処理で十分）
        if not is_final_layer:
            try:
                # [seq_len, hidden]
                # 新しい命名を優先、なければ旧命名を使用
                cached_mlp_input = bert_cache.mlp_input_cache.get(layer_idx)
                if cached_mlp_input is None:
                    cached_mlp_input = bert_cache.beta_cache.get(layer_idx)
                if cached_mlp_input is None:
                    raise ValueError(
                        f"Layer {layer_idx} のu（MLP入力）がキャッシュされていません"
                    )
                mlp_input_full = cached_mlp_input[0].to(device)
                # モデルのdtypeを取得して統一（float16/float32の不一致を回避）
                model_dtype = next(model_mlp.parameters()).dtype
                mlp_input_full = mlp_input_full.to(dtype=model_dtype)
                
                # ベースライン計算
                if baseline_method == "zero":
                    baseline_full = torch.zeros_like(mlp_input_full)
                # elif baseline_method == "self_input_token":
                #     # 自己入力トークンベースライン: 各トークンに対して、固定された参照トークンのATT出力をベースラインに設定
                #     # 理論: u_i^(l,h)(0) = u_{i'}^(l,h) （すべてのトークン位置iに対して、出力トークンi'のATT出力をベースラインに設定）
                #     # 固定された参照トークン（最初のトークン）のATT出力をベースラインに設定
                #     # 
                #     # 【理論的制約】MLPのIG計算において、Self Input Token Baselineは理論的に実装不可能です。
                #     # 各トークンiに対して、そのトークン自身のu_iをベースラインとして使用すると、
                #     # u_i - u_i = 0となり、すべてのIG値が必ず0になってしまいます。
                #     # 詳細は理論文書「4.IGの経路の定義について.md」の5.1.2.4節を参照。
                #     reference_token_idx = 0
                #     baseline_full = cached_mlp_input[0, reference_token_idx, :].clone().to(device).to(dtype=model_dtype)
                #     # すべてのトークン位置に同じベースラインをコピー
                #     baseline_full = baseline_full.unsqueeze(0).expand_as(mlp_input_full)
                else:
                    raise ValueError(f"未知のベースライン方法: {baseline_method}")

                # レイヤー参照
                if hasattr(model_mlp, "bert"):
                    encoder_layer = model_mlp.bert.encoder.layer[layer_idx]
                else:
                    encoder_layer = model_mlp.encoder.layer[layer_idx]

                # ベースラインの出力を事前計算 [1, seq_len, hidden]
                # 勾配計算のため、float32に変換（BF16/FP16では勾配計算が不安定な場合がある）
                original_dtype_for_base = baseline_full.dtype
                if original_dtype_for_base in (torch.bfloat16, torch.float16):
                    baseline_full_for_base = baseline_full.to(torch.float32)
                else:
                    baseline_full_for_base = baseline_full

                with torch.no_grad():
                    zero_mlp_input_exp = baseline_full_for_base.unsqueeze(
                        0
                    )  # [1, seq, hidden]
                    base_inter = encoder_layer.intermediate.dense(zero_mlp_input_exp)
                    base_inter = torch.nn.functional.gelu(base_inter)
                    base_mlp = encoder_layer.output.dense(base_inter)
                    base_final = _apply_layernorm_with_optional_residual(
                        encoder_layer=encoder_layer,
                        mlp_input=zero_mlp_input_exp,
                        mlp_output=base_mlp,
                        include_residual_connection=include_residual_connection,
                    )

                # ステップをバッチ化
                # 勾配計算のため、float32に変換（BF16/FP16では勾配計算が不安定な場合がある）
                # base_finalは既にfloat32で計算されている
                original_dtype = mlp_input_full.dtype
                if original_dtype in (torch.bfloat16, torch.float16):
                    mlp_input_full_fp32 = mlp_input_full.to(torch.float32)
                    baseline_full_fp32 = baseline_full.to(torch.float32)
                    base_final_fp32 = base_final  # 既にfloat32
                else:
                    mlp_input_full_fp32 = mlp_input_full
                    baseline_full_fp32 = baseline_full
                    base_final_fp32 = base_final

                alphas = (
                    torch.arange(
                        num_steps,
                        device=mlp_input_full_fp32.device,
                        dtype=torch.float32,
                    )
                    + 0.5
                ) / num_steps
                alphas = alphas.view(num_steps, 1, 1)
                interp_mlp_input = (
                    (
                        baseline_full_fp32.unsqueeze(0)
                        + alphas
                        * (
                            mlp_input_full_fp32.unsqueeze(0)
                            - baseline_full_fp32.unsqueeze(0)
                        )
                    )
                    .clone()
                    .detach()
                    .requires_grad_(True)
                )

                # 一括forward [num_steps, seq, hidden]
                # モデルがBF16/FP16の場合でも、forward内で自動的に変換される
                inter = encoder_layer.intermediate.dense(interp_mlp_input)
                inter = torch.nn.functional.gelu(inter)
                mlp_out = encoder_layer.output.dense(inter)
                final_out = _apply_layernorm_with_optional_residual(
                    encoder_layer=encoder_layer,
                    mlp_input=interp_mlp_input,
                    mlp_output=mlp_out,
                    include_residual_connection=include_residual_connection,
                )

                # 評価関数 ||z_i^{(l+1)}(u) - z_i^{(l+1)}(0)||_2 per step/token
                # 理論: M_i^(l)(a) = ||z_i^(l+1)(a) - z_i^(l+1)(0)||_2
                # ここで u はMLPへの入力（ATTの出力）
                diff = final_out - base_final_fp32  # [num_steps, seq, hidden]
                norms = torch.norm(diff, dim=-1)  # [num_steps, seq]

                # 理論的には各トークンについて個別にIGを計算する必要がある
                # 理論: IG_h,i^MLP = (u_i^(l,h) - u_i^(l,h),base) · ∫[0,1] ∂M_i^(l)(a)/∂u_i^(l,h) da
                # 各トークンについて個別に勾配を計算するため、各トークンのnormを個別に保持
                # 効率化のため、全トークンを一度に処理するが、各トークンについて個別に勾配を計算

                # 理論的に正しい実装: 各トークンについて個別に勾配を計算
                # 各ステップについて、各トークンのnormの合計をlossとして使用
                # これにより、各トークンについて個別に勾配が計算される
                # 注意: loss = norms.sum() は全ステップ・全トークンの合計だが、
                # 勾配計算では各トークンについて個別に勾配が計算される（PyTorchの自動微分の性質）
                loss = norms.sum()  # 全ステップ・全トークンの合計

                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=interp_mlp_input,
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True,
                )[
                    0
                ]  # [num_steps, seq, hidden]

                # 理論: IG_h,i^MLP = (u_i^(l,h) - u_i^(l,h),base) · ∫[0,1] ∂M_i^(l)(a)/∂u_i^(l,h) da
                # 数値的近似: IG_h,i^MLP ≈ (u_i^(l,h) - u_i^(l,h),base) · (1/m) Σ_k ∂M_i^(l)(a_k)/∂u_i^(l,h)
                # ここで a_k = k/m, k = 1, ..., m
                # 各ステップの勾配を合計して積分を近似
                # 各トークンについて個別に勾配を合計
                grad_sum = grads.sum(dim=0)  # [seq, hidden] - ステップ方向に合計

                # IG計算: (u_i - u_i,base) · grad_sum_i / num_steps
                # 各トークンについて個別にIGを計算
                step_contrib = (
                    grad_sum
                    * (mlp_input_full_fp32 - baseline_full_fp32)
                    * (1.0 / num_steps)
                )  # [seq, hidden]

                # 元のdtypeに戻す（メモリ節約のため）
                if original_dtype in (torch.bfloat16, torch.float16):
                    step_contrib = step_contrib.to(original_dtype)

                # ヘッド毎に集約 [seq, num_heads]
                num_heads = model_mlp.config.num_attention_heads
                head_dim = model_mlp.config.hidden_size // num_heads
                contrib_heads = step_contrib.view(
                    step_contrib.shape[0], num_heads, head_dim
                ).sum(
                    dim=-1
                )  # [seq, heads]

                # タスクに割当
                for idx, task in enumerate(tasks):
                    token_idx = task["token_idx"]
                    task_idx = idx
                    try:
                        # トークンインデックスがシーケンス長を超えないようにチェック
                        if token_idx >= contrib_heads.shape[0]:
                            results.append(
                                {
                                    "task_idx": task_idx,
                                    "word_idx": task.get("word_idx"),
                                    "contributions": None,
                                    "success": False,
                                    "error": f"Token index {token_idx} out of range (seq_len={contrib_heads.shape[0]})",
                                }
                            )
                            continue

                        vec = contrib_heads[token_idx].detach().cpu().numpy()
                        results.append(
                            {
                                "task_idx": task_idx,
                                "word_idx": task.get("word_idx"),
                                "contributions": vec,
                                "success": True,
                                "error": None,
                            }
                        )
                    except Exception as e:
                        results.append(
                            {
                                "task_idx": task_idx,
                                "word_idx": task.get("word_idx"),
                                "contributions": None,
                                "success": False,
                                "error": str(e),
                            }
                        )

                results.sort(key=lambda x: x["task_idx"])

                # 中間テンソルを削除（PyTorchの自動メモリ管理に任せる）
                del (
                    mlp_input_full,
                    baseline_full,
                    mlp_input_full_fp32,
                    baseline_full_fp32,
                    base_final_fp32,
                    interp_mlp_input,
                    inter,
                    mlp_out,
                    final_out,
                    diff,
                    norms,
                    grads,
                    grad_sum,
                    step_contrib,
                    contrib_heads,
                )

                return results
            except Exception:
                # ベクトル化に失敗した場合は従来処理にフォールバック
                pass

        # 従来のグルーピング処理（フォールバックまたは最終層）
        # GPUメモリ効率化のため、タスクをグループ化
        token_indices = [task["token_idx"] for task in tasks]
        head_indices = [task.get("head_idx") for task in tasks]

        token_groups = {}
        for i, token_idx in enumerate(token_indices):
            if token_idx not in token_groups:
                token_groups[token_idx] = []
            token_groups[token_idx].append(i)

        for token_idx, group_indices in token_groups.items():
            # 新しい命名を優先、なければ旧命名を使用
            cached_mlp_input = bert_cache.mlp_input_cache.get(layer_idx)
            if cached_mlp_input is None:
                cached_mlp_input = bert_cache.beta_cache.get(layer_idx)
            if cached_mlp_input is None:
                raise ValueError(
                    f"Layer {layer_idx} のu（MLP入力）がキャッシュされていません"
                )
            target_mlp_input = cached_mlp_input[:, token_idx, :].to(device)
            # モデルのdtypeを取得して統一（float16/float32の不一致を回避）
            model_dtype = next(model_mlp.parameters()).dtype
            target_mlp_input = target_mlp_input.to(dtype=model_dtype)
            
            # ベースライン計算
            if baseline_method == "zero":
                baseline_mlp_input = torch.zeros_like(target_mlp_input)
            # elif baseline_method == "self_input_token":
            #     # 自己入力トークンベースライン: 各トークン自身のATT出力をベースラインに設定
            #     # 
            #     # 【理論的制約】MLPのIG計算において、Self Input Token Baselineは理論的に実装不可能です。
            #     # 各トークンiに対して、そのトークン自身のu_iをベースラインとして使用すると、
            #     # u_i - u_i = 0となり、すべてのIG値が必ず0になってしまいます。
            #     # 詳細は理論文書「4.IGの経路の定義について.md」の5.1.2.4節を参照。
            #     reference_token_idx = token_idx  # 各トークン自身のATT出力をベースラインに設定
            #     baseline_mlp_input = cached_mlp_input[:, reference_token_idx, :].clone().to(device).to(dtype=model_dtype)
            else:
                baseline_mlp_input = torch.zeros_like(target_mlp_input)

            for task_idx in group_indices:
                task = tasks[task_idx]
                head_idx = task.get("head_idx")

                try:
                    mlp_model = MLPModel(
                        model_mlp,
                        layer_idx,
                        token_idx,
                        is_final_layer,
                        head_idx,
                        include_residual_connection=include_residual_connection,
                    )
                    ig = IntegratedGradients(mlp_model)
                    ig_attributions = _execute_ig_calculation(
                        ig, target_mlp_input, baseline_mlp_input, num_steps
                    )
                    ig_values = _process_ig_results(
                        ig_attributions,
                        model_mlp.config.num_attention_heads,
                        model_mlp.config.hidden_size,
                    )
                    result_numpy = ig_values.cpu().numpy()

                    # メモリクリーンアップ（頻度を下げるため削除 - 呼び出し側で管理）
                    del ig_values, ig_attributions

                    results.append(
                        {
                            "task_idx": task_idx,
                            "word_idx": task.get("word_idx"),
                            "contributions": result_numpy,
                            "success": True,
                            "error": None,
                        }
                    )
                except Exception as e:
                    results.append(
                        {
                            "task_idx": task_idx,
                            "word_idx": task.get("word_idx"),
                            "contributions": None,
                            "success": False,
                            "error": str(e),
                        }
                    )

        results.sort(key=lambda x: x["task_idx"])

        return results

    except Exception as e:
        # メモリ不足エラーの場合のみ、PyTorchに自動クリーンアップを任せる
        raise Exception(f"最適化バッチMLP IG計算エラー: {e}")


def _debug_layer_info(layer_idx: int, is_final_layer: bool):
    """層情報のデバッグ出力"""
    try:
        import streamlit as st

        # 層情報デバッグ出力をコメントアウト（保守性のため残しておく）
        # st.write(f"Layer {layer_idx} は最終層: {is_final_layer}")
    except ImportError:
        pass


def _debug_cache_info(cached_mlp_input: torch.Tensor, target_token_idx: int, config):
    """キャッシュ情報のデバッグ出力"""
    try:
        import streamlit as st

        # キャッシュ情報デバッグ出力をコメントアウト（保守性のため残しておく）
        # st.write(f"キャッシュされたu（ATT出力=MLP入力）のshape: {cached_mlp_input.shape}")
        # target_mlp_input = cached_mlp_input[:, target_token_idx, :]
        # st.write(f"トークン {target_token_idx} のu（MLP入力）を取得: {target_mlp_input.shape}")
        # st.write(
        #     f"ヘッド数: {config.num_attention_heads}, 各ヘッド次元: {config.hidden_size // config.num_attention_heads}"
        # )
    except ImportError:
        pass


def _debug_ig_start(
    is_final_layer: bool,
    target_mlp_input: torch.Tensor,
    baseline_mlp_input: torch.Tensor,
    num_steps: int,
):
    """IG計算開始のデバッグ出力"""
    try:
        import streamlit as st

        # IG計算開始デバッグ出力をコメントアウト（保守性のため残しておく）
        # st.write("=== Captum IG計算開始 ===")
        # if is_final_layer:
        #     st.write("最終層MLP処理: {u_i^(L,h)}_h → Output_i（特殊な扱い）")
        #     st.write("評価関数: Output_iを使ったノルム")
        # else:
        #     st.write("中間層MLP処理: {u_i^(l,h)}_h → z_i^(l+1)（標準的な扱い）")
        #     st.write("評価関数: ||z_i^(l+1)||_2")
        # st.write(f"target_mlp_input (u) shape: {target_mlp_input.shape}")
        # st.write(f"baseline_mlp_input (u=0) shape: {baseline_mlp_input.shape}")
        # st.write(f"num_steps: {num_steps}")
    except ImportError:
        pass


def _perform_theoretical_validation(
    mlp_model: MLPModel,
    target_mlp_input: torch.Tensor,
    baseline_mlp_input: torch.Tensor,
    is_final_layer: bool,
) -> float:
    """理論的検証の実行"""
    try:
        import streamlit as st

        # 理論的検証デバッグ出力をコメントアウト（保守性のため残しておく）
        # st.write("=== 理論的検証デバッグ ===")
        # 理論文書の正しい定義に基づく検証
        # 評価関数: M(a) = ||G(a) - G(0)||_2
        # つまり、M(1) = ||G(1) - G(0)||_2, M(0) = ||G(0) - G(0)||_2 = 0
        # ここで G はMLP関数、入力は u（ATT出力=MLP入力）
        # a=1での評価関数値（実際の入力での値）
        actual_input_value = mlp_model(target_mlp_input).item()
        # st.write(f"評価関数 M(1) = {actual_input_value:.6f}")

        # a=0での評価関数値（ベースラインでの値）
        baseline_value = mlp_model(baseline_mlp_input).item()
        # st.write(f"評価関数 M(0) = {baseline_value:.6f}")

        # 理論的な差分値の正しい定義
        # IGの理論的性質: ∑IG = M(1) - M(0) = M(1) (∵ M(0) = 0)
        theoretical_diff = actual_input_value  # M(1) = ||G(1) - G(0)||_2
        # st.write(f"理論的差分値 M(1) - M(0) = M(1) = {theoretical_diff:.6f}")

        # 評価関数の定義確認（最終層と中間層で異なる）
        _display_evaluation_function_definition(
            is_final_layer, actual_input_value, baseline_value, theoretical_diff
        )

        # グローバル変数として保存（後でIG計算で使用）
        st.session_state.actual_input_value = actual_input_value
        st.session_state.baseline_value = baseline_value
        st.session_state.theoretical_diff = theoretical_diff

        return theoretical_diff
    except ImportError:
        return 0.0


def _display_evaluation_function_definition(
    is_final_layer: bool,
    actual_input_value: float,
    baseline_value: float,
    theoretical_diff: float,
):
    """評価関数の定義表示"""
    try:
        import streamlit as st

        # 評価関数の定義表示をコメントアウト（保守性のため残しておく）
        # if is_final_layer:
        #     st.write("### 最終層の評価関数")
        #     st.write(
        #         "評価関数: $\\mathcal{M}_i^{(L)}(a) = ||\\text{Output}_i(a) - \\text{Output}_i(0)||_2$"
        #     )
        #     st.write(
        #         f"- $\\mathcal{{M}}_i^{{(L)}}(1) = ||\\text{{Output}}_i(1) - \\text{{Output}}_i(0)||_2 = {actual_input_value:.6f}$"
        #     )
        #     st.write(
        #         f"- $\\mathcal{{M}}_i^{{(L)}}(0) = ||\\text{{Output}}_i(0) - \\text{{Output}}_i(0)||_2 = {baseline_value:.6f}$"
        #     )
        #     st.write("### 理論的性質")
        #     st.write(
        #         "IGの総和は $\\mathcal{M}_i^{(L)}(1) - \\mathcal{M}_i^{(L)}(0) = \\mathcal{M}_i^{(L)}(1)$ に等しくなるべき"
        #     )
        #     st.write(f"期待値: $\\mathcal{{M}}_i^{{(L)}}(1) = {theoretical_diff:.6f}$")
        # else:
        #     st.write("### 中間層の評価関数")
        #     st.write(
        #         "評価関数: $\\mathcal{M}_{h'}^{(l)}(a) = ||\\alpha_i^{(l+1,h')}(a) - \\alpha_i^{(l+1,h')}(0)||_2$"
        #     )
        #     st.write(
        #         f"- $\\mathcal{{M}}_{{h'}}^{{(l)}}(1) = ||\\alpha_i^{{(l+1,h')}}(1) - \\alpha_i^{{(l+1,h')}}(0)||_2 = {actual_input_value:.6f}$"
        #     )
        #     st.write(
        #         f"- $\\mathcal{{M}}_{{h'}}^{{(l)}}(0) = ||\\alpha_i^{{(l+1,h')}}(0) - \\alpha_i^{{(l+1,h')}}(0)||_2 = {baseline_value:.6f}$"
        #     )
        #     st.write("### 理論的性質")
        #     st.write(
        #         "IGの総和は $\\mathcal{M}_{h'}^{(l)}(1) - \\mathcal{M}_{h'}^{(l)}(0) = \\mathcal{M}_{h'}^{(l)}(1)$ に等しくなるべき"
        #     )
        #     st.write(
        #         f"期待値: $\\mathcal{{M}}_{{h'}}^{{(l)}}(1) = {theoretical_diff:.6f}$"
        #     )
    except ImportError:
        pass


def _execute_ig_calculation(
    ig: IntegratedGradients,
    target_mlp_input: torch.Tensor,
    baseline_mlp_input: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """IG計算の実行"""
    try:
        import streamlit as st

        # IG計算実行デバッグ出力をコメントアウト（保守性のため残しておく）
        # st.write("Captum IG計算実行中...")
        # CaptumのIG計算はfloat32を要求するため、BF16/FP16の場合はfloat32に変換
        original_dtype = target_mlp_input.dtype
        if original_dtype in (torch.bfloat16, torch.float16):
            target_mlp_input_fp32 = target_mlp_input.to(torch.float32)
            baseline_mlp_input_fp32 = baseline_mlp_input.to(torch.float32)
        else:
            target_mlp_input_fp32 = target_mlp_input
            baseline_mlp_input_fp32 = baseline_mlp_input

        ig_attributions = ig.attribute(
            inputs=target_mlp_input_fp32,
            baselines=baseline_mlp_input_fp32,
            n_steps=num_steps,
            return_convergence_delta=False,
        )

        # 元のdtypeに戻す（メモリ節約のため）
        if original_dtype in (torch.bfloat16, torch.float16):
            ig_attributions = ig_attributions.to(original_dtype)

        # st.write("Captum IG計算成功")
        # st.write(f"IG計算結果 shape: {ig_attributions.shape}")
        # st.write(f"IG計算結果 dim: {ig_attributions.dim()}")
        # st.write(f"IG計算結果 dtype: {ig_attributions.dtype}")
        # st.write(f"IG計算結果 device: {ig_attributions.device}")

        return ig_attributions
    except ImportError:
        # CaptumのIG計算はfloat32を要求するため、BF16/FP16の場合はfloat32に変換
        original_dtype = target_mlp_input.dtype
        if original_dtype in (torch.bfloat16, torch.float16):
            target_mlp_input_fp32 = target_mlp_input.to(torch.float32)
            baseline_mlp_input_fp32 = baseline_mlp_input.to(torch.float32)
        else:
            target_mlp_input_fp32 = target_mlp_input
            baseline_mlp_input_fp32 = baseline_mlp_input

        ig_attributions = ig.attribute(
            inputs=target_mlp_input_fp32,
            baselines=baseline_mlp_input_fp32,
            n_steps=num_steps,
            return_convergence_delta=False,
        )

        # 元のdtypeに戻す（メモリ節約のため）
        if original_dtype in (torch.bfloat16, torch.float16):
            ig_attributions = ig_attributions.to(original_dtype)

        return ig_attributions


def _process_ig_results(
    ig_attributions: torch.Tensor, num_heads: int, hidden_size: int
) -> torch.Tensor:
    """IG結果の処理"""
    # 各特徴量の貢献度を取得（768次元）
    feature_contributions = ig_attributions.squeeze()  # [768]

    try:
        import streamlit as st

        # IG結果処理デバッグ出力をコメントアウト（保守性のため残しておく）
        # st.write(f"特徴量貢献度 shape: {feature_contributions.shape}")
        # st.write(f"特徴量貢献度 dim: {feature_contributions.dim()}")
        # st.write(f"特徴量貢献度 numel: {feature_contributions.numel()}")
        # st.write(f"特徴量貢献度 dtype: {feature_contributions.dtype}")
    except ImportError:
        pass

    # スカラー値は理論に反しヘッド別貢献を導出できないためエラー
    if feature_contributions.dim() == 0:
        raise RuntimeError(
            "IG attribution is scalar; cannot derive per-head contributions"
        )
    else:
        # ヘッドごとに貢献度を和して集約
        # [768] → [12, 64] → 各ヘッドの和
        head_dim = hidden_size // num_heads
        head_contributions = feature_contributions.view(num_heads, head_dim)  # [12, 64]

        try:
            import streamlit as st

            # st.write(f"ヘッド別特徴量貢献度 shape: {head_contributions.shape}")
        except ImportError:
            pass

        # 各ヘッドの貢献度を和で集約
        ig_values_list = []
        for head_idx in range(num_heads):
            head_contribution = head_contributions[head_idx].sum().item()
            ig_values_list.append(head_contribution)
            try:
                import streamlit as st

                # st.write(f"ヘッド {head_idx} のIG値: {head_contribution:.6f}")
            except ImportError:
                pass

    # 結果をテンソルに変換
    ig_values = torch.tensor(ig_values_list, device=feature_contributions.device)

    try:
        import streamlit as st

        # st.write(f"最終IG値形状: {ig_values.shape}")
    except ImportError:
        pass

    return ig_values


def _validate_ig_results(
    ig_values: torch.Tensor, theoretical_diff: float, is_final_layer: bool
):
    """IG結果の理論的検証"""
    try:
        import streamlit as st

        # IG値の理論的検証デバッグ出力をコメントアウト（保守性のため残しておく）
        # st.write("=== IG値の理論的検証 ===")
        # IG値の総和
        ig_sum = ig_values.sum().item()
        # st.write(f"IG値の総和: {ig_sum:.6f}")

        # 理論的な期待値（M(1)）
        theoretical_expected = theoretical_diff
        # st.write(f"理論的期待値 M(1): {theoretical_expected:.6f}")

        # 一致度の確認
        difference = abs(ig_sum - theoretical_expected)
        relative_error = (
            difference / abs(theoretical_expected)
            if theoretical_expected != 0
            else float("inf")
        )
        # st.write(f"絶対誤差: {difference:.6f}")
        # st.write(f"相対誤差: {relative_error:.6f}")

        # 理論的性質の確認
        if relative_error < 0.01:  # 1%以下の誤差
            st.success("✅ IG値の総和が理論的期待値と一致しています（誤差 < 1%）")
        elif relative_error < 0.05:  # 5%以下の誤差
            st.warning("⚠️ IG値の総和が理論的期待値とほぼ一致しています（誤差 < 5%）")
        else:
            st.error(
                f"❌ IG値の総和が理論的期待値と大きく異なっています（誤差: {relative_error:.2%}）"
            )

        # 詳細な比較をコメントアウト（保守性のため残しておく）
        # st.write("### 詳細比較")
        # st.write(f"- IG値の総和: {ig_sum:.6f}")
        # st.write(f"- 理論的期待値: {theoretical_expected:.6f}")
        # st.write(f"- 差: {ig_sum - theoretical_expected:.6f}")
        # st.write(f"- 相対誤差: {relative_error:.2%}")

        # 理論的背景の説明
        _display_theoretical_background(is_final_layer)

    except ImportError:
        pass


def _display_theoretical_background(is_final_layer: bool):
    """理論的背景の表示"""
    try:
        import streamlit as st

        st.write("### 理論的背景")
        st.write("**Integrated Gradientsの基本性質**:")
        st.write(
            "$$\\sum_{h} \\text{IG}_{h,h'}^{\\text{MLP}} = \\mathcal{M}_{h'}^{(l)}(1) - \\mathcal{M}_{h'}^{(l)}(0) = \\mathcal{M}_{h'}^{(l)}(1)$$"
        )
        st.write("（∵ $\\mathcal{M}_{h'}^{(l)}(0) = 0$）")

        if is_final_layer:
            st.write("**最終層の場合**:")
            st.write(
                "$$\\sum_{h} \\text{IG}_{h,i}^{\\text{MLP,L}} = \\mathcal{M}_i^{(L)}(1) = ||\\text{Output}_i(1) - \\text{Output}_i(0)||_2$$"
            )
            st.write(
                "これは、全ヘッドのu（ATT出力=MLP入力）が最終出力に与える影響の総和を表します。"
            )
        else:
            st.write("**中間層の場合**:")
            st.write(
                "$$\\sum_{h} \\text{IG}_{h}^{\\text{MLP}} = \\mathcal{M}_i^{(l)}(1) = ||z_i^{(l+1)}(1) - z_i^{(l+1)}(0)||_2$$"
            )
            st.write(
                "これは、全ヘッドのu（ATT出力=MLP入力）が次層のz（MLP出力）に与える影響の総和を表します。"
            )
    except ImportError:
        pass
