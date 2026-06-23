from __future__ import annotations

import logging
import os
import time
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from utils.calculations.ig.attention.attention_ig import (
    compute_attention_ig_global_analysis_multi_layer,
    compute_attention_ig_global_analysis_multi_layer_multi_token)
from utils.calculations.ig.mlp.mlp_ig import (
    compute_mlp_ig_optimized_batch, compute_mlp_ig_theoretical_with_cache)
from utils.cache.unified_cache import UnifiedCache

from .registry import ActiveComputationRegistry
from .tasks import IGTask, generate_task_key

logger = logging.getLogger(__name__)


def _wait_for_memory_available(
    device_id: int, max_wait_seconds: float = 30.0, check_interval: float = 2.0
) -> bool:
    """
    メモリが利用可能になるまで待機
    
    Args:
        device_id: GPUデバイスID
        max_wait_seconds: 最大待機時間（秒）
        check_interval: メモリチェック間隔（秒）
        
    Returns:
        メモリが利用可能になった場合はTrue、タイムアウトした場合はFalse
    """
    if not torch.cuda.is_available():
        return True
    
    start_time = time.time()
    while time.time() - start_time < max_wait_seconds:
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
            free_gb = free_bytes / (1024**3)
            # 最低5GBの空きメモリが必要
            if free_gb >= 5.0:
                return True
            # メモリがまだ不足している場合、少し待機
            time.sleep(check_interval)
        except Exception as e:
            logger.warning(f"GPU {device_id}のメモリチェック中にエラー: {e}")
            time.sleep(check_interval)
    
    logger.warning(
        f"GPU {device_id}のメモリが{max_wait_seconds}秒待機しても利用可能になりませんでした"
    )
    return False


def _check_memory_before_execution(
    device_id: int, required_memory_gb: float = 1.0
) -> bool:
    """
    実行前にメモリをチェック
    
    Args:
        device_id: GPUデバイスID
        required_memory_gb: 必要なメモリ量（GB）
        
    Returns:
        メモリが利用可能な場合はTrue、不足している場合はFalse
    """
    if not torch.cuda.is_available():
        return True
    
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
        free_gb = free_bytes / (1024**3)
        
        if free_gb < required_memory_gb:
            # メモリが不足している場合、待機（ログは削除）
            if _wait_for_memory_available(device_id, max_wait_seconds=30):
                return True
            else:
                return False
        return True
    except Exception as e:
        logger.warning(f"GPU {device_id}のメモリチェック中にエラー: {e}")
        return True  # エラーの場合は実行を許可（従来の動作を維持）


def _ensure_logging_setup() -> None:
    """統一ログ設定を使用してログをセットアップ（既にセットアップされている場合は追加設定をしない）"""
    root_logger = logging.getLogger()
    
    # 既に統一ログ設定がセットアップされているかチェック
    has_file_handler = any(
        isinstance(h, logging.FileHandler) 
        for h in root_logger.handlers
    )
    
    # 統一ログ設定がセットアップされていない場合のみ、簡易設定を追加
    if not has_file_handler:
        logs_dir = os.path.abspath("logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, "ig_calculation_debug.log")

        # 新しい実行時にログファイルをクリア（0から書き始める）
        # 注意: この関数は複数回呼ばれる可能性があるため、最初の呼び出し時のみ削除
        if os.path.exists(log_file) and not hasattr(_ensure_logging_setup, "_cleared"):
            try:
                os.remove(log_file)
                _ensure_logging_setup._cleared = True
            except Exception:
                pass  # 削除に失敗しても続行

        # Streamlit警告を完全に抑制
        import warnings

        warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")
        warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
        warnings.filterwarnings("ignore", message=".*Thread.*missing ScriptRunContext.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="streamlit.*")
        
        # 環境変数でStreamlit警告を抑制
        os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "ERROR")

        # Streamlitロガーの警告レベルを下げる
        for logger_name in [
            "streamlit.runtime.scriptrunner.script_run_context",
            "streamlit.runtime.caching",
            "streamlit.runtime.scriptrunner",
            "streamlit",
        ]:
            streamlit_logger = logging.getLogger(logger_name)
            streamlit_logger.setLevel(logging.CRITICAL)
            streamlit_logger.propagate = False

        # 統一ログ設定モジュールが利用可能な場合はそれを使用
        try:
            from utils.common.logging_setup import setup_unified_logging
            setup_unified_logging(
                log_file_path=log_file,
                log_level=logging.INFO,
                enable_console=False,  # このモジュールではコンソール出力を無効化
                enable_file=True,
                redirect_stdout=False,  # このモジュールでは標準出力をリダイレクトしない
            )
        except ImportError:
            # 統一ログ設定モジュールが利用できない場合は従来の設定を使用
            logging.basicConfig(
                filename=log_file,
                level=logging.INFO,  # DEBUGからINFOに変更（不要なDEBUGログを抑制）
                format="%(asctime)s - %(levelname)s - %(message)s",
                filemode="w",  # 追記モードから上書きモードに変更
                force=True,
            )


class _BaseExecutor:
    def __init__(
        self,
        cache: UnifiedCache,
        registry: ActiveComputationRegistry,
        *,
        is_h100: bool,
        baseline_method: str = "zero",
        input_type: str = "z",
        use_direct_computation: bool = False,
    ) -> None:
        self.cache = cache
        self.registry = registry
        self.is_h100 = is_h100
        self.baseline_method = baseline_method
        self.input_type = input_type
        self.use_direct_computation = use_direct_computation
        # 混合精度は使用しない（FP32のみ）

    def _wait_for_result(
        self,
        task_key: str,
        getter,
        *,
        max_attempts: int = 100,
        interval: float = 0.1,
    ):
        for _ in range(max_attempts):
            time.sleep(interval)
            cached = getter()
            if cached is not None:
                return cached
        return None


class AttentionExecutor(_BaseExecutor):
    def execute(
        self,
        task: IGTask,
        text: str,
        inputs: Dict[str, torch.Tensor],
        *,
        fallback_model: torch.nn.Module,
        model: Optional[torch.nn.Module],
        device_id: Optional[int],
        stream: Optional[torch.cuda.Stream],
        cached_hidden_states: Optional[Tuple] = None,  # 事前計算済みhidden states
    ) -> Optional[np.ndarray]:
        # 各スレッドで実行される関数内でもStreamlit警告を抑制
        import warnings
        warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")
        warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
        warnings.filterwarnings("ignore", message=".*Thread.*missing ScriptRunContext.*")
        
        task_key = generate_task_key(task, text)
        # INFOログは削減（大量に出力されるため）
        # logger.info("Attention IG最適化計算開始: %s", task_key)

        cached = self.cache.get(
            "attention_ig",
            text=text,
            layer_idx=task.layer_idx,
            token_idx=task.token_idx,
            head_idx=task.head_idx,
            num_steps=task.num_steps,
        )
        if cached is not None:
            return cached
        if self.registry.is_active(task_key):
            getter = lambda: self.cache.get(
                "attention_ig",
                text=text,
                layer_idx=task.layer_idx,
                token_idx=task.token_idx,
                head_idx=task.head_idx,
                num_steps=task.num_steps,
            )
            return self._wait_for_result(task_key, getter)
            return self._wait_for_result(task_key, getter)

        self.registry.start(task_key)
        try:
            result = self._compute_direct(
                task,
                inputs,
                fallback_model=fallback_model,
                model=model,
                device_id=device_id,
                stream=stream,
                cached_hidden_states=cached_hidden_states,  # キャッシュを渡す
            )
            # Noneが返された場合はエラーを発生（計算を完了させる）
            if result is None:
                raise RuntimeError(
                    f"Attention IG計算がNoneを返しました: L{task.layer_idx} T{task.token_idx} H{task.head_idx}"
                )
            
            if result is not None:
                self.cache.set(
                    "attention_ig",
                    result,
                    text=text,
                    layer_idx=task.layer_idx,
                    token_idx=task.token_idx,
                    head_idx=task.head_idx,
                    num_steps=task.num_steps,
                )
            return result
        finally:
            self.registry.finish(task_key)

    def _compute_direct(
        self,
        task: IGTask,
        inputs: Dict[str, torch.Tensor],
        *,
        fallback_model: torch.nn.Module,
        model: Optional[torch.nn.Module],
        device_id: Optional[int],
        stream: Optional[torch.cuda.Stream],
        cached_hidden_states: Optional[Tuple] = None,  # 事前計算済みhidden states
    ) -> Optional[np.ndarray]:
        # メモリエラー時は1回だけリトライ（メモリクリーンアップ後）
        result = None

        try:
            # 最初の試行
            try:
                _ensure_logging_setup()
                model = model or fallback_model
                target_device = next(model.parameters()).device
                
                # 事前メモリチェックは削除（パフォーマンス向上のため）
                # メモリエラーが発生した場合のみ処理する
                
                device_ctx = (
                    torch.cuda.device(target_device)
                    if target_device.type == "cuda"
                    else nullcontext()
                )

                # 混合精度は使用しない（FP32のみ）
                autocast_ctx = nullcontext()

                stream_ctx = (
                    torch.cuda.stream(stream)
                    if stream is not None and target_device.type == "cuda"
                    else nullcontext()
                )

                with device_ctx, stream_ctx:
                    local_inputs = inputs
                    if any(t.device != target_device for t in inputs.values()):
                        # Phase 2.1: 非同期データ転送の実装
                        # non_blocking=Trueでデータ転送と計算をオーバーラップ
                        local_inputs = {
                            k: v.to(target_device, non_blocking=True) for k, v in inputs.items()
                        }
                        # 同期は行わない（エラーが発生している可能性があるため）

                    with autocast_ctx:
                        # 単一レイヤーでもmulti_layer関数を使用（レイヤーリストで渡す）
                        batch_results = compute_attention_ig_global_analysis_multi_layer(
                            bert_model=model,
                            inputs=local_inputs,
                            layer_indices=[task.layer_idx],  # 単一レイヤーをリストで渡す
                            target_token_idx=task.token_idx,
                            target_head_idx=task.head_idx,
                            num_steps=task.num_steps,
                            debug=False,
                            cached_hidden_states=cached_hidden_states,
                            baseline_method=self.baseline_method,
                            input_type=self.input_type,
                            use_direct_computation=self.use_direct_computation,
                        )
                        # 結果を単一レイヤー形式に変換
                        result = batch_results.get(task.layer_idx, {})
                        
                        # 同期は行わない（エラーが発生している可能性があるため）

            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                error_msg = str(e)
                # CUDA illegal memory accessエラーの場合は即座に停止
                if "illegal memory access" in error_msg.lower() or "CUDA error" in error_msg:
                    logging.error(f"❌ CUDAエラーが発生しました。計算を停止します: {error_msg[:200]}")
                    raise RuntimeError(f"CUDAエラーにより計算を停止: {error_msg[:200]}")
                # メモリエラーの場合のみ1回リトライ
                if (
                    "out of memory" in error_msg.lower()
                    or "CUBLAS_STATUS_ALLOC_FAILED" in error_msg
                ):
                    logging.warning(
                        f"メモリエラー発生 (メモリクリーンアップ後にリトライ): "
                        f"L{task.layer_idx} T{task.token_idx} H{task.head_idx}, "
                        f"ステップ数: {task.num_steps}, "
                        f"エラー: {error_msg[:100]}"
                    )
                    
                    # GPUメモリ使用状況のログは削除（パフォーマンス向上のため）
                    # メモリクリーンアップ
                    if target_device.type == "cuda":
                        # すべてのGPUでメモリクリーンアップ
                        for gpu_id in range(torch.cuda.device_count()):
                            try:
                                torch.cuda.set_device(gpu_id)
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            except Exception:
                                pass
                        import gc
                        gc.collect()
                        
                        # メモリが解放されるまで待機（最大30秒、ログは削除）
                        if device_id is not None:
                            if not _wait_for_memory_available(device_id, max_wait_seconds=30):
                                # メモリが解放されない場合、エラーを発生させて失敗タスクとして記録
                                raise RuntimeError(
                                    f"GPU {device_id}のメモリが30秒待機しても利用可能になりませんでした。"
                                    f"タスクを失敗として記録します。"
                                )
                        else:
                            # device_idがNoneの場合は、少し待機してからリトライ
                            time.sleep(2.0)
                    else:
                        # CPUの場合もGCを実行
                        import gc
                        gc.collect()
                        time.sleep(1.0)

                    # 1回だけリトライ（同じ設定で）
                    try:
                        with device_ctx, stream_ctx:
                            with autocast_ctx:
                                # 単一レイヤーでもmulti_layer関数を使用（レイヤーリストで渡す）
                                batch_results = compute_attention_ig_global_analysis_multi_layer(
                                    bert_model=model,
                                    inputs=local_inputs,
                                    layer_indices=[task.layer_idx],  # 単一レイヤーをリストで渡す
                                    target_token_idx=task.token_idx,
                                    target_head_idx=task.head_idx,
                                    num_steps=task.num_steps,
                                    debug=False,
                                    cached_hidden_states=cached_hidden_states,
                                    baseline_method=self.baseline_method,
                                    input_type=self.input_type,
                                    use_direct_computation=self.use_direct_computation,
                                )
                                # 結果を単一レイヤー形式に変換
                                result = batch_results.get(task.layer_idx, {})
                    except Exception as retry_error:
                        # リトライも失敗した場合はエラーを発生
                        logging.error(
                            f"メモリエラー: リトライ後も失敗しました。計算を停止します。 "
                            f"L{task.layer_idx} T{task.token_idx} H{task.head_idx}"
                        )
                        raise RuntimeError(
                            f"IG計算がメモリエラーのため失敗しました: {str(retry_error)[:200]}"
                        ) from retry_error
                else:
                    # メモリエラー以外の場合は即座にエラーを返す
                    raise

            # resultがNoneの場合はエラーを発生
            if result is None or result.get("ig_values") is None:
                error_details = "不明"
                error_type = "Unknown"
                if result is not None and isinstance(result, dict):
                    error_details = result.get("error", "不明")
                    error_type = result.get("error_type", "Unknown")
                    # エラー情報を詳細にログに記録
                    logging.error(
                        "Attention IG計算失敗: L%d T%d H%s (device=%s), "
                        "error_type=%s, error_msg=%s",
                        task.layer_idx,
                        task.token_idx,
                        task.head_idx,
                        target_device if "target_device" in locals() else "unknown",
                        error_type,
                        error_details[:500] if isinstance(error_details, str) else str(error_details)[:500],
                    )
                else:
                    logging.error(
                        "Attention IG計算失敗: L%d T%d H%s (device=%s), result=None",
                        task.layer_idx,
                        task.token_idx,
                        task.head_idx,
                        target_device if "target_device" in locals() else "unknown",
                    )
                raise RuntimeError(
                    f"IG計算が完了しませんでした: L{task.layer_idx} T{task.token_idx} H{task.head_idx}, "
                    f"error_type={error_type}, error={error_details[:200] if isinstance(error_details, str) else str(error_details)[:200]}"
                )

            # stream.synchronize()を削除（バッチ処理を効率化）
            # 各タスクごとの同期は不要で、バッチ全体の完了後に同期する
            # if stream is not None and target_device.type == "cuda":
            #     stream.synchronize()

            ig_values = result.get("ig_values") if result else None

            # エラー情報があればログに記録
            if result and isinstance(result, dict) and result.get("error"):
                logging.warning(
                    "Attention IG計算でエラーが返されました: L%d T%d H%s, error=%s",
                    task.layer_idx,
                    task.token_idx,
                    task.head_idx,
                    result.get("error"),
                )

            # 結果の詳細をログに記録
            if ig_values is not None:
                import numpy as np

                if isinstance(ig_values, (list, np.ndarray)):
                    ig_array = (
                        np.array(ig_values)
                        if isinstance(ig_values, list)
                        else ig_values
                    )
                    non_zero_count = np.count_nonzero(ig_array)
                    max_val = float(np.max(ig_array))
                    min_val = float(np.min(ig_array))
                    # INFOログは削減（大量に出力されるため、エラーのみ記録）
                    # logging.info(
                    #     "Attention IG成功: L%d T%d H%s (device=%s), "
                    #     "len=%d, non_zero=%d, max=%.6f, min=%.6f",
                    #     task.layer_idx,
                    #     task.token_idx,
                    #     task.head_idx,
                    #     target_device,
                    #     len(ig_array),
                    #     non_zero_count,
                    #     max_val,
                    #     min_val,
                    # )
                else:
                    logging.warning(
                        "Attention IG結果の型が不正: L%d T%d H%s, type=%s",
                        task.layer_idx,
                        task.token_idx,
                        task.head_idx,
                        type(ig_values),
                    )
            else:
                # resultがNoneの場合はエラーを発生
                error_details = (
                    result.get("error", "不明")
                    if isinstance(result, dict) and result is not None
                    else "result=None"
                )
                logging.error(
                    "Attention IG計算失敗: L%d T%d H%s (device=%s), result=None, details=%s",
                    task.layer_idx,
                    task.token_idx,
                    task.head_idx,
                    target_device if "target_device" in locals() else "unknown",
                    error_details,
                )
                raise RuntimeError(
                    f"Attention IG計算がNoneを返しました: L{task.layer_idx} T{task.token_idx} H{task.head_idx}, "
                    f"error={error_details}"
                )

            return ig_values

        except Exception as exc:  # pragma: no cover - diagnostic logging path
            # エラーの詳細情報を記録して再発生
            import traceback

            error_msg = str(exc)
            error_type = type(exc).__name__
            tb_str = traceback.format_exc()

            logging.error(
                "Attention IG(vectorized)計算エラー: L%d T%d H%s (device=%s), "
                "error_type=%s, error_msg=%s",
                task.layer_idx,
                task.token_idx,
                task.head_idx,
                "unknown",
                error_type,
                error_msg,
            )
            # スタックトレースは最初の500文字のみ（ログが大きくなりすぎないように）
            logging.debug(
                "Attention IG計算エラー詳細 (L%d T%d H%s):\n%s",
                task.layer_idx,
                task.token_idx,
                task.head_idx,
                tb_str[:500],
            )
            # エラーを再発生（デフォルト値は返さない）
            raise

    def execute_multi_layer(
        self,
        tasks: List[IGTask],
        text: str,
        inputs: Dict[str, torch.Tensor],
        *,
        fallback_model: torch.nn.Module,
        model: Optional[torch.nn.Module],
        device_id: Optional[int],
        stream: Optional[torch.cuda.Stream],
        cached_hidden_states: Optional[Tuple] = None,
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        複数レイヤーのAttention IGを一度に計算（最適化版）
        
        同じヘッド×トークンのタスクを全レイヤーでまとめて処理します。
        
        Args:
            tasks: 同じヘッド×トークンのタスクリスト（異なるレイヤー）
            text: 入力テキスト
            inputs: 入力テンソル
            fallback_model: フォールバック用モデル
            model: 使用するモデル
            device_id: デバイスID
            stream: CUDAストリーム
            cached_hidden_states: 事前計算済みhidden states
            
        Returns:
            タスクキーから結果へのマッピング
        """
        if not tasks:
            return {}
        
        # キャッシュチェックとレジストリ登録
        task_keys = [generate_task_key(task, text) for task in tasks]
        results = {}
        tasks_to_compute = []
        task_indices = []
        
        for i, (task, task_key) in enumerate(zip(tasks, task_keys)):
            # キャッシュチェック
            cached = self.cache.get(
                "attention_ig",
                text=text,
                layer_idx=task.layer_idx,
                token_idx=task.token_idx,
                head_idx=task.head_idx,
                num_steps=task.num_steps,
            )
            if cached is not None:
                results[task_key] = cached
                continue
            
            # アクティブな計算を待つ
            if self.registry.is_active(task_key):
                getter = lambda tk=task_key: self.cache.get(
                    "attention_ig",
                    text=text,
                    layer_idx=task.layer_idx,
                    token_idx=task.token_idx,
                    head_idx=task.head_idx,
                    num_steps=task.num_steps,
                )
                cached = self._wait_for_result(task_key, getter)
                if cached is not None:
                    results[task_key] = cached
                    continue
            
            # 計算が必要なタスクを追加
            tasks_to_compute.append(task)
            task_indices.append(i)
            self.registry.start(task_key)
        
        if not tasks_to_compute:
            return results
        
        # 全レイヤーを一度に処理
        try:
            # 同じヘッド×トークンであることを確認
            if tasks_to_compute:
                first_task = tasks_to_compute[0]
                target_token_idx = first_task.token_idx
                target_head_idx = first_task.head_idx
                num_steps = first_task.num_steps
                
                if not all(
                    task.token_idx == target_token_idx
                    and task.head_idx == target_head_idx
                    and task.num_steps == num_steps
                    for task in tasks_to_compute
                ):
                    # 異なるヘッド×トークンの場合は個別処理
                    logging.warning(
                        "execute_multi_layer: タスクが同じヘッド×トークンではありません。個別処理に切り替えます。"
                    )
                    for task, task_key in zip(tasks_to_compute, [task_keys[i] for i in task_indices]):
                        result = self.execute(
                            task, text, inputs,
                            fallback_model=fallback_model,
                            model=model,
                            device_id=device_id,
                            stream=stream,
                            cached_hidden_states=cached_hidden_states,
                        )
                        results[task_key] = result
                    return results
                
                # レイヤーインデックスのリストを作成
                layer_indices = sorted(set(task.layer_idx for task in tasks_to_compute))
                
                # 全レイヤーを一度に処理
                batch_results = self._compute_multi_layer_direct(
                    layer_indices=layer_indices,
                    target_token_idx=target_token_idx,
                    target_head_idx=target_head_idx,
                    num_steps=num_steps,
                    inputs=inputs,
                    fallback_model=fallback_model,
                    model=model,
                    device_id=device_id,
                    stream=stream,
                    cached_hidden_states=cached_hidden_states,
                )
                
                # 結果をキャッシュに保存
                for task in tasks_to_compute:
                    task_key = generate_task_key(task, text)
                    if task.layer_idx in batch_results:
                        result = batch_results[task.layer_idx].get("ig_values")
                        if result is not None:
                            self.cache.set(
                                "attention_ig",
                                result,
                                text=text,
                                layer_idx=task.layer_idx,
                                token_idx=task.token_idx,
                                head_idx=task.head_idx,
                                num_steps=task.num_steps,
                            )
                            results[task_key] = result
                        else:
                            results[task_key] = None
                    else:
                        results[task_key] = None
            else:
                # 計算不要なタスクはNoneを設定
                for task_key in [task_keys[i] for i in task_indices]:
                    if task_key not in results:
                        results[task_key] = None
                        
        except Exception as e:
            logging.exception(
                "Attention IG複数レイヤー計算エラー (%d tasks): %s",
                len(tasks_to_compute),
                str(e),
            )
            # 計算に失敗したタスクに対してNoneを設定
            for task, task_key in zip(tasks_to_compute, [task_keys[i] for i in task_indices]):
                if task_key not in results:
                    results[task_key] = None
        finally:
            # レジストリから削除
            for task_key in [task_keys[i] for i in task_indices]:
                self.registry.finish(task_key)
        
        return results

    def _compute_multi_layer_direct(
        self,
        layer_indices: List[int],
        target_token_idx: int,
        target_head_idx: int,
        num_steps: int,
        inputs: Dict[str, torch.Tensor],
        *,
        fallback_model: torch.nn.Module,
        model: Optional[torch.nn.Module],
        device_id: Optional[int],
        stream: Optional[torch.cuda.Stream],
        cached_hidden_states: Optional[Tuple] = None,
    ) -> Dict[int, Dict]:
        """
        複数レイヤーのAttention IGを一度に計算（GPU最適化）
        
        Args:
            layer_indices: レイヤーインデックスリスト
            target_token_idx: 対象トークンインデックス
            target_head_idx: 対象ヘッドインデックス
            num_steps: 積分ステップ数
            inputs: 入力テンソル
            fallback_model: フォールバック用モデル
            model: 使用するモデル
            device_id: デバイスID
            stream: CUDAストリーム
            cached_hidden_states: 事前計算済みhidden states
            
        Returns:
            各レイヤーの結果（layer_idx -> {"ig_values": List[float], ...}）
        """
        if not layer_indices:
            return {}
        
        try:
            _ensure_logging_setup()
            model = model or fallback_model
            target_device = next(model.parameters()).device
            device_ctx = (
                torch.cuda.device(target_device)
                if target_device.type == "cuda"
                else nullcontext()
            )

            # 混合精度は使用しない（FP32のみ）
            autocast_ctx = nullcontext()

            stream_ctx = (
                torch.cuda.stream(stream)
                if stream is not None and target_device.type == "cuda"
                else nullcontext()
            )

            with device_ctx, stream_ctx:
                local_inputs = inputs
                if any(t.device != target_device for t in inputs.values()):
                    # Phase 2.1: 非同期データ転送の実装
                    # non_blocking=Trueでデータ転送と計算をオーバーラップ
                    local_inputs = {k: v.to(target_device, non_blocking=True) for k, v in inputs.items()}
                    # デバイス移動後の同期（複数GPU間でのデータ共有時は必要）
                    if target_device.type == "cuda" and stream is None:
                        torch.cuda.synchronize(target_device)

                with autocast_ctx:
                    # 複数レイヤーを一度に処理
                    batch_results = compute_attention_ig_global_analysis_multi_layer(
                        bert_model=model,
                        inputs=local_inputs,
                        layer_indices=layer_indices,
                        target_token_idx=target_token_idx,
                        target_head_idx=target_head_idx,
                        num_steps=num_steps,
                        debug=False,
                        cached_hidden_states=cached_hidden_states,
                        baseline_method=self.baseline_method,
                        input_type=self.input_type,
                        use_direct_computation=self.use_direct_computation,
                    )
                    # 計算後の同期は不要（PyTorchが自動管理）

            return batch_results

        except Exception as e:
            logging.exception(
                "Attention IG複数レイヤー計算エラー: layer_indices=%s, target_token=%d, target_head=%d, error=%s",
                layer_indices,
                target_token_idx,
                target_head_idx,
                str(e),
            )
            raise

    def execute_multi_layer_multi_token(
        self,
        tasks: List[IGTask],
        text: str,
        inputs: Dict[str, torch.Tensor],
        *,
        fallback_model: torch.nn.Module,
        model: Optional[torch.nn.Module],
        device_id: Optional[int],
        stream: Optional[torch.cuda.Stream],
        cached_hidden_states: Optional[Tuple] = None,
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        複数レイヤー×複数トークンのAttention IGを一度に計算（最適化版）
        
        同じヘッドのタスクを全レイヤー×全トークンでまとめて処理します。
        
        Args:
            tasks: 同じヘッドのタスクリスト（異なるレイヤー×異なるトークン）
            text: 入力テキスト
            inputs: 入力テンソル
            fallback_model: フォールバック用モデル
            model: 使用するモデル
            device_id: デバイスID
            stream: CUDAストリーム
            cached_hidden_states: 事前計算済みhidden states
            
        Returns:
            タスクキーから結果へのマッピング
        """
        if not tasks:
            return {}
        
        # キャッシュチェックとレジストリ登録
        task_keys = [generate_task_key(task, text) for task in tasks]
        results = {}
        tasks_to_compute = []
        task_indices = []
        
        for i, (task, task_key) in enumerate(zip(tasks, task_keys)):
            # キャッシュチェック
            cached = self.cache.get(
                "attention_ig",
                text=text,
                layer_idx=task.layer_idx,
                token_idx=task.token_idx,
                head_idx=task.head_idx,
                num_steps=task.num_steps,
            )
            if cached is not None:
                results[task_key] = cached
                continue
            
            # アクティブな計算を待つ
            if self.registry.is_active(task_key):
                getter = lambda tk=task_key: self.cache.get(
                    "attention_ig",
                    text=text,
                    layer_idx=task.layer_idx,
                    token_idx=task.token_idx,
                    head_idx=task.head_idx,
                    num_steps=task.num_steps,
                )
                cached = self._wait_for_result(task_key, getter)
                if cached is not None:
                    results[task_key] = cached
                    continue
            
            # 計算が必要なタスクを追加
            tasks_to_compute.append(task)
            task_indices.append(i)
            self.registry.start(task_key)
        
        if not tasks_to_compute:
            return results
        
        # 全レイヤー×全トークンを一度に処理
        try:
            # 同じヘッドであることを確認
            if tasks_to_compute:
                first_task = tasks_to_compute[0]
                target_head_idx = first_task.head_idx
                num_steps = first_task.num_steps
                
                if not all(
                    task.head_idx == target_head_idx
                    and task.num_steps == num_steps
                    for task in tasks_to_compute
                ):
                    # 異なるヘッドの場合は個別処理
                    logging.warning(
                        "execute_multi_layer_multi_token: タスクが同じヘッドではありません。個別処理に切り替えます。"
                    )
                    for task, task_key in zip(tasks_to_compute, [task_keys[i] for i in task_indices]):
                        result = self.execute(
                            task, text, inputs,
                            fallback_model=fallback_model,
                            model=model,
                            device_id=device_id,
                            stream=stream,
                            cached_hidden_states=cached_hidden_states,
                        )
                        results[task_key] = result
                    return results
                
                # レイヤーインデックスとトークンインデックスのリストを作成
                layer_indices = sorted(set(task.layer_idx for task in tasks_to_compute))
                token_indices = sorted(set(task.token_idx for task in tasks_to_compute))
                
                # 全レイヤー×全トークンを一度に処理
                batch_results = self._compute_multi_layer_multi_token_direct(
                    layer_indices=layer_indices,
                    token_indices=token_indices,
                    target_head_idx=target_head_idx,
                    num_steps=num_steps,
                    inputs=inputs,
                    fallback_model=fallback_model,
                    model=model,
                    device_id=device_id,
                    stream=stream,
                    cached_hidden_states=cached_hidden_states,
                )
                
                # 結果をキャッシュに保存
                for task in tasks_to_compute:
                    task_key = generate_task_key(task, text)
                    if task.layer_idx in batch_results and task.token_idx in batch_results[task.layer_idx]:
                        result = batch_results[task.layer_idx][task.token_idx].get("ig_values")
                        if result is not None:
                            self.cache.set(
                                "attention_ig",
                                result,
                                text=text,
                                layer_idx=task.layer_idx,
                                token_idx=task.token_idx,
                                head_idx=task.head_idx,
                                num_steps=task.num_steps,
                            )
                            results[task_key] = result
                        else:
                            results[task_key] = None
                    else:
                        results[task_key] = None
            else:
                # 計算不要なタスクはNoneを設定
                for task_key in [task_keys[i] for i in task_indices]:
                    if task_key not in results:
                        results[task_key] = None
                        
        except Exception as e:
            logging.exception(
                "Attention IG複数レイヤー×複数トークン計算エラー (%d tasks): %s",
                len(tasks_to_compute),
                str(e),
            )
            # 計算に失敗したタスクに対してNoneを設定
            for task, task_key in zip(tasks_to_compute, [task_keys[i] for i in task_indices]):
                if task_key not in results:
                    results[task_key] = None
        finally:
            # レジストリから削除
            for task_key in [task_keys[i] for i in task_indices]:
                self.registry.finish(task_key)
        
        return results

    def _compute_multi_layer_multi_token_direct(
        self,
        layer_indices: List[int],
        token_indices: List[int],
        target_head_idx: int,
        num_steps: int,
        inputs: Dict[str, torch.Tensor],
        *,
        fallback_model: torch.nn.Module,
        model: Optional[torch.nn.Module],
        device_id: Optional[int],
        stream: Optional[torch.cuda.Stream],
        cached_hidden_states: Optional[Tuple] = None,
    ) -> Dict[int, Dict[int, Dict]]:
        """
        複数レイヤー×複数トークンのAttention IGを一度に計算（GPU最適化）
        
        Args:
            layer_indices: レイヤーインデックスリスト
            token_indices: トークンインデックスリスト
            target_head_idx: 対象ヘッドインデックス
            num_steps: 積分ステップ数
            inputs: 入力テンソル
            fallback_model: フォールバック用モデル
            model: 使用するモデル
            device_id: デバイスID
            stream: CUDAストリーム
            cached_hidden_states: 事前計算済みhidden states
            
        Returns:
            各レイヤー×トークンの結果（layer_idx -> token_idx -> {"ig_values": List[float], ...}）
        """
        if not layer_indices or not token_indices:
            return {}
        
        try:
            _ensure_logging_setup()
            import time

            # loggingはファイル先頭で既にインポート済み
            logger = logging.getLogger(__name__)
            
            model = model or fallback_model
            target_device = next(model.parameters()).device
            
            # デバイス移動のタイミング記録は削除（パフォーマンス向上のため）
            device_ctx = (
                torch.cuda.device(target_device)
                if target_device.type == "cuda"
                else nullcontext()
            )

            # 混合精度は使用しない（FP32のみ）
            autocast_ctx = nullcontext()

            stream_ctx = (
                torch.cuda.stream(stream)
                if stream is not None and target_device.type == "cuda"
                else nullcontext()
            )

            with device_ctx, stream_ctx:
                local_inputs = inputs
                if any(t.device != target_device for t in inputs.values()):
                    # ログを削減（デバイス移動は通常の処理）
                    local_inputs = {k: v.to(target_device) for k, v in inputs.items()}
                
                # cached_hidden_statesも正しいデバイスに配置されているか確認
                if cached_hidden_states is not None:
                    # 最初の状態のデバイスを確認
                    first_state_device = None
                    if cached_hidden_states and len(cached_hidden_states) > 0:
                        first_state = cached_hidden_states[0]
                        if hasattr(first_state, 'device'):
                            first_state_device = first_state.device
                    
                    if first_state_device is None or first_state_device != target_device:
                        # ログを削減（デバイス移動は通常の処理）
                        cached_hidden_states = tuple(
                            state.to(target_device) if hasattr(state, 'to') else state
                            for state in cached_hidden_states
                        )

                with autocast_ctx:
                    # 複数レイヤー×複数トークンを一度に処理
                    batch_results = compute_attention_ig_global_analysis_multi_layer_multi_token(
                        bert_model=model,
                        inputs=local_inputs,
                        layer_indices=layer_indices,
                        target_token_indices=token_indices,
                        target_head_idx=target_head_idx,
                        num_steps=num_steps,
                        debug=False,
                        cached_hidden_states=cached_hidden_states,
                        baseline_method=self.baseline_method,
                        input_type=self.input_type,
                    )

            return batch_results

        except Exception as e:
            logging.exception(
                "Attention IG複数レイヤー×複数トークン計算エラー: layer_indices=%s, token_indices=%s, target_head=%d, error=%s",
                layer_indices,
                token_indices,
                target_head_idx,
                str(e),
            )
            raise


class MLPExecutor(_BaseExecutor):
    def execute(
        self,
        task: IGTask,
        text: str,
        inputs: Dict[str, torch.Tensor],
        *,
        fallback_model: torch.nn.Module,
        model: Optional[torch.nn.Module],
        device_id: Optional[int],
        stream: Optional[torch.cuda.Stream],
    ) -> Optional[np.ndarray]:
        # 各スレッドで実行される関数内でもStreamlit警告を抑制
        import warnings
        warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")
        warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
        warnings.filterwarnings("ignore", message=".*Thread.*missing ScriptRunContext.*")
        
        task_key = generate_task_key(task, text)

        cached = self.cache.get(
            "mlp_ig",
            text=text,
            layer_idx=task.layer_idx,
            token_idx=task.token_idx,
            num_steps=task.num_steps,
        )
        if cached is not None:
            return cached

        if self.registry.is_active(task_key):
            getter = lambda: self.cache.get(
                "mlp_ig",
                text=text,
                layer_idx=task.layer_idx,
                token_idx=task.token_idx,
                num_steps=task.num_steps,
            )
            return self._wait_for_result(task_key, getter)

        self.registry.start(task_key)
        try:
            result = self._compute_direct(
                task,
                inputs,
                fallback_model=fallback_model,
                model=model,
                device_id=device_id,
                stream=stream,
            )
            if result is not None:
                self.cache.set(
                    "mlp_ig",
                    result,
                    text=text,
                    layer_idx=task.layer_idx,
                    token_idx=task.token_idx,
                    num_steps=task.num_steps,
                )
            return result
        finally:
            self.registry.finish(task_key)

    def _compute_direct(
        self,
        task: IGTask,
        inputs: Dict[str, torch.Tensor],
        *,
        fallback_model: torch.nn.Module,
        model: Optional[torch.nn.Module],
        device_id: Optional[int],
        stream: Optional[torch.cuda.Stream],
    ) -> Optional[np.ndarray]:
        try:
            _ensure_logging_setup()
            model = model or fallback_model
            target_device = next(model.parameters()).device
            
            # 事前メモリチェックは削除（パフォーマンス向上のため）
            # メモリエラーが発生した場合のみ処理する
            
            device_ctx = (
                torch.cuda.device(target_device)
                if target_device.type == "cuda"
                else nullcontext()
            )

            # 混合精度は使用しない（FP32のみ）
            autocast_ctx = nullcontext()

            stream_ctx = (
                torch.cuda.stream(stream) if stream is not None else nullcontext()
            )
            with device_ctx, stream_ctx:
                local_inputs = inputs
                if any(t.device != target_device for t in inputs.values()):
                    # Phase 2.1: 非同期データ転送の実装
                    # non_blocking=Trueでデータ転送と計算をオーバーラップ
                    local_inputs = {k: v.to(target_device, non_blocking=True) for k, v in inputs.items()}

                with autocast_ctx:
                    # ログ設定は既に_ensure_logging_setup()で行われているため、重複設定を削除
                    # （パフォーマンス向上のため、毎回basicConfigを呼ばない）
                    # INFOログは削減（大量に出力されるため）
                    # logging.info(...)

                    tasks = [{"token_idx": task.token_idx, "head_idx": task.head_idx}]
                    batch_results = compute_mlp_ig_optimized_batch(
                        model,
                        task.layer_idx,
                        tasks,
                        num_steps=task.num_steps,
                        is_global_analysis=True,
                        baseline_method=self.baseline_method,
                    )

            # stream.synchronize()を削除（バッチ処理を効率化）
            # if stream is not None and target_device.type == "cuda":
            #     stream.synchronize()

            if batch_results and batch_results[0].get("success"):
                contrib = batch_results[0].get("contributions")
                # INFOログは削減（大量に出力されるため）
                # logging.info(...)
                return contrib

            logging.warning("MLP IG(batch)が失敗。従来経路へフォールバック")
            return compute_mlp_ig_theoretical_with_cache(
                model,
                task.layer_idx,
                task.token_idx,
                task.head_idx,
                task.num_steps,
                baseline_method=self.baseline_method,
            )

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            error_msg = str(e)
            # メモリエラーの場合、詳細なログを記録
            if (
                "out of memory" in error_msg.lower()
                or "CUBLAS_STATUS_ALLOC_FAILED" in error_msg
            ):
                logging.warning(
                    f"MLP IG計算でメモリエラー発生: "
                    f"L{task.layer_idx} T{task.token_idx}, "
                    f"ステップ数: {task.num_steps}, "
                    f"エラー: {error_msg[:100]}"
                )
                
                # GPUメモリ使用状況のログは削除（パフォーマンス向上のため）
            
            logging.exception(
                "MLP IG計算エラー L%d T%d H%s",
                task.layer_idx,
                task.token_idx,
                task.head_idx,
            )
            return None

    def execute_batch(
        self,
        tasks: List[IGTask],
        text: str,
        inputs: Dict[str, torch.Tensor],
        *,
        fallback_model: torch.nn.Module,
        model: Optional[torch.nn.Module],
        device_id: Optional[int],
        stream: Optional[torch.cuda.Stream],
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        複数タスクをまとめてバッチ処理
        
        Args:
            tasks: 処理するタスクリスト（同じレイヤーである必要がある）
            text: 入力テキスト
            inputs: 入力テンソル
            fallback_model: フォールバック用モデル
            model: 使用するモデル
            device_id: デバイスID
            stream: CUDAストリーム
            
        Returns:
            タスクキーから結果へのマッピング
        """
        if not tasks:
            return {}
        
        # キャッシュチェックとレジストリ登録
        task_keys = [generate_task_key(task, text) for task in tasks]
        results = {}
        tasks_to_compute = []
        task_indices = []
        
        for i, (task, task_key) in enumerate(zip(tasks, task_keys)):
            # キャッシュチェック
            cached = self.cache.get(
                "mlp_ig",
                text=text,
                layer_idx=task.layer_idx,
                token_idx=task.token_idx,
                num_steps=task.num_steps,
            )
            if cached is not None:
                results[task_key] = cached
                continue
            
            # アクティブな計算を待つ
            if self.registry.is_active(task_key):
                getter = lambda tk=task_key: self.cache.get(
                    "mlp_ig",
                    text=text,
                    layer_idx=task.layer_idx,
                    token_idx=task.token_idx,
                    num_steps=task.num_steps,
                )
                cached = self._wait_for_result(task_key, getter)
                if cached is not None:
                    results[task_key] = cached
                    continue
            
            # 計算が必要なタスクを追加
            tasks_to_compute.append(task)
            task_indices.append(i)
            self.registry.start(task_key)
        
        if not tasks_to_compute:
            return results
        
        # バッチ処理実行
        try:
            batch_results = self._compute_batch_direct(
                tasks_to_compute,
                inputs,
                fallback_model=fallback_model,
                model=model,
                device_id=device_id,
                stream=stream,
            )
            
            # 結果をキャッシュに保存
            for task, task_key, batch_result in zip(tasks_to_compute, [task_keys[i] for i in task_indices], batch_results):
                if batch_result is not None:
                    self.cache.set(
                        "mlp_ig",
                        batch_result,
                        text=text,
                        layer_idx=task.layer_idx,
                        token_idx=task.token_idx,
                        num_steps=task.num_steps,
                    )
                    results[task_key] = batch_result
                else:
                    # Noneの結果も記録する（エラーハンドリングのため）
                    results[task_key] = None
        except Exception as e:
            # 例外発生時も、すべてのタスクに対してNoneを設定してエラーハンドリングを可能にする
            logging.exception(
                "MLP IGバッチ計算エラー L%d (%d tasks): %s",
                tasks_to_compute[0].layer_idx if tasks_to_compute else -1,
                len(tasks_to_compute),
                str(e),
            )
            # 計算に失敗したタスクに対してNoneを設定
            for task, task_key in zip(tasks_to_compute, [task_keys[i] for i in task_indices]):
                if task_key not in results:
                    results[task_key] = None
        finally:
            # レジストリから削除
            for task_key in [task_keys[i] for i in task_indices]:
                self.registry.finish(task_key)
        
        return results

    def _compute_batch_direct(
        self,
        tasks: List[IGTask],
        inputs: Dict[str, torch.Tensor],
        *,
        fallback_model: torch.nn.Module,
        model: Optional[torch.nn.Module],
        device_id: Optional[int],
        stream: Optional[torch.cuda.Stream],
    ) -> List[Optional[np.ndarray]]:
        """
        複数タスクをまとめてバッチ処理（GPU最適化）
        
        Args:
            tasks: 処理するタスクリスト（同じレイヤーである必要がある）
            inputs: 入力テンソル
            fallback_model: フォールバック用モデル
            model: 使用するモデル
            device_id: デバイスID
            stream: CUDAストリーム
            
        Returns:
            各タスクの結果リスト
        """
        if not tasks:
            return []
        
        # 同じレイヤーであることを確認
        layer_idx = tasks[0].layer_idx
        num_steps = tasks[0].num_steps
        if not all(task.layer_idx == layer_idx and task.num_steps == num_steps for task in tasks):
            # 異なるレイヤーやステップ数の場合は個別処理
            return [self._compute_direct(task, inputs, fallback_model=fallback_model, model=model, device_id=device_id, stream=stream) for task in tasks]
        
        try:
            _ensure_logging_setup()
            model = model or fallback_model
            target_device = next(model.parameters()).device
            device_ctx = (
                torch.cuda.device(target_device)
                if target_device.type == "cuda"
                else nullcontext()
            )

            # 混合精度は使用しない（FP32のみ）
            autocast_ctx = nullcontext()

            stream_ctx = (
                torch.cuda.stream(stream) if stream is not None else nullcontext()
            )
            
            with device_ctx, stream_ctx:
                local_inputs = inputs
                if any(t.device != target_device for t in inputs.values()):
                    # Phase 2.1: 非同期データ転送の実装
                    # non_blocking=Trueでデータ転送と計算をオーバーラップ
                    local_inputs = {k: v.to(target_device, non_blocking=True) for k, v in inputs.items()}

                with autocast_ctx:
                    # 同期は行わない（エラーが発生している可能性があるため）
                    
                    # バッチ処理用のタスクリストを作成
                    batch_tasks = [
                        {"token_idx": task.token_idx, "head_idx": task.head_idx}
                        for task in tasks
                    ]
                    
                    batch_results = compute_mlp_ig_optimized_batch(
                        model,
                        layer_idx,
                        batch_tasks,
                        num_steps=num_steps,
                        is_global_analysis=True,
                        baseline_method=self.baseline_method,
                    )

            # 結果を抽出
            results = []
            for i, batch_result in enumerate(batch_results):
                if batch_result and batch_result.get("success"):
                    contrib = batch_result.get("contributions")
                    results.append(contrib)
                else:
                    # フォールバック処理
                    task = tasks[i]
                    fallback_result = compute_mlp_ig_theoretical_with_cache(
                        model,
                        task.layer_idx,
                        task.token_idx,
                        task.head_idx,
                        task.num_steps,
                        baseline_method=self.baseline_method,
                    )
                    results.append(fallback_result)
            
            return results

        except Exception:
            logging.exception(
                "MLP IGバッチ計算エラー L%d (%d tasks)",
                layer_idx,
                len(tasks),
            )
            # エラー時は個別処理にフォールバック
            return [
                self._compute_direct(
                    task, inputs, fallback_model=fallback_model, model=model, device_id=device_id, stream=stream
                )
                for task in tasks
            ]
