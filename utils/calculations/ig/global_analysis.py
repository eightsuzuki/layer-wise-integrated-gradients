# global_analysis.py
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from utils.calculations.ig.attention.attention_ig import (
    compute_attention_ig_global_analysis_multi_layer,
)
from utils.calculations.ig.mlp.mlp_ig import compute_mlp_ig_theoretical_with_cache
from utils.calculations.ig.resource_planner import plan_gpu_batch_size

_GA_LOGGER_INITIALIZED = False


def _detect_optimal_precision() -> Tuple[str, torch.dtype]:
    """
    利用可能なGPUに応じて最適な精度を自動検出

    Returns:
        (precision_name, dtype): 精度名とdtype

    Priority:
        1. BF16: H100, A100 (Ampere以降, compute capability >= 8.0)
        2. FP16: V100, T4, RTX系 (compute capability >= 7.0)
        3. FP32: CPU, 古いGPU
    """
    if not torch.cuda.is_available():
        return ("FP32 (CPU)", torch.float32)

    try:
        # 現在のGPUのプロパティを取得
        device_props = torch.cuda.get_device_properties(0)
        compute_capability = device_props.major + device_props.minor / 10

        # BF16サポートチェック（Ampere以降）
        supports_bf16 = False
        if hasattr(torch.cuda, "is_bf16_supported"):
            try:
                supports_bf16 = torch.cuda.is_bf16_supported()
            except:
                supports_bf16 = compute_capability >= 8.0
        else:
            supports_bf16 = compute_capability >= 8.0

        if supports_bf16:
            return (f"BF16 ({device_props.name})", torch.bfloat16)

        # FP16サポートチェック（Volta以降）
        if compute_capability >= 7.0:
            return (f"FP16 ({device_props.name})", torch.float16)

        # 古いGPUの場合はFP32
        return (f"FP32 ({device_props.name}, 旧GPU)", torch.float32)

    except Exception as e:
        # フォールバック
        return ("FP32 (fallback)", torch.float32)


def _get_ga_logger() -> logging.Logger:
    """統一ログ設定を使用してロガーを取得"""
    global _GA_LOGGER_INITIALIZED
    logger = logging.getLogger("global_analysis")
    if not _GA_LOGGER_INITIALIZED:
        logger.setLevel(logging.NOTSET)  # ルートロガーのレベルを使用

        # 統一ログ設定モジュールが利用可能な場合はそれを使用
        root_logger = logging.getLogger()
        has_file_handler = any(
            isinstance(h, logging.FileHandler) for h in root_logger.handlers
        )

        if not has_file_handler:
            try:
                from utils.common.logging_setup import setup_unified_logging

                logs_dir = os.path.abspath("logs")
                os.makedirs(logs_dir, exist_ok=True)
                log_file = os.path.join(logs_dir, "ig_calculation_debug.log")
                setup_unified_logging(
                    log_file_path=log_file,
                    log_level=logging.INFO,
                    enable_console=False,
                    enable_file=True,
                    redirect_stdout=False,
                )
            except ImportError:
                # 統一ログ設定モジュールが利用できない場合は従来の設定を使用
                try:
                    os.makedirs("logs", exist_ok=True)
                    fh = logging.FileHandler(
                        "logs/ig_calculation_debug.log", encoding="utf-8"
                    )
                    fh.setLevel(logging.DEBUG)
                    fmt = logging.Formatter(
                        fmt="%(asctime)s [%(levelname)s] GA: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                    fh.setFormatter(fmt)
                    if not any(
                        isinstance(h, logging.FileHandler) for h in logger.handlers
                    ):
                        logger.addHandler(fh)
                except Exception:
                    pass  # ロガー設定失敗時は標準ロガーのみ

        _GA_LOGGER_INITIALIZED = True
    return logger


def _get_model_device(model) -> str:
    try:
        if model is None:
            return "<none>"
        return str(next(model.parameters()).device)
    except Exception:
        return "<unknown>"


def _get_inputs_devices(inputs) -> Dict[str, str]:
    try:
        return {k: str(v.device) for k, v in inputs.items()}
    except Exception:
        return {}


def _calculate_optimal_batch_size(*args, **kwargs):
    raise RuntimeError(
        "_calculate_optimal_batch_size is deprecated. Use plan_gpu_batch_size from resource_planner."
    )


def _auto_configure_runtime(
    *,
    total_calculations: int,
    batch_size: Optional[int],
    max_workers: Optional[int],
    auto_batch_size: bool,
    use_mixed_precision: Optional[bool],
    enable_tuning: bool,
    logger: logging.Logger,
) -> Tuple[int, int, bool, bool, Dict[str, Any]]:
    """
    利用可能なGPUリソースを検出してランタイム設定を動的に調整

    Returns:
        (batch_size, max_workers, auto_batch_size, use_mixed_precision, runtime_info)
    """
    runtime_info: Dict[str, Any] = {
        "tuning_enabled": bool(enable_tuning),
        "gpu_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "inputs": {
            "total_calculations": int(total_calculations),
            "requested_batch_size": batch_size if batch_size is not None else None,
            "requested_auto_batch_size": bool(auto_batch_size),
            "requested_max_workers": max_workers if max_workers is not None else None,
            "requested_use_mixed_precision": (
                None if use_mixed_precision is None else bool(use_mixed_precision)
            ),
        },
        "devices": [],
        "aggregate": {},
        "torch_flags": {},
    }

    # batch_sizeとmax_workersがNoneの場合はデフォルト値を設定
    if batch_size is None:
        batch_size = 8  # デフォルト値
    if max_workers is None:
        max_workers = 4  # デフォルト値

    tuned_batch_size = max(int(batch_size), 1)
    tuned_max_workers = max(int(max_workers), 1)
    tuned_auto_batch = bool(auto_batch_size)
    tuned_precision: Optional[bool] = use_mixed_precision

    if not torch.cuda.is_available():
        if tuned_precision is None:
            tuned_precision = False
        runtime_info["summary"] = "GPU未検出: CPUモードで実行"
        logger.debug(
            "Runtime auto-config (CPU): batch=%d max_workers=%d mixed_precision=%s",
            tuned_batch_size,
            tuned_max_workers,
            tuned_precision,
        )
        return (
            tuned_batch_size,
            tuned_max_workers,
            tuned_auto_batch,
            tuned_precision,
            runtime_info,
        )

    device_count = runtime_info["device_count"]
    aggregate_sm = 0
    aggregate_free_gb = 0.0
    aggregate_total_gb = 0.0
    any_tf32 = False
    any_fp16 = False
    any_bf16 = False

    original_device = torch.cuda.current_device()
    try:
        for device_idx in range(device_count):
            torch.cuda.set_device(device_idx)
            props = torch.cuda.get_device_properties(device_idx)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
            except AttributeError:
                total_bytes = props.total_memory
                reserved_bytes = torch.cuda.memory_reserved(device_idx)
                allocated_bytes = torch.cuda.memory_allocated(device_idx)
                free_bytes = max(total_bytes - max(reserved_bytes, allocated_bytes), 0)
            free_gb = free_bytes / 1024**3
            total_gb = total_bytes / 1024**3

            aggregate_sm += props.multi_processor_count
            aggregate_free_gb += free_gb
            aggregate_total_gb += total_gb
            any_tf32 = any_tf32 or props.major >= 8
            any_fp16 = any_fp16 or props.major >= 7

            supports_bf16 = False
            if hasattr(torch.cuda, "is_bf16_supported"):
                try:
                    supports_bf16 = torch.cuda.is_bf16_supported()
                except Exception:
                    supports_bf16 = False
            any_bf16 = any_bf16 or supports_bf16

            runtime_info["devices"].append(
                {
                    "index": device_idx,
                    "name": props.name,
                    "total_memory_gb": round(total_gb, 2),
                    "free_memory_gb": round(free_gb, 2),
                    "multi_processor_count": props.multi_processor_count,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "supports_bf16": supports_bf16,
                    "supports_tf32": props.major >= 8,
                }
            )
    finally:
        torch.cuda.set_device(original_device)

    runtime_info["aggregate"] = {
        "sm_count": aggregate_sm,
        "total_memory_gb": round(aggregate_total_gb, 2),
        "free_memory_gb": round(aggregate_free_gb, 2),
    }

    # 混合精度の自動判定
    if tuned_precision is None:
        # 最適な精度を自動検出
        precision_name, precision_dtype = _detect_optimal_precision()
        tuned_precision = precision_dtype != torch.float32
        runtime_info["auto_detected_precision"] = {
            "name": precision_name,
            "dtype": str(precision_dtype),
            "enabled": tuned_precision,
        }

    if enable_tuning:
        if tuned_auto_batch:
            sm_based_floor = max(aggregate_sm // 4, 8)
            gpu_based_floor = device_count * 8
            recommended_floor = max(tuned_batch_size, sm_based_floor, gpu_based_floor)
            tuned_batch_size = min(
                max(recommended_floor, 8), max(1, total_calculations)
            )

        # H100の場合は大幅にワーカー数を増やす
        if device_count > 0 and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if "H100" in device_name:
                tuned_max_workers = max(
                    tuned_max_workers, min(512, device_count * 128)
                )  # H100なら最大512まで
            elif "A100" in device_name:
                tuned_max_workers = max(tuned_max_workers, min(256, device_count * 64))
            else:
                tuned_max_workers = max(tuned_max_workers, min(128, device_count * 32))
        else:
            tuned_max_workers = max(tuned_max_workers, min(32, device_count * 8))

    # Torch backend tuning
    try:
        torch.backends.cudnn.benchmark = True
        runtime_info["torch_flags"]["cudnn_benchmark"] = True
    except Exception:
        runtime_info["torch_flags"]["cudnn_benchmark"] = False

    try:
        torch.backends.cuda.matmul.allow_tf32 = any_tf32
        runtime_info["torch_flags"]["matmul.allow_tf32"] = bool(
            torch.backends.cuda.matmul.allow_tf32
        )
    except Exception:
        runtime_info["torch_flags"]["matmul.allow_tf32"] = False

    try:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        runtime_info["torch_flags"][
            "matmul.allow_fp16_reduced_precision_reduction"
        ] = True
    except Exception:
        runtime_info["torch_flags"][
            "matmul.allow_fp16_reduced_precision_reduction"
        ] = False

    try:
        torch.set_float32_matmul_precision("high" if any_tf32 else "medium")
        runtime_info["torch_flags"]["float32_matmul_precision"] = (
            "high" if any_tf32 else "medium"
        )
    except Exception:
        runtime_info["torch_flags"]["float32_matmul_precision"] = "default"

    tuned_precision = bool(tuned_precision)
    runtime_info["selected"] = {
        "batch_size_floor": tuned_batch_size,
        "max_workers": tuned_max_workers,
        "use_mixed_precision": bool(tuned_precision),
        "auto_batch_size": tuned_auto_batch,
    }

    device_summary_parts = []
    for dev in runtime_info["devices"]:
        device_summary_parts.append(
            f"{dev['name']}#{dev['index']}({dev['free_memory_gb']:.1f}/{dev['total_memory_gb']:.1f}GB)"
        )
    device_summary = ", ".join(device_summary_parts) if device_summary_parts else "N/A"
    runtime_info["summary"] = (
        f"GPU {device_count}台: {device_summary} | "
        f"バッチ基準≥{tuned_batch_size} | "
        f"ワーカー≥{tuned_max_workers} | "
        f"混合精度 {'ON' if tuned_precision else 'OFF'}"
    )
    logger.debug("Runtime auto-config: %s", runtime_info)

    return (
        tuned_batch_size,
        tuned_max_workers,
        tuned_auto_batch,
        tuned_precision,
        runtime_info,
    )


def compute_global_ig_analysis(
    unified_model,
    tokenizer,
    text: str,
    num_steps: int,
    progress_callback=None,
    batch_size: Optional[int] = None,
    max_workers: Optional[int] = None,
    auto_batch_size: bool = True,
    use_mixed_precision: Optional[bool] = None,
    auto_configure: bool = True,
    # 診断/ドライラン
    diagnostic: bool = False,
    dry_run: bool = False,
    diagnostic_probe: bool = True,
    reset_cache: bool = True,
    baseline_method: str = "zero",  # ベースライン選択方法
    input_type: str = "v",  # 入力タイプ（"z": 入力埋め込み, "v": Valueベクトル）
    use_direct_computation: bool = True,  # 直接計算を使用するか（input_type="v"の場合）
    # 後方互換性（非推奨）
    model_lightning=None,
    model_attn=None,
    model_mlp=None,
) -> Dict:
    """
    全体分析：すべてのLayerのATTとMLPのすべてのHeadのすべてのTokenのIG計算（GPU最適化版）

    理論文書「4.IGの経路の定義について.md」の5.3節に基づき、
    input_type="v"かつuse_direct_computation=Trueの場合、線形性を利用した直接計算を使用します。

    Args:
        unified_model: 統合BERTモデル（UnifiedBertModel）
        tokenizer: トークナイザー
        text: 入力テキスト
        num_steps: 積分分割数（直接計算の場合は使用されない）
        progress_callback: 進捗コールバック関数
        batch_size: GPUバッチサイズ
        max_workers: 並列処理数
        auto_batch_size: GPUメモリに応じて自動調整するか
        use_mixed_precision: 混合精度計算を使用するか（Noneの場合は自動判定）
        auto_configure: 利用可能なGPUに応じて設定を自動調整するか
        baseline_method: ベースライン選択方法（"zero", "self_input_token"）
        input_type: 入力タイプ（"z": 入力埋め込み, "v": Valueベクトル）
        use_direct_computation: 直接計算を使用するか（input_type="v"の場合のみ有効）
        model_lightning: [非推奨] 後方互換性のため残存
        model_attn: [非推奨] 後方互換性のため残存
        model_mlp: [非推奨] 後方互換性のため残存

    Returns:
        Dict: 分析結果

    Note:
        - H100, A100, V100など各種GPUに自動対応
        - CPU環境でもフォールバック動作可能
        - 混合精度: BF16 (H100), FP16 (V100/A100), FP32 (CPU/旧GPU)
        - input_type="v"かつuse_direct_computation=Trueの場合、IGの数値積分は不要（理論文書5.3節）
    """

    # 後方互換性チェック
    if model_lightning is not None:
        import warnings

        warnings.warn(
            "model_lightning引数は非推奨です。unified_modelを使用してください。",
            DeprecationWarning,
            stacklevel=2,
        )
        if unified_model is None:
            unified_model = model_lightning

    logger = _get_ga_logger()
    # GPU0固定
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    except Exception:
        pass
    start_time = time.time()

    # キャッシュ初期化（要求時）
    if reset_cache:
        try:
            from utils.cache.bert_cache import clear_bert_cache
            from utils.cache.unified_cache import clear_all_cache

            clear_all_cache()
            clear_bert_cache()
        except Exception as e:
            logger.warning("キャッシュ初期化エラー: %s", e)

    # 入力テンソルの準備（dry_run時は簡易トークン化を許容）
    if tokenizer is not None:
        inputs = tokenizer(text, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    else:
        inputs = None
        tokens = text.strip().split()

    total_layers = getattr(
        getattr(model_lightning, "config", None), "num_hidden_layers", 12
    )
    total_heads = getattr(
        getattr(model_lightning, "config", None), "num_attention_heads", 12
    )
    total_tokens = len(tokens)
    total_calculations = total_layers * total_tokens * (total_heads + 1)

    # 予測・診断（dry_run対応）
    def _count_tasks(layers: int, tokens_n: int, heads: int) -> Dict[str, int]:
        attn = layers * tokens_n * heads
        mlp = max(layers - 1, 0) * tokens_n + 1
        return {"attention": attn, "mlp": mlp, "total": attn + mlp}

    task_counts = _count_tasks(total_layers, total_tokens, total_heads)

    # デバイス情報ログ
    gpu_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu_available else "CPU"
    logger.debug(
        f"env: gpu_available={gpu_available}, device={device_name}, layers={total_layers}, heads={total_heads}, tokens={total_tokens}, steps={num_steps}"
    )

    (
        batch_size,
        max_workers,
        auto_batch_size,
        use_mixed_precision,
        runtime_config,
    ) = _auto_configure_runtime(
        total_calculations=total_calculations,
        batch_size=batch_size,
        max_workers=max_workers,
        auto_batch_size=auto_batch_size,
        use_mixed_precision=use_mixed_precision,
        enable_tuning=auto_configure,
        logger=logger,
    )

    if auto_batch_size:
        planned_batch_size = plan_gpu_batch_size(
            total_calculations,
            base_batch_size=batch_size,
            use_mixed_precision=use_mixed_precision,
            memory_safety_margin=0.9,
        )
        batch_size = max(1, planned_batch_size)
    runtime_config.setdefault("selected", {})["batch_size_final"] = batch_size
    runtime_config.setdefault("selected", {})["max_workers_final"] = max_workers
    runtime_config.setdefault("selected", {})[
        "use_mixed_precision"
    ] = use_mixed_precision
    runtime_config.setdefault("selected", {})["auto_batch_size"] = auto_batch_size
    runtime_summary_message = runtime_config.get("summary")

    if dry_run:
        diagnostics = {
            "gpu_available": gpu_available,
            "device": device_name,
            "task_counts": task_counts,
            "predicted_total_seconds": None,
            "attn_probe_sec": None,
            "mlp_probe_sec": None,
            "probe_used": False,
            "batch_size": batch_size,
            "runtime": runtime_config,
        }
        logger.debug(f"dry_run: tasks={task_counts}")
        return {
            "text": text,
            "tokens": tokens,
            "num_steps": num_steps,
            "layers": {},
            "diagnostics": diagnostics,
            "dry_run": True,
            "runtime": runtime_config,
        }

    def _describe_model(model) -> str:
        if model is None:
            return "<none>"
        try:
            class_name = model.__class__.__name__
        except Exception:
            class_name = str(type(model))
        device = _get_model_device(model)
        return f"{class_name}@{device}"

    input_preview = (text or "").strip()
    input_preview = " ".join(input_preview.split())
    if len(input_preview) > 80:
        input_preview = f"{input_preview[:77]}..."
    start_notice = (
        "Global分析開始: "
        f"model={_describe_model(model_lightning)}, "
        f"attn={_describe_model(model_attn)}, "
        f"mlp={_describe_model(model_mlp)}, "
        f"steps={num_steps}, tokens={total_tokens}, total_calc={total_calculations}"
    )
    input_notice = (
        f"入力テキスト: {input_preview}" if input_preview else "入力テキスト: <empty>"
    )
    input_notice_shown = False

    tqdm_bar = None
    orig_progress_callback = progress_callback
    disable_tqdm = os.getenv("TQDM_DISABLE", "0") == "1"
    if not disable_tqdm:
        try:
            from tqdm import tqdm

            tqdm.write(start_notice)
            tqdm.write(input_notice)
            input_notice_shown = True

            tqdm_bar = tqdm(
                total=max(total_calculations, 1),
                desc="IG計算",
                unit="calc",
                leave=False,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )
        except Exception:
            tqdm_bar = None

    if not input_notice_shown:
        # tqdmが無効な場合はコンソールにも出力しない（ログ氾濫防止）
        pass

    last_progress = {
        "current": 0,
        "total": max(total_calculations, 1),
        "message": "",
    }

    def progress_notify(current: int, total: int, message: str):
        if tqdm_bar:
            if total and total > 0:
                new_total = max(int(total), 1)
                if new_total != last_progress["total"]:
                    tqdm_bar.reset(total=new_total)
                    last_progress["total"] = new_total

            target = max(int(current), 0)
            delta = target - last_progress["current"]
            if delta < 0:
                tqdm_bar.reset()
                tqdm_bar.update(target)
            elif delta > 0:
                tqdm_bar.update(delta)

            message_text = ""
            if message:
                message_text = " " + message.strip()[:78]
            if message_text != last_progress["message"]:
                tqdm_bar.set_postfix_str(message_text)
                tqdm_bar.refresh()
            elif delta == 0:
                tqdm_bar.refresh()

            last_progress["current"] = target
            last_progress["message"] = message_text
        if orig_progress_callback:
            orig_progress_callback(current, total, message)

    progress_callback = progress_notify
    if runtime_summary_message and progress_callback:
        progress_callback(0, total_calculations, runtime_summary_message)

    # デバイス管理（unified_model使用）
    from utils.calculations.shared.ig_calculations import (
        ensure_model_on_device,
        ensure_tensors_on_device,
    )

    ensure_model_on_device(unified_model)
    inputs = ensure_tensors_on_device(inputs)

    logger.debug(
        f"placement: unified_model_device={_get_model_device(unified_model)}, inputs_devices={_get_inputs_devices(inputs)}"
    )

    # GPUメモリと並列処理能力に応じたバッチ設定を共有
    precision_info = " (混合精度)" if use_mixed_precision else ""
    if auto_batch_size and progress_callback:
        progress_callback(
            0, total_calculations, f"適応的バッチサイズ: {batch_size}{precision_info}"
        )
    logger.debug(f"batch_size_selected={batch_size}{precision_info}")

    # 全体の計算回数は既に算出済み

    # 診断用のプローブ（1サンプル計測）
    attn_probe_sec = None
    mlp_probe_sec = None
    predicted_total_seconds = None
    if diagnostic and diagnostic_probe:
        try:
            # attention probe
            t0 = time.time()
            _ = compute_attention_ig_global_analysis_multi_layer(
                bert_model=model_attn,
                inputs=inputs,
                layer_indices=[0],
                target_token_idx=0,
                target_head_idx=0,
                num_steps=num_steps,
                debug=False,
                baseline_method=baseline_method,  # baseline_methodを渡す
                input_type=input_type,  # 入力タイプを渡す
                use_direct_computation=use_direct_computation,  # 直接計算フラグを渡す
            )
            # 結果は辞書形式で返される: {layer_idx: {"ig_values": List[float], ...}}
            attn_probe_sec = max(time.time() - t0, 1e-6)
        except Exception as e:
            logger.debug(f"probe(attention) failed: {e}")
        try:
            # mlp probe
            t0 = time.time()
            _ = compute_mlp_ig_theoretical_with_cache(
                model_mlp,
                0,
                0,
                None,
                num_steps,
                is_global_analysis=True,
                baseline_method=baseline_method,
            )
            mlp_probe_sec = max(time.time() - t0, 1e-6)
        except Exception as e:
            logger.debug(f"probe(mlp) failed: {e}")

        if attn_probe_sec is not None and mlp_probe_sec is not None:
            predicted_total_seconds = (
                attn_probe_sec * task_counts["attention"]
                + mlp_probe_sec * task_counts["mlp"]
            )
            logger.debug(
                f"probe: attn={attn_probe_sec:.4f}s, mlp={mlp_probe_sec:.4f}s, predicted_total={predicted_total_seconds:.1f}s, tasks={task_counts}"
            )

    # 混合精度の設定
    precision_context = None
    precision_dtype = torch.float32
    scaler = None

    if use_mixed_precision and torch.cuda.is_available():
        # 最適な精度を自動検出
        precision_name, precision_dtype = _detect_optimal_precision()

        # AutocastのContext Managerを作成
        precision_context = torch.cuda.amp.autocast(dtype=precision_dtype)

        # GradScalerは勾配計算時のみ必要（IG計算では不要だが念のため）
        if precision_dtype == torch.float16:
            scaler = torch.cuda.amp.GradScaler()

        if progress_callback:
            progress_callback(
                0, total_calculations, f"混合精度を有効化: {precision_name}"
            )

    # 進捗表示の初期化
    if progress_callback:
        precision_info = " (混合精度)" if use_mixed_precision else ""
        progress_callback(
            0, total_calculations, f"BERT層出力キャッシュ処理中...{precision_info}"
        )

    # BERT層出力のキャッシュ処理を実行
    from utils.cache.bert_cache import cache_bert_layer_outputs

    try:
        # 混合精度を適用してキャッシュ処理
        if precision_context is not None:
            with precision_context:
                cache_bert_layer_outputs(
                    unified_model,
                    inputs,
                    tokenizer,
                    text,
                    max_layers=unified_model.config.num_hidden_layers,
                )
        else:
            cache_bert_layer_outputs(
                unified_model,
                inputs,
                tokenizer,
                text,
                max_layers=unified_model.config.num_hidden_layers,
            )
        if progress_callback:
            # キャッシュ完了を通知
            progress_callback(0, total_calculations, "キャッシュ完了、分析開始...")
        logger.debug("cache_bert_layer_outputs: done")
    except Exception as e:
        raise Exception(f"BERT層出力キャッシュ処理に失敗しました: {e}")

    # 分析結果を格納する辞書
    analysis_results = {
        "text": text,
        "tokens": tokens,
        "num_steps": num_steps,
        "created_at": datetime.now().isoformat(),
        "layers": {},
        "runtime": runtime_config,
        "diagnostics": {"runtime": runtime_config},
    }

    # 全体の計算回数
    current_calculation = 0

    # バッチ処理用のタスクリストを作成
    tasks = []

    # 各層の分析タスクを生成
    for layer_idx in range(total_layers):
        layer_key = f"layer_{layer_idx}"
        analysis_results["layers"][layer_key] = {
            "layer_idx": layer_idx,
            "is_final_layer": layer_idx == total_layers - 1,
            "attention": {},
            "mlp": {},
        }

        # Attention分析タスクを生成
        for head_idx in range(total_heads):
            head_key = f"head_{head_idx}"
            analysis_results["layers"][layer_key]["attention"][head_key] = {
                "head_idx": head_idx,
                "tokens": {},
            }

            for token_idx in range(total_tokens):
                token_key = f"token_{token_idx}"
                task = {
                    "type": "attention",
                    "layer_idx": layer_idx,
                    "head_idx": head_idx,
                    "token_idx": token_idx,
                    "layer_key": layer_key,
                    "head_key": head_key,
                    "token_key": token_key,
                    "token": tokens[token_idx],
                }
                tasks.append(task)

        # MLP分析タスクを生成
        is_final_layer = layer_idx == total_layers - 1
        if is_final_layer:
            # 最終層のMLP分析（全トークンで同じ結果）
            tasks.append(
                {
                    "type": "mlp_final",
                    "layer_idx": layer_idx,
                    "layer_key": layer_key,
                    "is_final_layer": True,
                }
            )
        else:
            # 中間層のMLP分析（出力は z_i^(l+1) としてヘッド非分割で評価）
            for token_idx in range(total_tokens):
                token_key = f"token_{token_idx}"
                analysis_results["layers"][layer_key]["mlp"].setdefault("tokens", {})
                analysis_results["layers"][layer_key]["mlp"]["tokens"].setdefault(
                    token_key, {}
                )

                task = {
                    "type": "mlp_intermediate",
                    "layer_idx": layer_idx,
                    "token_idx": token_idx,
                    "layer_key": layer_key,
                    "token_key": token_key,
                    "is_final_layer": False,
                }
                tasks.append(task)

    # 最適化されたバッチ処理でタスクを実行（重複排除・統一キャッシュ）
    from utils.calculations.ig.optimized_ig import IGTask, get_ig_calculator

    # タスクを最適化されたIGタスクに変換
    ig_tasks = []
    for task in tasks:
        ig_task = IGTask(
            task_id=f"{task['type']}_{task['layer_idx']}_{task.get('token_idx', 0)}_{task.get('head_idx', 0)}",
            task_type=task["type"]
            .replace("mlp_final", "mlp")
            .replace("mlp_intermediate", "mlp"),
            layer_idx=task["layer_idx"],
            token_idx=task.get("token_idx", 0),
            head_idx=task.get("head_idx"),
            num_steps=num_steps,
        )
        ig_tasks.append(ig_task)

    # 最適化IG計算器で実行（unified_model使用）
    ig_calculator = get_ig_calculator(unified_model, tokenizer)

    # 混合精度コンテキストをIG計算器に渡す
    if precision_context is not None:
        ig_calculator.set_precision_context(precision_context)

    def optimized_progress_callback(current, total, message):
        if progress_callback:
            # 進捗値を1.0以下に制限
            adjusted_current = min(current, total)
            progress_callback(
                current_calculation + adjusted_current, total_calculations, message
            )

    batch_results_dict = ig_calculator.compute_batch_ig(
        ig_tasks, text, optimized_progress_callback
    )

    # 結果を元の形式に変換
    from utils.calculations.ig.optimized_runtime.tasks import generate_task_key

    batch_results = []
    missing_keys = []
    found_keys = []

    for task, ig_task in zip(tasks, ig_tasks):
        # generate_task_keyを使って正しいキーを生成
        task_key = generate_task_key(ig_task, text)

        if task_key in batch_results_dict:
            result_data = batch_results_dict[task_key]
            found_keys.append(task_key)

            # 結果がNoneまたはsuccessがFalseの場合の処理
            if (
                result_data.get("success", False)
                and result_data.get("result") is not None
            ):
                batch_results.append(
                    {
                        **task,
                        "contributions": result_data["result"],
                        "success": True,
                    }
                )
            else:
                # 結果がNoneまたは失敗の場合、警告を記録してスキップ
                logging.warning(
                    f"IGタスクの結果がNoneまたは失敗: {task_key} "
                    f"(L{task['layer_idx']} T{task.get('token_idx', 0)} H{task.get('head_idx', '-')})"
                )
                # エラー情報があれば記録
                error_msg = result_data.get("error", "結果がNone")
                logging.warning(f"エラー詳細: {error_msg}")
                # このタスクはスキップ（後でエラーハンドリング）
                batch_results.append(
                    {
                        **task,
                        "contributions": None,
                        "success": False,
                        "error": error_msg,
                    }
                )
        else:
            # 結果が得られない場合は警告を記録してスキップ
            missing_keys.append(task_key)
            logging.warning(
                f"IG batch result missing for task {task['type']} "
                f"L{task['layer_idx']} T{task.get('token_idx', 0)} H{task.get('head_idx', '-')} "
                f"(task_key: {task_key})"
            )
            batch_results.append(
                {
                    **task,
                    "contributions": None,
                    "success": False,
                    "error": "結果が見つかりません",
                }
            )

    # デバッグ情報をログに記録
    if missing_keys:
        logging.warning(
            f"batch_results_dictに存在しないタスクキー数: {len(missing_keys)}/{len(tasks)}"
        )
        logging.warning(f"batch_results_dictのキー数: {len(batch_results_dict)}")
        logging.warning(
            f"batch_results_dictのキー例（最初の10個）: {list(batch_results_dict.keys())[:10]}"
        )
        logging.warning(f"見つからなかったキー例（最初の10個）: {missing_keys[:10]}")

    # 結果を分析結果辞書に統合
    failed_tasks = []
    for result in batch_results:
        if result["type"] == "attention":
            layer_key = result["layer_key"]
            head_key = result["head_key"]
            token_key = result["token_key"]

            contributions = result["contributions"]
            # Attention Relevanceを計算（理論準拠・均等分配を廃止）
            if contributions is not None and len(contributions) > 0:
                total_positive = sum(max(0, c) for c in contributions)
                if total_positive > 0:
                    relevance = [max(0, c) / total_positive for c in contributions]
                else:
                    relevance = [0.0] * len(contributions)
            else:
                # contributionsがNoneの場合はエラーを記録してスキップ
                error_msg = result.get("error", "contributionsがNone")
                logging.error(
                    f"Attention contributions is None for layer {result['layer_key']} "
                    f"head {result['head_key']} token {result['token_key']}: {error_msg}"
                )
                failed_tasks.append(result)
                continue

            analysis_results["layers"][layer_key]["attention"][head_key]["tokens"][
                token_key
            ] = {
                "token_idx": result["token_idx"],
                "token": result["token"],
                "contributions": contributions,
                "relevance": relevance,
            }

        elif result["type"] == "mlp_final":
            layer_key = result["layer_key"]
            mlp_ig = result["contributions"]

            if mlp_ig is not None:
                contributions = (
                    mlp_ig.tolist() if isinstance(mlp_ig, np.ndarray) else mlp_ig
                )
                # MLP Final Relevanceを計算
                if contributions is not None and len(contributions) > 0:
                    total_positive = sum(max(0, c) for c in contributions)
                    if total_positive > 0:
                        relevance = [max(0, c) / total_positive for c in contributions]
                    else:
                        relevance = [0.0] * len(contributions)
                else:
                    # contributionsがNoneの場合はエラーを記録してスキップ
                    error_msg = result.get("error", "contributionsがNoneまたは空")
                    logging.error(
                        f"MLP Final contributions is None for layer {result['layer_key']}: {error_msg}"
                    )
                    failed_tasks.append(result)
                    continue

                for token_idx in range(total_tokens):
                    token_key = f"token_{token_idx}"
                    analysis_results["layers"][layer_key]["mlp"]["tokens"] = (
                        analysis_results["layers"][layer_key]["mlp"].get("tokens", {})
                    )
                    analysis_results["layers"][layer_key]["mlp"]["tokens"][
                        token_key
                    ] = {"contributions": contributions, "relevance": relevance}
            else:
                # mlp_igがNoneの場合はエラーを記録してスキップ
                error_msg = result.get("error", "mlp_igがNone")
                logging.error(
                    f"MLP Final IG is None for layer {result['layer_key']}: {error_msg}"
                )
                failed_tasks.append(result)

        elif result["type"] == "mlp_intermediate":
            # 中間層は z_i^(l+1) としてヘッド非分割で保存
            layer_key = result["layer_key"]
            token_key = result["token_key"]

            contributions = result["contributions"]
            if contributions is not None and len(contributions) > 0:
                total_positive = sum(max(0, c) for c in contributions)
                if total_positive > 0:
                    relevance = [max(0, c) / total_positive for c in contributions]
                else:
                    relevance = [0.0] * len(contributions)
            else:
                # contributionsがNoneの場合はエラーを記録してスキップ
                error_msg = result.get("error", "contributionsがNoneまたは空")
                logging.error(
                    f"MLP Intermediate contributions is None for layer {result['layer_key']} "
                    f"token {result['token_key']}: {error_msg}"
                )
                failed_tasks.append(result)
                continue

            analysis_results["layers"][layer_key]["mlp"]["tokens"][token_key] = {
                "contributions": contributions,
                "relevance": relevance,
            }

    # 失敗したタスクがあれば即座にエラーを発生させて停止
    if failed_tasks:
        error_details = []
        for failed_task in failed_tasks[:10]:  # 最初の10個のみ詳細を記録
            error_details.append(
                f"{failed_task.get('type')} L{failed_task.get('layer_idx')} "
                f"T{failed_task.get('token_idx', '-')} H{failed_task.get('head_idx', '-')}: "
                f"{failed_task.get('error', '不明')}"
            )
        if len(failed_tasks) > 10:
            error_details.append(
                f"... 他{len(failed_tasks) - 10}個のタスクも失敗しています"
            )

        error_msg = (
            f"❌ {len(failed_tasks)}個のIGタスクが失敗しました。計算を停止します。\n"
            f"失敗したタスク:\n" + "\n".join(error_details)
        )
        logging.error(error_msg)
        raise RuntimeError(error_msg)

    # 付帯診断を添付
    if diagnostic:
        actual_total_seconds = max(time.time() - start_time, 0.0)
        diagnostics_block = analysis_results.setdefault("diagnostics", {})
        diagnostics_block.update(
            {
                "gpu_available": gpu_available,
                "device": device_name,
                "task_counts": task_counts,
                "predicted_total_seconds": predicted_total_seconds,
                "actual_total_seconds": actual_total_seconds,
                "attn_probe_sec": attn_probe_sec,
                "mlp_probe_sec": mlp_probe_sec,
                "probe_used": diagnostic_probe,
            }
        )
        diagnostics_block.setdefault("runtime", runtime_config)
        logger.debug(
            f"done: actual_total={actual_total_seconds:.1f}s, predicted_total={predicted_total_seconds}"
        )

    if progress_callback:
        progress_callback(total_calculations, total_calculations, "IG計算完了")

    if tqdm_bar:
        try:
            from tqdm import tqdm as _tqdm  # type: ignore

            _tqdm.write(f"✅ IG計算完了: {total_calculations}/{total_calculations}")
        except Exception:
            pass
        tqdm_bar.close()
    else:
        # 静かなモードではコンソール出力も抑制
        logger.debug("IG計算完了")

    return analysis_results


def _execute_tasks_in_batches(
    tasks: List[Dict],
    model_attn,
    model_mlp,
    inputs,
    tokens: List[str],
    num_steps: int,
    batch_size: int,
    max_workers: int,
    progress_callback,
    current_calculation: int,
    total_calculations: int,
    baseline_method: str = "zero",  # ベースライン方法
    input_type: str = "v",  # 入力タイプ
    use_direct_computation: bool = True,  # 直接計算を使用するか
) -> List[Dict]:
    """
    タスクをバッチ処理で実行（GPU最適化版）
    """
    results = []

    # 開始時間を記録
    start_time = time.time()
    last_update_time = start_time

    # タスクをバッチに分割
    batches = [tasks[i : i + batch_size] for i in range(0, len(tasks), batch_size)]

    # 並列処理でバッチを実行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 各バッチを並列実行
        future_to_batch = {
            executor.submit(
                _process_batch,
                batch,
                model_attn,
                model_mlp,
                inputs,
                tokens,
                num_steps,
                baseline_method,
                input_type,
                use_direct_computation,
            ): batch
            for batch in batches
        }

        # 結果を収集
        for future in as_completed(future_to_batch):
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as e:
                # バッチ処理でエラーが発生した場合は再発生させる
                raise RuntimeError(f"Batch processing failed: {e}")

            # 進捗更新
            current_calculation += len(batch_results)
            if progress_callback:
                # 現在時刻を取得
                current_time = time.time()

                # バッチごとの進捗を計算
                total_batches = len(batches)
                completed_batches = len(results) // batch_size
                batch_progress = (len(results) % batch_size) / batch_size * 100

                # 予測時間を計算
                elapsed_time = current_time - start_time
                if current_calculation > 0:
                    # 1計算あたりの平均時間
                    avg_time_per_calc = elapsed_time / current_calculation
                    # 残り計算数
                    remaining_calcs = total_calculations - current_calculation
                    # 予測残り時間
                    estimated_remaining = avg_time_per_calc * remaining_calcs

                    # 時間フォーマット
                    def format_time(seconds):
                        if seconds < 60:
                            return f"{seconds:.0f}秒"
                        elif seconds < 3600:
                            minutes = seconds / 60
                            return f"{minutes:.0f}分"
                        else:
                            hours = seconds / 3600
                            return f"{hours:.1f}時間"

                    # 全体進捗の予測時間
                    total_estimated = elapsed_time + estimated_remaining
                    progress_message = (
                        f"Batch {completed_batches}/{total_batches} "
                        f"({batch_progress:.1f}%) - "
                        f"全体進捗: {current_calculation}/{total_calculations} "
                        f"({current_calculation/total_calculations*100:.1f}%) - "
                        f"経過時間: {format_time(elapsed_time)} - "
                        f"予測残り時間: {format_time(estimated_remaining)} - "
                        f"予測完了時間: {format_time(total_estimated)}"
                    )
                else:
                    progress_message = (
                        f"Batch {completed_batches}/{total_batches} "
                        f"({batch_progress:.1f}%) - "
                        f"全体進捗: {current_calculation}/{total_calculations} "
                        f"({current_calculation/total_calculations*100:.1f}%)"
                    )

                progress_callback(
                    current_calculation, total_calculations, progress_message
                )

                # 更新頻度を制限（1秒に1回）
                if current_time - last_update_time >= 1.0:
                    last_update_time = current_time

    return results


def _process_batch(
    batch: List[Dict],
    model_attn,
    model_mlp,
    inputs,
    tokens: List[str],
    num_steps: int,
    baseline_method: str = "zero",  # ベースライン方法
    input_type: str = "v",  # 入力タイプ
    use_direct_computation: bool = True,  # 直接計算を使用するか
) -> List[Dict]:
    """
    バッチ内のタスクを処理
    """
    results = []

    for task in batch:
        try:
            if task["type"] == "attention":
                # Attention分析（全体分析用、verificationスキップ）
                # 理論文書5.3節に基づき、input_type="v"で直接計算を使用
                result_dict = compute_attention_ig_global_analysis_multi_layer(
                    bert_model=model_attn,
                    inputs=inputs,
                    layer_indices=[task["layer_idx"]],
                    target_token_idx=task["token_idx"],
                    target_head_idx=task["head_idx"],
                    num_steps=num_steps,
                    debug=False,
                    baseline_method=baseline_method,  # baseline_methodを渡す
                    input_type=input_type,  # 入力タイプを渡す
                    use_direct_computation=use_direct_computation,  # 直接計算フラグを渡す
                )
                # 結果は辞書形式で返される: {layer_idx: {"ig_values": List[float], ...}}
                result = result_dict.get(task["layer_idx"], {}) if result_dict else {}

                if result.get("ig_values") is not None:
                    attention_contributions = result["ig_values"]
                    results.append(
                        {
                            "type": "attention",
                            "layer_key": task["layer_key"],
                            "head_key": task["head_key"],
                            "token_key": task["token_key"],
                            "token_idx": task["token_idx"],
                            "token": task["token"],
                            "contributions": (
                                attention_contributions.tolist()
                                if isinstance(attention_contributions, np.ndarray)
                                else attention_contributions
                            ),
                        }
                    )
                else:
                    raise RuntimeError(
                        f"Attention IG computation failed for layer {task['layer_idx']} head {task['head_idx']} token {task['token_idx']}"
                    )

            elif task["type"] == "mlp_final":
                # 最終層MLP分析
                mlp_ig = compute_mlp_ig_theoretical_with_cache(
                    model_mlp,
                    task["layer_idx"],
                    0,  # 仮のトークン番号（最終層では全トークンで同じ結果）
                    None,  # 最終層の場合はtarget_head_idxは不要
                    num_steps,
                    is_global_analysis=True,
                    baseline_method=baseline_method,
                )

                results.append(
                    {
                        "type": "mlp_final",
                        "layer_key": task["layer_key"],
                        "contributions": mlp_ig,
                    }
                )

            elif task["type"] == "mlp_intermediate":
                # 中間層MLP分析（出力は z_i^(l+1) としてヘッド非分割で評価）
                mlp_ig = compute_mlp_ig_theoretical_with_cache(
                    model_mlp,
                    task["layer_idx"],
                    task["token_idx"],
                    None,  # ヘッド非分割
                    num_steps,
                    is_global_analysis=True,
                    baseline_method=baseline_method,
                )

                results.append(
                    {
                        "type": "mlp_intermediate",
                        "layer_key": task["layer_key"],
                        "token_key": task["token_key"],
                        "contributions": (
                            mlp_ig.tolist()
                            if isinstance(mlp_ig, np.ndarray)
                            else mlp_ig
                        ),
                    }
                )

        except Exception as e:
            # エラーを再発生させる（均等分配のフォールバックを廃止）
            raise RuntimeError(
                f"Task processing failed for {task['type']} layer {task['layer_idx']} "
                f"token {task.get('token_idx', 'N/A')} head {task.get('head_idx', 'N/A')}: {e}"
            )

    return results


def save_global_analysis_cache(
    analysis_results: Dict,
    text: str,
    num_steps: int,
    cache_dir: str = "cache/global_analysis",
) -> bool:
    """
    全体分析結果をキャッシュに保存

    Args:
        analysis_results: 分析結果
        text: 入力テキスト
        num_steps: 積分分割数
        cache_dir: キャッシュディレクトリ

    Returns:
        bool: 保存成功フラグ
    """
    logger = _get_ga_logger()
    try:
        logger.debug("キャッシュ保存開始: %s", cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        # キャッシュファイル名の生成
        import hashlib

        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_filename = f"global_analysis_{text_hash}_steps{num_steps}.json"
        cache_path = os.path.join(cache_dir, cache_filename)

        logger.debug("キャッシュファイルパス: %s", cache_path)

        # NumPy配列とスカラー値をPythonネイティブ型に変換する関数
        def convert_numpy_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
                return (
                    float(obj)
                    if isinstance(obj, (np.float32, np.float64))
                    else int(obj)
                )
            elif isinstance(obj, dict):
                return {key: convert_numpy_arrays(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_arrays(item) for item in obj]
            else:
                return obj

        # NumPy配列を変換してからJSON形式で保存
        serializable_results = convert_numpy_arrays(analysis_results)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)

        logger.debug("キャッシュ保存完了: %s", cache_path)
        return True
    except Exception as e:
        logger.warning("キャッシュ保存に失敗しました: %s", e)
        raise Exception(f"キャッシュ保存に失敗しました: {e}")


def load_global_analysis_cache(
    text: str, num_steps: int, cache_dir: str = "cache/global_analysis"
) -> Optional[Dict]:
    """
    全体分析結果をキャッシュから読み込み

    Args:
        text: 入力テキスト
        num_steps: 積分分割数
        cache_dir: キャッシュディレクトリ

    Returns:
        Optional[Dict]: 分析結果（キャッシュが存在しない場合はNone）
    """
    logger = _get_ga_logger()
    try:
        import hashlib

        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_filename = f"global_analysis_{text_hash}_steps{num_steps}.json"
        cache_path = os.path.join(cache_dir, cache_filename)

        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return None
    except Exception as e:
        logger.warning("キャッシュ読み込みエラー: %s", e)
        return None


def is_global_analysis_cached(
    text: str, num_steps: int, cache_dir: str = "cache/global_analysis"
) -> bool:
    """
    全体分析結果がキャッシュされているかチェック

    Args:
        text: 入力テキスト
        num_steps: 積分分割数
        cache_dir: キャッシュディレクトリ

    Returns:
        bool: キャッシュ存在フラグ
    """
    try:
        import hashlib

        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_filename = f"global_analysis_{text_hash}_steps{num_steps}.json"
        cache_path = os.path.join(cache_dir, cache_filename)

        return os.path.exists(cache_path)
    except Exception:
        return False


def get_global_analysis_summary(analysis_results: Dict) -> Dict:
    """
    全体分析結果のサマリーを取得

    Args:
        analysis_results: 分析結果

    Returns:
        Dict: サマリー情報
    """
    try:
        layers = analysis_results.get("layers", {})
        total_layers = len(layers)
        tokens = analysis_results.get("tokens", [])
        num_tokens = len(tokens)

        # 期待値を決定（最初のレイヤーから実際の値を取得）
        expected_heads = None
        if layers:
            first_layer = next(iter(layers.values()))
            first_attention = first_layer.get("attention", {})
            expected_heads = len(first_attention) if first_attention else None

        # 各層の情報を集計
        layer_summaries = {}
        validation_errors = []

        for layer_key, layer_data in layers.items():
            layer_idx = layer_data.get("layer_idx", 0)
            is_final_layer = layer_data.get("is_final_layer", False)
            layer_validation_errors = []

            # Attention分析の集計
            attention_data = layer_data.get("attention", {})
            attention_heads = len(attention_data)

            # Attention分析の検証（期待値が確定している場合のみ）
            if expected_heads is not None and attention_heads != expected_heads:
                error_msg = (
                    f"Layer {layer_idx}: Attentionヘッド数が期待値と異なります "
                    f"(期待: {expected_heads}, 実際: {attention_heads})"
                )
                validation_errors.append(error_msg)
                layer_validation_errors.append(error_msg)

            # 各ヘッドのトークン数を確認
            for head_key, head_data in attention_data.items():
                head_tokens = head_data.get("tokens", {})
                if len(head_tokens) != num_tokens:
                    error_msg = (
                        f"Layer {layer_idx} {head_key}: トークン数が期待値と異なります "
                        f"(期待: {num_tokens}, 実際: {len(head_tokens)})"
                    )
                    validation_errors.append(error_msg)
                    layer_validation_errors.append(error_msg)

            # MLP分析の集計
            mlp_data = layer_data.get("mlp", {})
            mlp_tokens = mlp_data.get("tokens", {})
            mlp_tokens_count = len(mlp_tokens)

            # MLP分析の検証
            if is_final_layer:
                # 最終層は全トークンで同じ結果（1つの結果を全トークンに適用）
                # 実際にはtokensキーにデータが格納されているか確認
                if mlp_tokens_count == 0:
                    error_msg = f"Layer {layer_idx} (最終層): MLPデータが存在しません"
                    validation_errors.append(error_msg)
                    layer_validation_errors.append(error_msg)
            else:
                # 中間層は各トークンごとに結果が必要
                if mlp_tokens_count != num_tokens:
                    error_msg = (
                        f"Layer {layer_idx} (中間層): MLPトークン数が期待値と異なります "
                        f"(期待: {num_tokens}, 実際: {mlp_tokens_count})"
                    )
                    validation_errors.append(error_msg)
                    layer_validation_errors.append(error_msg)

            layer_summaries[layer_key] = {
                "layer_idx": layer_idx,
                "is_final_layer": is_final_layer,
                "attention_heads": attention_heads,
                "mlp_tokens": mlp_tokens_count,
                "validation_errors": layer_validation_errors,
            }

        summary = {
            "text": analysis_results.get("text", ""),
            "tokens": tokens,
            "num_steps": analysis_results.get("num_steps", 0),
            "total_layers": total_layers,
            "created_at": analysis_results.get("created_at", ""),
            "layer_summaries": layer_summaries,
            "validation_passed": len(validation_errors) == 0,
            "validation_errors": validation_errors,
        }

        return summary
    except Exception as e:
        return {"error": str(e), "validation_passed": False}
