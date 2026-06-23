"""モデル読み込みの共通ユーティリティ"""
import torch
from transformers import AutoTokenizer
from typing import Tuple, Optional, Any


def load_bert_model_and_tokenizer(
    model_name: str = "bert-base-uncased",
    device: Optional[str] = None
) -> Tuple[Any, AutoTokenizer]:
    """
    BERTモデルとトークナイザーを読み込み
    
    Args:
        model_name: モデル名
        device: デバイス（Noneの場合は自動選択）
    
    Returns:
        (model, tokenizer)
    """
    try:
        from utils.common.unified_bert_model import load_unified_model
    except Exception:  # pragma: no cover - fallback for moved module path
        from utils.unified_bert_model import load_unified_model
    
    # モデル読み込み
    model = load_unified_model(model_name)
    
    # トークナイザー読み込み
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # デバイス設定
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device != "cpu":
        model = model.to(device)
    
    return model, tokenizer


def get_device_info() -> dict:
    """
    デバイス情報を取得
    
    Returns:
        デバイス情報の辞書
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": []
    }
    
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        
        for i in range(info["device_count"]):
            device_info = {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": torch.cuda.get_device_properties(i).total_memory / 1024**3,
                "allocated_memory_gb": torch.cuda.memory_allocated(i) / 1024**3,
                "reserved_memory_gb": torch.cuda.memory_reserved(i) / 1024**3,
            }
            device_info["free_memory_gb"] = (
                device_info["total_memory_gb"] - device_info["reserved_memory_gb"]
            )
            info["devices"].append(device_info)
    
    return info

