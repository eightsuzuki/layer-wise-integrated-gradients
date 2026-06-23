# device_utils.py
"""
デバイス管理ユーティリティ
PyTorch Lightningモデルとテンソルのデバイス統一を管理
"""

from typing import Dict

import lightning as L
import torch


def ensure_model_on_device(model: L.LightningModule) -> torch.device:
    """PyTorch Lightningモデルのデバイス管理"""
    # Lightningモデルのデバイスを取得
    if hasattr(model, "device"):
        return model.device
    else:
        # フォールバック: パラメータからデバイスを取得
        return next(model.parameters()).device


def ensure_tensors_on_device(
    inputs: Dict[str, torch.Tensor], model: L.LightningModule
) -> Dict[str, torch.Tensor]:
    """入力テンソルのデバイス管理（Lightningモデルに合わせる）"""
    device = ensure_model_on_device(model)
    # デバイス移動の重複を削減: 既にデバイス移動済みのテンソルは再度移動しない
    result = {}
    for k, v in inputs.items():
        if v.device != device:
            result[k] = v.to(device)
        else:
            result[k] = v  # 既に正しいデバイスにある場合はそのまま使用
    return result
