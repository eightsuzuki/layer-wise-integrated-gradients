import json
import os
from typing import Any, Dict, Optional, Callable


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _hash_text(text: str) -> str:
    # 既存のハッシュ関数と整合を取るためにutils.cache_utilsがあれば利用
    try:
        from utils.cache_utils import hash_text  # type: ignore
        return hash_text(text)
    except Exception:
        import hashlib
        return hashlib.md5(text.encode("utf-8")).hexdigest()


def z2z_cache_path(text: str, num_steps: int) -> str:
    text_hash = _hash_text(text)
    cache_dir = "cache/z2z_global_analysis"
    _ensure_dir(cache_dir)
    return os.path.join(
        cache_dir, f"z2z_global_analysis_{text_hash}_steps{num_steps}.json"
    )


def load_z2z_global_analysis_cache(text: str, num_steps: int) -> Optional[Dict[str, Any]]:
    path = z2z_cache_path(text, num_steps)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def save_z2z_global_analysis_cache(data: Dict[str, Any], text: str, num_steps: int) -> bool:
    try:
        path = z2z_cache_path(text, num_steps)
        with open(path, "w") as f:
            json.dump(data, f)
        return True
    except Exception:
        return False


def compute_z2z_from_global_cache(
    global_cache: Dict[str, Any],
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    既存のglobal_analysis（attention/mlp）キャッシュから、
    各層のz^(l)→z^(l+1)の統合Relevance（z2z）を構築する。

    合成式（単純合成）:
      Z2Z[l][src_token→dst_token] = sum_h R_Attn[l][h][src_token][dst_token] * R_MLP[l][dst_token][h]
    """
    layers = global_cache.get("layers", {})
    tokens = global_cache.get("tokens", [])
    num_layers = len([k for k in layers.keys() if k.startswith("layer_")])

    z2z_layers: Dict[str, Any] = {}

    for l in range(num_layers):
        layer_key = f"layer_{l}"
        layer_data = layers.get(layer_key, {})
        attention = layer_data.get("attention", {}) or {}
        mlp = layer_data.get("mlp", {}) or {}

        # R_MLP[l][dst_token][h] 形式へ正規化
        mlp_tokens = mlp.get("tokens", mlp)
        R_MLP: Dict[int, Dict[int, float]] = {}
        for dst_token in range(len(tokens)):
            tok_key = f"token_{dst_token}"
            if tok_key in mlp_tokens:
                tok_data = mlp_tokens[tok_key]
                if "relevance" in tok_data and isinstance(tok_data["relevance"], list):
                    # ヘッド配列
                    R_MLP[dst_token] = {
                        h: float(val)
                        for h, val in enumerate(tok_data["relevance"]) if val is not None
                    }
                else:
                    # 旧スキーマ head_x.relevance[dst_token]
                    tmp: Dict[int, float] = {}
                    for h in range(12):
                        head_key = f"head_{h}"
                        if head_key in tok_data:
                            rel_list = tok_data[head_key].get("relevance", [])
                            if len(rel_list) > 0:
                                tmp[h] = float(rel_list[0])
                    if tmp:
                        R_MLP[dst_token] = tmp

        # R_Attn[l][h][src_token][dst_token]
        R_Attn: Dict[int, Dict[int, Dict[int, float]]] = {}
        for h in range(12):
            head_key = f"head_{h}"
            if head_key in attention:
                head_data = attention[head_key]
                tok_map = head_data.get("tokens", {}) or {}
                R_Attn[h] = {}
                for src_token in range(len(tokens)):
                    src_key = f"token_{src_token}"
                    if src_key in tok_map:
                        rel_list = tok_map[src_key].get("relevance", [])
                        # rel_list[dst_token] が寄与度
                        R_Attn[h][src_token] = {
                            dst_t: float(val) for dst_t, val in enumerate(rel_list)
                            if isinstance(val, (int, float)) and val > 0
                        }

        # 合成
        z2z_tokens: Dict[str, Any] = {}
        for src_token in range(len(tokens)):
            combined: Dict[int, float] = {}
            # 各dst_tokenについて合成
            for dst_token in range(len(tokens)):
                total = 0.0
                # hで合成
                for h in range(12):
                    attn_val = R_Attn.get(h, {}).get(src_token, {}).get(dst_token, 0.0)
                    mlp_val = R_MLP.get(dst_token, {}).get(h, 0.0)
                    if attn_val > 0 and mlp_val > 0:
                        total += attn_val * mlp_val
                if total > 0:
                    combined[dst_token] = total
            z2z_tokens[f"token_{src_token}"] = {"relevance": [combined.get(i, 0.0) for i in range(len(tokens))]}

        z2z_layers[layer_key] = {
            "z2z": {
                "tokens": z2z_tokens
            }
        }

        if progress_callback:
            progress_callback(l + 1, num_layers, f"z2z集計: Layer {l} 完了")

    return {
        "tokens": tokens,
        "layers": z2z_layers,
        "schema": "z2z_global_analysis_v1"
    }


def compute_and_cache_z2z_analysis(
    model_lightning,
    model_attn,
    model_mlp,
    tokenizer,
    text: str,
    num_steps: int = 32,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    既存のglobal_analysisが無ければ作成し、それを基にz2z集計を行い、
    cache/z2z_global_analysis に保存して返す。
    """
    from utils.calculations.ig.global_analysis import (
        is_global_analysis_cached,
        load_global_analysis_cache,
        compute_global_ig_analysis,
        save_global_analysis_cache,
    )

    # まず既存global_analysisを用意
    if not is_global_analysis_cached(text, num_steps):
        ga = compute_global_ig_analysis(
            model_lightning=model_lightning,
            model_attn=model_attn,
            model_mlp=model_mlp,
            tokenizer=tokenizer,
            text=text,
            num_steps=num_steps,
            progress_callback=progress_callback,
        )
        save_global_analysis_cache(ga, text, num_steps)

    base_cache = load_global_analysis_cache(text, num_steps)
    if base_cache is None:
        raise RuntimeError("global_analysisの読み込みに失敗しました")

    # z2z集計
    z2z = compute_z2z_from_global_cache(base_cache, progress_callback)
    if not save_z2z_global_analysis_cache(z2z, text, num_steps):
        raise RuntimeError("z2z_global_analysisの保存に失敗しました")

    return z2z


