"""シンプルなキャッシュ/ハッシュユーティリティ.

本プロジェクトで参照されている `hash_text`, `hash_inputs`, `cache_result`
の最小実装を提供します。外部依存を避けつつ、安定したハッシュ文字列を
返すために SHA-256 を使用します。
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache, wraps
from typing import Any, Callable, Iterable, Mapping, Tuple

__all__ = ["hash_text", "hash_inputs", "cache_result"]


def _stable_serialize(args: Iterable[Any], kwargs: Mapping[str, Any]) -> str:
    """引数をソート済みの JSON 文字列に直列化する（非直列化可能な型は repr）。"""

    def _default(o: Any) -> str:
        return repr(o)

    try:
        return json.dumps({"args": list(args), "kwargs": dict(sorted(kwargs.items()))}, sort_keys=True, default=_default)
    except Exception:
        # 万一 JSON 化できなくても落とさない
        return repr((list(args), dict(sorted(kwargs.items()))))


def hash_text(text: str) -> str:
    """テキストから SHA-256 ハッシュを返す。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_inputs(*args: Any, **kwargs: Any) -> str:
    """引数全体をまとめてハッシュ化する。"""
    payload = _stable_serialize(args, kwargs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_result(func: Callable) -> Callable:
    """簡易デコレータ: 関数結果をメモリ上でキャッシュする。

    引数は JSON 直列化＋ハッシュ化でキーを作る。特に設定不要の
    lru_cache を用いる軽量版。
    """

    cached = lru_cache(maxsize=None)(func)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return cached(*args, **kwargs)

    # lru_cache の管理メソッドを透過させる
    wrapper.cache_info = cached.cache_info  # type: ignore[attr-defined]
    wrapper.cache_clear = cached.cache_clear  # type: ignore[attr-defined]
    return wrapper















