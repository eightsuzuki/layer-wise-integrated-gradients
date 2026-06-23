"""
IG計算のコア機能
"""

from .baseline_computation import compute_baseline_embeddings
from .embedding_extraction import extract_embeddings_fast
from .value_extraction import extract_value_vectors

__all__ = [
    "compute_baseline_embeddings",
    "extract_embeddings_fast",
    "extract_value_vectors",
]

# 後方互換性のため、古い関数名もエクスポート
_compute_baseline_embeddings = compute_baseline_embeddings
_extract_embeddings_fast = extract_embeddings_fast
_extract_value_vectors = extract_value_vectors

