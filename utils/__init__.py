"""
Utils モジュール - 後方互換性ラッパー統合

このモジュールは、整理されたコードベースへの後方互換性を提供します。
実際の実装は以下の場所にあります：
- bert_hooks: utils.common.bert_hooks
- cache_utils: utils.cache.cache_utils
- clark2019_*: utils.clark2019_attention_analysis.*
- ig_attention_adapter: utils.ptb_dependency.ig_attention.adapter
- unified_bert_model: utils.common.unified_bert_model
- visualization: utils.common.visualization.visualization
- wikitext_cache: utils.cache.wikitext_cache
"""

# 後方互換性ラッパー: bert_hooks
try:
    from .common.bert_hooks import (
        BertWithHooks,
        BertWithMLPHooks,
        load_attn_model,
        load_mlp_model,
        DEVICE,
    )
except Exception:
    # オプション依存。未存在でも致命的にしない
    pass

# 後方互換性ラッパー: cache_utils
try:
    from .cache.cache_utils import cache_result, hash_inputs, hash_text
except Exception:
    pass

# 後方互換性ラッパー: clark2019関連
try:
    from .clark2019_attention_analysis.clark2019_batch_analyzer import run_clark2019_analysis
    from .clark2019_attention_analysis.clark2019_analysis_core import *
    from .clark2019_attention_analysis.clark2019_metrics import *
    from .cache.clark2019_cache import *
except Exception:
    pass

# attention_reproductionは別途utils/attention_reproduction.pyでラッパー提供

# 後方互換性ラッパー: ig_attention_adapter
try:
    from .ptb_dependency.ig_attention.adapter import (
        extract_ig_attention_maps,
        is_cached_ig_attention,
        load_ig_attention_cache,
        save_ig_attention_cache,
        delete_ig_attention_cache,
        delete_all_ig_attention_cache,
    )
except Exception:
    pass

# 後方互換性ラッパー: unified_bert_model
try:
    from .common.unified_bert_model import load_unified_model, UnifiedBertModel
except Exception:
    pass

# 後方互換性ラッパー: visualization
try:
    from .common.visualization.visualization import (
        plot_head_graph,
        plot_important_path,
        visualize_graph,
    )
except Exception:
    pass

# 後方互換性ラッパー: wikitext_cache
try:
    from .cache.wikitext_cache import (
        load_wikitext2_cached,
        get_wikitext_cache_info,
        clear_wikitext_cache,
    )
except Exception:
    pass

# LRP関連
try:
    from .calculations.lrp.important_path_analysis import (
        compute_important_paths,
        format_path_summary,
        get_path_statistics,
        is_important_paths_cached,
        load_important_paths_cache,
        save_important_paths_cache,
    )
except Exception:
    pass

__all__ = [
    # bert_hooks
    "BertWithHooks",
    "BertWithMLPHooks",
    "load_attn_model",
    "load_mlp_model",
    "DEVICE",
    # cache_utils
    "cache_result",
    "hash_inputs",
    "hash_text",
    # clark2019
    "run_clark2019_analysis",
    # ig_attention_adapter
    "extract_ig_attention_maps",
    "is_cached_ig_attention",
    "load_ig_attention_cache",
    "save_ig_attention_cache",
    "delete_ig_attention_cache",
    "delete_all_ig_attention_cache",
    # unified_bert_model
    "load_unified_model",
    "UnifiedBertModel",
    # visualization
    "plot_head_graph",
    "plot_important_path",
    "visualize_graph",
    # wikitext_cache
    "load_wikitext2_cached",
    "get_wikitext_cache_info",
    "clear_wikitext_cache",
    # LRP
    "compute_important_paths",
    "format_path_summary",
    "get_path_statistics",
    "is_important_paths_cached",
    "load_important_paths_cache",
    "save_important_paths_cache",
]
