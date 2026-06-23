# Paper reproduction (Experiment A)

Scripts for **flow-consistency** comparison (layer-direct z2z vs ATT+MLP composed z2z) on Penn Treebank **dev** sentences.

The [arXiv preprint](https://arxiv.org/abs/2606.21564) describes this as Experiment A: Stanford Dependencies format, sentence indices **0–1699** (1,700 sentences), within-layer \(L_2\) consistency. The paper text cites Treebank-3 and the bibliography lists:

- **Marcus et al. (1999), Treebank-3, LDC Catalog No. LDC99T42** (`marcus1999treebank`)

This repository does **not** include the corpus. The public demo ships only **two** excerpt sentences — see [examples/paper_demo/DATA_NOTICE.md](../examples/paper_demo/DATA_NOTICE.md).

## Obtain PTB

1. Register with the [Linguistic Data Consortium](https://www.ldc.upenn.edu/).
2. License [Treebank-3 (LDC99T42)](https://catalog.ldc.upenn.edu/LDC99T42).
3. Prepare a `dev.txt` in Stanford Dependencies layout (or equivalent used by your cache).

## Environment

```bash
export PTB_DEPPARSE_DIR=/path/to/depparse    # contains dev.txt
export PTB_CACHE_ROOT=/path/to/cache/ptb_ig_analysis   # optional, default: cache/ptb_ig_analysis
export MONOREPO_ROOT=/path/to/bert_token_embedding_visualization   # for prepare_att_mlp.sh only
```

## Quick path (existing cache on disk)

If you already have caches (e.g. from the development monorepo):

```bash
export PTB_CACHE_ROOT=/home/data/eight/bert_token_embedding_visualization/cache/ptb_ig_analysis

python scripts/reproduce/compare_layer_vs_composed.py --start 410 --end 410
```

## Full Experiment A (PTB dev, samples 0–1699)

```bash
# 1) ATT + MLP caches (requires monorepo batch runners + PTB)
bash scripts/reproduce/prepare_att_mlp.sh --split dev --end 1699

# 2) Layer-direct z2z (zero / ITB / ITB-zeroRatio baselines — run per baseline)
python scripts/reproduce/run_layer_direct_ig.py --split dev --start_sample 0 --end_sample 1699 --baseline_method zero

# 3) Composed z2z
python scripts/reproduce/compose_z2z.py --split dev --start 0 --end 1699

# 4) L2 comparison
python scripts/reproduce/compare_layer_vs_composed.py --split dev --start 0 --end 1699 \
  --csv-out results/summary_layer_vs_composed.csv

# 5) Top-3 per reference group (tab_decomp_top)
python scripts/reproduce/export_decomp_table.py results/summary_layer_vs_composed.csv \
  --out results/decomp_top3_by_group.csv
```

## Visualization (no LDC license required)

Precomputed demo for samples **16** and **410** only:

- [examples/paper_demo/README.md](../examples/paper_demo/README.md)
- `python scripts/build_demo_html.py`

## Citing the data

Use the bibtex in [DATA_NOTICE.md](../examples/paper_demo/DATA_NOTICE.md) (`marcus1999treebank`) — consistent with the paper references.
