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
export PTB_CACHE_ROOT=/path/to/your/cache/ptb_ig_analysis

python scripts/reproduce/compare_layer_vs_composed.py --start 410 --end 410
```

## Full Experiment A (PTB dev, samples 0-1699)

```bash
# 1) ATT + MLP caches.
#    The MLP side runs from this repo; the ATT side still delegates to the
#    monorepo runner (--skip-att if you already have those caches).
#    --device auto picks the CUDA card with the most free memory, which matters
#    on a shared machine; --device cuda:N pins one.
bash scripts/reproduce/prepare_att_mlp.sh --split dev --end 1699 --device auto

# 2) Layer-direct z2z (the reference side; run once per baseline)
python scripts/reproduce/run_layer_direct_ig.py --split dev \
    --start-sample 0 --end-sample 1699 --baseline-method zero --device auto
python scripts/reproduce/run_layer_direct_ig.py --split dev \
    --start-sample 0 --end-sample 1699 --baseline-method self_input_token --device auto

# 3) Composed z2z (affine composition, Eq. layer-decomp-hat)
python scripts/reproduce/compose_z2z.py --split dev --start 0 --end 1699

# 4) L2 comparison
python scripts/reproduce/compare_layer_vs_composed.py --split dev --start 0 --end 1699 \
    --csv-out results/summary_layer_vs_composed.csv

# 5) Top-3 per reference group (tab_decomp_top)
python scripts/reproduce/export_decomp_table.py results/summary_layer_vs_composed.csv \
    --out results/decomp_top3_by_group.csv
```

### What has to match for the numbers to come out right

Three settings decide whether step 4 reproduces the published table. All three
are the defaults, but they are easy to change by accident:

| Setting | Paper | Why |
|---|---|---|
| Composition | `--mode affine` | Eq. layer-decomp-hat chains **unit-sum allocations**, normalizing each boundary column by its own IG total. `--mode prod` chains raw scores and is dominated by columns with a large total. |
| MLP cache | `--mlp-suffix __headspacefix` | Attribution must be taken with respect to `u^{(l,h)}`, before `W_o`. Slicing the post-projection residual into `head_dim` blocks does not give heads. |
| ATT steps | 256 | The z->u boundary does not satisfy completeness at 32 steps. The MLP boundary does, so it stays at 32. |

The comparison in step 4 also aligns the two sides before measuring: the
layer-direct cache is in subword space and the composed cache is in word space,
so subword rows and columns are summed into words first. Columns that are
identically zero on either side are skipped rather than scored against the zero
vector -- such a column means the attribution for that output token was never
produced, and averaging it in would mix a missing measurement into the result.

### Sharing a GPU

Every step takes `--device`. `auto` (the default) picks the CUDA card with the
most free memory and drops to CPU rather than squeezing onto a card that is
nearly full; `cuda:N` pins one. To use several cards, run disjoint shards --
each skips sentences already written, so runs resume after an interrupt:

```bash
python scripts/reproduce/run_mlp_head_space_ig.py --start 0   --end 849  --device cuda:0 &
python scripts/reproduce/run_mlp_head_space_ig.py --start 850 --end 1699 --device cuda:1 &
wait
```

## Visualization (no LDC license required)

Precomputed demo for samples **16** and **410** only:

- [examples/paper_demo/README.md](../examples/paper_demo/README.md)
- `python scripts/build_demo_html.py`

## Citing the data

Use the bibtex in [DATA_NOTICE.md](../examples/paper_demo/DATA_NOTICE.md) (`marcus1999treebank`) — consistent with the paper references.
