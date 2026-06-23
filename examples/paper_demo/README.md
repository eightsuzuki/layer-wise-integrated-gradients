# Paper demo — z2z visualization

Precomputed within-layer (z2z) maps for **two PTB dev sentences** used in the interactive demo.

> **Data licensing:** This demo includes *limited excerpts* from Penn Treebank Treebank-3 (LDC99T42).  
> Read **[DATA_NOTICE.md](DATA_NOTICE.md)** before redistributing or citing the sentence text.  
> The [arXiv preprint](https://arxiv.org/abs/2606.21564) cites Treebank-3 in its bibliography (`marcus1999treebank`); Experiment A uses PTB dev indices 0–1699 (not shipped here).

## GitHub Pages

- [Main demo](../../docs/githubpage/z2z_token_contribution.html) — sample dropdown (`00016` / `00410`)
- [Sample 00016](../../docs/githubpage/z2z_token_contribution_sample_00016.html)
- [Sample 00410](../../docs/githubpage/z2z_token_contribution_sample_00410.html) (default paper example)

Default attribution source on Pages: `layer_ig_zero` (layer-direct IG, zero baseline).

## Regenerate (requires private PTB cache)

```bash
export PTB_CACHE_ROOT=/path/to/cache/ptb_ig_analysis
python scripts/build_paper_demos.py
python scripts/build_demo_html.py
```

Local comparison across all baselines (not published to Pages by default):

```bash
python scripts/build_demo_html.py --all-sources
```

## Files

| Path | Description |
|------|-------------|
| [`DATA_NOTICE.md`](DATA_NOTICE.md) | PTB copyright, LDC license, citations |
| [`manifest.json`](manifest.json) | Sample sentences and attribution source IDs |
| `json/sample_*__*.json` | One sentence + precomputed matrices per file |
| `lig_z2z_zero.json` | Legacy alias (`sample_00410`, zero baseline) |

## Full paper reproduction (1,700 sentences)

Not included in this folder. See [docs/REPRODUCTION.md](../../docs/REPRODUCTION.md) and obtain [LDC99T42](https://catalog.ldc.upenn.edu/LDC99T42).
