# Data notice — Penn Treebank (PTB) excerpts in this repository

This document describes **what Treebank-3 text appears in the public LIG release**, how it may be used, and how it relates to the paper experiments and bibliography.

## Summary

| Item | In this repository? |
|------|---------------------|
| Full PTB corpus (`dev.txt`, MRG, etc.) | **No** — obtain [LDC99T42](https://catalog.ldc.upenn.edu/LDC99T42) from LDC |
| Interactive demo sentences (2 excerpts) | **Yes** — under `examples/paper_demo/json/` |
| Precomputed LIG matrices for those sentences | **Yes** — same directory |
| Experiment A reproduction (PTB dev 0–1699) | **Scripts only** — requires your own LDC-licensed copy |

The **arXiv preprint** evaluates on PTB dev (Stanford Dependencies; sentence indices 0–1699) and cites Treebank-3 in the references as Marcus et al. (1999), LDC Catalog No. **LDC99T42** (`marcus1999treebank` in `en/refs.bib`).

---

## What is Treebank-3?

**Treebank-3** (LDC Catalog No. [LDC99T42](https://catalog.ldc.upenn.edu/LDC99T42)) is distributed by the [Linguistic Data Consortium (LDC)](https://www.ldc.upenn.edu/). It contains annotated English newswire (Wall Street Journal and other sources). This project uses the **Stanford Dependencies** conversion of PTB for dependency parsing experiments (see paper §Experiments).

You must have a valid LDC license to download or use the full corpus. This GitHub repository does **not** substitute for that license.

---

## Copyright and LDC license

- Treebank-3 is subject to the [LDC User Agreement](https://catalog.ldc.upenn.edu/license/ldc-non-members-agreement.pdf).
- LDC members may include **limited excerpts** in articles, reports, and other documents that describe **non-commercial research results** (see LDC guidelines for your membership terms).
- Copyright notices (from LDC documentation):
  - Portions © 1987–1989 **Dow Jones & Company, Inc.**
  - Portions © 1993–1995, 1999 **Trustees of the University of Pennsylvania**

**If you reuse demo sentence text**, retain appropriate attribution and comply with LDC terms. Do not redistribute additional PTB material through this repository.

---

## What this repository publishes

### Demo sentences (limited excerpts)

Only **two** PTB dev sentences are embedded in public JSON for the GitHub Pages visualization. They are identified in [`manifest.json`](manifest.json):

| Sample ID | PTB dev index | Role |
|-----------|---------------|------|
| `sample_00016` | 16 | Additional demo sentence (Congress / RTC) |
| `sample_00410` | 410 | Default paper / figure example |

Each published JSON file stores:

- The **sentence text** (excerpt only)
- **Precomputed LIG** z→z (and related) attribution tensors for `bert-base-uncased`
- Metadata (baseline, route, tokenization) — not the full PTB annotation file

No `dev.txt`, no WSJ MRG files, and no 1,700-sentence dump are included.

### Attribution variants

See [`manifest.json`](manifest.json) for `layer_ig_*` and `composed_*` source IDs. Published GitHub Pages demos use a subset; the full matrix of variants may exist locally after running `scripts/build_paper_demos.py` with a private PTB cache.

---

## Paper experiments vs. public demo

| | Paper (arXiv preprint) | This repository (public) |
|--|------------------------|---------------------------|
| **Data** | PTB dev, indices **0–1699** (1,700 sentences) | **2 sentences** only in `json/` |
| **Task** | Within-layer \(L_2\) consistency (Experiment A) | Interactive z2z visualization |
| **Model** | BERT-base-uncased | Same (for published JSON) |
| **Citation** | `\cite{marcus1999treebank}` in text + bibliography | This file + bibtex below |

Full Experiment A reproduction: [docs/REPRODUCTION.md](../../docs/REPRODUCTION.md) (requires LDC-licensed PTB on your machine).

---

## How to obtain PTB for reproduction

1. Register with LDC and purchase / access [LDC99T42](https://catalog.ldc.upenn.edu/LDC99T42).
2. Prepare Stanford Dependencies `dev.txt` (or equivalent) and set:
   ```bash
   export PTB_DEPPARSE_DIR=/path/to/depparse   # contains dev.txt
   export PTB_CACHE_ROOT=/path/to/cache/ptb_ig_analysis
   ```
3. Follow [docs/REPRODUCTION.md](../../docs/REPRODUCTION.md).

The development monorepo (private) may already contain batch runners; the public release ships **reproduction scripts** and `utils/reproduce/ptb_loader.py` only.

---

## Citation — Treebank-3

Use the **same entry as the paper** (recommended):

```bibtex
@misc{marcus1999treebank,
  author       = {Marcus, Mitchell P. and Santorini, Beatrice and Marcinkiewicz, Mary Ann and Taylor, Ann},
  title        = {Treebank-3},
  howpublished = {Web Download},
  publisher    = {Linguistic Data Consortium},
  address      = {Philadelphia},
  year         = {1999},
  note         = {LDC Catalog No. LDC99T42. https://catalog.ldc.upenn.edu/LDC99T42}
}
```

## Citation — LIG software

```bibtex
@article{suzuki2026lig,
  title         = {LIG: Layer-wise Integrated Gradients for Within-Layer Flow Analysis in Transformers},
  author        = {Suzuki, Eight and Hino, Hideitsu and Murata, Noboru},
  year          = {2026},
  eprint        = {2606.21564},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

---

## Contact

Questions about **LIG code or demo JSON**: open an issue on [GitHub](https://github.com/eightsuzuki/layer-wise-integrated-gradients/issues).

Questions about **LDC licensing or corpus access**: contact the [Linguistic Data Consortium](https://www.ldc.upenn.edu/).
