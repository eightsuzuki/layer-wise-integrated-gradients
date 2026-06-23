# Public Release Manifest

`layer-wise-integrated-gradients/` is the **public staging directory**.  
Sync from the parent monorepo with `bash scripts/sync_from_monorepo.sh`, then push to [GitHub](https://github.com/eightsuzuki/layer-wise-integrated-gradients).

---

## What is published

| Category | Paths |
|----------|--------|
| **Public API** | `lig/` (`explain()`, CLI, encoder adapters) |
| **Visualization** | `lig/viz/`, `lig/demo/`, `examples/paper_demo/`, `docs/githubpage/` |
| **Reproduction (PTB-gated)** | `scripts/reproduce/`, `utils/reproduce/ptb_loader.py`, `docs/REPRODUCTION.md` |
| **Core IG** | `utils/calculations/ig/`, `utils/calculations/shared/` |
| **Infrastructure** | `utils/common/`, `utils/cache/` |
| **Docs** | `docs/DOCKER.md`, `docs/PYPI.md`, `docs/DECODER_DESIGN.md`, … |
| **Docker** | `Dockerfile*`, `docker-compose*.yml` |
| **Tests** | `test/test_lig_api.py`, `test/test_encoder_models.py`, `test/test_layer_*.py` |
| **GitHub Pages** | `.github/workflows/pages.yml`, `docs/index.html` |

---

## What is NOT published

| Excluded | Reason |
|----------|--------|
| **Penn Treebank corpus files** (`dev.txt`, …) | LDC license — users obtain LDC99T42 separately |
| **VAIG** (`**/vaig.py`) | Separate work; not in this public release |
| Full `utils/ptb_dependency/`, `utils/experiments/ptb_dependency/` | UAS / heavy PTB batch (monorepo) |
| `app.py`, `pages/` (monorepo Streamlit suite) | Full interactive UI |
| `utils/important_path/` | Greedy / MCF path search |
| `BERT_IG_baselin_paper/` | Private paper sources |

**Included as limited excerpts:** demo sentences + LIG JSON in `examples/paper_demo/json/` (see `DATA_NOTICE.md`).

`prepare_att_mlp.sh` delegates to the parent monorepo when present; otherwise point `PTB_CACHE_ROOT` at an existing cache.

---

## Sync workflow

```bash
bash layer-wise-integrated-gradients/scripts/sync_from_monorepo.sh
cd layer-wise-integrated-gradients
git add -A && git commit -m "Sync release from monorepo"
git push origin main
```

---

## Pre-release checklist

- [ ] `manifest.txt` has no PTB corpus paths
- [ ] `bash scripts/run_regression_tests.sh` passes
- [ ] `python scripts/check_no_otb_in_release.py` passes (release scope)
- [ ] `python scripts/build_paper_demos.py && python scripts/build_demo_html.py`
- [ ] No `data/depparse/`, `cache/`, or LDC corpus files staged
