# Layer-wise Integrated Gradients (LIG)

<p align="center">
  <img src="logo/LIG-LOGO.png" alt="LIG logo" width="660">
</p>

**Set-to-set Integrated Gradients for within-layer flow analysis in Transformers.**

<p align="center">
  <a href="https://eightsuzuki.github.io/layer-wise-integrated-gradients/index.html">
    <strong>Project website &amp; interactive demo</strong>
  </a>
  <br>
  <sub>Notation · figures · install · within-layer z2z visualization</sub>
</p>

[![Website](https://img.shields.io/badge/demo-interactive%20z2z%20map-0d9488)](https://eightsuzuki.github.io/layer-wise-integrated-gradients/index.html)
[![PyPI version](https://img.shields.io/pypi/v/layer-wise-integrated-gradients)](https://pypi.org/project/layer-wise-integrated-gradients/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/code-GitHub-181717)](https://github.com/eightsuzuki/layer-wise-integrated-gradients)

LIG applies Integrated Gradients (IG) at ATT and MLP module boundaries inside each Transformer layer, composes within-layer token-to-token contributions, and compares them with layer-whole attribution under an $L_2$ diagnostic — without model-specific retraining or per-operation interpreter design.

- **Paper:** Coming soon · **arXiv:** [2606.21564](https://arxiv.org/abs/2606.21564)

This release implements **L2-scalarized set-to-set IG** at module boundaries (z→u, u→z, z→z), with baselines **ITB** (`self_input_token`), **ITB-zeroRatio** (`itb_zero_ratio`), **ITB-mapRatio** (`itb_map_ratio`), **Zero**, and **ATTITBa=0** (`att_itb_a0`) for MLP.

---

## Quick start

Install **PyTorch first** (CUDA/CPU wheel index differs), then LIG:

```bash
# GPU (CUDA 12.1 example)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# From PyPI
pip install layer-wise-integrated-gradients

# Or from GitHub
pip install "layer-wise-integrated-gradients @ git+https://github.com/eightsuzuki/layer-wise-integrated-gradients.git"

# Development clone
git clone https://github.com/eightsuzuki/layer-wise-integrated-gradients.git
cd layer-wise-integrated-gradients
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
```

### Explain one sentence → JSON

```python
from lig import explain

result = explain(
    "The cat sat on the mat.",
    model="bert-base-uncased",
    num_steps=32,
    granularity="all",           # att (z→u), mlp (u→z), layer (z→z)
    layers=[0, 11],
)
```

```bash
lig explain "The cat sat on the mat." --steps 32 --granularity all -o attributions.json
```

### Supported models

| Family | Models | `granularity` |
|--------|--------|---------------|
| BERT-style | BERT, RoBERTa, DeBERTa, ELECTRA, XLM-RoBERTa, … | z→u, u→z, z→z |
| Block-only | MPNet, DistilBERT, ModernBERT, Switch MoE encoder, Mamba | **z→z (layer) only** |
| GPT-2 (decoder) | z→u, u→z, z→z (`granularity="all"`) | — |

GPT-2 example:

```python
explain(
    "Hello world",
    model="gpt2",
    granularity="all",
    baseline_att="self_input_token",   # or itb_zero_ratio / itb_map_ratio
    baseline_mlp="att_itb_a0",         # or zero
    layers=[0],
)
```

ATT ITB self-contribution variants (see method section in the upcoming paper):

```python
explain("Hello world", baseline_att="itb_zero_ratio", granularity="att", layers=[0])
explain("Hello world", baseline_att="itb_map_ratio", granularity="att", layers=[0])
explain("Hello world", baseline_layer="itb_zero_ratio", granularity="layer", layers=[0])
```

Other decoders (Llama, …): [docs/DECODER_DESIGN.md](docs/DECODER_DESIGN.md).

### Inspect boundaries (no IG run)

```python
from lig import describe_boundaries

# Config-only (no weight download)
describe_boundaries("gpt2", load_weights=False)

# Full module introspection
describe_boundaries("bert-base-uncased", load_weights=True, device="cpu")
```

```bash
python examples/detect_model_boundaries.py bert-base-uncased gpt2 distilbert-base-uncased
```

### Docker

```bash
docker compose -f docker-compose.cpu.yml build
docker compose -f docker-compose.cpu.yml run --rm lig explain "Hello world" \
  --steps 4 --granularity layer --layers 0 --target-tokens 1 -o /output/out.json
```

See [docs/DOCKER.md](docs/DOCKER.md).

### Dev setup (uv)

```bash
bash scripts/ops/setup_uv_env.sh          # GPU torch + pip install -e ".[dev]"
bash scripts/ops/setup_uv_env.sh --cpu    # CPU torch
source .venv/bin/activate
```

### Regression tests

```bash
bash scripts/run_regression_tests.sh
```

### Paper demo (no full PTB download)

The public repo ships **two PTB dev excerpts** (indices 16 and 410) with precomputed LIG JSON — not the 1,700-sentence Experiment A set.

- **GitHub Pages**: [project site](https://eightsuzuki.github.io/layer-wise-integrated-gradients/index.html) · [standalone demo](https://eightsuzuki.github.io/layer-wise-integrated-gradients/githubpage/z2z_token_contribution.html) — samples `00016` / `00410`; routes (direct z→z / composed ATT×MLP; Zero / ITB / ITB-zeroRatio)
- **Data & licensing**: [examples/paper_demo/DATA_NOTICE.md](examples/paper_demo/DATA_NOTICE.md) (LDC terms, citations aligned with the paper bibliography)
- **Regenerate** (private PTB cache required): `python scripts/build_paper_demos.py && python scripts/build_demo_html.py`
- **Streamlit**: `pip install -e ".[demo]"` then `streamlit run lig/demo/streamlit_app.py`
- JSON: `examples/paper_demo/json/` · manifest: [manifest.json](examples/paper_demo/manifest.json)

### Paper reproduction (Experiment A)

Experiment A in the [arXiv preprint](https://arxiv.org/abs/2606.21564) uses **PTB dev sentences 0–1699** (Stanford Dependencies) and cites Treebank-3 as Marcus et al. (1999), LDC99T42 (`marcus1999treebank` in the paper references).

You must obtain the corpus from LDC yourself. See [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

---

## Features

- **One-call API** — `explain(text)` → JSON with z→u, u→z, and z→z attributions  
- **CLI** — `lig explain "..." -o out.json`  
- **Model-agnostic block boundaries** — auto-detect z (residual stream) and u (ATT output / MLP input) from module layout; BERT-family + GPT-2 full granularity; block-level z→z for MPNet, DistilBERT, Switch, Mamba  
- **Boundary introspection** — `describe_boundaries("bert-base-uncased")` or `python examples/detect_model_boundaries.py`
- **Configurable** — integration steps, baselines, granularity, layers, target tokens

---

## Project layout

```
.
├── lig/                  # Public API + encoder adapters + viz demo
├── examples/paper_demo/  # One-sentence PTB excerpt + LIG JSON
├── docs/                 # DOCKER, PyPI, REPRODUCTION, Pages demo
├── utils/calculations/ig/  # Core LIG implementations
├── scripts/
│   ├── reproduce/        # Experiment A (PTB-gated)
│   ├── build_demo_html.py
│   └── run_regression_tests.sh
└── test/
```

---

## Citation

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

## Data and licensing

| Component | License / terms |
|-----------|-----------------|
| **LIG source code** | [MIT](LICENSE) |
| **Demo PTB excerpts** (2 sentences in `examples/paper_demo/json/`) | [LDC User Agreement](https://catalog.ldc.upenn.edu/license/ldc-non-members-agreement.pdf); see [DATA_NOTICE.md](examples/paper_demo/DATA_NOTICE.md) |
| **Full PTB for Experiment A** | Not distributed here — [LDC99T42](https://catalog.ldc.upenn.edu/LDC99T42) |

When citing the evaluation data, use the Treebank-3 entry from [DATA_NOTICE.md](examples/paper_demo/DATA_NOTICE.md) (same as the paper bibliography).

## License

[MIT License](LICENSE) for this software.

## PyPI

Package metadata and publish steps: [docs/PYPI.md](docs/PYPI.md).

Maintainers: sync core IG code from the parent monorepo with `bash scripts/sync_from_monorepo.sh` when needed. See [MANIFEST.md](MANIFEST.md).
