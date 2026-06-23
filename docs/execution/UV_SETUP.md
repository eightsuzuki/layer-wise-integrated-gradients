# UV environment setup (public LIG release)

Docker-free setup for running `lig explain` and regression tests.

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- (Optional GPU) CUDA-capable PyTorch wheels

## Setup

From the repository root:

```bash
bash scripts/ops/setup_uv_env.sh
source .venv/bin/activate
pip install -e .
```

## Regression tests

```bash
bash scripts/run_regression_tests.sh
```

## Scope

This public release covers **LIG** only (L2-scalarized IG, ITB / Zero / ATTITBa=0 baselines).

PTB batch reproduction and VAIG live in the private development monorepo — not in this repository.
