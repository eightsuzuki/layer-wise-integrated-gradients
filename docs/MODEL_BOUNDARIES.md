# Model-agnostic block boundaries

LIG treats each Transformer block as a residual stream with ATT and MLP module boundaries:

- **z^(l)** — block input (residual stream before attention)
- **u^(l)** — ATT output / MLP input
- **z^(l+1)** — block output

## Detection

`lig.boundaries.detect_boundaries(model)` inspects the first block module:

| Layout | Typical models | `granularity` |
|--------|----------------|---------------|
| `post_ln_encoder` | BERT, RoBERTa, DeBERTa, ELECTRA | att, mlp, layer |
| `pre_ln_decoder` | GPT-2, GPT-NeoX (planned) | att, mlp, layer |
| `block_only` | MPNet, DistilBERT, ModernBERT, Switch, Mamba | layer only |

Known families can override introspection via `model_type` (e.g. MPNet has BERT-like submodules but uses layer-only IG today).

## API

```python
from lig import describe_boundaries, explain

# Config-only (no weights)
describe_boundaries("gpt2", load_weights=False)

# Full introspection
describe_boundaries("bert-base-uncased", load_weights=True, device="cpu")

# IG JSON includes boundary metadata
explain("Hello", model="gpt2", granularity="all", layers=[0])
```

CLI helper:

```bash
python examples/detect_model_boundaries.py bert-base-uncased gpt2
```

## IG hook points

`resolve_hook_modules(layer, boundaries)` returns modules at ATT/MLP boundaries for debugging or optional forward hooks. The public `explain()` path uses direct forwards (not PyTorch hooks).

## Remaining work

- Wire Llama/Mistral/Gemma via `pre_ln_decoder` introspection + causal IG paths
- Auto-enable ATT/MLP for architectures with non-BERT FFN layout when IG forward is implemented
- Unified decoder adapter using `boundaries.u_from_z` / `forward_block` (GPT-2 still uses dedicated paths in `lig/api.py`)
