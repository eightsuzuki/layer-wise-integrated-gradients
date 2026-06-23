# Decoder-only models (GPT, Llama, …) — design & roadmap

LIG today ships **encoder** support (`lig.explain` on BERT, RoBERTa, …).  
Decoder-only causal LMs are **designed but not implemented** in v0.1.

---

## Why decoders differ

| | Encoder (BERT-style) | Decoder (GPT / Llama-style) |
|---|---------------------|----------------------------|
| Attention | Bidirectional + padding mask | **Causal** mask (token $i$ sees $\le i$) |
| Block layout | `encoder.layer[i].attention` + `intermediate` | `transformer.h[i].attn` + `mlp` (GPT-2) or `model.layers[i]` (Llama) |
| z / u / z′ | Same as paper | Same **concept**, different module paths |
| IG baseline | ITB on token $j$ | ITB still valid; causal mask in forward |

The **set-to-set IG at module boundaries** (z→u, u→z, z→z) applies to both; only the forward hooks and masks change.

---

## Module mapping (GPT-2)

```
hidden_states[l]  ──►  z^(l)  ──►  Block.attn  ──►  u^(l)  ──►  Block.mlp  ──►  z^(l+1)
```

| LIG node | GPT-2 module |
|----------|----------------|
| z | input to `transformer.h[l]` (residual stream) |
| u | after `attn` + residual |
| z′ | after `mlp` + residual |

Llama / Mistral / Qwen2: `model.layers[l].self_attn` + `model.layers[l].mlp`.

---

## Planned API (unchanged surface)

```python
from lig import explain

# Future — same call signature
result = explain(
    "The cat sat on the mat.",
    model="gpt2",  # or meta-llama/Llama-3.2-1B
    granularity="all",
    ...
)
```

`lig.adapters.load_adapter()` already detects decoder `model_type` and returns `DecoderAdapter`;  
`explain()` will call decoder-specific IG once `utils/calculations/ig` paths are generalized.

---

## Implementation phases

### Phase 1 — GPT-2 (smallest decoder) ✅

- [x] Causal attention mask + block forward (`utils/calculations/ig/gpt2/block_forward.py`)
- [x] z→u (ATT): `gpt2_attention_models.py` + embedding baseline interpolation
- [x] u→z (MLP): `mlp/mlp_lig_ig.py`, `mlp/gpt2_mlp_lig_ig.py` (`baseline_mlp='zero' | 'att_itb_a0'` for GPT-2)
- [x] z→z (layer): `z2z/gpt2_layer_direct_ig.py`
- [x] Test: `explain(..., model="gpt2", granularity="all", layers=[0])`

### Phase 2 — Llama family

- [ ] RMSNorm + SwiGLU MLP path in MLP wrapper
- [ ] GQA / MQA head layout in ATT wrapper

### Phase 3 — Composition & PTB-optional demos

- [ ] z2z compose (ATT × MLP) for decoder
- [ ] Optional: single-sentence demos only (no PTB required)

---

## Code layout (current)

```
lig/adapters/
  encoder.py    # BERT-family — production
  decoder.py    # load GPT-2 / stub for Llama family
  factory.py    # load_adapter()
lig/api.py      # _run_explain_gpt2() for z→u, u→z, z→z
```

`test/test_lig_api.py` — RoBERTa regression; GPT-2 `granularity="all"` smoke test.

---

## References

- Paper: module boundaries at ATT and MLP (encoder experiments on BERT-base + PTB)
- Existing GPT-2 hooks (parent monorepo): `utils/ptb_dependency/ig_attention/gpt2_*.py` (PTB UAS only; not yet unified into `lig`)
