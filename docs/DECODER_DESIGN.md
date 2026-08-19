# Decoder-only models (GPT, Llama, Gemma3, …) — design & roadmap

LIG ships **encoder** support and selected **decoder** paths (`lig.explain`).

---

## Why decoders differ

| | Encoder (BERT-style) | Decoder (GPT / Llama / Gemma3) |
|---|---------------------|----------------------------|
| Attention | Bidirectional + padding mask | **Causal** (and sliding-window) mask |
| Block layout | `encoder.layer[i].attention` + `intermediate` | GPT-2 `h[i]`; Llama `layers[i]`; Gemma3 pre+post RMSNorm |
| z / u / z′ | Same as paper | Same **concept**, different module paths |
| IG baseline | ITB on token $j$ | ITB still valid; causal/sliding mask in forward |

The **set-to-set IG at module boundaries** (z→u, u→z, z→z) applies to both; only the forward paths and masks change.

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

## Module mapping (Gemma3 pre+post-LN)

```
z  → input_layernorm → attention core → concat heads = u
u  → output projection → post_attention_layernorm → +z
   → pre_feedforward_layernorm → mlp → post_feedforward_layernorm → +residual = z′
```

| LIG node | Gemma3 |
|----------|--------|
| z | input to `language_model.layers[l]` |
| u | concatenated attention heads before the output projection (`n_head * head_dim`) |
| z′ | block output |

Notes:

- Dual RoPE (global / local) and alternating sliding / full attention layers
- Both ATT output and downstream input use the concatenated-head boundary before
  the attention output projection. The downstream `u→z′` map includes that
  projection, post-attention RMSNorm, the fixed `z` residual, and the FFN.
- `n_head * head_dim` may differ from `hidden_size`
  (e.g. 8×256 = 2048 ≠ 2560 on gemma-3-4b-it).
- **ATT IG input space**: Gemma3 z→u interpolates the *token embeddings* and
  replays blocks `0 .. l-1` (as `gpt2_attention_models.py` does), while the
  GPT-2 / Llama adapters in `lig/adapters/decoder_ig/` interpolate z^(l)
  directly. This is not cosmetic: Gemma3's `u` sits before `o_proj` and has no
  residual term, so RMSNorm makes it scale invariant in z (`u(a·z) = u(z)`), and
  interpolating z^(l) would collapse the `zero` / `itb_zero_ratio` baselines to
  ~zero attributions (pinned by `test/test_gemma3_block_parity.py`). Gemma3 ATT
  scores are therefore embedding-space attributions — do not compare them
  head-to-head with GPT-2 / Llama z→u values.
- Multimodal checkpoints (`model_type=gemma3`, e.g. `google/gemma-3-4b-it`) use
  `model.language_model` for LIG (public API registers `gemma3` only)

Llama / Mistral / Qwen2 (not yet wired): `model.layers[l].self_attn` + `mlp`.

---

## Public API

```python
from lig import explain

result = explain(
    "The cat sat on the mat.",
    model="gpt2",  # or "google/gemma-3-4b-it"
    granularity="all",
    layers=[0],
    target_tokens=[1],
    target_head=0,  # recommended for Gemma3 ATT
)
```

`lig.adapters.load_adapter()` already detects decoder `model_type` and returns `DecoderAdapter`;  
`explain()` routes GPT-2 and Llama-family models through `load_decoder_ig_adapter()`,
and Gemma3 through its dedicated pre+post-LN path.

---

## Implementation phases

### Phase 1 — GPT-2 (smallest decoder) ✅

- [x] Causal attention mask + block forward (`utils/calculations/ig/gpt2/block_forward.py`)
- [x] z→u (ATT): `gpt2_attention_models.py` + embedding baseline interpolation
- [x] u→z (MLP): `mlp/gpt2_mlp_lig_ig.py` (`baseline_mlp='zero' | 'att_itb_a0'`)
- [x] z→z (layer): `z2z/gpt2_layer_direct_ig.py`
- [x] Test: `explain(..., model="gpt2", granularity="all", layers=[0])`

### Phase 2 — Llama family ✅

- [x] RMSNorm + SwiGLU MLP path (`lig/adapters/decoder_ig/llama.py`)
- [x] GQA: per-head z→u via head-dim slices
- [x] `load_decoder_ig_adapter()` + `lig.explain(model="meta-llama/...")`

### Phase 2b — Gemma3 ✅ (this release)

- [x] Pre+post RMSNorm block forward (`utils/calculations/ig/gemma3/block_forward.py`)
- [x] GQA + pre-`o_proj` head ATT IG (`attention/gemma3_attention_models.py`)
- [x] SwiGLU / Gemma3MLP path (`mlp/gemma3_mlp_lig_ig.py`)
- [x] Layer-direct IG (`z2z/gemma3_layer_direct_ig.py`)
- [x] `lig.explain(..., model="google/gemma-3-4b-it")` via `_run_explain_gemma3`

### Phase 3 — Composition & demos

- [ ] z2z compose (ATT × MLP) for decoder
- [ ] Optional: single-sentence demos only (no PTB required)

---

## Code layout (current)

```
lig/adapters/
  encoder.py    # BERT-family — production
  decoder.py    # load GPT-2 / Llama family / Gemma3
  decoder_ig/
    factory.py  # load_decoder_ig_adapter()
    gpt2.py     # GPT2Adapter
    llama.py    # LlamaIGAdapter (Llama 2/3, Mistral, Qwen2, Gemma)
lig/api.py      # _run_explain_decoder() for z→u, u→z, z→z
utils/calculations/ig/llama/  # block forward
utils/calculations/ig/z2z/llama_layer_direct_ig.py
utils/calculations/ig/
  gpt2/         # GPT-2 block helpers
  gemma3/       # Gemma3 pre+post-LN helpers
```

`test/test_lig_api.py` — GPT-2 smoke; Gemma3-4b-it marked `@pytest.mark.slow`.

---

## References

- Paper: module boundaries at ATT and MLP (encoder experiments on BERT-base + PTB)
- Transformers Gemma3 modeling (4.57.x): pre+post RMSNorm + dual RoPE
