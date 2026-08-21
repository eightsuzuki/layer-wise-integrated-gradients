"""Regression tests for the GPT-2 decoder IG adapter.

GPT-2 escaped the 2026-08-19 causal-mask audit — ``eager_attention_forward``
applies ``module.bias`` internally, so ``attention_mask=None`` is still causal
here — but its per-head path did not: ``_split_heads`` / ``_attn`` were removed
in the transformers attention-interface refactor, and a bare ``except`` fell
back to slicing the *post*-c_proj vector, which mixes all heads.
"""

from __future__ import annotations

import pytest

TINY_GPT2 = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def tiny_gpt2_adapter():
    pytest.importorskip("torch")
    from lig.adapters.decoder_ig import GPT2Adapter

    adapter = GPT2Adapter(TINY_GPT2, device="cpu")
    adapter.cache(adapter.encode("one two three four five six"))
    return adapter


def test_gpt2_attention_output_ignores_future_tokens(tiny_gpt2_adapter):
    import torch

    ad = tiny_gpt2_adapter
    z = ad.get_z(0)
    target = 2
    perturbed = z.clone()
    perturbed[:, target + 1 :, :] += 100.0
    assert torch.allclose(
        ad.attention_output(0, z, target), ad.attention_output(0, perturbed, target), atol=1e-4
    )


def test_gpt2_per_head_outputs_reconstruct_attention(tiny_gpt2_adapter):
    """c_proj(concat of per-head outputs) == the full attention output."""
    import torch

    ad = tiny_gpt2_adapter
    z = ad.get_z(0)
    target = 3
    full = ad.attention_output(0, z, target)
    heads = torch.cat(
        [ad.attention_output(0, z, target, head_idx=h) for h in range(ad.num_heads)], dim=-1
    )
    c_proj = ad.decoder.h[0].attn.c_proj
    assert torch.allclose(c_proj(heads), full, atol=1e-4)


def test_gpt2_per_head_is_not_a_residual_slice(tiny_gpt2_adapter):
    """The old fallback returned ``full[:, :head_dim]`` for head 0."""
    import torch

    ad = tiny_gpt2_adapter
    z = ad.get_z(0)
    target = 3
    full = ad.attention_output(0, z, target)
    head0 = ad.attention_output(0, z, target, head_idx=0)
    assert not torch.allclose(head0, full[:, : ad.head_dim], atol=1e-6)


def test_gpt2_attention_ig_zero_future_attribution(tiny_gpt2_adapter):
    """No post-hoc zeroing: causality alone must drive future attributions to 0."""
    import numpy as np

    ad = tiny_gpt2_adapter
    target = 3
    result = ad.compute_attention_ig(0, target_token_idx=target, baseline="zero", num_steps=32)
    assert result.verification["relative_error"] < 0.02
    assert np.abs(result.values[target + 1 :]).max() == 0.0
