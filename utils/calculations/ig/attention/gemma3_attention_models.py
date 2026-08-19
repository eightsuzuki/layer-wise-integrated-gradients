"""
Gemma 3 Attention Integrated Gradients (z -> u), pre-o_proj head space.

Unlike GPT-2, Gemma3 has ``n_head * head_dim != hidden_size``, so per-head
scores are taken from the attention output **before** ``o_proj``
(``[batch, head_dim]`` or concatenated heads). This is not residual-stream
space; compare heads relatively rather than against ``u`` L2 norms.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from utils.calculations.ig.gemma3.block_forward import (
    attn_pre_oproj_output,
    forward_gemma3_block,
    get_gemma3_text_model,
    layer_attention_mask,
    rope_position_embeddings,
)


class Gemma3AttentionModel(nn.Module):
    """
    Captum-compatible ATT IG model for Gemma3.

    Interpolates token embeddings (inputs_embeds / scaled wte), runs blocks
    ``0 .. layer_idx-1``, then evaluates the target-layer attention core
    (pre-``o_proj``) at ``target_token_idx`` / ``target_head_idx``.

    The IG input is the embedding, not z^(l) as in the GPT-2 / Llama adapters
    (``lig/adapters/decoder_ig/``): Gemma3's u sits before ``o_proj`` and has no
    residual term, so RMSNorm makes it scale invariant in z
    (``u(a * z) == u(z)``). Interpolating z would give a constant path and ~zero
    attributions for the ``zero`` and ``itb_zero_ratio`` baselines. Scores are
    therefore embedding-space attributions and are not directly comparable with
    the GPT-2 / Llama z→u numbers.
    """

    def __init__(
        self,
        text_model: nn.Module,
        layer_idx: int,
        target_token_idx: int,
        target_head_idx: Optional[int] = None,
        use_last_token: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.text_model = get_gemma3_text_model(text_model)
        self.layer_idx = layer_idx
        self.target_token_idx = target_token_idx
        self.target_head_idx = target_head_idx
        self.use_last_token = use_last_token
        self.debug = debug

        self.device = next(self.text_model.parameters()).device
        cfg = self.text_model.config
        self.num_heads = int(cfg.num_attention_heads)
        self.head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
        self.hidden_size = int(cfg.hidden_size)

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = input_embeddings.shape
        actual_target_idx = seq_len - 1 if self.use_last_token else self.target_token_idx

        global_pe, local_pe = rope_position_embeddings(self.text_model, input_embeddings)
        hidden = input_embeddings
        for block_idx in range(self.layer_idx):
            mask = layer_attention_mask(
                self.text_model, block_idx, seq_len, input_embeddings.device, input_embeddings.dtype
            )
            hidden = forward_gemma3_block(
                self.text_model.layers[block_idx],
                hidden,
                mask,
                global_pe,
                local_pe,
            )

        block = self.text_model.layers[self.layer_idx]
        mask = layer_attention_mask(
            self.text_model, self.layer_idx, seq_len, input_embeddings.device, input_embeddings.dtype
        )
        pe = local_pe if block.self_attn.is_sliding else global_pe
        normed = block.input_layernorm(hidden)
        pre_oproj = attn_pre_oproj_output(
            block,
            normed,
            mask,
            pe,
            target_head_idx=self.target_head_idx,
        )
        target = pre_oproj[:, actual_target_idx, :]
        return torch.norm(target, dim=-1)


def create_gemma3_attention_model(
    text_model: nn.Module,
    layer_idx: int,
    target_token_idx: int,
    target_head_idx: Optional[int] = None,
    use_last_token: bool = False,
    debug: bool = False,
) -> Gemma3AttentionModel:
    return Gemma3AttentionModel(
        text_model=text_model,
        layer_idx=layer_idx,
        target_token_idx=target_token_idx,
        target_head_idx=target_head_idx,
        use_last_token=use_last_token,
        debug=debug,
    )
