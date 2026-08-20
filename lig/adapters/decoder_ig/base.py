"""Common interface for decoder-model attribution adapters."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

COMPLETENESS_TOLERANCE = 0.02
"""Relative completeness error above which an IG attribution is not usable.

Not a cosmetic threshold.  An unconverged attribution can still *look* right:
on Qwen2.5-7B block 17 at ``num_steps=64`` the relative error is 1.005 while the
source ranking scores the highest Spearman of any method tested; at 1024 steps
the error falls to 0.036 and the ranking drops to its true value.  Required step
counts are model-dependent and differ by 16x between Llama and Qwen.
"""


def check_completeness(relative_error: float, *, where: str) -> bool:
    """Return whether IG completeness holds, warning loudly when it does not.

    Callers used to have to notice ``is_valid`` in a dict; nothing forced them
    to, and a silently unconverged run reads as a confident result.
    """
    is_valid = bool(relative_error < COMPLETENESS_TOLERANCE)
    if not is_valid:
        warnings.warn(
            f"{where}: IG completeness is violated by a relative "
            f"{relative_error:.3g} (tolerance {COMPLETENESS_TOLERANCE}). "
            "The attribution has not converged; raise num_steps until this "
            "clears before reading any ranking from it.",
            RuntimeWarning,
            stacklevel=3,
        )
    return is_valid


class DecoderIGAdapter(ABC):
    """Model-structure adapter used by decoder IG implementations.

    The adapter hides where a decoder model stores layer inputs, attention
    outputs, MLP inputs, masks, and head dimensions. Attribution code should
    depend on this interface rather than on model-specific module names.
    """

    @abstractmethod
    def encode(self, text: str, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """Tokenize text and move tensors to the adapter device."""

    @abstractmethod
    def cache(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Run/capture one forward pass and cache intermediate tensors."""

    @abstractmethod
    def get_z(self, layer_idx: int) -> torch.Tensor:
        """Return the residual stream entering decoder block ``layer_idx``."""

    @abstractmethod
    def attention_output(
        self,
        layer_idx: int,
        z: torch.Tensor,
        target_token_idx: int,
        head_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Return the attention output vector for a target token/head."""

    @abstractmethod
    def get_mlp_input(self, layer_idx: int) -> torch.Tensor:
        """Return the residual stream entering the MLP sublayer."""

    @abstractmethod
    def mlp_output(
        self,
        layer_idx: int,
        mlp_input: torch.Tensor,
    ) -> torch.Tensor:
        """Return the residual stream after applying the MLP sublayer."""
