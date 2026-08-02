"""Common interface for decoder-model attribution adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


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
