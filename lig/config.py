from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Union

Granularity = Literal["att", "mlp", "layer", "all"]
AttBaseline = Literal["zero", "self_input_token", "itb_zero_ratio", "itb_map_ratio"]
MlpBaseline = Literal["zero", "att_itb_a0"]
LayerBaseline = Literal["zero", "self_input_token", "itb_zero_ratio"]

_LIG_RELEASE_ATT_BASELINES = frozenset({"zero", "self_input_token", "itb_zero_ratio", "itb_map_ratio"})
_LIG_RELEASE_MLP_BASELINES = frozenset({"zero", "att_itb_a0"})
_LIG_RELEASE_LAYER_BASELINES = frozenset({"zero", "self_input_token", "itb_zero_ratio"})


@dataclass
class LIGConfig:
    """Configuration for a single LIG explanation run (public release scope)."""

    model: str = "bert-base-uncased"
    num_steps: int = 32
    granularity: Union[Granularity, List[Granularity]] = "all"
    baseline_att: AttBaseline = "self_input_token"
    baseline_mlp: MlpBaseline = "zero"
    baseline_layer: LayerBaseline = "self_input_token"
    layers: Optional[List[int]] = None
    target_tokens: Optional[List[int]] = None
    target_head: Optional[int] = None
    device: Optional[str] = None
    include_residual_connection: bool = True

    def resolved_granularity(self) -> List[str]:
        if self.granularity == "all":
            return ["att", "mlp", "layer"]
        if isinstance(self.granularity, str):
            return [self.granularity]
        return list(self.granularity)


def validate_release_baselines(cfg: LIGConfig) -> None:
    """Reject baselines outside the public LIG release."""
    if cfg.baseline_att not in _LIG_RELEASE_ATT_BASELINES:
        raise ValueError(
            f"baseline_att must be one of {sorted(_LIG_RELEASE_ATT_BASELINES)}. "
            f"Got: {cfg.baseline_att!r}. "
            "ITB-zeroRatio=itb_zero_ratio, ITB-mapRatio=itb_map_ratio."
        )
    if cfg.baseline_mlp not in _LIG_RELEASE_MLP_BASELINES:
        raise ValueError(
            f"baseline_mlp must be one of {sorted(_LIG_RELEASE_MLP_BASELINES)} "
            f"(ATTITBa=0=att_itb_a0). Got: {cfg.baseline_mlp!r}."
        )
    if cfg.baseline_layer not in _LIG_RELEASE_LAYER_BASELINES:
        raise ValueError(
            f"baseline_layer must be one of {sorted(_LIG_RELEASE_LAYER_BASELINES)}. "
            f"Got: {cfg.baseline_layer!r}. "
            "LAYER-ITB-zeroRatio=itb_zero_ratio."
        )
