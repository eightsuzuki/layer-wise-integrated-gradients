"""Guards for the public LIG release scope."""

from __future__ import annotations

_UNSUPPORTED_BASELINE = "self_output_token"


def reject_otb_baseline(baseline_method: str) -> None:
    if baseline_method == _UNSUPPORTED_BASELINE:
        raise ValueError(
            f"baseline {baseline_method!r} is not supported in the public "
            "layer-wise-integrated-gradients release."
        )
