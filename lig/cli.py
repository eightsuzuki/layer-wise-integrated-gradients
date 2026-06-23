#!/usr/bin/env python3
"""CLI for Layer-wise Integrated Gradients."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lig",
        description="Layer-wise Integrated Gradients — explain encoder models in one command.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("explain", help="Compute LIG attributions for one sentence")
    ex.add_argument("text", help="Input sentence")
    ex.add_argument("--model", default="bert-base-uncased", help="Hugging Face model id")
    ex.add_argument("--steps", type=int, default=32, help="IG integration steps")
    ex.add_argument(
        "--granularity",
        default="all",
        help="att, mlp, layer, or all (comma-separated for multiple)",
    )
    ex.add_argument("--baseline-att", default="self_input_token", choices=["zero", "self_input_token", "itb_zero_ratio", "itb_map_ratio"])
    ex.add_argument("--baseline-mlp", default="zero", choices=["zero", "att_itb_a0"])
    ex.add_argument("--baseline-layer", default="self_input_token", choices=["zero", "self_input_token", "itb_zero_ratio"])
    ex.add_argument("--layers", default=None, help="Comma-separated layer indices (default: all)")
    ex.add_argument("--target-tokens", default=None, help="Comma-separated token indices (default: all)")
    ex.add_argument("--target-head", type=int, default=None, help="Attention head index (default: all heads aggregated)")
    ex.add_argument("--device", default=None, help="cuda / cpu (auto if omitted)")
    ex.add_argument("-o", "--output", default=None, help="Write JSON to file (default: stdout)")
    ex.add_argument("--no-residual", action="store_true", help="Disable MLP residual in u->z path")

    args = parser.parse_args(argv)

    if args.command == "explain":
        from lig import explain

        granularity = args.granularity
        if "," in granularity:
            granularity = [g.strip() for g in granularity.split(",")]

        layers = _parse_int_list(args.layers)
        target_tokens = _parse_int_list(args.target_tokens)

        result = explain(
            args.text,
            model=args.model,
            num_steps=args.steps,
            granularity=granularity,
            baseline_att=args.baseline_att,
            baseline_mlp=args.baseline_mlp,
            baseline_layer=args.baseline_layer,
            layers=layers,
            target_tokens=target_tokens,
            target_head=args.target_head,
            device=args.device,
            include_residual_connection=not args.no_residual,
        )
        payload = json.dumps(result, indent=2, ensure_ascii=False)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(payload)
                f.write("\n")
        else:
            print(payload)
        return 0

    return 1


def _parse_int_list(s: str | None) -> list[int] | None:
    if s is None or s.strip() == "":
        return None
    return [int(x.strip()) for x in s.split(",")]


if __name__ == "__main__":
    sys.exit(main())
