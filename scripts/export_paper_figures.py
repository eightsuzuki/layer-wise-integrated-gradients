#!/usr/bin/env python3
"""Export paper figures as web PNGs into docs/figures/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = (
    ROOT.parent
    / "BERT_IG_baselin_paper/iconip2026/en/images"
)
OUT = ROOT / "docs/figures"

FIGURES = (
    ("transformer_overview/transformer_node_viz.pdf", "transformer_node_viz.png", 1100),
    ("ig_views/z2z_z2u_u2z_two_views.pdf", "z2z_z2u_u2z_two_views.png", 1100),
    ("ig_views/z2z_layerwise_ig_map_contribution.pdf", "z2z_layerwise_ig_map_contribution.png", 700),
)


def export_figures(src_dir: Path, out_dir: Path) -> None:
    try:
        import fitz  # PyMuPDF
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Install export deps in the project venv:\n"
            "  python -m pip install pymupdf pillow\n"
            f"({exc})"
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    for rel, name, max_w in FIGURES:
        pdf = src_dir / rel
        if not pdf.exists():
            print(f"Skip missing: {pdf}")
            continue
        doc = fitz.open(pdf)
        page = doc[0]
        zoom = min(1.5, max_w / page.rect.width)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        out = out_dir / name
        pix.save(out)
        im = Image.open(out)
        if im.width > max_w:
            ratio = max_w / im.width
            im = im.resize((max_w, int(im.height * ratio)), Image.Resampling.LANCZOS)
        im.save(out, optimize=True, compress_level=9)
        print(f"Wrote {out} ({im.width}×{im.height}, {out.stat().st_size // 1024} KB)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_SRC,
        help="Paper en/images directory (default: monorepo BERT_IG_baselin_paper path)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help="Output directory for PNG files",
    )
    args = parser.parse_args()
    if not args.src.is_dir():
        print(f"Source not found: {args.src}", file=sys.stderr)
        return 1
    export_figures(args.src, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
