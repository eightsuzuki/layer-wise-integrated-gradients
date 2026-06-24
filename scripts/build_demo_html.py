#!/usr/bin/env python3
"""Build GitHub Pages z2z demo HTML from examples/paper_demo/manifest.json."""

from __future__ import annotations

import argparse
import html as html_module
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / "examples/paper_demo/manifest.json"
DATA_DIR = ROOT / "examples/paper_demo/json"
GITHUBPAGE_DIR = ROOT / "docs/githubpage"


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def export_name(sample_id: str, source_id: str) -> str:
    return f"{sample_id}__{source_id}.json"


def html_name(sample_id: str, source_id: str) -> str:
    return f"z2z_token_contribution_{sample_id}__{source_id}.html"


def pages_html_name(sample_id: str) -> str:
    return f"z2z_token_contribution_{sample_id}.html"


def unified_html_name() -> str:
    return "z2z_token_contribution.html"


def visualization_legend_html(route: str = "layer_direct") -> str:
    if route == "composed":
        intro = (
            "Within-layer token-to-token contributions on the layer-direct "
            "<em>z</em>→<em>z</em> map (token vectors <em>z</em> at layer input — see the "
            "<a href=\"../index.html#notation\">project page notation</a>). "
            "Each column is a target token position; "
            "circles show <strong>source → target</strong> contribution in that layer."
        )
    else:
        intro = (
            "Within-layer token-to-token contributions on the layer-direct "
            "<em>z</em>→<em>z</em> map (token vectors <em>z</em> at layer input — see the "
            "<a href=\"../index.html#notation\">project page notation</a>). "
            "Each column is a target token position; "
            "circles show <strong>source → target</strong> contribution in that layer."
        )
    return (
        '<div class="demo-legend">'
        '<p class="demo-legend-title"><strong>How to read</strong></p>'
        f'<p class="demo-legend-intro">{intro}</p>'
        '<ul class="demo-legend-list">'
        "<li><strong>Left label on each box</strong>: "
        "<em>target</em> token (receives the contribution)</li>"
        "<li><strong>Top / bottom token labels</strong> (horizontal): "
        "<em>source</em> token — circle position shows who contributed</li>"
        "<li><strong>Vertical axis</strong>: layers 0–11 (bottom row = layer 0, "
        "top row = layer 11; lower layers are closer to the input)</li>"
        "<li><strong>Circle size</strong>: contribution from source → target within the same layer</li>"
        "<li><strong>Path thickness</strong>: inter-layer contribution paths "
        '(toggle with "Show inter-layer contribution paths" in Display options)</li>'
        "<li><strong>Red highlight</strong>: clicked target token</li>"
        "</ul>"
        "</div>"
    )


def description_unified(sample: dict, source: dict) -> str:
    sample_line = f"{sample['index']:05d}"
    route = source.get("route", "layer_direct")
    sentence = sample.get("display_text") or ""
    if route == "composed":
        method_rows = (
            "<dt>Route</dt><dd>Composed (z→u × u→z)</dd>"
            f"<dt>ATT baseline (z→u)</dt><dd>{html_module.escape(source.get('att_label', ''))}</dd>"
            f"<dt>MLP baseline (u→z)</dt><dd>{html_module.escape(source.get('mlp_label', ''))}</dd>"
        )
    else:
        method_rows = (
            "<dt>Route</dt><dd>Direct LIG (layer z→z)</dd>"
            f"<dt>Baseline</dt><dd>{html_module.escape(source.get('label', source['id']))}</dd>"
        )
    return (
        '<p class="demo-lead"><strong>Within-layer token contributions (z→z)</strong></p>'
        '<dl class="demo-meta">'
        f"<dt>Sample</dt><dd>{html_module.escape(sample_line)}</dd>"
        f"{method_rows}"
        "</dl>"
        '<p class="demo-sentence-label">Input sentence</p>'
        f'<blockquote class="demo-sentence">{html_module.escape(sentence)}</blockquote>'
        + visualization_legend_html(route)
    )


def description_variant(sample: dict, source: dict) -> str:
    return description_unified(sample, source)


def embed_html_name() -> str:
    return "z2z_embed.html"


def _demo_source_specs(sources: list[dict]) -> list[dict]:
    return [
        {
            "id": s["id"],
            "label": s.get("label", s["id"]),
            "route": s.get("route", "layer_direct"),
            "att_baseline": s.get("att_baseline"),
            "mlp_baseline": s.get("mlp_baseline"),
            "att_label": s.get("att_label"),
            "mlp_label": s.get("mlp_label"),
        }
        for s in sources
    ]


def _compact_z2z_data(z2z_data: list, *, ndigits: int = 5) -> list:
    from lig.viz.z2z_contribution import compact_z2z_matrix

    return compact_z2z_matrix(z2z_data, ndigits=ndigits)


def _export_demo_data_chunk(
    out_dir: Path,
    sample_id: str,
    source_id: str,
    z2z_data: list,
    tokens: list[str],
) -> str:
    """Write compact JSON chunk; return URL path relative to githubpage HTML."""
    chunk_dir = out_dir / "demo_data"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{sample_id}__{source_id}.json"
    payload = {"z2zData": _compact_z2z_data(z2z_data), "tokens": tokens}
    (chunk_dir / fname).write_text(
        json.dumps(payload, separators=(",", ":")),
        encoding="utf-8",
    )
    return f"demo_data/{fname}"


def _load_demo_matrix(
    samples: list[dict],
    sources: list[dict],
    data_dir: Path,
    *,
    out_dir: Path,
    initial_sample_id: str,
    initial_source_id: str,
) -> tuple[dict, dict[str, str], str]:
    from lig.viz.z2z_contribution import load_demo_payload

    demo_matrix: dict = {}
    sample_labels: dict[str, str] = {}
    default_sample_id = next((s["id"] for s in samples if s.get("default")), samples[0]["id"])

    for entry in samples:
        sample_id = entry["id"]
        sample_labels[sample_id] = f"{entry['index']:05d}"
        demo_matrix[sample_id] = {}
        for source in sources:
            json_path = data_dir / export_name(sample_id, source["id"])
            if not json_path.exists():
                print(f"Skip missing: {json_path}")
                continue
            z2z_data, tokens, _text = load_demo_payload(str(json_path))
            source_id = source["id"]
            description_html = description_unified(entry, source)
            data_url = _export_demo_data_chunk(out_dir, sample_id, source_id, z2z_data, tokens)
            if sample_id == initial_sample_id and source_id == initial_source_id:
                demo_matrix[sample_id][source_id] = {
                    "z2z_data": _compact_z2z_data(z2z_data),
                    "tokens": tokens,
                    "description_html": description_html,
                }
            else:
                demo_matrix[sample_id][source_id] = {
                    "data_url": data_url,
                    "description_html": description_html,
                }
    return demo_matrix, sample_labels, default_sample_id


def _render_unified_page(
    demo_matrix: dict,
    sources: list[dict],
    sample_labels: dict[str, str],
    default_sample_id: str,
    default_source: dict,
    *,
    circle_max_style: str,
    show_site_banner: bool = True,
    embed_mode: bool = False,
) -> str:
    from lig.viz.z2z_contribution import render_z2z_multi_html

    return render_z2z_multi_html(
        demo_matrix,
        demo_sources=_demo_source_specs(sources),
        demo_sample_labels=sample_labels,
        initial_sample_id=default_sample_id,
        initial_source_id=default_source["id"],
        title="",
        layout_mode="normal",
        circle_max_style=circle_max_style,
        show_site_banner=show_site_banner,
        embed_mode=embed_mode,
    )


def build_unified_page(
    manifest: dict,
    samples: list[dict],
    *,
    data_dir: Path,
    out_dir: Path,
    sources: list[dict],
    default_source: dict,
    circle_max_style: str,
) -> Path:
    default_sample_id = next((s["id"] for s in samples if s.get("default")), samples[0]["id"])
    demo_matrix, sample_labels, default_sample_id = _load_demo_matrix(
        samples,
        sources,
        data_dir,
        out_dir=out_dir,
        initial_sample_id=default_sample_id,
        initial_source_id=default_source["id"],
    )

    page = _render_unified_page(
        demo_matrix,
        sources,
        sample_labels,
        default_sample_id,
        default_source,
        circle_max_style=circle_max_style,
        show_site_banner=True,
        embed_mode=False,
    )
    out_path = out_dir / unified_html_name()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    print(f"Wrote {out_path}")

    embed_page = _render_unified_page(
        demo_matrix,
        sources,
        sample_labels,
        default_sample_id,
        default_source,
        circle_max_style=circle_max_style,
        show_site_banner=False,
        embed_mode=True,
    )
    embed_path = out_dir / embed_html_name()
    embed_path.write_text(embed_page, encoding="utf-8")
    print(f"Wrote {embed_path}")

    for entry in samples:
        alias = out_dir / pages_html_name(entry["id"])
        shutil.copy2(out_path, alias)
        print(f"Wrote {alias} (alias)")

    from scripts.build_index_html import build_index_html

    build_index_html(ROOT / "docs" / "index.html")

    return out_path


def build_variant_page(
    json_path: Path,
    *,
    description: str,
    out_path: Path,
    circle_max_style: str,
) -> None:
    from lig.viz.z2z_contribution import load_demo_payload, render_z2z_html

    z2z_data, tokens, _ = load_demo_payload(str(json_path))
    page = render_z2z_html(
        z2z_data,
        tokens,
        title="",
        description=description,
        layout_mode="normal",
        circle_max_style=circle_max_style,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    print(f"Wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=GITHUBPAGE_DIR)
    parser.add_argument("--sample", help="Build only this sample id")
    parser.add_argument("--source", help="Build only this cache source id")
    parser.add_argument(
        "--all-sources",
        action="store_true",
        help="Also build per-source variant HTML (local comparison)",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    samples = manifest["samples"]
    if args.sample:
        samples = [s for s in samples if s["id"] == args.sample]
        if not samples:
            raise SystemExit(f"Unknown sample: {args.sample}")

    all_sources = manifest["sources"]
    page_sources = [s for s in all_sources if s.get("publish", True)]
    default_source = next((s for s in page_sources if s.get("default")), page_sources[0])

    if not args.all_sources and not args.source:
        build_unified_page(
            manifest,
            samples,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            sources=page_sources,
            default_source=default_source,
            circle_max_style="slot_cap",
        )
        return 0

    from lig.viz.z2z_contribution import load_demo_payload

    sources = all_sources
    if args.source:
        sources = [s for s in sources if s["id"] == args.source]
    for entry in samples:
        sample_id = entry["id"]
        for source in sources:
            variant_json = args.data_dir / export_name(sample_id, source["id"])
            if not variant_json.exists():
                raise SystemExit(f"Missing demo JSON: {variant_json}")
            _, _, vtext = load_demo_payload(str(variant_json))
            build_variant_page(
                variant_json,
                description=description_variant(entry, source),
                out_path=args.out_dir / html_name(sample_id, source["id"]),
                circle_max_style="slot_cap",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
