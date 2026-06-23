"""Publication metadata shared by the project page and demo HTML."""

from __future__ import annotations

import html as html_module
import json
from dataclasses import dataclass
from pathlib import Path

# (display name, affiliation superscripts, optional profile URL)
PUBLICATION_AUTHORS: tuple[tuple[str, str, str | None], ...] = (
    ("Eight Suzuki", "1,3", "https://github.com/eightsuzuki"),
    ("Hideitsu Hino", "1,2", "https://sites.google.com/view/hinohide/"),
    ("Noboru Murata", "1", "https://noboru-murata.github.io/"),
)

PUBLICATION_AFFILIATIONS: tuple[str, ...] = (
    "Waseda University",
    "The Institute of Statistical Mathematics",
    "Fujitsu Limited",
)

PUBLICATION_CONTACT_EMAIL = "suzuki8@akane.waseda.jp"

GITHUB_PAGES_BASE = "https://eightsuzuki.github.io/layer-wise-integrated-gradients"
PYPI_PROJECT_URL = "https://pypi.org/project/layer-wise-integrated-gradients/"
GITHUB_REPO_URL = "https://github.com/eightsuzuki/layer-wise-integrated-gradients"
ARXIV_ID = "2606.21564"
ARXIV_URL = f"https://arxiv.org/abs/{ARXIV_ID}"
PAPER_URL: str | None = None  # conference proceedings — not published yet

ARXIV_BIBTEX = """@article{suzuki2026lig,
  title         = {LIG: Layer-wise Integrated Gradients for Within-Layer Flow Analysis in Transformers},
  author        = {Suzuki, Eight and Hino, Hideitsu and Murata, Noboru},
  year          = {2026},
  eprint        = {2606.21564},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}"""
PAPER_TITLE = (
    "LIG: Layer-wise Integrated Gradients for Within-Layer Flow Analysis in Transformers"
)
SITE_TITLE = "LIG — Layer-wise Integrated Gradients for Transformer Explainability"
SITE_DESCRIPTION = (
    "Layer-wise Integrated Gradients (LIG) attributes token-to-token influence "
    "inside Transformer layers using set-to-set Integrated Gradients at Attention "
    "and MLP boundaries. Interactive BERT demo, Python package, and arXiv preprint."
)
SITE_KEYWORDS = (
    "integrated gradients, layer-wise integrated gradients, LIG, transformers, "
    "explainability, interpretability, attribution, BERT, within-layer flow, "
    "attention, MLP, token contribution, z2z, feature attribution, NLP"
)
DEMO_PAGE_TITLE = "LIG Interactive Demo — Within-Layer z2z Token Contribution Maps"
DEMO_PAGE_DESCRIPTION = (
    "Explore within-layer token-to-token contribution maps (z2z) computed with "
    "Layer-wise Integrated Gradients on BERT-base-uncased. Switch PTB samples, "
    "layers, and attribution routes in an interactive visualization."
)
EMBED_PAGE_TITLE = "LIG Demo Embed — Within-Layer z2z Token Contributions"
EMBED_PAGE_DESCRIPTION = (
    "Embedded within-layer z2z token contribution visualization for "
    "Layer-wise Integrated Gradients (LIG) on Transformer models."
)
LOGO_ALT_TEXT = "LIG logo — Layer-wise Integrated Gradients for Transformer explainability"
SOCIAL_PREVIEW_IMAGE = f"{GITHUB_PAGES_BASE}/logo/LIG-LOGO.png"
SOCIAL_PREVIEW_IMAGE_WIDTH = 1415
SOCIAL_PREVIEW_IMAGE_HEIGHT = 417
LANDING_PAGE_URL = f"{GITHUB_PAGES_BASE}/"
LANDING_PAGE_CANONICAL = f"{GITHUB_PAGES_BASE}/index.html"
DEMO_PAGE_URL = f"{GITHUB_PAGES_BASE}/githubpage/z2z_token_contribution.html"
EMBED_PAGE_URL = f"{GITHUB_PAGES_BASE}/githubpage/z2z_embed.html"
SITEMAP_URLS: tuple[str, ...] = (
    LANDING_PAGE_URL,
    LANDING_PAGE_CANONICAL,
    DEMO_PAGE_URL,
    EMBED_PAGE_URL,
)


@dataclass(frozen=True)
class PublicationMeta:
    authors: tuple[tuple[str, str, str | None], ...]
    affiliations: tuple[str, ...]


def _seo_indent_line(indent: str, line: str) -> str:
    return f"{indent}{line}" if indent else line


def canonical_link_html(*, page_url: str, indent: str = "  ") -> str:
    u = html_module.escape(page_url)
    return _seo_indent_line(indent, f'<link rel="canonical" href="{u}">')


def meta_keywords_html(*, keywords: str = SITE_KEYWORDS, indent: str = "  ") -> str:
    k = html_module.escape(keywords)
    return _seo_indent_line(indent, f'<meta name="keywords" content="{k}">')


def social_meta_tags_html(
    *,
    title: str = SITE_TITLE,
    description: str = SITE_DESCRIPTION,
    page_url: str = LANDING_PAGE_CANONICAL,
    indent: str = "  ",
) -> str:
    """Open Graph and Twitter Card tags for link previews (absolute GitHub Pages URLs)."""
    t = html_module.escape(title)
    d = html_module.escape(description)
    u = html_module.escape(page_url)
    img = html_module.escape(SOCIAL_PREVIEW_IMAGE)
    alt = html_module.escape(LOGO_ALT_TEXT)
    w = SOCIAL_PREVIEW_IMAGE_WIDTH
    h = SOCIAL_PREVIEW_IMAGE_HEIGHT
    i = indent
    return f"""{i}<meta property="og:title" content="{t}">
{i}<meta property="og:description" content="{d}">
{i}<meta property="og:url" content="{u}">
{i}<meta property="og:type" content="website">
{i}<meta property="og:site_name" content="Layer-wise Integrated Gradients (LIG)">
{i}<meta property="og:locale" content="en_US">
{i}<meta property="og:image" content="{img}">
{i}<meta property="og:image:width" content="{w}">
{i}<meta property="og:image:height" content="{h}">
{i}<meta property="og:image:alt" content="{alt}">
{i}<meta name="twitter:card" content="summary_large_image">
{i}<meta name="twitter:title" content="{t}">
{i}<meta name="twitter:description" content="{d}">
{i}<meta name="twitter:image" content="{img}">
{i}<meta name="twitter:image:alt" content="{alt}">"""


def _json_ld_authors() -> list[dict[str, str]]:
    authors: list[dict[str, str]] = []
    for name, _marks, profile_url in PUBLICATION_AUTHORS:
        entry: dict[str, str] = {"@type": "Person", "name": name}
        if profile_url:
            entry["url"] = profile_url
        authors.append(entry)
    return authors


def json_ld_graph_html(*, page_url: str, indent: str = "  ") -> str:
    """JSON-LD graph: SoftwareSourceCode + ScholarlyArticle (arXiv preprint)."""
    graph = [
        {
            "@type": "SoftwareSourceCode",
            "@id": f"{GITHUB_PAGES_BASE}/#software",
            "name": "layer-wise-integrated-gradients",
            "alternateName": "LIG",
            "description": SITE_DESCRIPTION,
            "url": page_url,
            "codeRepository": GITHUB_REPO_URL,
            "downloadUrl": PYPI_PROJECT_URL,
            "programmingLanguage": "Python",
            "license": "https://spdx.org/licenses/MIT.html",
            "author": _json_ld_authors(),
            "citation": {
                "@id": f"{ARXIV_URL}#paper",
            },
        },
        {
            "@type": "ScholarlyArticle",
            "@id": f"{ARXIV_URL}#paper",
            "headline": PAPER_TITLE,
            "name": PAPER_TITLE,
            "author": _json_ld_authors(),
            "datePublished": "2026",
            "isPartOf": {"@type": "WebSite", "name": "arXiv", "url": "https://arxiv.org"},
            "identifier": {
                "@type": "PropertyValue",
                "propertyID": "arXiv",
                "value": ARXIV_ID,
            },
            "sameAs": ARXIV_URL,
            "url": ARXIV_URL,
        },
    ]
    payload = {
        "@context": "https://schema.org",
        "@graph": graph,
    }
    script = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f'{indent}<script type="application/ld+json">{script}</script>'


def seo_head_extras_html(
    *,
    title: str = SITE_TITLE,
    description: str = SITE_DESCRIPTION,
    page_url: str = LANDING_PAGE_CANONICAL,
    indent: str = "  ",
    include_json_ld: bool = True,
) -> str:
    """Keywords, canonical URL, social cards, and optional JSON-LD."""
    parts = [
        meta_keywords_html(indent=indent),
        canonical_link_html(page_url=page_url, indent=indent),
        social_meta_tags_html(
            title=title,
            description=description,
            page_url=page_url,
            indent=indent,
        ),
    ]
    if include_json_ld:
        parts.append(json_ld_graph_html(page_url=page_url, indent=indent))
    return "\n".join(parts)


def robots_txt_content() -> str:
    sitemap = f"{GITHUB_PAGES_BASE}/sitemap.xml"
    return f"""User-agent: *
Allow: /

Sitemap: {sitemap}
"""


def sitemap_xml_content() -> str:
    from datetime import date

    today = date.today().isoformat()
    url_blocks = []
    for loc in SITEMAP_URLS:
        url_blocks.append(
            "  <url>\n"
            f"    <loc>{html_module.escape(loc)}</loc>\n"
            f"    <lastmod>{today}</lastmod>\n"
            "  </url>"
        )
    body = "\n".join(url_blocks)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{body}\n"
        "</urlset>\n"
    )


def write_seo_static_files(docs_dir: Path) -> tuple[Path, Path]:
    """Write docs/robots.txt and docs/sitemap.xml for GitHub Pages."""
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    robots_path = docs_dir / "robots.txt"
    sitemap_path = docs_dir / "sitemap.xml"
    robots_path.write_text(robots_txt_content(), encoding="utf-8")
    sitemap_path.write_text(sitemap_xml_content(), encoding="utf-8")
    return robots_path, sitemap_path


def publication_meta() -> PublicationMeta:
    return PublicationMeta(
        authors=PUBLICATION_AUTHORS,
        affiliations=PUBLICATION_AFFILIATIONS,
    )


def publication_paper_button_html() -> str:
    """Paper button — active when PAPER_URL is set, otherwise disabled 'soon'."""
    if PAPER_URL:
        url = html_module.escape(PAPER_URL)
        return (
            f'<a href="{url}" target="_blank" rel="noopener" class="pub-btn">Paper</a>'
        )
    return """<span class="pub-btn is-disabled" title="Conference paper coming soon">
          <span>Paper</span><span class="pub-soon">soon</span>
        </span>"""


def publication_arxiv_button_html() -> str:
    """Active arXiv preprint link."""
    url = html_module.escape(ARXIV_URL)
    return (
        f'<a href="{url}" target="_blank" rel="noopener" class="pub-btn">arXiv</a>'
    )


def publication_authors_block_html() -> str:
    """Author line + numbered affiliations for landing page and demo banner."""
    meta = publication_meta()
    author_bits: list[str] = []
    for idx, (name, marks, profile_url) in enumerate(meta.authors):
        if idx:
            author_bits.append('<span class="pub-author-sep">·</span>')
        label = f"{html_module.escape(name)}<sup>{marks}</sup>"
        if profile_url:
            url = html_module.escape(profile_url)
            label = (
                f'<a href="{url}" target="_blank" rel="noopener noreferrer">{label}</a>'
            )
        author_bits.append(f'<span class="pub-author">{label}</span>')
    authors_line = "\n          ".join(author_bits)

    affil_items = "\n          ".join(
        f"<li><sup>{idx}</sup> {affil}</li>"
        for idx, affil in enumerate(meta.affiliations, start=1)
    )

    return f"""
      <div class="pub-authors-block">
        <p class="pub-authors-line">
          {authors_line}
        </p>
        <ol class="pub-affil-list">
          {affil_items}
        </ol>
      </div>
"""
