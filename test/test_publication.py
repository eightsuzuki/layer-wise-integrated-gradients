"""Tests for publication metadata used on the project site."""

from __future__ import annotations

from pathlib import Path

from lig.publication import (
    ARXIV_ID,
    ARXIV_URL,
    DEMO_PAGE_URL,
    GITHUB_PAGES_BASE,
    LANDING_PAGE_CANONICAL,
    PUBLICATION_AFFILIATIONS,
    PUBLICATION_AUTHORS,
    SITE_KEYWORDS,
    canonical_link_html,
    json_ld_graph_html,
    meta_keywords_html,
    publication_arxiv_button_html,
    publication_authors_block_html,
    publication_paper_button_html,
    robots_txt_content,
    seo_head_extras_html,
    sitemap_xml_content,
    social_meta_tags_html,
    write_seo_static_files,
)

PAPER_TEX = (
    Path(__file__).resolve().parents[2]
    / "BERT_IG_baselin_paper/iconip2026/en/main.tex"
)


def test_publication_authors_match_paper_tex():
    if not PAPER_TEX.is_file():
        return

    tex = PAPER_TEX.read_text(encoding="utf-8").replace("\n", " ")
    for name, marks, _profile_url in PUBLICATION_AUTHORS:
        assert f"{name}\\inst{{{marks}}}" in tex

    for affil in PUBLICATION_AFFILIATIONS:
        assert affil in tex


def test_publication_authors_block_html_includes_fujitsu():
    html = publication_authors_block_html()
    assert "Fujitsu Limited" in html
    assert "Eight Suzuki<sup>1,3</sup>" in html
    assert "Hideitsu Hino<sup>1,2</sup>" in html
    assert "Waseda University" in html
    assert "The Institute of Statistical Mathematics" in html


def test_publication_authors_block_html_links_profiles():
    html = publication_authors_block_html()
    assert 'href="https://github.com/eightsuzuki"' in html
    assert 'href="https://sites.google.com/view/hinohide/"' in html
    assert 'href="https://noboru-murata.github.io/"' in html
    assert 'rel="noopener noreferrer"' in html


def test_publication_arxiv_button_is_active_link():
    html = publication_arxiv_button_html()
    assert f'href="{ARXIV_URL}"' in html
    assert ARXIV_ID in ARXIV_URL
    assert "is-disabled" not in html
    assert "pub-soon" not in html


def test_publication_paper_button_stays_soon_until_proceedings():
    html = publication_paper_button_html()
    assert "is-disabled" in html
    assert "pub-soon" in html
    assert "ICONIP" not in html


def test_seo_meta_includes_keywords_and_canonical():
    html = seo_head_extras_html()
    assert 'rel="canonical"' in html
    assert LANDING_PAGE_CANONICAL in html
    assert 'name="keywords"' in html
    assert "integrated gradients" in SITE_KEYWORDS
    assert 'property="og:title"' in html
    assert 'name="twitter:card"' in html
    assert "application/ld+json" in html


def test_social_meta_tags_include_image_and_url():
    html = social_meta_tags_html(
        page_url=DEMO_PAGE_URL,
    )
    assert DEMO_PAGE_URL in html
    assert 'property="og:image"' in html
    assert 'name="twitter:image"' in html
    assert 'property="og:site_name"' in html


def test_json_ld_includes_software_and_scholarly_article():
    html = json_ld_graph_html(page_url=LANDING_PAGE_CANONICAL)
    assert "SoftwareSourceCode" in html
    assert "ScholarlyArticle" in html
    assert ARXIV_ID in html
    assert ARXIV_URL in html
    assert "https://github.com/eightsuzuki" in html


def test_robots_and_sitemap_reference_github_pages():
    robots = robots_txt_content()
    sitemap = sitemap_xml_content()
    assert "Allow: /" in robots
    assert f"{GITHUB_PAGES_BASE}/sitemap.xml" in robots
    assert LANDING_PAGE_CANONICAL in sitemap
    assert DEMO_PAGE_URL in sitemap


def test_write_seo_static_files(tmp_path: Path):
    robots_path, sitemap_path = write_seo_static_files(tmp_path)
    assert robots_path.is_file()
    assert sitemap_path.is_file()
    assert "Sitemap:" in robots_path.read_text(encoding="utf-8")
    assert "<urlset" in sitemap_path.read_text(encoding="utf-8")


def test_canonical_and_keywords_helpers_escape_urls():
    url = 'https://example.com/?q="x"'
    assert "&quot;" in canonical_link_html(page_url=url)
    assert "&quot;" in meta_keywords_html(keywords='foo, "bar"')
