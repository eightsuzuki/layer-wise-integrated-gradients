#!/usr/bin/env python3
"""Build GitHub Pages landing (docs/index.html)."""

from __future__ import annotations

from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DOCS = ROOT / "docs"


def _ref_entry_html(*, cite_key: str, author_year_line: str, bibtex: str) -> str:
    return f"""        <div class="lig-ref-item">
          <p class="lig-ref-entry">{author_year_line}</p>
          <details class="lig-bibtex-details">
            <summary>BibTeX (<code>{cite_key}</code>)</summary>
            <pre class="lig-code"><code>{bibtex}</code></pre>
          </details>
        </div>"""


def _references_section_html() -> str:
  from lig.publication import ARXIV_URL

  entries = [
      dict(
          cite_key="sundararajan2017",
          author_year_line=(
              "Sundararajan, Mukund; Taly, Ankur; Yan, Qiqi (2017). "
              "<em>Axiomatic Attribution for Deep Networks</em>. "
              "Proceedings of ICML."
          ),
          bibtex="""@article{sundararajan2017,
  title        = {Axiomatic Attribution for Deep Networks},
  author       = {Sundararajan, Mukund and Taly, Ankur and Yan, Qiqi},
  journal      = {Proceedings of the 34th International Conference on Machine Learning},
  year         = {2017},
  pages        = {3319--3328}
}""",
      ),
      dict(
          cite_key="bach2015",
          author_year_line=(
              "Bach, Sebastian; Binder, Alexander; Montavon, Grégoire; et al. (2015). "
              "<em>On Pixel-Wise Explanations for Non-Linear Classifier Decisions "
              "by Layer-Wise Relevance Propagation</em>. PLOS ONE."
          ),
          bibtex="""@article{bach2015,
  title        = {On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation},
  author       = {Bach, Sebastian and Binder, Alexander and Montavon, Gr{\\'e}goire and Klauschen, Frederick and M{\\\"u}ller, Klaus-Robert and Samek, Wojciech},
  journal      = {PLOS ONE},
  year         = {2015},
  volume       = {10},
  number       = {7},
  pages        = {e0130140}
}""",
      ),
      dict(
          cite_key="montavon2019",
          author_year_line=(
              "Montavon, Grégoire; Lapuschkin, Sebastian; Binder, Alexander; et al. (2019). "
              "<em>Layer-Wise Relevance Propagation: An Overview</em>. "
              "Springer (Explainable AI)."
          ),
          bibtex="""@article{montavon2019,
  title        = {Layer-Wise Relevance Propagation: An Overview},
  author       = {Montavon, Gr{\\'e}goire and Lapuschkin, Sebastian and Binder, Alexander and Samek, Wojciech and M{\\\"u}ller, Klaus-Robert},
  journal      = {Explainable {AI}: Interpreting, Explaining and Visualizing Deep Learning},
  year         = {2019},
  pages        = {193--209},
  publisher    = {Springer}
}""",
      ),
      dict(
          cite_key="vaswani2017",
          author_year_line=(
              "Vaswani, Ashish; Shazeer, Noam; Parmar, Niki; et al. (2017). "
              "<em>Attention Is All You Need</em>. NeurIPS."
          ),
          bibtex="""@inproceedings{vaswani2017,
  title        = {Attention Is All You Need},
  author       = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, {\\L}ukasz and Polosukhin, Illia},
  booktitle    = {Advances in Neural Information Processing Systems (NeurIPS)},
  year         = {2017},
  pages        = {5998--6008}
}""",
      ),
      dict(
          cite_key="devlin2019",
          author_year_line=(
              "Devlin, Jacob; Chang, Ming-Wei; Lee, Kenton; Toutanova, Kristina (2019). "
              "<em>BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</em>. "
              "NAACL-HLT."
          ),
          bibtex="""@inproceedings{devlin2019,
  title        = {{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author       = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle    = {Proceedings of NAACL-HLT},
  year         = {2019},
  pages        = {4171--4186}
}""",
      ),
      dict(
          cite_key="achtibat2024attnlrp",
          author_year_line=(
              "Achtibat, Reduan; Hatefi, Sayed Mohammad Vakilzadeh; Dreyer, Maximilian; "
              "et al. (2024). "
              "<em>AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers</em>. "
              "ICML."
          ),
          bibtex="""@inproceedings{achtibat2024attnlrp,
  title        = {{AttnLRP}: {Attention-Aware} {Layer-Wise} {Relevance} {Propagation} for {Transformers}},
  author       = {Achtibat, Reduan and Hatefi, Sayed Mohammad Vakilzadeh and Dreyer, Maximilian and Samek, Wojciech and Lapuschkin, Sebastian},
  booktitle    = {Proceedings of the International Conference on Machine Learning (ICML)},
  year         = {2024}
}""",
      ),
      dict(
          cite_key="sturmfels2020",
          author_year_line=(
              "Sturmfels, Pascal; Lundberg, Scott; Lee, Su-In (2020). "
              "<em>Visualizing the Impact of Feature Attribution Baselines</em>. Distill."
          ),
          bibtex="""@article{sturmfels2020,
  title        = {Visualizing the Impact of Feature Attribution Baselines},
  author       = {Sturmfels, Pascal and Lundberg, Scott and Lee, Su-In},
  journal      = {Distill},
  year         = {2020}
}""",
      ),
      dict(
          cite_key="marcus1999treebank",
          author_year_line=(
              "Marcus, Mitchell P.; Santorini, Beatrice; Marcinkiewicz, Mary Ann; Taylor, Ann (1999). "
              "<em>Treebank-3</em>. Linguistic Data Consortium, Philadelphia. "
              "LDC Catalog No.&nbsp;<a href=\"https://catalog.ldc.upenn.edu/LDC99T42\">LDC99T42</a>."
          ),
          bibtex="""@misc{marcus1999treebank,
  author       = {Marcus, Mitchell P. and Santorini, Beatrice and Marcinkiewicz, Mary Ann and Taylor, Ann},
  title        = {Treebank-3},
  howpublished = {Web Download},
  publisher    = {Linguistic Data Consortium},
  address      = {Philadelphia},
  year         = {1999},
  note         = {LDC Catalog No. LDC99T42. https://catalog.ldc.upenn.edu/LDC99T42}
}""",
      ),
  ]
  blocks = "\n".join(
      _ref_entry_html(
          cite_key=entry["cite_key"],
          author_year_line=entry["author_year_line"],
          bibtex=entry["bibtex"],
      )
      for entry in entries
  )
  return f"""      <div class="lig-subsection" id="references">
        <p class="lig-eyebrow">References</p>
        <h2 class="lig-section-title">Paper bibliography</h2>
        <p class="lig-abstract">
          Key citations from the LIG paper — methods, models, baselines, and evaluation data.
          Full bibliography: <a href="{ARXIV_URL}" target="_blank" rel="noopener">arXiv preprint</a>.
        </p>
{blocks}
      </div>"""


def build_index_html(out_path: Path | None = None) -> Path:
    from lig.publication import (
        ARXIV_BIBTEX,
        ARXIV_URL,
        GITHUB_REPO_URL,
        LOGO_ALT_TEXT,
        PUBLICATION_CONTACT_EMAIL,
        PYPI_PROJECT_URL,
        SITE_DESCRIPTION,
        SITE_TITLE,
        LANDING_PAGE_CANONICAL,
        seo_head_extras_html,
        publication_arxiv_button_html,
        publication_authors_block_html,
        publication_paper_button_html,
        write_seo_static_files,
    )

    authors_block = publication_authors_block_html()
    github = GITHUB_REPO_URL
    pypi = PYPI_PROJECT_URL
    out_path = out_path or DOCS / "index.html"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{SITE_TITLE}</title>
  <meta name="description" content="{SITE_DESCRIPTION}">
{seo_head_extras_html(title=SITE_TITLE, description=SITE_DESCRIPTION, page_url=LANDING_PAGE_CANONICAL)}
  <meta name="theme-color" content="#0d9488">
  <link rel="icon" type="image/png" href="logo/LIG-LOGO-web.png">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="lig-hero.css">
</head>
<body class="lig-site">
  <div class="lig-particles-zone">
    <div id="lig-particles" class="lig-hero-particles" aria-hidden="true"></div>
    <section class="lig-hero-banner">
      <div class="lig-hero-inner lig-hero-inner--wide">
      <img class="lig-hero-logo" src="logo/LIG-LOGO.png" alt="{LOGO_ALT_TEXT}">
      <h1 class="pub-title">
        Layer-wise Integrated Gradients<br>
        for within-layer flow in Transformers
      </h1>
      <p class="pub-subtitle">
        Set-to-set Integrated Gradients at Attention and MLP boundaries inside each Transformer
        layer — with an interactive demo and a Python package. Theory and experiments are in the
        <a href="{ARXIV_URL}" target="_blank" rel="noopener">arXiv preprint</a>.
      </p>
      {authors_block}
      <div class="pub-links">
        {publication_paper_button_html()}
        {publication_arxiv_button_html()}
        <a href="{github}" target="_blank" rel="noopener" class="pub-btn">Code</a>
        <a href="{pypi}" target="_blank" rel="noopener" class="pub-btn">PyPI</a>
        <a href="#demo" class="pub-btn">Interactive demo</a>
        <a href="mailto:{PUBLICATION_CONTACT_EMAIL}" class="pub-btn">Contact</a>
      </div>
      </div>
    </section>

  <section class="lig-section" id="overview">
    <div class="lig-section-inner">
      <p class="lig-eyebrow">Overview</p>
      <h2 class="lig-section-title">What is LIG?</h2>
      <div class="lig-abstract">
        <p>
          <strong>Layer-wise Integrated Gradients (LIG)</strong> attributes how tokens influence
          each other <em>inside</em> a Transformer layer — not only across layers — by applying
          Integrated Gradients at Attention and MLP module boundaries.
        </p>
        <p class="lig-paper-note">
          Definitions, baselines, L₂ diagnostics, and PTB experiments are in the paper.
          This page covers notation, figures, install, and the visualization.
        </p>
        <p>
          LIG is <strong>model-agnostic at the block level</strong> — the same API covers
          BERT-style encoders (ATT/MLP split), block-only models (layer granularity only),
          and GPT-2 decoders, with reproducible PTB experiments in the
          <a href="{github}">release repository</a>.
          Boundary detection is automatic: <code>describe_boundaries(model_id)</code> reports
          residual-stream nodes <em>z</em>, attention outputs <em>u</em>, and IG hook points.
        </p>
      </div>
    </div>
  </section>

  <section class="lig-section lig-section--soft" id="notation">
    <div class="lig-section-inner">
      <p class="lig-eyebrow">Notation</p>
      <h2 class="lig-section-title">Reading the demo</h2>
      <div class="lig-abstract">
        <p>
          Each Transformer layer processes a <strong>set of token vectors</strong> on the residual stream.
          <strong><em>z</em><sub><em>i</em></sub><sup>(<em>l</em>)</sup></strong> is the representation of token
          <em>i</em> at the input of layer&nbsp;<em>l</em>.
          After multi-head attention (ATT), head&nbsp;<em>h</em> produces
          <strong><em>u</em><sub><em>i</em></sub><sup>(<em>l</em>,<em>h</em>)</sup></strong> for each token;
          MLP then updates the stream to <em>z</em><sup>(<em>l</em>+1)</sup>.
        </p>
        <p>
          With layer&nbsp;<em>l</em> fixed, one block is
          <em>z</em><sup>(<em>l</em>)</sup> → ATT → <em>u</em> → MLP → <em>z</em><sup>(<em>l</em>+1)</sup>
          (see the flow below).
          LIG attributes <em>token-to-token</em> influence inside that block using Integrated Gradients at the
          ATT and MLP module boundaries.
        </p>
        <p>
          The interactive demo plots <strong>within-layer <em>z</em>→<em>z</em> maps</strong> (route label
          <strong>z2z</strong>): how much each input token contributes to each output token after the full
          layer block.
          Labels <strong>z2u</strong> and <strong>u2z</strong> refer to the ATT and MLP steps in between when
          you switch to the composed route in the demo.
        </p>
      </div>
      <div class="lig-flow" aria-label="Layer stack: Layer l-1 through Layer l+1">
        <span class="lig-flow-node lig-flow-node--layer">Layer <em>l</em>−1</span>
        <span class="lig-flow-arrow" aria-hidden="true">→</span>
        <span class="lig-flow-var"><em>z</em><sup>(<em>l</em>)</sup></span>
        <span class="lig-flow-arrow" aria-hidden="true">→</span>
        <div class="lig-flow-node lig-flow-node--layer lig-flow-node--layer-block">
          <span class="lig-flow-layer-title">Layer <em>l</em></span>
          <div class="lig-flow-layer-inner">
            <span class="lig-flow-node lig-flow-node--module">ATT</span>
            <span class="lig-flow-arrow" aria-hidden="true">→</span>
            <span class="lig-flow-var"><em>u</em><sup>(<em>l</em>,<em>h</em>)</sup></span>
            <span class="lig-flow-arrow" aria-hidden="true">→</span>
            <span class="lig-flow-node lig-flow-node--module">MLP</span>
          </div>
        </div>
        <span class="lig-flow-arrow" aria-hidden="true">→</span>
        <span class="lig-flow-var"><em>z</em><sup>(<em>l</em>+1)</sup></span>
        <span class="lig-flow-arrow" aria-hidden="true">→</span>
        <span class="lig-flow-node lig-flow-node--layer">Layer <em>l</em>+1</span>
      </div>
      <div class="lig-route-cards" aria-label="Contribution route notation">
        <article class="lig-route-card lig-route-card--att">
          <div class="lig-route-card-head">
            <span class="lig-route-formula"><em>z</em> → <em>u</em></span>
            <span class="lig-route-tag">z2u · Attention</span>
          </div>
          <p>How much each input token <em>z</em> contributes to each attention output <em>u</em> (per head).</p>
        </article>
        <article class="lig-route-card lig-route-card--mlp">
          <div class="lig-route-card-head">
            <span class="lig-route-formula"><em>u</em> → <em>z</em></span>
            <span class="lig-route-tag">u2z · MLP</span>
          </div>
          <p>How much each attention output <em>u</em> (per head) contributes to the next-layer token vector.</p>
        </article>
        <article class="lig-route-card lig-route-card--layer">
          <div class="lig-route-card-head">
            <span class="lig-route-formula"><em>z</em> → <em>z</em></span>
            <span class="lig-route-tag">z2z · Layer</span>
          </div>
          <p>Token-to-token contribution for the whole layer block — measured directly (top path) or as the product of z2u and u2z (bottom path).</p>
        </article>
      </div>
      <div class="lig-figure-block">
        <div class="lig-figure-intro">
          <h3 class="lig-figure-kicker">Two views within layer <em>l</em> (Fig. 2)</h3>
          <p>
            LIG can attribute within-layer flow in two ways. They answer the same question — how tokens
            influence each other inside one layer — but follow different paths through the block.
          </p>
          <dl class="lig-route-dl">
            <dt><strong>Layer-direct · z2z</strong> <span class="lig-route-dl-tag">top path</span></dt>
            <dd>
              Apply Integrated Gradients once on the layer-whole map
              <em>z</em><sup>(<em>l</em>)</sup> → <em>z</em><sup>(<em>l</em>+1)</sup>.
              The demo’s default view uses this route: each cell shows how much source token <em>i</em>
              contributes to target token <em>j</em> within the same layer.
            </dd>
            <dt><strong>Composed · z2u × u2z</strong> <span class="lig-route-dl-tag">bottom path</span></dt>
            <dd>
              Measure ATT (<em>z</em>→<em>u</em>, z2u) and MLP (<em>u</em>→<em>z</em>, u2z) separately, multiply
              per-head contributions, then sum over heads.
              The paper compares this composition with the layer-direct map under an <em>L</em><sub>2</sub>
              criterion (details in the paper).
            </dd>
          </dl>
        </div>
        <figure class="lig-figure lig-figure--wide">
          <img
            src="figures/z2z_z2u_u2z_two_views.png"
            alt="Top: layer-direct z-to-z path. Bottom: composed Attention z-to-u and MLP u-to-z paths."
            loading="lazy"
          >
        </figure>
      </div>
    </div>
  </section>

  <section class="lig-section lig-demo-section" id="demo">
    <div class="lig-section-inner">
      <p class="lig-eyebrow">Visualization</p>
      <h2 class="lig-section-title">Within-layer contribution map</h2>
      <p class="lig-demo-lead">
        Samples <code>00016</code> / <code>00410</code>. Use <strong>Display options</strong>
        (top-right) to toggle paths and adjust circle sizes.
        <a href="githubpage/z2z_token_contribution.html">Click the preview below</a>
        or use <strong>Open in new tab</strong> if you prefer a separate window.
      </p>
      <div class="lig-btn-row">
        <a href="githubpage/z2z_token_contribution.html" target="_blank" rel="noopener" class="pub-btn">Open in new tab</a>
        <a href="{github}/blob/main/examples/paper_demo/DATA_NOTICE.md" target="_blank" rel="noopener" class="pub-btn">DATA_NOTICE.md</a>
      </div>
    </div>
  </section>
  </div>

  <div class="lig-demo-frame-wrap" id="ligDemoFrameWrap">
    <iframe
      class="lig-demo-frame"
      src="githubpage/z2z_embed.html"
      title="LIG z2z token contribution demo"
      loading="lazy"
      referrerpolicy="no-referrer-when-downgrade"
      tabindex="-1"
    ></iframe>
    <a
      href="githubpage/z2z_token_contribution.html"
      class="lig-demo-frame-hit"
      aria-label="Open interactive within-layer contribution map"
    >Open interactive demo</a>
  </div>

  <section class="lig-section" id="data">
    <div class="lig-section-inner">
      <p class="lig-eyebrow">Data</p>
      <h2 class="lig-section-title">Penn Treebank (PTB)</h2>
      <div class="lig-abstract">
        <p>
          The paper's <strong>Experiment A</strong> evaluates within-layer flow consistency on
          Penn Treebank development sentences in Stanford Dependencies format (indices
          <code>0</code>–<code>1699</code>).
          This site ships only <strong>two excerpt sentences</strong> for the interactive demo
          (<code>00016</code> and <code>00410</code>).
        </p>
      </div>
      <div class="lig-highlight-box">
        The full Treebank-3 corpus is <strong>not</strong> included in this repository.
        Reproducing Experiment A requires an
        <a href="https://catalog.ldc.upenn.edu/LDC99T42">LDC license</a> for Treebank-3
        (LDC99T42). See <a href="{github}/blob/main/examples/paper_demo/DATA_NOTICE.md">DATA_NOTICE.md</a>
        for licensing details and what is redistributed here.
      </div>
      <div class="lig-btn-row">
        <a href="{github}/blob/main/examples/paper_demo/DATA_NOTICE.md" target="_blank" rel="noopener" class="pub-btn pub-btn--ghost">DATA_NOTICE.md</a>
        <a href="{github}/blob/main/docs/REPRODUCTION.md" target="_blank" rel="noopener" class="pub-btn pub-btn--ghost">Reproduction guide</a>
      </div>
{_references_section_html()}
    </div>
  </section>

  <section class="lig-section lig-section--soft" id="resources">
    <div class="lig-section-inner">
      <p class="lig-eyebrow">Resources</p>
      <h2 class="lig-section-title">Install &amp; reproduction</h2>
      <div class="lig-grid lig-grid--two">
        <article class="lig-card">
          <h3>Install from PyPI</h3>
          <p>
            Install <strong>PyTorch</strong> first (CUDA or CPU wheel), then the package.
            CLI: <code>lig explain "…"</code>
          </p>
          <pre class="lig-code"><code>pip install torch  # pick your CUDA/CPU index
pip install layer-wise-integrated-gradients</code></pre>
          <div class="lig-btn-row">
            <a href="{pypi}" target="_blank" rel="noopener" class="pub-btn">PyPI</a>
            <a href="{github}" target="_blank" rel="noopener" class="pub-btn pub-btn--ghost">GitHub</a>
          </div>
        </article>
        <article class="lig-card">
          <h3>Python API</h3>
          <p>One-call attribution to JSON — z→u, u→z, and z→z inside each layer.</p>
          <pre class="lig-code"><code>from lig import explain

explain(
    "The cat sat on the mat.",
    model="bert-base-uncased",
    num_steps=32,
    granularity="all",
    layers=[0, 11],
)</code></pre>
          <div class="lig-btn-row">
            <a href="{github}#quick-start" target="_blank" rel="noopener" class="pub-btn pub-btn--ghost">README</a>
            <a href="{github}/blob/main/docs/LIG_COMPUTATION.md" target="_blank" rel="noopener" class="pub-btn pub-btn--ghost">Computation guide</a>
          </div>
        </article>
        <article class="lig-card">
          <h3>Experiment A (PTB)</h3>
          <p>
            Reproduce the paper's PTB dev evaluation (indices 0–1699) with the scripts in the
            repository. Data licensing and the Treebank-3 citation are summarized in the
            <a href="#data">Penn Treebank section</a>.
          </p>
          <div class="lig-btn-row">
            <a href="#data" class="pub-btn pub-btn--ghost">PTB &amp; licensing</a>
            <a href="{github}/blob/main/docs/REPRODUCTION.md" target="_blank" rel="noopener" class="pub-btn pub-btn--ghost">Reproduction guide</a>
          </div>
        </article>
        <article class="lig-card">
          <h3>Model boundaries</h3>
          <p>
            Inspect residual-stream nodes <em>z</em>, attention outputs <em>u</em>, and hook points
            without running IG.
          </p>
          <pre class="lig-code"><code>from lig import describe_boundaries

describe_boundaries("gpt2", load_weights=False)</code></pre>
          <div class="lig-btn-row">
            <a href="{github}/blob/main/docs/MODEL_BOUNDARIES.md" target="_blank" rel="noopener" class="pub-btn pub-btn--ghost">Boundary guide</a>
          </div>
        </article>
      </div>
      <div class="lig-subsection">
      <p class="lig-eyebrow">Cite</p>
      <h2 class="lig-section-title">BibTeX</h2>
      <p class="lig-abstract">
        Cite the arXiv preprint (<a href="{ARXIV_URL}" target="_blank" rel="noopener">2606.21564</a>).
        Conference paper BibTeX will be added when available.
      </p>
      <pre class="lig-code"><code>{ARXIV_BIBTEX}</code></pre>
      </div>
    </div>
  </section>

  <footer class="lig-footer">
    <nav class="lig-footer-nav" aria-label="Site links">
      <a href="{pypi}" target="_blank" rel="noopener">PyPI</a>
      <a href="{github}" target="_blank" rel="noopener">GitHub</a>
      <a href="githubpage/z2z_token_contribution.html">Demo</a>
      <a href="#data">PTB data</a>
      <a href="#references">References</a>
      <a href="{github}/blob/main/LICENSE" target="_blank" rel="noopener">MIT License</a>
    </nav>
    <small>Layer-wise Integrated Gradients · Waseda University</small>
  </footer>
  <script src="js/particles.min.js"></script>
  <script>
    (function () {{
      function initHeroParticles() {{
        var el = document.getElementById('lig-particles');
        if (!el || typeof particlesJS !== 'function') return;
        particlesJS('lig-particles', {{
          particles: {{
            number: {{ value: 80, density: {{ enable: true, value_area: 680 }} }},
            color: {{ value: ['#0d9488', '#14b8a6', '#5eead4'] }},
            shape: {{ type: 'circle' }},
            opacity: {{
              value: 0.52,
              random: true,
              anim: {{ enable: true, speed: 0.8, opacity_min: 0.22, sync: false }}
            }},
            size: {{ value: 3.2, random: true }},
            line_linked: {{
              enable: true,
              distance: 145,
              color: '#0d9488',
              opacity: 0.3,
              width: 1
            }},
            move: {{
              enable: true,
              speed: 1.1,
              direction: 'none',
              random: true,
              straight: false,
              out_mode: 'out',
              bounce: false
            }}
          }},
          interactivity: {{
            detect_on: 'canvas',
            events: {{
              onhover: {{ enable: true, mode: 'grab' }},
              onclick: {{ enable: false }},
              resize: true
            }},
            modes: {{
              grab: {{ distance: 120, line_linked: {{ opacity: 0.35 }} }}
            }}
          }},
          retina_detect: true
        }});
      }}
      if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', initHeroParticles);
      }} else {{
        initHeroParticles();
      }}

      var frame = document.querySelector('.lig-demo-frame');
      if (!frame) return;
      var MIN_PX = 480;
      var DEFAULT_PX = 720;
      var MAX_VH = 92;
      var lastContentHeight = 0;
      function vhPx(vh) {{
        return Math.round(window.innerHeight * vh / 100);
      }}
      function clampFrameHeight(contentPx) {{
        var maxH = vhPx(MAX_VH);
        if (!contentPx) return Math.min(DEFAULT_PX, maxH);
        var h = Math.min(Math.ceil(contentPx) + 8, maxH);
        return Math.max(MIN_PX, h);
      }}
      function applyFrameHeight(contentPx) {{
        if (contentPx) lastContentHeight = contentPx;
        var h = clampFrameHeight(lastContentHeight);
        frame.style.height = h + 'px';
        frame.style.minHeight = h + 'px';
      }}
      applyFrameHeight(0);
      window.addEventListener('message', function (ev) {{
        if (ev.data && ev.data.type === 'lig-demo-height' && typeof ev.data.height === 'number') {{
          applyFrameHeight(ev.data.height);
        }}
      }});
      window.addEventListener('resize', function () {{
        applyFrameHeight(lastContentHeight);
      }});
    }})();
  </script>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    robots_path, sitemap_path = write_seo_static_files(DOCS)
    print(f"Wrote {robots_path}")
    print(f"Wrote {sitemap_path}")
    print(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    build_index_html()
