"""Z2Z layer-wise token contribution visualization (standalone HTML)."""

from __future__ import annotations

import html as html_module
import json as json_module
from typing import Any, Dict, List, Optional

from lig.publication import (
    DEMO_PAGE_DESCRIPTION,
    DEMO_PAGE_TITLE,
    DEMO_PAGE_URL,
    EMBED_PAGE_DESCRIPTION,
    EMBED_PAGE_TITLE,
    EMBED_PAGE_URL,
    LOGO_ALT_TEXT,
    PAPER_TITLE,
    PUBLICATION_CONTACT_EMAIL,
    seo_head_extras_html,
    publication_arxiv_button_html,
    publication_authors_block_html,
    publication_paper_button_html,
)


def load_demo_payload(path: str) -> tuple[list, list[str], str]:
    """Load z2z matrices and tokens from paper demo or lig explain JSON."""
    import json
    from pathlib import Path
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tokens = data.get("tokens", [])
    if "z2z" in data and isinstance(data["z2z"], list):
        return data["z2z"], tokens, data.get("text", "")
    layers = data.get("layers", {})
    z2z_data: list = []
    for i in range(12):
        layer = layers.get(str(i)) or layers.get(f"layer_{i}")
        if not layer:
            break
        z2z = layer.get("z2z", {})
        matrix = z2z.get("matrix")
        if matrix is not None:
            z2z_data.append(matrix)
    return z2z_data, tokens, data.get("text", "")


def compact_z2z_matrix(
    z2z_data: List[List[List[float]]],
    *,
    ndigits: int = 5,
) -> List[List[List[float]]]:
    """Round z2z values for smaller embedded demo JSON (visual fidelity preserved)."""
    return [
        [[round(float(v), ndigits) for v in row] for row in layer]
        for layer in z2z_data
    ]


def demo_site_head_assets(*, css_href: str = "../lig-hero.css") -> str:
    """Shared hero stylesheet and Inter font for GitHub Pages demos."""
    css = html_module.escape(css_href)
    return f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="{css}">
"""


def demo_site_particles_scripts(*, js_href: str = "../js/particles.min.js") -> str:
    """particles.js loader and init for standalone demo hero banners."""
    src = html_module.escape(js_href)
    return f"""
  <script src="{src}"></script>
  <script>
    (function () {{
      function initHeroParticles() {{
        var el = document.getElementById('lig-particles');
        if (!el || typeof particlesJS !== 'function') return;
        particlesJS('lig-particles', {{
          particles: {{
            number: {{ value: 45, density: {{ enable: true, value_area: 680 }} }},
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
              onhover: {{ enable: false }},
              onclick: {{ enable: false }},
              resize: true
            }}
          }},
          retina_detect: true
        }});
      }}
      function scheduleHeroParticles() {{
        if (typeof requestIdleCallback === 'function') {{
          requestIdleCallback(initHeroParticles, {{ timeout: 1200 }});
        }} else {{
          setTimeout(initHeroParticles, 200);
        }}
      }}
      if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', scheduleHeroParticles);
      }} else {{
        scheduleHeroParticles();
      }}
    }})();
  </script>
"""


def demo_site_particles_zone_open() -> str:
    """Opening wrapper for particles.js backdrop on standalone demo pages."""
    return """
<div class="lig-particles-zone lig-particles-zone--demo">
  <div id="lig-particles" class="lig-hero-particles" aria-hidden="true"></div>
"""


def demo_site_particles_zone_close() -> str:
    return "</div>\n"


def demo_site_banner_html(
    *,
    logo_href: str = "../logo/LIG-LOGO.png",
    home_href: str = "../index.html",
    demo_href: str = "../index.html",
    github_href: str = "https://github.com/eightsuzuki/layer-wise-integrated-gradients",
) -> str:
    """Compact header for GitHub Pages z2z demos."""
    logo = html_module.escape(logo_href)
    home = html_module.escape(home_href)
    demo = html_module.escape(demo_href)
    github = html_module.escape(github_href)
    contact = html_module.escape(PUBLICATION_CONTACT_EMAIL)
    return f"""
<section class="lig-hero-banner lig-hero-banner--demo">
  <div class="lig-hero-inner">
    <a href="{home}"><img class="lig-hero-logo" src="{logo}" alt="{html_module.escape(LOGO_ALT_TEXT)}"></a>
    <h1 class="pub-title">{html_module.escape(PAPER_TITLE)}</h1>
    <p class="pub-subtitle">Set-to-set Integrated Gradients at Attention and MLP module boundaries inside each Transformer layer.</p>
    {publication_authors_block_html()}
    <div class="pub-links">
      {publication_paper_button_html()}
      {publication_arxiv_button_html()}
      <a href="{github}" target="_blank" rel="noopener" class="pub-btn">Code</a>
      <a href="{demo}" class="pub-btn">Project page</a>
      <a href="mailto:{contact}" class="pub-btn">Contact</a>
    </div>
  </div>
</section>
"""


def _parse_initial_nav(
    source_id: str,
    source_entries: List[Dict[str, Any]],
) -> Dict[str, str]:
    by_id = {s["id"]: s for s in source_entries}
    src = by_id.get(source_id) or source_entries[0]
    if src.get("route") == "composed":
        return {
            "route": "composed",
            "layer_source_id": "",
            "att_baseline": src.get("att_baseline", "zero"),
            "mlp_baseline": src.get("mlp_baseline", "zero"),
        }
    return {
        "route": "layer_direct",
        "layer_source_id": src["id"],
        "att_baseline": "zero",
        "mlp_baseline": "zero",
    }


def _build_demo_nav_html(
    sample_entries: Dict[str, Dict[str, Any]],
    source_entries: List[Dict[str, Any]],
    current_sample_id: str,
    current_source_id: str,
) -> str:
    sample_options = []
    for sid, entry in sample_entries.items():
        label = html_module.escape(str(entry.get("label", sid)))
        selected = " selected" if sid == current_sample_id else ""
        sample_options.append(f'<option value="{html_module.escape(sid)}"{selected}>{label}</option>')

    init = _parse_initial_nav(current_source_id, source_entries)
    layer_direct = [s for s in source_entries if s.get("route", "layer_direct") == "layer_direct"]

    def _opts(items: List[tuple[str, str]], current: str) -> str:
        return "\n".join(
            f'<option value="{html_module.escape(val)}"{" selected" if val == current else ""}>'
            f"{html_module.escape(label)}</option>"
            for val, label in items
        )

    layer_opts = _opts([(s["id"], s.get("label", s["id"])) for s in layer_direct], init["layer_source_id"])
    att_opts = _opts(
        [
            ("zero", "Zero"),
            ("itb_raw", "ITB (raw)"),
            ("itb_map", "ITB-mapRatio"),
            ("itb_zero_ratio", "ITB-zeroRatio"),
        ],
        init["att_baseline"],
    )
    mlp_opts = _opts(
        [
            ("zero", "Zero"),
            ("attitba0", "ATTITBa=0"),
        ],
        init["mlp_baseline"],
    )

    route_direct_sel = " selected" if init["route"] == "layer_direct" else ""
    route_composed_sel = " selected" if init["route"] == "composed" else ""
    direct_style = "" if init["route"] == "layer_direct" else ' style="display:none"'
    composed_style = "" if init["route"] == "composed" else ' style="display:none"'

    catalog_json = json_module.dumps(
        {
            s["id"]: {
                "route": s.get("route", "layer_direct"),
                "att_baseline": s.get("att_baseline"),
                "mlp_baseline": s.get("mlp_baseline"),
                "att_label": s.get("att_label"),
                "mlp_label": s.get("mlp_label"),
                "label": s.get("label", s["id"]),
            }
            for s in source_entries
        }
    )

    return f"""
<div class="z2z-demo-nav">
  <div class="z2z-demo-nav__group">
    <label for="demo-sample-select">Sample</label>
    <select id="demo-sample-select" onchange="applyDemoSelection()">
    {"\\n".join(sample_options)}
    </select>
  </div>
  <div class="z2z-demo-nav__group">
    <label for="demo-route-select">Route</label>
    <select id="demo-route-select" onchange="onDemoRouteChange()">
      <option value="layer_direct"{route_direct_sel}>Direct LIG (layer z→z)</option>
      <option value="composed"{route_composed_sel}>Composed (z→u × u→z)</option>
    </select>
  </div>
  <div class="z2z-demo-nav__group" id="demo-direct-group"{direct_style}>
    <label for="demo-layer-baseline-select">Layer baseline</label>
    <select id="demo-layer-baseline-select" onchange="applyDemoSelection()">
    {layer_opts}
    </select>
  </div>
  <div class="z2z-demo-nav__group z2z-demo-nav__group--composed" id="demo-composed-group"{composed_style}>
    <p class="z2z-demo-nav__hint">z→u (ATT) and u→z (MLP) baselines are independent — 4 × 2 = 8 combinations.</p>
    <div class="z2z-demo-nav__composed-row">
      <div class="z2z-demo-nav__subgroup">
        <label for="demo-att-baseline-select">ATT baseline (z→u)</label>
        <select id="demo-att-baseline-select" onchange="applyDemoSelection()" title="Integrated Gradients baseline for attention z→u">
        {att_opts}
        </select>
      </div>
      <span class="z2z-demo-nav__times" aria-hidden="true">×</span>
      <div class="z2z-demo-nav__subgroup">
        <label for="demo-mlp-baseline-select">MLP baseline (u→z)</label>
        <select id="demo-mlp-baseline-select" onchange="applyDemoSelection()" title="Integrated Gradients baseline for MLP u→z (ATTITBa=0 uses ITB attention output at a=0)">
        {mlp_opts}
        </select>
      </div>
    </div>
    <p class="z2z-demo-nav__combo" id="demo-composed-summary" style="display:none"></p>
  </div>
</div>
<script>
  const demoSourceCatalog = {catalog_json};
</script>
<style>
  .z2z-demo-nav {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px 16px;
    align-items: flex-end;
    margin-bottom: 12px;
  }}
  .z2z-demo-nav__group {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: min(200px, 100%);
  }}
  .z2z-demo-nav__group--composed {{
    min-width: min(320px, 100%);
  }}
  .z2z-demo-nav__hint {{
    margin: 0 0 6px;
    font-size: 11px;
    color: #64748b;
    line-height: 1.45;
  }}
  .z2z-demo-nav__composed-row {{
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    gap: 8px 10px;
  }}
  .z2z-demo-nav__subgroup {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    flex: 1 1 140px;
  }}
  .z2z-demo-nav__times {{
    font-size: 1.25rem;
    font-weight: 700;
    color: #94a3b8;
    padding-bottom: 10px;
    line-height: 1;
  }}
  .z2z-demo-nav__combo {{
    margin: 8px 0 0;
    font-size: 13px;
    font-weight: 600;
    color: #4338ca;
  }}
  .z2z-demo-nav__combo span {{
    color: #64748b;
    font-weight: 500;
  }}
  .z2z-demo-nav__group label {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #64748b;
  }}
  .z2z-demo-nav select {{
    font-size: 14px;
    padding: 8px 10px;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    background: #fff;
    color: #0f172a;
  }}
  .z2z-demo-nav select:disabled {{
    opacity: 0.45;
    cursor: not-allowed;
  }}
</style>
"""


def _build_sample_selector_html(
    demo_samples: Dict[str, Dict[str, Any]],
    current_sample_id: str,
) -> str:
    """Legacy single-axis sample nav (sample only)."""
    options = []
    for sid, entry in demo_samples.items():
        label = html_module.escape(f"{sid} — {entry.get('label', '')}")
        selected = " selected" if sid == current_sample_id else ""
        options.append(f'<option value="{html_module.escape(sid)}"{selected}>{label}</option>')
    options_html = "\n".join(options)
    return f"""
<div class="demo-nav">
  <label for="demo-sample-select"><strong>Sample:</strong></label>
  <select id="demo-sample-select" onchange="applyDemoSelection()">
    {options_html}
  </select>
</div>
<style>
  .demo-nav {{
    position: fixed;
    top: 10px;
    left: 10px;
    z-index: 10001;
    background: rgba(255, 255, 255, 0.95);
    padding: 10px 14px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
    font-size: 14px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
    max-width: min(96vw, 720px);
  }}
  .demo-nav select {{
    flex: 1;
    min-width: 140px;
    font-size: 13px;
  }}
</style>
"""


def create_interactive_visualization(
    z2z_data: List[List[List[float]]],
    tokens: List[str],
    layer: int = 0,
    target_token: Optional[int] = None,
    num_layers: int = 12,
    show_paths: bool = False,
    align_horizontal: bool = False,
    title: str = "Layer-wise token contributions (z2z IG)",
    description: str = None,
    layout_mode: str = "normal",
    circle_max_style: str = "slot_cap",
    demo_samples: Optional[Dict[str, Dict[str, Any]]] = None,
    initial_sample_id: Optional[str] = None,
    demo_matrix: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    demo_sources: Optional[List[Dict[str, Any]]] = None,
    demo_sample_labels: Optional[Dict[str, str]] = None,
    initial_source_id: Optional[str] = None,
    favicon_href: Optional[str] = "../logo/LIG-LOGO-web.png",
    show_site_banner: bool = False,
    embed_mode: bool = False,
    logo_href: str = "../logo/LIG-LOGO.png",
) -> str:
    """インタラクティブな可視化のHTMLを生成 - Layer 0のToken 0のみ"""

    use_slot_cap_js = "true" if circle_max_style == "slot_cap" else "false"

    # データをJSON形式に変換
    import html as html_module
    import json as json_module

    z2z_json = json_module.dumps(compact_z2z_matrix(z2z_data), separators=(',', ':'))
    tokens_json = json_module.dumps(tokens, separators=(',', ':'))

    escaped_title = html_module.escape(title)
    if embed_mode:
        seo_title = EMBED_PAGE_TITLE
        seo_description = EMBED_PAGE_DESCRIPTION
        seo_page_url = EMBED_PAGE_URL
    else:
        seo_title = DEMO_PAGE_TITLE if not title else title
        seo_description = DEMO_PAGE_DESCRIPTION
        seo_page_url = DEMO_PAGE_URL
    page_title = html_module.escape(seo_title if not title else title) or html_module.escape(DEMO_PAGE_TITLE)
    favicon_link = (
        f'<link rel="icon" type="image/png" href="{html_module.escape(favicon_href)}">'
        if favicon_href
        else ""
    )
    hero_head_assets = demo_site_head_assets() if show_site_banner else ""
    if description:
        escaped_description = description.replace("\n", "<br>")
    else:
        escaped_description = None

    multi_sample = (demo_matrix is not None and len(demo_matrix) > 0) or (
        demo_samples is not None and len(demo_samples) > 0
    )
    if demo_matrix:
        initial_sample_id = initial_sample_id or next(iter(demo_matrix))
        initial_source_id = initial_source_id or (
            demo_sources[0]["id"] if demo_sources else next(iter(demo_matrix[initial_sample_id]))
        )
        sample_labels = {
            sid: {"label": (demo_sample_labels or {}).get(sid, sid)}
            for sid in demo_matrix
        }
        demo_payload = {
            sample_id: {
                source_id: (
                    {
                        "dataUrl": entry["data_url"],
                        "descriptionHtml": entry.get("description_html", ""),
                    }
                    if entry.get("data_url")
                    else {
                        "z2zData": compact_z2z_matrix(entry["z2z_data"]),
                        "tokens": entry["tokens"],
                        "descriptionHtml": entry.get("description_html", ""),
                    }
                )
                for source_id, entry in sources.items()
            }
            for sample_id, sources in demo_matrix.items()
        }
        script_data_js = f"""
            const demoMatrix = {json_module.dumps(demo_payload, separators=(',', ':'))};
            const demoLoadPromises = new Map();
            let currentSampleId = {json_module.dumps(initial_sample_id)};
            let currentSourceId = {json_module.dumps(initial_source_id)};
            function isDemoEntryLoaded(entry) {{
                return entry && Array.isArray(entry.z2zData);
            }}
            async function ensureDemoLoaded(sampleId, sourceId) {{
                const entry = demoMatrix[sampleId] && demoMatrix[sampleId][sourceId];
                if (!entry) return null;
                if (isDemoEntryLoaded(entry)) return entry;
                const cacheKey = sampleId + '\\0' + sourceId;
                if (demoLoadPromises.has(cacheKey)) return demoLoadPromises.get(cacheKey);
                const promise = fetch(entry.dataUrl)
                    .then(resp => {{
                        if (!resp.ok) throw new Error('Failed to load demo data');
                        return resp.json();
                    }})
                    .then(data => {{
                        entry.z2zData = data.z2zData || data.z2z;
                        entry.tokens = data.tokens;
                        delete entry.dataUrl;
                        demoLoadPromises.delete(cacheKey);
                        return entry;
                    }})
                    .catch(err => {{
                        demoLoadPromises.delete(cacheKey);
                        throw err;
                    }});
                demoLoadPromises.set(cacheKey, promise);
                return promise;
            }}
            function getCurrentDemo() {{
                return demoMatrix[currentSampleId][currentSourceId];
            }}
            let z2zData = isDemoEntryLoaded(getCurrentDemo()) ? getCurrentDemo().z2zData : null;
            let tokens = isDemoEntryLoaded(getCurrentDemo()) ? getCurrentDemo().tokens : null;
        """
        sample_selector_html = _build_demo_nav_html(
            sample_labels,
            demo_sources or [],
            initial_sample_id,
            initial_source_id,
        )
        init_demo = demo_matrix[initial_sample_id][initial_source_id]
        init_desc = init_demo.get("description_html", "") or (escaped_description or "")
        description_block = (
            f'<div id="visualization-description" class="visualization-description">{init_desc}</div>'
        )
    elif multi_sample:
        initial_sample_id = initial_sample_id or next(iter(demo_samples))
        demo_payload = {
            sid: {
                "_only": {
                    "z2zData": entry["z2z_data"],
                    "tokens": entry["tokens"],
                    "descriptionHtml": entry.get("description_html", ""),
                }
            }
            for sid, entry in demo_samples.items()
        }
        script_data_js = f"""
            const demoMatrix = {json_module.dumps(demo_payload)};
            let currentSampleId = {json_module.dumps(initial_sample_id)};
            let currentSourceId = "_only";
            function getCurrentDemo() {{
                return demoMatrix[currentSampleId][currentSourceId];
            }}
            let z2zData = getCurrentDemo().z2zData;
            let tokens = getCurrentDemo().tokens;
        """
        sample_selector_html = _build_sample_selector_html(demo_samples, initial_sample_id)
        init_desc = demo_samples[initial_sample_id].get("description_html", "") or (
            escaped_description or ""
        )
        description_block = (
            f'<div id="visualization-description" class="visualization-description">{init_desc}</div>'
        )
    else:
        script_data_js = f"""
            const z2zData = {z2z_json};
            const tokens = {tokens_json};
        """
        sample_selector_html = ""
        description_block = (
            f'<div class="visualization-description">{escaped_description}</div>'
            if escaped_description
            else ""
        )

    switch_sample_js = ""
    if multi_sample:
        switch_sample_js = """
            function resolveDemoSourceId() {
                const routeSel = document.getElementById('demo-route-select');
                const route = routeSel ? routeSel.value : 'layer_direct';
                if (route === 'layer_direct') {
                    const layerSel = document.getElementById('demo-layer-baseline-select');
                    return layerSel ? layerSel.value : currentSourceId;
                }
                const att = document.getElementById('demo-att-baseline-select')?.value;
                const mlp = document.getElementById('demo-mlp-baseline-select')?.value;
                for (const [sid, meta] of Object.entries(demoSourceCatalog)) {
                    if (meta.route === 'composed' && meta.att_baseline === att && meta.mlp_baseline === mlp) {
                        return sid;
                    }
                }
                return currentSourceId;
            }
            function updateComposedSummary() {
                const el = document.getElementById('demo-composed-summary');
                const routeSel = document.getElementById('demo-route-select');
                if (!el || !routeSel) return;
                if (routeSel.value !== 'composed') {
                    el.style.display = 'none';
                    return;
                }
                const sampleId = document.getElementById('demo-sample-select')?.value || currentSampleId;
                const sourceId = resolveDemoSourceId();
                const meta = demoSourceCatalog[sourceId] || {};
                const att = meta.att_label || document.getElementById('demo-att-baseline-select')?.value || '';
                const mlp = meta.mlp_label || document.getElementById('demo-mlp-baseline-select')?.value || '';
                const available = demoMatrix[sampleId] && demoMatrix[sampleId][sourceId];
                el.innerHTML = available
                    ? `Selected: <strong>ATT ${att}</strong> <span>×</span> <strong>MLP ${mlp}</strong>`
                    : 'No cached data for this ATT × MLP pair on the current sample.';
                el.style.display = '';
            }
            function onDemoRouteChange() {
                const route = document.getElementById('demo-route-select').value;
                const directGroup = document.getElementById('demo-direct-group');
                const composedGroup = document.getElementById('demo-composed-group');
                if (directGroup) directGroup.style.display = route === 'layer_direct' ? '' : 'none';
                if (composedGroup) composedGroup.style.display = route === 'composed' ? '' : 'none';
                updateComposedSummary();
                applyDemoSelection();
            }
            function refreshBaselineAvailability() {
                const sampleId = document.getElementById('demo-sample-select')?.value || currentSampleId;
                const available = new Set(Object.keys(demoMatrix[sampleId] || {}));
                const layerSel = document.getElementById('demo-layer-baseline-select');
                if (layerSel) {
                    for (const opt of layerSel.options) {
                        opt.disabled = !available.has(opt.value);
                    }
                }
                const attSel = document.getElementById('demo-att-baseline-select');
                const mlpSel = document.getElementById('demo-mlp-baseline-select');
                if (attSel && mlpSel) {
                    for (const mlpOpt of mlpSel.options) {
                        const att = attSel.value;
                        let found = false;
                        for (const [sid, meta] of Object.entries(demoSourceCatalog)) {
                            if (meta.route !== 'composed') continue;
                            if (meta.att_baseline === att && meta.mlp_baseline === mlpOpt.value && available.has(sid)) {
                                found = true;
                                break;
                            }
                        }
                        mlpOpt.disabled = !found;
                    }
                    for (const attOpt of attSel.options) {
                        let any = false;
                        for (const mlpOpt of mlpSel.options) {
                            if (mlpOpt.disabled) continue;
                            for (const [sid, meta] of Object.entries(demoSourceCatalog)) {
                                if (meta.route !== 'composed') continue;
                                if (meta.att_baseline === attOpt.value && meta.mlp_baseline === mlpOpt.value && available.has(sid)) {
                                    any = true;
                                    break;
                                }
                            }
                            if (any) break;
                        }
                        attOpt.disabled = !any;
                    }
                    if (mlpSel.selectedOptions[0]?.disabled) {
                        const first = Array.from(mlpSel.options).find(o => !o.disabled);
                        if (first) mlpSel.value = first.value;
                    }
                    if (attSel.selectedOptions[0]?.disabled) {
                        const first = Array.from(attSel.options).find(o => !o.disabled);
                        if (first) attSel.value = first.value;
                    }
                }
                updateComposedSummary();
            }
            function applyDemoSelection() {
                const sampleSel = document.getElementById('demo-sample-select');
                const sampleId = sampleSel ? sampleSel.value : currentSampleId;
                refreshBaselineAvailability();
                updateComposedSummary();
                const sourceId = resolveDemoSourceId();
                if (!demoMatrix[sampleId] || !demoMatrix[sampleId][sourceId]) return;
                if (sampleId === currentSampleId && sourceId === currentSourceId && isDemoEntryLoaded(getCurrentDemo())) return;
                const vizRoot = document.getElementById('z2z-layout-root');
                if (vizRoot) vizRoot.classList.add('z2z-demo-loading');
                ensureDemoLoaded(sampleId, sourceId).then(demo => {
                    if (!demo) return;
                    currentSampleId = sampleId;
                    currentSourceId = sourceId;
                    z2zData = demo.z2zData;
                    tokens = demo.tokens;
                    const descEl = document.getElementById('visualization-description');
                    if (descEl) descEl.innerHTML = demo.descriptionHtml;
                    selectedToken = null;
                    clickedLayer = null;
                    recalcLayoutMetrics();
                    clearVisualization();
                    buildVisualization(() => {
                        updateTokenAxisLabels();
                        requestAnimationFrame(() => {
                            updateCirclePositionsAndDrawPaths();
                            syncContainerHeight();
                            syncVizScrollCenter();
                        });
                    });
                }).catch(() => {
                    const descEl = document.getElementById('visualization-description');
                    if (descEl) descEl.insertAdjacentHTML('beforeend', '<p class="demo-load-error">Failed to load visualization data.</p>');
                }).finally(() => {
                    if (vizRoot) vizRoot.classList.remove('z2z-demo-loading');
                });
            }
            function switchSample(sampleId) {
                if (sampleId) {
                    const sampleSel = document.getElementById('demo-sample-select');
                    if (sampleSel) sampleSel.value = sampleId;
                    applyDemoSelection();
                }
            }
            document.addEventListener('DOMContentLoaded', () => {
                refreshBaselineAvailability();
                updateComposedSummary();
            });
        """

    if layout_mode == "rotate_90_cw":
        layout_root_class = "z2z-layout-root z2z-layout-rotate-90"
    else:
        layout_root_class = "z2z-layout-root z2z-layout-normal"

    z2z_layout_rotate90_js = "true" if layout_mode == "rotate_90_cw" else "false"

    selected_token_js = "null" if target_token is None else str(int(target_token))

    site_banner_block = demo_site_banner_html(logo_href=logo_href) if show_site_banner else ""
    # particles.js is reserved for the landing page (docs/index.html) only.
    site_particles_zone_open = ""
    site_particles_zone_close = ""
    site_particles_scripts = ""
    body_class = "lig-demo-embed" if embed_mode else "lig-site"
    class_attr = f' class="{body_class}"'
    html_class_attr = ' class="lig-iframe-embed"' if embed_mode else ""
    controls_panel_class = (
        "controls controls-panel is-collapsed"
        if embed_mode
        else "controls controls-panel is-expanded"
    )
    display_options_toggle_expanded = "false" if embed_mode else "true"
    display_options_body_inert = " inert" if embed_mode else ""

    html = f"""
    <!DOCTYPE html>
    <html lang="en"{html_class_attr}>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{page_title}</title>
        <meta name="description" content="{html_module.escape(seo_description)}">
{seo_head_extras_html(title=seo_title, description=seo_description, page_url=seo_page_url, indent="        ", include_json_ld=not embed_mode)}
        {favicon_link}
        {hero_head_assets}
        <style>
            html.lig-iframe-embed,
            html.lig-iframe-embed body {{
                min-height: 0;
                height: auto;
                overflow-x: auto;
                overflow-y: auto;
            }}
            body {{
                font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: {'8px 12px 16px' if embed_mode else '0 0 24px'};
                min-height: 0;
                {'background-color: transparent;' if embed_mode else ''}
            }}
            body.lig-demo-embed .z2z-page-header {{
                margin-top: 0;
            }}
            .contribution-wrapper {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 0;
                position: absolute;
            }}
            .token-label {{
                font-size: 14px;
                color: #4a5f63;
                white-space: nowrap;
                width: 120px;
                flex-shrink: 0;
                text-align: left;
                cursor: pointer;
                overflow: hidden;
                text-overflow: ellipsis;
            }}
            .token-label-num {{
                font-weight: 700;
                color: #7a9196;
            }}
            .token-label-text {{
                font-weight: 400;
                color: #94a3b8;
                font-size: 12px;
            }}
            .z2z-layout-root.z2z-layout-rotate-90 .token-label {{
                transform: rotate(-90deg);
                transform-origin: center center;
            }}
            .contribution-box {{
                display: inline-block;
                background-color: rgba(232, 232, 232, 0.25);
                border: 1px solid rgba(0, 0, 0, 0.15);
                border-radius: 8px;
                padding: 5px 10px;
                cursor: pointer;
                position: relative;
                overflow: visible; /* 円がはみ出しても表示 */
            }}
            .contribution-box:hover {{
                background-color: rgba(255, 0, 0, 0.12);
            }}
            .contribution-box.selected {{
                border: 2px solid #FF0000;
            }}
            .contribution-box.layer-focused {{
                border: 2px solid #FF0000;
            }}
            .contribution-bar {{
                display: flex;
                flex-direction: row;
                flex-wrap: nowrap;
                align-items: center;
                gap: 0;
            }}
            .contribution-circle-slot {{
                flex: 0 0 auto;
                box-sizing: border-box;
            }}
            .tooltip {{
                position: fixed;
                background-color: rgba(0, 0, 0, 0.9);
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
                white-space: nowrap;
                font-size: 12px;
                pointer-events: none;
                z-index: 999999;
                display: none;
            }}
            .contribution-item {{
                margin-left: 0;
            }}
            .layer-label {{
                position: absolute;
                top: -15px;
                left: 0px;
                font-weight: 700;
                font-size: 20px;
                color: #000;
                padding: 5px 10px;
                z-index: 11990;
                opacity: 1;
            }}
            .z2z-layout-root.z2z-layout-rotate-90 .layer-label {{
                transform: rotate(-90deg);
                transform-origin: center center;
            }}
            #tokenAxisLabelsTop,
            #tokenAxisLabelsBottom {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 0;
                pointer-events: none;
                z-index: 1000;
            }}
            .token-axis-label {{
                position: absolute;
                font-size: 12px;
                color: #333;
                white-space: nowrap;
                pointer-events: auto;
            }}
            .token-axis-label-top {{
                transform-origin: left bottom;
            }}
            .token-axis-label-bottom {{
                transform-origin: left top;
            }}
            .axis-role-caption {{
                position: absolute;
                pointer-events: none;
                z-index: 1001;
                line-height: 1.35;
            }}
            .axis-role-caption-target {{
                left: 0;
                font-size: 14px;
                font-weight: 700;
                color: #0f172a;
                text-align: left;
                white-space: normal;
                transform: rotate(-90deg);
                transform-origin: left center;
            }}
            .axis-role-caption-source {{
                font-size: 12px;
                font-weight: 700;
                color: #1e293b;
                letter-spacing: 0.01em;
                white-space: nowrap;
                text-align: center;
                transform: translateX(-50%);
                max-width: calc(100% - 16px);
            }}
            .z2z-layout-root.z2z-layout-rotate-90 .axis-role-caption-target {{
                transform: rotate(-90deg);
                transform-origin: left center;
            }}
            .z2z-layout-root.z2z-layout-rotate-90 .axis-role-caption-source {{
                transform: translateX(-50%);
            }}
            /* transformはインラインスタイルで制御するため、CSSでは削除 */
            .controls {{
                /* layout via .controls-panel */
            }}
            .visualization-title {{
                text-align: center;
                font-size: 28px;
                font-weight: bold;
                color: #333;
                margin: 20px 0;
                padding: 15px;
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
            .visualization-description {{
                text-align: left;
                font-size: 13px;
                color: #333;
                margin: 0 auto 15px auto;
                padding: 12px 18px;
                max-width: 900px;
                background-color: rgba(250, 250, 250, 0.95);
                border-radius: 6px;
                line-height: 1.6;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            }}
            .visualization-description p {{
                margin: 4px 0;
                color: #333;
            }}
            .visualization-description p:first-child {{
                margin-top: 0;
                margin-bottom: 6px;
            }}
            .visualization-description strong {{
                color: #000;
                font-weight: 600;
            }}
            .demo-lead {{
                margin: 0 0 12px 0;
                font-size: 15px;
            }}
            .demo-meta {{
                display: grid;
                grid-template-columns: auto 1fr;
                gap: 6px 14px;
                margin: 0 0 14px 0;
                font-size: 13px;
            }}
            .demo-meta dt {{
                margin: 0;
                color: #64748b;
                font-weight: 600;
            }}
            .demo-meta dd {{
                margin: 0;
                color: #0f172a;
                line-height: 1.5;
            }}
            .demo-sentence-label {{
                margin: 0 0 6px 0;
                font-size: 12px;
                font-weight: 600;
                color: #64748b;
                letter-spacing: 0.03em;
            }}
            .demo-sentence {{
                margin: 0 0 14px 0;
                padding: 12px 16px;
                background: #f1f5f9;
                border-left: 3px solid #3b82f6;
                border-radius: 0 8px 8px 0;
                font-size: 14px;
                line-height: 1.75;
                color: #1e293b;
            }}
            .demo-legend {{
                margin: 0;
                font-size: 12px;
                color: #475569;
                line-height: 1.65;
            }}
            .demo-legend-title {{
                margin: 0 0 6px 0;
                font-size: 13px;
                color: #334155;
            }}
            .demo-legend-intro {{
                margin: 0 0 10px 0;
            }}
            .demo-legend-list {{
                margin: 0 0 10px 0;
                padding-left: 1.2em;
            }}
            .demo-legend-list li {{
                margin-bottom: 4px;
            }}
            .demo-legend-list li strong {{
                color: #1e293b;
            }}
            .demo-legend-note {{
                margin: 0;
                font-size: 11px;
                color: #64748b;
                font-style: italic;
            }}
            .control-item {{
                margin-bottom: 10px;
            }}
            .control-item label {{
                display: flex;
                align-items: center;
                cursor: pointer;
                color: #334155;
            }}
            .control-item input[type="checkbox"] {{
                margin-right: 8px;
                cursor: pointer;
            }}
            .control-item input[type="range"] {{
                width: 100%;
                margin-top: 4px;
            }}
            .control-item--slider label {{
                display: block;
                font-size: 12px;
                color: #475569;
            }}
            .control-item--slider .slider-value {{
                float: right;
                font-weight: 600;
                color: #0f172a;
            }}
            .circle-size-curve-block {{
                width: 100%;
                max-width: 100%;
                box-sizing: border-box;
            }}
            .circle-size-curve-hint {{
                margin: 0 0 8px 0;
                font-size: 11px;
                color: #64748b;
                line-height: 1.4;
                width: 100%;
                max-width: 100%;
                box-sizing: border-box;
                overflow-wrap: break-word;
            }}
            .circle-size-curve-label {{
                display: block;
                font-size: 12px;
                font-weight: 600;
                color: #334155;
                margin-bottom: 4px;
            }}
            .circle-size-curve-wrap {{
                margin-bottom: 10px;
            }}
            .circle-size-curve-svg {{
                width: 100%;
                height: auto;
                display: block;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                background: #f8fafc;
                touch-action: none;
                user-select: none;
            }}
            .circle-size-curve-actions {{
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
            }}
            .curve-preset-btns {{
                display: flex;
                flex-wrap: wrap;
                gap: 4px;
                margin-bottom: 6px;
                width: 100%;
                box-sizing: border-box;
            }}
            .curve-preset-btn {{
                flex: 1 1 calc(33.333% - 4px);
                min-width: min(4.5rem, 100%);
                max-width: 100%;
                box-sizing: border-box;
                font-size: 10px;
                font-weight: 600;
                padding: 4px 6px;
                border-radius: 5px;
                border: 1px solid #cbd5e1;
                background: #ffffff;
                color: #334155;
                cursor: pointer;
                white-space: normal;
                line-height: 1.25;
                text-align: center;
                overflow-wrap: anywhere;
                transition: background 0.15s, color 0.15s, border-color 0.15s;
            }}
            .curve-preset-btn:hover {{
                border-color: #94a3b8;
                background: #f8fafc;
            }}
            .curve-preset-btn.is-active {{
                background: #0f172a;
                color: #ffffff;
                border-color: #0f172a;
            }}
            .control-reset-btn {{
                font-size: 12px;
                padding: 6px 12px;
                border-radius: 6px;
                border: 1px solid #cbd5e1;
                background: #ffffff;
                color: #334155;
                cursor: pointer;
                transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
            }}
            .control-reset-btn:hover {{
                background: #f1f5f9;
            }}
            .control-display-reset {{
                margin-top: 4px;
            }}
            .controls-panel-toggle {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                width: 100%;
                margin: 0;
                padding: 8px 12px;
                border: none;
                border-bottom: 1px solid #e2e8f0;
                background: #f8fafc;
                color: #0f172a;
                font-size: 12px;
                font-weight: 700;
                cursor: pointer;
                border-radius: 12px 12px 0 0;
                transition: border-radius 0.28s ease, border-color 0.28s ease, background 0.15s ease;
            }}
            .controls-panel-toggle:hover {{
                background: #f1f5f9;
            }}
            .controls-panel.is-collapsed .controls-panel-toggle {{
                border-bottom-color: transparent;
                border-radius: 12px;
            }}
            .controls-panel-body {{
                display: grid;
                grid-template-rows: 0fr;
                overflow: hidden;
                transition: grid-template-rows 0.28s ease;
            }}
            .controls-panel.is-expanded .controls-panel-body {{
                grid-template-rows: 1fr;
            }}
            .controls-panel-body-inner {{
                min-height: 0;
                overflow: hidden;
                opacity: 0;
                padding: 0 12px;
                visibility: hidden;
                transition: opacity 0.22s ease, padding 0.28s ease, visibility 0s linear 0.28s;
            }}
            .controls-panel.is-expanded .controls-panel-body-inner {{
                opacity: 1;
                padding: 10px 12px 12px;
                visibility: visible;
                transition: opacity 0.22s ease 0.04s, padding 0.28s ease, visibility 0s linear 0s;
            }}
            .controls-panel-body .controls-panel-heading {{
                display: none;
            }}
            .controls-panel-chevron {{
                font-size: 10px;
                color: #64748b;
                transition: transform 0.28s ease;
            }}
            .controls-panel.is-expanded .controls-panel-chevron {{
                transform: rotate(180deg);
            }}
            @media (prefers-reduced-motion: reduce) {{
                .controls-panel-body,
                .controls-panel-body-inner,
                .controls-panel-chevron,
                .controls-panel-toggle {{
                    transition: none !important;
                }}
                .controls-panel.is-collapsed .controls-panel-body-inner {{
                    visibility: hidden;
                    opacity: 0;
                }}
                .controls-panel.is-expanded .controls-panel-body-inner {{
                    visibility: visible;
                    opacity: 1;
                }}
            }}
            body.lig-demo-embed .z2z-viz-section,
            html.lig-iframe-embed .z2z-viz-section {{
                margin: 8px auto 0;
                width: 100%;
                max-width: 100%;
                padding-inline: max(8px, env(safe-area-inset-left, 0px)) max(8px, env(safe-area-inset-right, 0px));
            }}
            body.lig-demo-embed .controls-panel,
            html.lig-iframe-embed .controls-panel {{
                position: fixed;
                top: max(8px, env(safe-area-inset-top, 0px));
                right: max(8px, env(safe-area-inset-right, 0px));
                left: auto;
                grid-column: auto;
                grid-row: auto;
                justify-self: auto;
                align-self: auto;
                z-index: 10;
                width: min(220px, calc(100vw - 16px));
                max-width: calc(100vw - 16px);
                max-height: calc(100vh - 16px);
                margin: 0;
                font-size: 12px;
                box-shadow: 0 6px 20px rgba(15, 23, 42, 0.14);
            }}
            body.lig-demo-embed .controls-panel.is-expanded .controls-panel-body,
            html.lig-iframe-embed .controls-panel.is-expanded .controls-panel-body {{
                max-height: min(50vh, 380px);
            }}
            body.lig-demo-embed .controls-panel.is-expanded .controls-panel-body-inner,
            html.lig-iframe-embed .controls-panel.is-expanded .controls-panel-body-inner {{
                overflow-y: auto;
            }}
            body.lig-demo-embed .circle-size-curve-block,
            html.lig-iframe-embed .circle-size-curve-block {{
                max-width: 100%;
            }}
            body.lig-demo-embed .circle-size-curve-hint,
            html.lig-iframe-embed .circle-size-curve-hint {{
                font-size: 10px;
            }}
            body.lig-demo-embed .z2z-page-header,
            html.lig-iframe-embed .z2z-page-header {{
                position: sticky;
                top: 0;
                z-index: 10001;
                max-width: none;
                padding-right: max(8px, min(220px, calc(100vw - 16px)) + 12px);
                background: #fff;
            }}
            body.lig-demo-embed .demo-legend-list,
            html.lig-iframe-embed .demo-legend-list,
            body.lig-demo-embed .demo-legend-intro,
            html.lig-iframe-embed .demo-legend-intro {{
                display: none;
            }}
            body.lig-demo-embed .demo-sentence,
            html.lig-iframe-embed .demo-sentence {{
                font-size: 13px;
                padding: 8px 12px;
                margin-bottom: 8px;
            }}
            .z2z-page-header {{
                max-width: min(96vw, 52rem);
                margin: 0 auto 36px auto;
                padding: 16px 20px;
                background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
            }}
            .layer-label {{
                cursor: pointer;
                user-select: none;
                color: #000;
                opacity: 1;
            }}
            .z2z-viz-section {{
                display: grid;
                grid-template-columns: minmax(0, 1fr);
                grid-template-rows: auto;
                width: 100%;
                max-width: 100%;
                min-width: 0;
                margin: 24px auto 0 auto;
                padding-inline: max(12px, env(safe-area-inset-left, 0px)) max(12px, env(safe-area-inset-right, 0px));
                box-sizing: border-box;
                position: relative;
                overflow: visible;
            }}
            #z2z-layout-root.z2z-demo-loading {{
                opacity: 0.42;
                pointer-events: none;
                position: relative;
            }}
            #z2z-layout-root.z2z-demo-loading::after {{
                content: '';
                position: absolute;
                top: 42%;
                left: 50%;
                width: 36px;
                height: 36px;
                margin: -18px 0 0 -18px;
                border: 3px solid #e2e8f0;
                border-top-color: #0d9488;
                border-radius: 50%;
                animation: z2z-demo-spin 0.75s linear infinite;
                z-index: 20000;
                pointer-events: none;
            }}
            @keyframes z2z-demo-spin {{
                to {{ transform: rotate(360deg); }}
            }}
            .demo-load-error {{
                color: #b91c1c;
                margin-top: 8px;
                font-size: 14px;
            }}
            .z2z-viz-scroll {{
                display: flex;
                grid-column: 1;
                grid-row: 1;
                align-items: flex-start;
                position: relative;
                z-index: 1;
                width: 100%;
                max-width: 100%;
                min-width: 0;
                overflow-x: auto;
                overflow-y: visible;
                -webkit-overflow-scrolling: touch;
                scroll-padding-inline: max(12px, env(safe-area-inset-left, 0px)) max(12px, env(safe-area-inset-right, 0px));
            }}
            .z2z-viz-scroll-lead,
            .z2z-viz-scroll-trail {{
                flex: 1 0 0;
                min-width: 0;
                pointer-events: none;
                background: transparent;
            }}
            .z2z-viz-scroll-lead {{
                min-width: var(--z2z-left-gutter, 0px);
            }}
            .z2z-viz-scroll-body {{
                flex: 0 0 auto;
            }}
            #container.z2z-container--animate {{
                transition: width 0.42s cubic-bezier(0.4, 0, 0.2, 1);
            }}
            .duplicate-overall-container {{
                transition: opacity 0.35s ease, transform 0.42s cubic-bezier(0.4, 0, 0.2, 1);
                will-change: transform, opacity;
            }}
            .duplicate-overall-container.z2z-dup-entering,
            .duplicate-overall-container.z2z-dup-leaving {{
                opacity: 0;
                transform: translateX(28px);
            }}
            body.lig-site:not(.lig-demo-embed) .z2z-viz-section {{
                background: var(--lig-bg);
                margin-top: 0;
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }}
            .controls-panel {{
                grid-column: 1;
                grid-row: 1;
                justify-self: end;
                align-self: start;
                position: sticky;
                top: max(12px, env(safe-area-inset-top, 0px));
                z-index: 10;
                width: min(280px, calc(100vw - 24px));
                max-width: min(280px, calc(100% - 8px));
                margin: max(12px, env(safe-area-inset-top, 0px)) max(0px, env(safe-area-inset-right, 0px)) 0 0;
                box-sizing: border-box;
                max-height: calc(100vh - 24px);
                overflow: hidden;
                background: #ffffff;
                color: #0f172a;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
                padding: 0;
                font-size: 13px;
                pointer-events: auto;
            }}
            .controls-panel.is-expanded .controls-panel-body {{
                max-height: calc(100vh - 24px - 40px);
            }}
            .controls-panel.is-expanded .controls-panel-body-inner {{
                overflow-y: auto;
                scrollbar-gutter: stable;
            }}
            .controls-panel h3 {{
                margin: 0 0 10px 0;
                font-size: 13px;
                font-weight: 700;
                color: #0f172a;
            }}
            .controls-panel hr {{
                border: none;
                border-top: 1px solid #e2e8f0;
                margin: 12px 0;
            }}
            @media (prefers-color-scheme: dark) {{
                .token-label {{
                    color: #4a5f63;
                }}
                .token-label-num {{
                    color: #7a9196;
                }}
                .layer-label {{
                    color: #000;
                }}
                .controls-panel {{
                    background: #ffffff;
                    color: #0f172a;
                    border-color: #e2e8f0;
                }}
                .controls-panel h3 {{
                    color: #0f172a;
                }}
                .controls-panel .control-item label,
                .controls-panel .control-item--slider label {{
                    color: #334155;
                }}
                .controls-panel .control-item--slider .slider-value {{
                    color: #0f172a;
                }}
            }}
            .z2z-layout-root {{
                box-sizing: border-box;
                margin: 0 auto;
                flex: 0 1 auto;
                order: 1;
                min-width: min-content;
                max-width: none;
                overflow: visible;
            }}
            #container {{
                position: relative;
                isolation: isolate;
                overflow: visible;
                margin-left: auto;
                margin-right: auto;
                max-width: none;
                box-sizing: border-box;
            }}
            .z2z-layout-normal {{
                margin-left: auto;
                margin-right: auto;
                width: fit-content;
                max-width: none;
            }}
            .visualization-description {{
                margin-left: auto;
                margin-right: auto;
                max-width: min(96vw, 52rem);
            }}
            .controls {{
                margin-left: 0;
                margin-right: 0;
                width: auto;
            }}
            @media (max-width: 960px) {{
                .controls-panel {{
                    width: min(240px, calc(100vw - 24px));
                    max-width: min(240px, calc(100% - 8px));
                    font-size: 12px;
                }}
                .controls-panel.is-expanded .controls-panel-body {{
                    max-height: min(55vh, 420px);
                }}
                .curve-preset-btn {{
                    flex: 1 1 calc(50% - 4px);
                    min-width: min(5rem, calc(50% - 4px));
                }}
            }}
            @media (max-width: 520px) {{
                body.lig-demo-embed .controls-panel,
                html.lig-iframe-embed .controls-panel {{
                    width: min(220px, calc(100vw - 16px));
                    max-width: calc(100vw - 16px);
                }}
                body.lig-demo-embed .z2z-page-header,
                html.lig-iframe-embed .z2z-page-header {{
                    padding-right: max(8px, env(safe-area-inset-right, 0px));
                }}
                .curve-preset-btn {{
                    flex: 1 1 100%;
                    min-width: 100%;
                }}
            }}
            .z2z-layout-rotate-90 {{
                transform: rotate(90deg);
                transform-origin: center center;
                margin: 8vh auto 12vh auto;
                width: fit-content;
                max-width: none;
            }}
        </style>
    </head>
    <body{class_attr}>
        {site_particles_zone_open}{site_banner_block}
        <div class="z2z-page-header">
        {sample_selector_html}
        {f'<h2 class="visualization-title">{escaped_title}</h2>' if escaped_title else ''}
        {description_block}
        </div>{site_particles_zone_close}
        <div class="z2z-viz-section">
        <div class="z2z-viz-scroll" id="z2zVizScroll">
            <div class="z2z-viz-scroll-lead" aria-hidden="true"></div>
            <div class="z2z-viz-scroll-body">
        <div id="z2z-layout-root" class="{layout_root_class}">
        <div id="container" style="position: relative; padding-top: 100px; padding-left: 160px; margin-left: auto; margin-right: auto;">
            <svg id="pathSvg" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 11000;"></svg>
            <div id="targetTokenAxisCaption" class="axis-role-caption axis-role-caption-target" aria-hidden="true">Target token<br>(receives)</div>
            <div id="tokenAxisCaptionTop" class="axis-role-caption axis-role-caption-source" aria-hidden="true">Source token (contributes)</div>
            <div id="tokenAxisCaptionBottom" class="axis-role-caption axis-role-caption-source" aria-hidden="true">Source token (contributes)</div>
            <div id="tokenAxisLabelsTop"></div>
            <div id="tokenAxisLabelsBottom"></div>
        </div>
        </div>
            </div>
            <div class="z2z-viz-scroll-trail" aria-hidden="true"></div>
        </div>
        <div class="{controls_panel_class}" id="displayOptionsPanel">
            <button type="button" class="controls-panel-toggle" id="displayOptionsToggle" aria-expanded="{display_options_toggle_expanded}" aria-controls="displayOptionsBody">
                <span>Display options</span>
                <span class="controls-panel-chevron" aria-hidden="true">▼</span>
            </button>
            <div class="controls-panel-body" id="displayOptionsBody"{display_options_body_inert}>
            <div class="controls-panel-body-inner">
            <h3 class="controls-panel-heading">Display options</h3>
            <div class="control-item">
                <label>
                    <input type="checkbox" id="showPathsCheckbox" {('checked' if show_paths else '')}>
                    Show inter-layer contribution paths
                </label>
            </div>
            <div class="control-item" id="alignHorizontalControl" style="display: none;">
                <label>
                    <input type="checkbox" id="alignHorizontalCheckbox" {('checked' if align_horizontal else '')}>
                    Align token boxes within each layer
                </label>
            </div>
            <div class="control-item">
                <label class="circle-size-curve-label">Circle size mapping</label>
                <p class="circle-size-curve-hint" id="circleSizeCurveHint">Green = pivot at 100/N (drag ↕ to bend). Orange = left of pivot, blue = right — drag each handle independently. Dashed line marks 100/N.</p>
                <div class="circle-size-curve-block">
                <div class="curve-preset-btns" role="group" aria-label="Circle size mapping preset">
                    <button type="button" class="curve-preset-btn" data-curve-preset="linear" title="Diameter grows linearly with share">Linear</button>
                    <button type="button" class="curve-preset-btn" data-curve-preset="curve" title="Diameter grows with √share (area ∝ share)">Curve</button>
                    <button type="button" class="curve-preset-btn is-active" data-curve-preset="pivot" title="Bend the curve vertically at the 100/N pivot; orange/blue handles shape each side">Pivot</button>
                </div>
                <div class="circle-size-curve-wrap">
                    <svg id="circleSizeCurveSvg" class="circle-size-curve-svg" viewBox="0 0 280 170" aria-label="Circle size mapping curve"></svg>
                </div>
                </div>
            </div>
            <div class="control-item control-item--slider">
                <label>Min diameter (px) <span class="slider-value" id="circleSizeMinPxVal">0.1</span></label>
                <input type="range" id="circleSizeMinPx" min="0.1" max="8" step="0.1" value="0.1">
            </div>
            <div class="control-item circle-size-curve-actions">
                <button type="button" class="control-reset-btn" id="resetCircleSizeCurveBtn">Reset curve</button>
            </div>
            <hr>
            <h3>Layout</h3>
            <div class="control-item control-item--slider">
                <label>Token X offset (px) <span class="slider-value" id="tokenOffsetXVal">6.7</span></label>
                <input type="range" id="tokenOffsetX" min="0" max="20" step="0.5" value="6.7">
            </div>
            <div class="control-item control-item--slider">
                <label>Token Y offset (px) <span class="slider-value" id="tokenOffsetYVal">5</span></label>
                <input type="range" id="tokenOffsetY" min="0" max="30" step="1" value="5">
            </div>
            <div class="control-item control-item--slider">
                <label>Layer spacing (px) <span class="slider-value" id="layerSpacingVal">150</span></label>
                <input type="range" id="layerSpacing" min="80" max="250" step="5" value="150">
            </div>
            <div class="control-item control-display-reset">
                <button type="button" class="control-reset-btn" id="resetDisplayOptionsBtn">Reset display options</button>
            </div>
            </div>
            </div>
        </div>
        </div>
        
        <script>
            const z2zLayoutRotate90 = {z2z_layout_rotate90_js};
            const useSlotCap = {use_slot_cap_js};
            const isEmbedPage = {'true' if embed_mode else 'false'};
{script_data_js}
            const container = document.getElementById('container');
            function scheduleDeferred(fn, timeoutMs) {{
                if (typeof requestIdleCallback === 'function') {{
                    requestIdleCallback(fn, {{ timeout: timeoutMs || 400 }});
                }} else {{
                    setTimeout(fn, 0);
                }}
            }}
            function throttle(fn, waitMs) {{
                let timer = null;
                let pendingArgs = null;
                return function throttled() {{
                    pendingArgs = arguments;
                    if (timer !== null) return;
                    timer = setTimeout(() => {{
                        timer = null;
                        fn.apply(null, pendingArgs);
                        pendingArgs = null;
                    }}, waitMs);
                }};
            }}
            let globalTooltipEl = null;
            let circleTooltipEl = null;
            let updateSelectedToken = null;
            let circleTooltipHoverEl = null;
            let circleSizeChartRendered = false;
            let vizBuildGeneration = 0;
            const VIZ_LAYERS_PER_FRAME = 2;
            let layoutRecalcScheduled = false;

            function ensureVizTooltips() {{
                if (!globalTooltipEl) {{
                    globalTooltipEl = document.createElement('div');
                    globalTooltipEl.className = 'tooltip';
                    document.body.appendChild(globalTooltipEl);
                }}
                if (!circleTooltipEl) {{
                    circleTooltipEl = document.createElement('div');
                    circleTooltipEl.className = 'tooltip';
                    circleTooltipEl.style.backgroundColor = 'rgba(128, 128, 128, 0.9)';
                    circleTooltipEl.style.zIndex = '1000000';
                    document.body.appendChild(circleTooltipEl);
                }}
            }}

            function bindCircleTooltipDelegation() {{
                if (container.dataset.circleTooltipBound) return;
                container.dataset.circleTooltipBound = '1';
                container.addEventListener('mouseover', (e) => {{
                    let circle = e.target.closest('.contribution-circle');
                    if (!circle) {{
                        const slot = e.target.closest('.contribution-circle-slot');
                        if (slot) circle = slot.querySelector('.contribution-circle');
                    }}
                    if (!circle || !container.contains(circle)) return;
                    if (circle === circleTooltipHoverEl) return;
                    circleTooltipHoverEl = circle;
                    const idx = parseInt(circle.dataset.contributionTokenIdx, 10);
                    const pct = circle.dataset.contributionPct || '';
                    circleTooltipEl.textContent = `Token ${{idx}}: ${{tokenLabelAt(idx)}} (${{pct}}%)`;
                    const rect = circle.getBoundingClientRect();
                    circleTooltipEl.style.left = (rect.left + rect.width / 2) + 'px';
                    circleTooltipEl.style.top = (rect.top - 5) + 'px';
                    circleTooltipEl.style.transform = 'translate(-50%, -100%)';
                    circleTooltipEl.style.display = 'block';
                }});
                container.addEventListener('mouseout', (e) => {{
                    const related = e.relatedTarget;
                    if (related && (related.closest('.contribution-circle-slot') || related.closest('.contribution-circle'))) return;
                    circleTooltipHoverEl = null;
                    circleTooltipEl.style.display = 'none';
                }});
                container.addEventListener('mousemove', (e) => {{
                    if (!circleTooltipHoverEl || circleTooltipEl.style.display === 'none') return;
                    const rect = circleTooltipHoverEl.getBoundingClientRect();
                    circleTooltipEl.style.left = (rect.left + rect.width / 2) + 'px';
                    circleTooltipEl.style.top = (rect.top - 5) + 'px';
                }});
            }}

            function scheduleRecalcLayoutMetrics() {{
                if (layoutRecalcScheduled) return;
                layoutRecalcScheduled = true;
                requestAnimationFrame(() => {{
                    layoutRecalcScheduled = false;
                    recalcLayoutMetrics();
                }});
            }}

            function ensureCircleSizeCurveChart() {{
                if (circleSizeChartRendered) return;
                circleSizeChartRendered = true;
                renderCircleSizeCurveChart();
            }}
            const pathSvg = document.getElementById('pathSvg');
            const vizScrollEl = document.getElementById('z2zVizScroll');
            const vizScrollLeadEl = vizScrollEl ? vizScrollEl.querySelector('.z2z-viz-scroll-lead') : null;
            const vizScrollTrailEl = vizScrollEl ? vizScrollEl.querySelector('.z2z-viz-scroll-trail') : null;
            const vizScrollBodyEl = vizScrollEl ? vizScrollEl.querySelector('.z2z-viz-scroll-body') : null;
            const VIZ_LAYOUT_TRANSITION_MS = 420;
            let vizLayoutTransitionTimer = null;

            let numTokens, numTokensToShow, numLayersToShow, topLayer;
            let circleSlotSize, circleCenterOffset, circleMaxDiameter, maxCircleSizeFixed;
            let selectedToken = {selected_token_js};
            let clickedLayer = null;
            let showPaths = {str(show_paths).lower()};
            let alignHorizontal = {str(align_horizontal).lower()};

            let topLayerTokenPositions = [];
            let bottomLayerTokenPositions = [];
            let circlePositions = {{}};
            let duplicateCirclePositions = {{}};

            let layoutTokenOffsetX = 20 / 3;
            let layoutTokenOffsetY = 5;
            let layoutLayerSpacing = 150;
            const layoutTopLabelOffset = 75;
            const duplicatePanelTopOffset = 40;
            const duplicatePanelDisplayOptionsGap = 12;
            const layoutBottomLabelOffset = 12;
            const layoutContainerBottomMargin = 48;
            let layoutBaseTop = 140;
            let focusedLayer = null;
            const CIRCLE_SLOT_PX = 14;
            const PATH_SVG_Z_INDEX = 11000;
            const WRAPPER_Z_INDEX_BASE = 12000;
            const VIZ_INTERACTIVE_SELECTOR =
                '.contribution-wrapper, .contribution-box, .contribution-bar, .contribution-circle-slot, .contribution-circle, .layer-label, .token-axis-label, .token-label, .duplicate-overall-container, .duplicate-layer-container, .duplicate-axis-container';

            const CIRCLE_CURVE_GAMMA_DEFAULT = 0.35;
            const CIRCLE_CURVE_PRESET_DEFAULT = 'pivot';
            const CIRCLE_SIZE_FLOOR_PX = 0.1;
            const CIRCLE_SIZE_MIN_PX_DEFAULT = 0.1;
            const CIRCLE_SIZE_FOCUS_PX_DEFAULT = 8;
            const CIRCLE_SIZE_MAX_PX_DEFAULT = 42;
            const CIRCLE_CURVE_PRESETS = {{
                linear: {{ id: 'linear', label: 'Linear', gamma: 1 }},
                curve: {{ id: 'curve', label: 'Curve', gamma: CIRCLE_CURVE_GAMMA_DEFAULT }},
                pivot: {{ id: 'pivot', label: 'Pivot' }},
            }};
            const PIVOT_BEND_X_FRAC_DEFAULT = 0.38;
            const PIVOT_BEND_Y_FRAC_DEFAULT = 0.32;
            const PIVOT_BEND_HIGH_X_FRAC_DEFAULT = 0.38;
            const PIVOT_BEND_HIGH_Y_FRAC_DEFAULT = 0.35;
            const PIVOT_BEND_Y_NORM_MIN = -0.08;
            const PIVOT_BEND_Y_NORM_MAX = 1.55;
            const CURVE_FOCUS_CHART_T = 0.78;
            const CURVE_LOW_X_GAMMA = 0.55;
            const CURVE_CHART = {{ width: 280, height: 170, margin: {{ top: 14, right: 14, bottom: 36, left: 38 }} }};

            function getCurveFocusShare() {{
                const n = Math.max(2, getSequenceLength());
                return 1 / n;
            }}

            function shareToChartT(shareRatio) {{
                const focus = getCurveFocusShare();
                const s = Math.max(0, Math.min(1, shareRatio));
                if (focus <= 0 || focus >= 1) return s;
                if (s <= focus) {{
                    const u = focus > 0 ? s / focus : 0;
                    return Math.pow(u, CURVE_LOW_X_GAMMA) * CURVE_FOCUS_CHART_T;
                }}
                const u = (s - focus) / (1 - focus);
                return CURVE_FOCUS_CHART_T + u * (1 - CURVE_FOCUS_CHART_T);
            }}

            function chartTToShare(chartT) {{
                const focus = getCurveFocusShare();
                const t = Math.max(0, Math.min(1, chartT));
                if (focus <= 0 || focus >= 1) return t;
                if (t <= CURVE_FOCUS_CHART_T) {{
                    const u = CURVE_FOCUS_CHART_T > 0 ? t / CURVE_FOCUS_CHART_T : 0;
                    return Math.pow(u, 1 / CURVE_LOW_X_GAMMA) * focus;
                }}
                const u = (t - CURVE_FOCUS_CHART_T) / (1 - CURVE_FOCUS_CHART_T);
                return focus + u * (1 - focus);
            }}

            function getCurveChartTickTs() {{
                const low = [
                    0,
                    0.12 * CURVE_FOCUS_CHART_T,
                    0.28 * CURVE_FOCUS_CHART_T,
                    0.48 * CURVE_FOCUS_CHART_T,
                    0.68 * CURVE_FOCUS_CHART_T,
                    CURVE_FOCUS_CHART_T,
                ];
                const high = [
                    CURVE_FOCUS_CHART_T + 0.35 * (1 - CURVE_FOCUS_CHART_T),
                    CURVE_FOCUS_CHART_T + 0.7 * (1 - CURVE_FOCUS_CHART_T),
                    1,
                ];
                return [...new Set([...low, ...high])].sort((a, b) => a - b);
            }}

            function evalCircleCurveYFromGamma(x, gamma) {{
                const focus = getCurveFocusShare();
                const yFocus = getCircleSizeFocusNormY();
                const s = Math.max(0, Math.min(1, x));
                if (focus <= 0 || focus >= 1) {{
                    return s <= 0 ? 0 : Math.pow(s, gamma);
                }}
                if (s <= focus) {{
                    const t = focus > 0 ? s / focus : 0;
                    return yFocus * Math.pow(Math.max(0, Math.min(1, t)), gamma);
                }}
                const t = (s - focus) / (1 - focus);
                return yFocus + (1 - yFocus) * Math.pow(Math.max(0, Math.min(1, t)), gamma);
            }}

            function makeDefaultCircleSizeCurveAnchors() {{
                const focus = getCurveFocusShare();
                const anchors = [0];
                for (let i = 1; i <= 16; i++) {{
                    anchors.push(focus * (i / 16));
                }}
                [1.25, 1.75, 2.5, 4, 8, 16].forEach(mult => {{
                    const v = focus * mult;
                    if (v > focus && v < 1) anchors.push(v);
                }});
                [0.15, 0.35, 0.55, 0.75, 1.0].forEach(v => {{
                    if (v > focus) anchors.push(v);
                }});
                anchors.push(1);
                const uniq = [...new Set(anchors.map(v => +Math.max(0, Math.min(1, v)).toFixed(6)))];
                uniq.sort((a, b) => a - b);
                return uniq;
            }}

            function cloneCircleSizeCurve(points) {{
                return points.map(p => ({{ x: p.x, y: p.y }}));
            }}

            function getCircleSizeFocusNormY() {{
                const minD = circleSizeMinPx;
                const maxD = getCircleMaxDiameterPx();
                const span = maxD - minD;
                if (span <= 0) return 0;
                // Normalized curve height at 100/N; mapShareToDiameter maps that anchor to CIRCLE_SIZE_FOCUS_PX_DEFAULT.
                return Math.max(0, Math.min(1, (CIRCLE_SIZE_FOCUS_PX_DEFAULT - minD) / span));
            }}

            function makeCircleSizeCurveFromGamma(gamma) {{
                return makeDefaultCircleSizeCurveAnchors().map(x => ({{
                    x: x,
                    y: evalCircleCurveYFromGamma(x, gamma),
                }}));
            }}

            function makeDefaultCircleSizeCurve() {{
                return makeCircleSizeCurveFromGamma(CIRCLE_CURVE_PRESETS.curve.gamma);
            }}

            function clonePivotBendControls(controls) {{
                return {{
                    pivot: {{ xShare: controls.pivot.xShare, yNorm: controls.pivot.yNorm }},
                    low: {{ xShare: controls.low.xShare, yNorm: controls.low.yNorm }},
                    high: {{ xShare: controls.high.xShare, yNorm: controls.high.yNorm }},
                }};
            }}

            function makeDefaultPivotBendControls() {{
                const focus = getCurveFocusShare();
                const yFocus = getCircleSizeFocusNormY();
                return {{
                    pivot: {{
                        xShare: focus,
                        yNorm: yFocus,
                    }},
                    low: {{
                        xShare: focus * PIVOT_BEND_X_FRAC_DEFAULT,
                        yNorm: yFocus * PIVOT_BEND_Y_FRAC_DEFAULT,
                    }},
                    high: {{
                        xShare: focus + (1 - focus) * PIVOT_BEND_HIGH_X_FRAC_DEFAULT,
                        yNorm: yFocus + (1 - yFocus) * PIVOT_BEND_HIGH_Y_FRAC_DEFAULT,
                    }},
                }};
            }}

            function getPivotAnchorSharePoint() {{
                return {{
                    x: pivotBendControls.pivot.xShare,
                    y: pivotBendControls.pivot.yNorm,
                }};
            }}

            function getPivotLowControlSharePoint() {{
                return {{
                    x: pivotBendControls.low.xShare,
                    y: pivotBendControls.low.yNorm,
                }};
            }}

            function getPivotHighControlSharePoint() {{
                return {{
                    x: pivotBendControls.high.xShare,
                    y: pivotBendControls.high.yNorm,
                }};
            }}

            function clampPivotAnchor() {{
                const c = pivotBendControls.pivot;
                c.xShare = getCurveFocusShare();
                c.yNorm = Math.max(
                    PIVOT_BEND_Y_NORM_MIN,
                    Math.min(PIVOT_BEND_Y_NORM_MAX, c.yNorm));
            }}

            function clampPivotBendSide(side) {{
                const pivotX = pivotBendControls.pivot.xShare;
                const c = pivotBendControls[side];
                c.yNorm = Math.max(
                    PIVOT_BEND_Y_NORM_MIN,
                    Math.min(PIVOT_BEND_Y_NORM_MAX, c.yNorm));
                if (side === 'low') {{
                    c.xShare = Math.max(0, Math.min(Math.max(0, pivotX - 1e-6), c.xShare));
                }} else {{
                    c.xShare = Math.max(Math.min(1, pivotX + 1e-6), Math.min(1, c.xShare));
                }}
            }}

            function evalQuadraticBezierYAtShare(x, p0x, p0y, p1x, p1y, p2x, p2y) {{
                const target = Math.max(p0x, Math.min(p2x, x));
                if (Math.abs(p2x - p0x) < 1e-9) return p0y + (p2y - p0y) * (target - p0x) / (p2x - p0x || 1);
                let lo = 0;
                let hi = 1;
                for (let i = 0; i < 40; i++) {{
                    const t = (lo + hi) / 2;
                    const bx = (1 - t) * (1 - t) * p0x + 2 * (1 - t) * t * p1x + t * t * p2x;
                    if (bx < target) lo = t;
                    else hi = t;
                }}
                const t = (lo + hi) / 2;
                return (1 - t) * (1 - t) * p0y + 2 * (1 - t) * t * p1y + t * t * p2y;
            }}

            function evalPivotCurveY(x) {{
                const pivot = getPivotAnchorSharePoint();
                const s = Math.max(0, Math.min(1, x));
                if (pivot.x <= 0 || pivot.x >= 1) {{
                    return s <= 0 ? 0 : s;
                }}
                if (s <= pivot.x) {{
                    const c = getPivotLowControlSharePoint();
                    return evalQuadraticBezierYAtShare(s, 0, 0, c.x, c.y, pivot.x, pivot.y);
                }}
                const c = getPivotHighControlSharePoint();
                return evalQuadraticBezierYAtShare(s, pivot.x, pivot.y, c.x, c.y, 1, 1);
            }}

            let circleSizeMinPx = CIRCLE_SIZE_MIN_PX_DEFAULT;
            let circleSizeCurvePoints = makeDefaultCircleSizeCurve();
            let circleSizeCurvePreset = CIRCLE_CURVE_PRESET_DEFAULT;
            let pivotBendControls = makeDefaultPivotBendControls();
            let activeCurveDragIndex = null;
            let activePivotBendDrag = null;
            let curveDragAxis = null;
            let curveDragStartClient = null;
            const CURVE_DRAG_LOCK_PX = 5;
            const CURVE_HANDLE_HIT_RADIUS = 12;
            let curveChartRedrawPending = false;
            const displayOptionsDefaults = {{
                showPaths: {str(show_paths).lower()},
                alignHorizontal: {str(align_horizontal).lower()},
                circleSizeMinPx: CIRCLE_SIZE_MIN_PX_DEFAULT,
                circleSizeCurve: cloneCircleSizeCurve(makeDefaultCircleSizeCurve()),
                pivotBendControls: makeDefaultPivotBendControls(),
            }};

            function updateCircleSizeCurveHint() {{
                const hint = document.getElementById('circleSizeCurveHint');
                if (!hint) return;
                if (circleSizeCurvePreset === 'pivot') {{
                    hint.textContent = 'Green = pivot at 100/N (drag ↕ to bend). Orange = left of pivot, blue = right — drag each handle independently. Dashed line marks 100/N.';
                    return;
                }}
                hint.textContent = 'X: % of column max — left 2/3 expanded below ~100/N (dashed). Drag handles: ↔ contribution share, ↕ diameter (locks to your drag direction).';
            }}

            function updateCurvePresetButtons() {{
                document.querySelectorAll('[data-curve-preset]').forEach(btn => {{
                    const active = circleSizeCurvePreset !== null
                        && btn.dataset.curvePreset === circleSizeCurvePreset;
                    btn.classList.toggle('is-active', active);
                    btn.setAttribute('aria-pressed', active ? 'true' : 'false');
                }});
            }}

            function applyCircleSizeCurvePreset(presetId) {{
                const preset = CIRCLE_CURVE_PRESETS[presetId];
                if (!preset) return;
                circleSizeCurvePreset = presetId;
                if (presetId === 'pivot') {{
                    pivotBendControls = makeDefaultPivotBendControls();
                }} else {{
                    circleSizeCurvePoints = cloneCircleSizeCurve(
                        makeCircleSizeCurveFromGamma(preset.gamma));
                }}
                updateCurvePresetButtons();
                updateCircleSizeCurveHint();
                renderCircleSizeCurveChart();
                refreshAllCircleSizes();
            }}

            function markCircleSizeCurveCustom() {{
                circleSizeCurvePreset = null;
                updateCurvePresetButtons();
            }}

            function getCircleMaxDiameterPx() {{
                const maxD = useSlotCap ? circleMaxDiameter : maxCircleSizeFixed;
                return Number.isFinite(maxD) ? maxD : CIRCLE_SIZE_MAX_PX_DEFAULT;
            }}

            function getCurvePlotRect() {{
                const m = CURVE_CHART.margin;
                return {{
                    left: m.left,
                    top: m.top,
                    width: CURVE_CHART.width - m.left - m.right,
                    height: CURVE_CHART.height - m.top - m.bottom,
                }};
            }}

            function interpolateCircleCurve(x) {{
                if (circleSizeCurvePreset === 'curve') {{
                    return evalCircleCurveYFromGamma(x, CIRCLE_CURVE_PRESETS.curve.gamma);
                }}
                if (circleSizeCurvePreset === 'linear') {{
                    return evalCircleCurveYFromGamma(x, 1);
                }}
                if (circleSizeCurvePreset === 'pivot') {{
                    return evalPivotCurveY(x);
                }}
                const pts = circleSizeCurvePoints;
                if (!pts.length) return 0;
                if (x <= pts[0].x) return pts[0].y;
                if (x >= pts[pts.length - 1].x) return pts[pts.length - 1].y;
                for (let i = 0; i < pts.length - 1; i++) {{
                    const a = pts[i];
                    const b = pts[i + 1];
                    if (x >= a.x && x <= b.x) {{
                        const t = (x - a.x) / (b.x - a.x || 1);
                        return a.y + t * (b.y - a.y);
                    }}
                }}
                return 0;
            }}

            function buildCircleSizeCurvePathD() {{
                if (circleSizeCurvePreset === 'curve' || circleSizeCurvePreset === 'pivot') {{
                    const steps = 160;
                    const parts = [];
                    for (let i = 0; i <= steps; i++) {{
                        const chartT = i / steps;
                        const xShare = chartTToShare(chartT);
                        const px = curveChartTToPx(chartT);
                        const py = curveChartYFromDiameter(mapShareToDiameter(xShare));
                        parts.push((i === 0 ? 'M' : 'L') + px.toFixed(2) + ' ' + py.toFixed(2));
                    }}
                    return parts.join(' ');
                }}
                const pts = circleSizeCurvePoints;
                if (!pts.length) return '';
                return pts.map((pt, i) => {{
                    const px = curveChartX(pt.x);
                    const py = curveChartYFromDiameter(mapShareToDiameter(pt.x));
                    return (i === 0 ? 'M' : 'L') + px.toFixed(2) + ' ' + py.toFixed(2);
                }}).join(' ');
            }}

            function diameterToCurveNormY(shareRatio, diameterPx) {{
                const minD = circleSizeMinPx;
                const maxD = getCircleMaxDiameterPx();
                const focus = getCurveFocusShare();
                const focusD = CIRCLE_SIZE_FOCUS_PX_DEFAULT;
                const s = Math.max(0, Math.min(1, shareRatio));
                const d = Math.max(minD, Math.min(maxD, diameterPx));
                if (focus <= 0 || focus >= 1 || maxD <= minD) {{
                    return (d - minD) / (maxD - minD || 1);
                }}
                const yAtFocus = interpolateCircleCurve(focus);
                let shape;
                if (s <= focus) {{
                    const spanBelow = focusD - minD;
                    shape = spanBelow > 0 ? (d - minD) / spanBelow : 0;
                    return yAtFocus * Math.max(0, Math.min(1, shape));
                }}
                const spanAbove = maxD - focusD;
                shape = spanAbove > 0 ? (d - focusD) / spanAbove : 0;
                return yAtFocus + (1 - yAtFocus) * Math.max(0, Math.min(1, shape));
            }}

            function mapShareToDiameter(shareRatio) {{
                const minD = circleSizeMinPx;
                const maxD = getCircleMaxDiameterPx();
                const focus = getCurveFocusShare();
                const focusD = CIRCLE_SIZE_FOCUS_PX_DEFAULT;
                const s = Math.max(0, Math.min(1, shareRatio));
                const yNorm = interpolateCircleCurve(s);
                let diameter;
                if (focus <= 0 || focus >= 1 || maxD <= minD) {{
                    diameter = minD + (maxD - minD) * yNorm;
                }} else {{
                    const yAtFocus = interpolateCircleCurve(focus);
                    if (s <= focus) {{
                        const shape = yAtFocus > 0 ? yNorm / yAtFocus : (focus > 0 ? s / focus : 0);
                        diameter = minD + (focusD - minD) * Math.max(0, Math.min(1, shape));
                    }} else {{
                        const spanAbove = 1 - yAtFocus;
                        const shape = spanAbove > 0
                            ? (yNorm - yAtFocus) / spanAbove
                            : ((s - focus) / (1 - focus));
                        diameter = focusD + (maxD - focusD) * Math.max(0, Math.min(1, shape));
                    }}
                }}
                if (useSlotCap) {{
                    diameter = Math.min(diameter, circleMaxDiameter);
                }}
                return diameter;
            }}

            const PATH_WIDTH_MAX_MAIN = 6;
            const PATH_WIDTH_MAX_DUP = 6;
            const PATH_WIDTH_GAMMA = 2;
            // 低〜中貢献度は n^2 に近いまま、高貢献度だけ頭打ちにして太線を抑える
            const PATH_WIDTH_HIGH_COMPRESSION = 0.38;

            function contributionToPathWidth(normalizedValue, minLineWidth, maxLineWidth) {{
                const n = Math.max(0, Math.min(1, normalizedValue));
                const nPow = Math.pow(n, PATH_WIDTH_GAMMA);
                const scale = nPow * (1 - PATH_WIDTH_HIGH_COMPRESSION * nPow);
                return minLineWidth + scale * (maxLineWidth - minLineWidth);
            }}

            function formatCircleSizeMinPx(v) {{
                const n = parseFloat(v);
                if (!Number.isFinite(n)) return String(CIRCLE_SIZE_FLOOR_PX);
                const rounded = Math.round(n * 10) / 10;
                return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
            }}

            function applyCircleDiameter(circle, diameter) {{
                const d = Math.max(CIRCLE_SIZE_FLOOR_PX, Number.isFinite(diameter) ? diameter : CIRCLE_SIZE_FLOOR_PX);
                circle.style.width = d + 'px';
                circle.style.height = d + 'px';
                circle.style.minWidth = d + 'px';
                circle.style.minHeight = d + 'px';
                circle.style.maxWidth = d + 'px';
                circle.style.maxHeight = d + 'px';
                circle.style.left = (circleCenterOffset - d / 2) + 'px';
                circle.style.top = (circleCenterOffset - d / 2) + 'px';
            }}

            function updateCircleDiametersFromCurve() {{
                document.querySelectorAll('.contribution-circle').forEach(circle => {{
                    const shareRatio = parseFloat(circle.dataset.shareRatio || '0');
                    const diameter = mapShareToDiameter(shareRatio);
                    applyCircleDiameter(circle, diameter);
                    const layer = circle.dataset.layerIdx;
                    const tokenIdx = circle.dataset.tokenIdx;
                    const srcIdx = circle.dataset.contributionTokenIdx;
                    if (circlePositions[layer] && circlePositions[layer][tokenIdx] && circlePositions[layer][tokenIdx][srcIdx]) {{
                        circlePositions[layer][tokenIdx][srcIdx].size = diameter;
                    }}
                    if (duplicateCirclePositions[layer] && duplicateCirclePositions[layer][tokenIdx] && duplicateCirclePositions[layer][tokenIdx][srcIdx]) {{
                        duplicateCirclePositions[layer][tokenIdx][srcIdx].size = diameter;
                    }}
                }});
            }}

            function refreshAllCircleSizes() {{
                updateCircleDiametersFromCurve();
                updateCirclePositionsAndDrawPaths();
            }}

            function curveChartX(shareRatio) {{
                const plot = getCurvePlotRect();
                return plot.left + shareToChartT(shareRatio) * plot.width;
            }}

            function curveChartTToPx(chartT) {{
                const plot = getCurvePlotRect();
                return plot.left + chartT * plot.width;
            }}

            function curveClientToChartT(clientX) {{
                const svg = document.getElementById('circleSizeCurveSvg');
                if (!svg) return 0;
                const rect = svg.getBoundingClientRect();
                const sx = (clientX - rect.left) / rect.width * CURVE_CHART.width;
                const plot = getCurvePlotRect();
                return Math.max(0, Math.min(1, (sx - plot.left) / plot.width));
            }}

            function curveChartYFromDiameter(diameterPx) {{
                const plot = getCurvePlotRect();
                const minD = circleSizeMinPx;
                const maxD = getCircleMaxDiameterPx();
                const t = (diameterPx - minD) / (maxD - minD || 1);
                return plot.top + (1 - t) * plot.height;
            }}

            function curveChartYFromNorm(yNorm) {{
                const minD = circleSizeMinPx;
                const maxD = getCircleMaxDiameterPx();
                return curveChartYFromDiameter(minD + (maxD - minD) * yNorm);
            }}

            function curveClientToDiameter(clientY) {{
                const svg = document.getElementById('circleSizeCurveSvg');
                if (!svg) return circleSizeMinPx;
                const rect = svg.getBoundingClientRect();
                const sy = (clientY - rect.top) / rect.height * CURVE_CHART.height;
                const plot = getCurvePlotRect();
                const minD = circleSizeMinPx;
                const maxD = getCircleMaxDiameterPx();
                return maxD - ((sy - plot.top) / plot.height) * (maxD - minD);
            }}

            function curveClientToNormY(clientY) {{
                const minD = circleSizeMinPx;
                const maxD = getCircleMaxDiameterPx();
                const diameterPx = curveClientToDiameter(clientY);
                return (diameterPx - minD) / (maxD - minD || 1);
            }}

            function clampCurvePointY(index, yNorm) {{
                return Math.max(0, Math.min(1, yNorm));
            }}

            function clampCurvePointX(index, xShare) {{
                const eps = 1e-5;
                const minX = index > 0 ? circleSizeCurvePoints[index - 1].x + eps : 0;
                const maxX = index < circleSizeCurvePoints.length - 1 ? circleSizeCurvePoints[index + 1].x - eps : 1;
                return Math.max(minX, Math.min(maxX, xShare));
            }}

            function formatShareAxisLabel(shareRatio) {{
                const pct = shareRatio * 100;
                const n = Math.max(2, getSequenceLength());
                if (Math.abs(shareRatio - 1 / n) < 1 / n * 0.08) {{
                    return '100/' + n;
                }}
                if (pct < 1) return pct.toFixed(1);
                if (pct < 10) return pct.toFixed(1);
                return String(Math.round(pct));
            }}

            function renderCircleSizeCurveChart() {{
                if (circleSizeCurvePreset === 'pivot') {{
                    clampPivotAnchor();
                    clampPivotBendSide('low');
                    clampPivotBendSide('high');
                }}
                const svg = document.getElementById('circleSizeCurveSvg');
                if (!svg) return;
                circleSizeChartRendered = true;
                const plot = getCurvePlotRect();
                const minD = circleSizeMinPx;
                const maxD = getCircleMaxDiameterPx();
                const ns = 'http://www.w3.org/2000/svg';
                svg.innerHTML = '';

                const axis = document.createElementNS(ns, 'g');
                axis.setAttribute('stroke', '#94a3b8');
                axis.setAttribute('stroke-width', '1');
                axis.setAttribute('fill', 'none');

                const xAxis = document.createElementNS(ns, 'line');
                xAxis.setAttribute('x1', String(plot.left));
                xAxis.setAttribute('y1', String(plot.top + plot.height));
                xAxis.setAttribute('x2', String(plot.left + plot.width));
                xAxis.setAttribute('y2', String(plot.top + plot.height));
                axis.appendChild(xAxis);

                const yAxis = document.createElementNS(ns, 'line');
                yAxis.setAttribute('x1', String(plot.left));
                yAxis.setAttribute('y1', String(plot.top));
                yAxis.setAttribute('x2', String(plot.left));
                yAxis.setAttribute('y2', String(plot.top + plot.height));
                axis.appendChild(yAxis);
                svg.appendChild(axis);

                const labelStyle = 'font-size:10px;fill:#64748b;font-family:system-ui,sans-serif';
                const xLabel = document.createElementNS(ns, 'text');
                xLabel.setAttribute('x', String(plot.left + plot.width / 2));
                xLabel.setAttribute('y', String(CURVE_CHART.height - 8));
                xLabel.setAttribute('text-anchor', 'middle');
                xLabel.setAttribute('style', labelStyle);
                xLabel.textContent = '% of column max (expanded below ~100/N)';
                svg.appendChild(xLabel);

                const yLabel = document.createElementNS(ns, 'text');
                yLabel.setAttribute('x', String(10));
                yLabel.setAttribute('y', String(plot.top + plot.height / 2));
                yLabel.setAttribute('text-anchor', 'middle');
                yLabel.setAttribute('transform', `rotate(-90 10 ${{plot.top + plot.height / 2}})`);
                yLabel.setAttribute('style', labelStyle);
                yLabel.textContent = 'Diameter (px)';
                svg.appendChild(yLabel);

                const focusShare = getCurveFocusShare();
                const focusLineX = curveChartTToPx(CURVE_FOCUS_CHART_T);
                const focusGuide = document.createElementNS(ns, 'line');
                focusGuide.setAttribute('x1', String(focusLineX));
                focusGuide.setAttribute('y1', String(plot.top));
                focusGuide.setAttribute('x2', String(focusLineX));
                focusGuide.setAttribute('y2', String(plot.top + plot.height));
                focusGuide.setAttribute('stroke', '#0d9488');
                focusGuide.setAttribute('stroke-width', '1');
                focusGuide.setAttribute('stroke-dasharray', '4 3');
                focusGuide.setAttribute('opacity', '0.55');
                svg.appendChild(focusGuide);

                const focusTag = document.createElementNS(ns, 'text');
                focusTag.setAttribute('x', String(focusLineX));
                focusTag.setAttribute('y', String(plot.top - 4));
                focusTag.setAttribute('text-anchor', 'middle');
                focusTag.setAttribute('style', labelStyle + ';fill:#0f766e;font-weight:600');
                const nTok = Math.max(2, getSequenceLength());
                focusTag.textContent = '100/' + nTok + ' (' + (focusShare * 100).toFixed(1) + '%)';
                svg.appendChild(focusTag);

                getCurveChartTickTs().forEach(t => {{
                    const share = chartTToShare(t);
                    const x = curveChartTToPx(t);
                    const inLowRegion = t <= CURVE_FOCUS_CHART_T + 1e-6;
                    const tickLen = inLowRegion ? 5 : 4;
                    const tick = document.createElementNS(ns, 'line');
                    tick.setAttribute('x1', String(x));
                    tick.setAttribute('y1', String(plot.top + plot.height));
                    tick.setAttribute('x2', String(x));
                    tick.setAttribute('y2', String(plot.top + plot.height + tickLen));
                    tick.setAttribute('stroke', '#94a3b8');
                    tick.setAttribute('stroke-width', inLowRegion ? '1.25' : '1');
                    svg.appendChild(tick);
                    const txt = document.createElementNS(ns, 'text');
                    txt.setAttribute('x', String(x));
                    txt.setAttribute('y', String(plot.top + plot.height + 14));
                    txt.setAttribute('text-anchor', 'middle');
                    txt.setAttribute('style', labelStyle + (inLowRegion ? ';font-size:9px' : ''));
                    txt.textContent = formatShareAxisLabel(share);
                    svg.appendChild(txt);
                }});

                const focusD = CIRCLE_SIZE_FOCUS_PX_DEFAULT;
                [...new Set([minD, focusD, maxD])]
                    .filter(d => d >= minD && d <= maxD)
                    .sort((a, b) => a - b)
                    .forEach(d => {{
                    const y = curveChartYFromDiameter(d);
                    const isFocusD = Math.abs(d - focusD) < 0.05;
                    const tick = document.createElementNS(ns, 'line');
                    tick.setAttribute('x1', String(plot.left - 4));
                    tick.setAttribute('y1', String(y));
                    tick.setAttribute('x2', String(plot.left));
                    tick.setAttribute('y2', String(y));
                    tick.setAttribute('stroke', isFocusD ? '#0d9488' : '#94a3b8');
                    tick.setAttribute('stroke-width', isFocusD ? '1.5' : '1');
                    svg.appendChild(tick);
                    const txt = document.createElementNS(ns, 'text');
                    txt.setAttribute('x', String(plot.left - 6));
                    txt.setAttribute('y', String(y + 3));
                    txt.setAttribute('text-anchor', 'end');
                    txt.setAttribute('style', labelStyle + (isFocusD ? ';fill:#0f766e;font-weight:600' : ''));
                    txt.textContent = formatCircleSizeMinPx(d);
                    svg.appendChild(txt);
                }});

                const path = document.createElementNS(ns, 'path');
                path.setAttribute('d', buildCircleSizeCurvePathD());
                path.setAttribute('fill', 'none');
                path.setAttribute('stroke', '#2563eb');
                path.setAttribute('stroke-width', '2');
                path.setAttribute('stroke-linecap', 'round');
                svg.appendChild(path);

                if (circleSizeCurvePreset === 'pivot') {{
                    const pivot = getPivotAnchorSharePoint();
                    const pivotPx = curveChartX(pivot.x);
                    const pivotPy = curveChartYFromNorm(pivot.y);
                    const bendLow = getPivotLowControlSharePoint();
                    const bendLowPx = curveChartX(bendLow.x);
                    const bendLowPy = curveChartYFromNorm(bendLow.y);
                    const bendHigh = getPivotHighControlSharePoint();
                    const bendHighPx = curveChartX(bendHigh.x);
                    const bendHighPy = curveChartYFromNorm(bendHigh.y);

                    const guideStyle = 'stroke:#cbd5e1;stroke-width:1;stroke-dasharray:3 3;fill:none;opacity:0.85';
                    const lowGuide = document.createElementNS(ns, 'path');
                    lowGuide.setAttribute('d', `M ${{curveChartX(0)}} ${{curveChartYFromNorm(0)}} L ${{bendLowPx}} ${{bendLowPy}} L ${{pivotPx}} ${{pivotPy}}`);
                    lowGuide.setAttribute('style', guideStyle);
                    svg.appendChild(lowGuide);
                    const highGuide = document.createElementNS(ns, 'path');
                    highGuide.setAttribute('d', `M ${{pivotPx}} ${{pivotPy}} L ${{bendHighPx}} ${{bendHighPy}} L ${{curveChartX(1)}} ${{curveChartYFromNorm(1)}}`);
                    highGuide.setAttribute('style', guideStyle);
                    svg.appendChild(highGuide);

                    [
                        {{ side: 'pivot', px: pivotPx, py: pivotPy, fill: '#0f766e', r: 5 }},
                        {{ side: 'low', px: bendLowPx, py: bendLowPy, fill: '#ea580c', r: 6 }},
                        {{ side: 'high', px: bendHighPx, py: bendHighPy, fill: '#2563eb', r: 6 }},
                    ].forEach(({{ side, px, py, fill, r }}) => {{
                        const bendGroup = document.createElementNS(ns, 'g');
                        bendGroup.dataset.pivotBend = side;
                        const bendHit = document.createElementNS(ns, 'circle');
                        bendHit.setAttribute('cx', String(px));
                        bendHit.setAttribute('cy', String(py));
                        bendHit.setAttribute('r', String(CURVE_HANDLE_HIT_RADIUS));
                        bendHit.setAttribute('fill', 'transparent');
                        bendHit.setAttribute('stroke', 'none');
                        bendHit.style.cursor = side === 'pivot' ? 'ns-resize' : 'grab';
                        bendGroup.appendChild(bendHit);
                        const bendHandle = document.createElementNS(ns, 'circle');
                        bendHandle.setAttribute('cx', String(px));
                        bendHandle.setAttribute('cy', String(py));
                        bendHandle.setAttribute('r', String(r));
                        bendHandle.setAttribute('fill', fill);
                        bendHandle.setAttribute('stroke', '#ffffff');
                        bendHandle.setAttribute('stroke-width', '1.5');
                        bendHandle.style.cursor = side === 'pivot' ? 'ns-resize' : 'grab';
                        bendGroup.appendChild(bendHandle);
                        svg.appendChild(bendGroup);
                    }});
                    return;
                }}

                circleSizeCurvePoints.forEach((pt, index) => {{
                    const cx = curveChartX(pt.x);
                    const cy = curveChartYFromDiameter(mapShareToDiameter(pt.x));
                    const isEndpoint = index === 0 || index === circleSizeCurvePoints.length - 1;
                    const group = document.createElementNS(ns, 'g');
                    group.dataset.curveIndex = String(index);
                    if (!isEndpoint) {{
                        const hit = document.createElementNS(ns, 'circle');
                        hit.setAttribute('cx', String(cx));
                        hit.setAttribute('cy', String(cy));
                        hit.setAttribute('r', String(CURVE_HANDLE_HIT_RADIUS));
                        hit.setAttribute('fill', 'transparent');
                        hit.setAttribute('stroke', 'none');
                        hit.style.cursor = 'grab';
                        group.appendChild(hit);
                    }}
                    const handle = document.createElementNS(ns, 'circle');
                    handle.setAttribute('cx', String(cx));
                    handle.setAttribute('cy', String(cy));
                    handle.setAttribute('r', isEndpoint ? '4' : '5.5');
                    handle.setAttribute('fill', isEndpoint ? '#64748b' : '#2563eb');
                    handle.setAttribute('stroke', '#ffffff');
                    handle.setAttribute('stroke-width', '1.5');
                    handle.setAttribute('pointer-events', isEndpoint ? 'none' : 'all');
                    if (!isEndpoint) {{
                        handle.style.cursor = 'grab';
                    }}
                    group.appendChild(handle);
                    svg.appendChild(group);
                }});
            }}

            function scheduleCircleSizeCurveRedraw() {{
                if (curveChartRedrawPending) return;
                curveChartRedrawPending = true;
                requestAnimationFrame(() => {{
                    curveChartRedrawPending = false;
                    renderCircleSizeCurveChart();
                }});
            }}

            function bindCircleSizeCurveChart() {{
                const svg = document.getElementById('circleSizeCurveSvg');
                if (!svg) return;

                const pickHandle = (evt) => {{
                    let el = evt.target;
                    while (el && el !== svg) {{
                        const bendSide = el.dataset && el.dataset.pivotBend;
                        if (bendSide === 'pivot' || bendSide === 'low' || bendSide === 'high') {{
                            return bendSide;
                        }}
                        if (el.dataset && el.dataset.curveIndex !== undefined) {{
                            const index = parseInt(el.dataset.curveIndex, 10);
                            if (index <= 0 || index >= circleSizeCurvePoints.length - 1) return null;
                            return index;
                        }}
                        el = el.parentNode;
                    }}
                    return null;
                }};

                const onDown = (evt) => {{
                    const picked = pickHandle(evt);
                    if (picked === null) return;
                    evt.preventDefault();
                    const clientX = evt.touches ? evt.touches[0].clientX : evt.clientX;
                    const clientY = evt.touches ? evt.touches[0].clientY : evt.clientY;
                    if (picked === 'pivot' || picked === 'low' || picked === 'high') {{
                        activePivotBendDrag = picked;
                        activeCurveDragIndex = null;
                    }} else {{
                        activePivotBendDrag = null;
                        activeCurveDragIndex = picked;
                    }}
                    curveDragAxis = null;
                    curveDragStartClient = {{ x: clientX, y: clientY }};
                    svg.style.cursor = 'grabbing';
                }};

                const onMove = (evt) => {{
                    if (activePivotBendDrag === 'pivot' || activePivotBendDrag === 'low' || activePivotBendDrag === 'high') {{
                        const clientX = evt.touches ? evt.touches[0].clientX : evt.clientX;
                        const clientY = evt.touches ? evt.touches[0].clientY : evt.clientY;
                        if (activePivotBendDrag === 'pivot') {{
                            const pivot = pivotBendControls.pivot;
                            pivot.yNorm = curveClientToNormY(clientY);
                            clampPivotAnchor();
                            clampPivotBendSide('low');
                            clampPivotBendSide('high');
                        }} else {{
                            const side = pivotBendControls[activePivotBendDrag];
                            side.xShare = chartTToShare(curveClientToChartT(clientX));
                            side.yNorm = curveClientToNormY(clientY);
                            clampPivotBendSide(activePivotBendDrag);
                        }}
                        scheduleCircleSizeCurveRedraw();
                        updateCircleDiametersFromCurve();
                        return;
                    }}
                    if (activeCurveDragIndex === null) return;
                    const clientX = evt.touches ? evt.touches[0].clientX : evt.clientX;
                    const clientY = evt.touches ? evt.touches[0].clientY : evt.clientY;
                    if (curveDragAxis === null && curveDragStartClient) {{
                        const dx = clientX - curveDragStartClient.x;
                        const dy = clientY - curveDragStartClient.y;
                        if (Math.hypot(dx, dy) < CURVE_DRAG_LOCK_PX) return;
                        curveDragAxis = Math.abs(dx) >= Math.abs(dy) ? 'x' : 'y';
                    }}
                    const index = activeCurveDragIndex;
                    if (curveDragAxis === 'x') {{
                        const chartT = curveClientToChartT(clientX);
                        circleSizeCurvePoints[index].x = clampCurvePointX(index, chartTToShare(chartT));
                    }} else if (curveDragAxis === 'y') {{
                        const xShare = circleSizeCurvePoints[index].x;
                        const diameterPx = curveClientToDiameter(clientY);
                        circleSizeCurvePoints[index].y = clampCurvePointY(
                            index, diameterToCurveNormY(xShare, diameterPx));
                    }}
                    markCircleSizeCurveCustom();
                    scheduleCircleSizeCurveRedraw();
                    updateCircleDiametersFromCurve();
                }};

                const onUp = () => {{
                    if (activeCurveDragIndex !== null || activePivotBendDrag !== null) {{
                        updateCirclePositionsAndDrawPaths();
                    }}
                    activeCurveDragIndex = null;
                    activePivotBendDrag = null;
                    curveDragAxis = null;
                    curveDragStartClient = null;
                    svg.style.cursor = '';
                }};

                svg.addEventListener('mousedown', onDown);
                svg.addEventListener('touchstart', onDown, {{ passive: false }});
                window.addEventListener('mousemove', onMove);
                window.addEventListener('touchmove', onMove, {{ passive: false }});
                window.addEventListener('mouseup', onUp);
                window.addEventListener('touchend', onUp);
            }}

            function resetCircleSizeCurve() {{
                applyCircleSizeCurvePreset(circleSizeCurvePreset || CIRCLE_CURVE_PRESET_DEFAULT);
            }}

            function resetDisplayOptions() {{
                showPaths = displayOptionsDefaults.showPaths;
                alignHorizontal = displayOptionsDefaults.alignHorizontal;
                showPathsCheckbox.checked = showPaths;
                alignHorizontalCheckbox.checked = alignHorizontal;
                alignHorizontalControl.style.display = showPaths ? 'block' : 'none';
                circleSizeMinPx = displayOptionsDefaults.circleSizeMinPx;
                const minSlider = document.getElementById('circleSizeMinPx');
                const minVal = document.getElementById('circleSizeMinPxVal');
                if (minSlider) minSlider.value = String(circleSizeMinPx);
                if (minVal) minVal.textContent = formatCircleSizeMinPx(circleSizeMinPx);
                pivotBendControls = clonePivotBendControls(displayOptionsDefaults.pivotBendControls);
                applyCircleSizeCurvePreset(CIRCLE_CURVE_PRESET_DEFAULT);
            }}

            function bindDisplayOptionsPanelToggle() {{
                const panel = document.getElementById('displayOptionsPanel');
                const toggle = document.getElementById('displayOptionsToggle');
                const body = document.getElementById('displayOptionsBody');
                if (!panel || !toggle) return;
                const setExpanded = (expanded) => {{
                    panel.classList.toggle('is-collapsed', !expanded);
                    panel.classList.toggle('is-expanded', expanded);
                    toggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
                    if (body) {{
                        if (expanded) body.removeAttribute('inert');
                        else body.setAttribute('inert', '');
                    }}
                    if (expanded) {{
                        requestAnimationFrame(() => ensureCircleSizeCurveChart());
                    }}
                }};
                toggle.addEventListener('click', () => {{
                    setExpanded(panel.classList.contains('is-collapsed'));
                    requestAnimationFrame(repositionDuplicatePanelIfPresent);
                }});
            }}

            function bindCircleSizeControls() {{
                const minSlider = document.getElementById('circleSizeMinPx');
                const minVal = document.getElementById('circleSizeMinPxVal');
                if (minSlider) {{
                    minSlider.addEventListener('input', () => {{
                        circleSizeMinPx = parseFloat(minSlider.value);
                        if (minVal) minVal.textContent = formatCircleSizeMinPx(minSlider.value);
                        renderCircleSizeCurveChart();
                        refreshAllCircleSizes();
                    }});
                }}
                const resetCurveBtn = document.getElementById('resetCircleSizeCurveBtn');
                if (resetCurveBtn) resetCurveBtn.addEventListener('click', resetCircleSizeCurve);
                document.querySelectorAll('[data-curve-preset]').forEach(btn => {{
                    btn.addEventListener('click', () => {{
                        applyCircleSizeCurvePreset(btn.dataset.curvePreset);
                    }});
                }});
                updateCurvePresetButtons();
                updateCircleSizeCurveHint();
                const resetDisplayBtn = document.getElementById('resetDisplayOptionsBtn');
                if (resetDisplayBtn) resetDisplayBtn.addEventListener('click', resetDisplayOptions);
                bindCircleSizeCurveChart();
                if (!document.getElementById('displayOptionsPanel')?.classList.contains('is-collapsed')) {{
                    ensureCircleSizeCurveChart();
                }}
            }}

            function getSequenceLength() {{
                if (!z2zData || z2zData.length === 0) {{
                    return Array.isArray(tokens) ? tokens.length : 0;
                }}
                const layer0 = z2zData[0];
                if (!Array.isArray(layer0) || layer0.length === 0) {{
                    return Array.isArray(tokens) ? tokens.length : 0;
                }}
                const row0 = layer0[0];
                if (Array.isArray(row0) && row0.length > 0) {{
                    return Math.max(layer0.length, row0.length);
                }}
                return layer0.length;
            }}

            function tokenLabelAt(idx) {{
                if (Array.isArray(tokens) && idx >= 0 && idx < tokens.length) {{
                    return tokens[idx];
                }}
                return String(idx);
            }}

            const HOVER_HIGHLIGHT_RED = '#FF0000';
            const HOVER_BOX_BG = 'rgba(255, 0, 0, 0.12)';

            function createTokenLabelElement(tokenIdx) {{
                const el = document.createElement('div');
                el.className = 'token-label';
                const num = document.createElement('span');
                num.className = 'token-label-num';
                num.textContent = String(tokenIdx);
                const text = document.createElement('span');
                text.className = 'token-label-text';
                text.textContent = ' ' + tokenLabelAt(tokenIdx);
                el.appendChild(num);
                el.appendChild(text);
                return el;
            }}

            function styleTokenLabelNum(tokenLabel, highlight) {{
                const num = tokenLabel.querySelector('.token-label-num');
                const text = tokenLabel.querySelector('.token-label-text');
                if (!num) return;
                if (highlight) {{
                    num.style.color = HOVER_HIGHLIGHT_RED;
                    num.style.fontWeight = 'bold';
                    if (text) {{
                        text.style.color = HOVER_HIGHLIGHT_RED;
                        text.style.fontWeight = 'bold';
                    }}
                    return;
                }}
                const wrapper = tokenLabel.closest('.contribution-wrapper');
                const wrapperTokenIdx = wrapper ? parseInt(wrapper.dataset.tokenIdx, 10) : -1;
                if (selectedToken !== null && wrapperTokenIdx === selectedToken) {{
                    num.style.color = HOVER_HIGHLIGHT_RED;
                    num.style.fontWeight = 'bold';
                    if (text) {{
                        text.style.color = HOVER_HIGHLIGHT_RED;
                        text.style.fontWeight = 'bold';
                    }}
                }} else {{
                    num.style.color = '#7a9196';
                    num.style.fontWeight = '700';
                    if (text) {{
                        text.style.color = '#94a3b8';
                        text.style.fontWeight = '400';
                    }}
                }}
            }}

            const getVisualLayerIndex = (layer) => (numLayersToShow - 1 - layer);
            const getLayerTop = (layer) => getVisualLayerIndex(layer) * layoutLayerSpacing + layoutBaseTop;
            const getTokenHorizontalSpacing = () => (alignHorizontal ? 0 : layoutTokenOffsetX);
            const getTokenVerticalSpacing = () => (alignHorizontal ? 0 : layoutTokenOffsetY);
            const getTopAxisPosition = () => getLayerTop(topLayer) - layoutTopLabelOffset;

            function getDuplicatePanelTop(layerAlignedTop) {{
                const baseTop = layerAlignedTop + duplicatePanelTopOffset;
                const panel = document.getElementById('displayOptionsPanel');
                if (!panel) {{
                    return baseTop;
                }}
                const containerRect = container.getBoundingClientRect();
                const panelRect = panel.getBoundingClientRect();
                const minTopBelowPanel = panelRect.bottom - containerRect.top + duplicatePanelDisplayOptionsGap;
                return Math.max(baseTop, minTopBelowPanel);
            }}

            function repositionDuplicatePanelIfPresent() {{
                const duplicateOverallContainer = document.querySelector('.duplicate-overall-container');
                if (!duplicateOverallContainer || selectedToken === null) return;
                const layerAlignedTop = getLayerTop(topLayer) - layoutTopLabelOffset;
                duplicateOverallContainer.style.top = getDuplicatePanelTop(layerAlignedTop) + 'px';
            }}

            function estimateVisualizationBottom() {{
                const lastRow = Math.max(0, numTokensToShow - 1);
                return getLayerTop(0) + lastRow * getTokenVerticalSpacing() + circleSlotSize + 28 + layoutBottomLabelOffset;
            }}

            function measureVisualizationBottom() {{
                let bottom = estimateVisualizationBottom();
                container.querySelectorAll(
                    '.contribution-wrapper:not([data-is-duplicate="true"]), .token-axis-label-bottom, .duplicate-overall-container'
                ).forEach(el => {{
                    const relBottom = el.offsetTop + el.offsetHeight;
                    if (relBottom > bottom) bottom = relBottom;
                }});
                return bottom;
            }}

            function measureEmbedDocumentHeight() {{
                const contentBottom = measureVisualizationBottom();
                const padBottom = layoutContainerBottomMargin;
                let layoutFromContainer = contentBottom + padBottom;
                let node = container;
                while (node) {{
                    layoutFromContainer += node.offsetTop;
                    node = node.offsetParent;
                }}
                const scrollH = Math.max(
                    document.body.scrollHeight,
                    document.documentElement.scrollHeight
                );
                return Math.ceil(Math.max(scrollH, layoutFromContainer));
            }}

            function reportEmbedHeight() {{
                if (window.parent === window) return;
                if (!isEmbedPage && new URLSearchParams(window.location.search).get('embed') !== '1') return;
                window.parent.postMessage(
                    {{ type: 'lig-demo-height', height: measureEmbedDocumentHeight() }},
                    '*'
                );
            }}

            function syncContainerHeight() {{
                const contentBottom = measureVisualizationBottom();
                const padBottom = layoutContainerBottomMargin;
                container.style.paddingBottom = padBottom + 'px';
                container.style.minHeight = (contentBottom + padBottom) + 'px';
                reportEmbedHeight();
            }}

            function computeWrapperZIndex(layer, tokenIdx) {{
                const base = numTokensToShow - tokenIdx;
                if (focusedLayer !== null) {{
                    return WRAPPER_Z_INDEX_BASE + (layer === focusedLayer ? base + 5000 : base);
                }}
                if (selectedToken !== null && tokenIdx === selectedToken) {{
                    return WRAPPER_Z_INDEX_BASE + numTokensToShow + 1000;
                }}
                return WRAPPER_Z_INDEX_BASE + base;
            }}

            function hasCircleCoords(circleData) {{
                return circleData && typeof circleData.x === 'number' && typeof circleData.y === 'number';
            }}

            function registerDuplicateCirclePositions(duplicateContributionBox, layerIdx, wrapperTokenIdx) {{
                const duplicateCircles = duplicateContributionBox.querySelectorAll('.contribution-circle');
                let totalContribution = 0;
                for (let srcIdx = 0; srcIdx < numTokens; srcIdx++) {{
                    if (z2zData[layerIdx] && z2zData[layerIdx][wrapperTokenIdx] && z2zData[layerIdx][wrapperTokenIdx][srcIdx] !== undefined) {{
                        totalContribution += Math.abs(z2zData[layerIdx][wrapperTokenIdx][srcIdx]);
                    }}
                }}

                duplicateCircles.forEach(circle => {{
                    const circleIdx = parseInt(circle.dataset.contributionTokenIdx || circle.dataset.tokenIdx, 10);
                    const circleValue = (z2zData[layerIdx] && z2zData[layerIdx][wrapperTokenIdx] && z2zData[layerIdx][wrapperTokenIdx][circleIdx] !== undefined)
                        ? z2zData[layerIdx][wrapperTokenIdx][circleIdx]
                        : 0;
                    const contributionPercent = totalContribution > 0 ? ((Math.abs(circleValue) / totalContribution) * 100).toFixed(2) : '0.00';
                    const duplicateLayerIdx = parseInt(circle.dataset.layerIdx || String(layerIdx), 10);
                    const duplicateTokenIdx = parseInt(circle.dataset.tokenIdx || String(wrapperTokenIdx), 10);

                    if (!duplicateCirclePositions[duplicateLayerIdx]) {{
                        duplicateCirclePositions[duplicateLayerIdx] = {{}};
                    }}
                    if (!duplicateCirclePositions[duplicateLayerIdx][duplicateTokenIdx]) {{
                        duplicateCirclePositions[duplicateLayerIdx][duplicateTokenIdx] = {{}};
                    }}
                    duplicateCirclePositions[duplicateLayerIdx][duplicateTokenIdx][circleIdx] = {{
                        element: circle,
                        value: circleValue,
                        size: parseFloat(circle.style.width) || mapShareToDiameter(parseFloat(circle.dataset.shareRatio || '0')),
                    }};
                    circle.dataset.contributionPct = contributionPercent + '%';
                }});
            }}


            function refreshTokenLabelColors() {{
                document.querySelectorAll('.contribution-wrapper .token-label').forEach(tokenLabel => {{
                    if (tokenLabel.dataset.hoverHighlighted === '1') return;
                    styleTokenLabelNum(tokenLabel, false);
                }});
            }}

            function refreshTokenAxisLabelColors() {{
                document.querySelectorAll('.token-axis-label-top, .token-axis-label-bottom').forEach(axisLabel => {{
                    if (axisLabel.dataset.hoverHighlighted === '1') return;
                    const tokenIdx = parseInt(axisLabel.dataset.tokenIdx, 10);
                    if (selectedToken !== null && tokenIdx === selectedToken) {{
                        axisLabel.style.color = HOVER_HIGHLIGHT_RED;
                        axisLabel.style.fontWeight = 'bold';
                    }} else {{
                        axisLabel.style.color = '#333';
                        axisLabel.style.fontWeight = 'normal';
                    }}
                }});
            }}

            function applyBoxHoverHighlight(tokenIdx, boxEl) {{
                boxEl.dataset.originalBackgroundColor = boxEl.style.backgroundColor || '';
                boxEl.style.backgroundColor = HOVER_BOX_BG;
                boxEl.querySelectorAll('[data-token-idx]').forEach(circle => {{
                    circle.dataset.originalOpacity = circle.style.opacity || '1.0';
                    circle.style.opacity = '1.0';
                }});
                document.querySelectorAll(
                    `.token-axis-label-top[data-token-idx="${{tokenIdx}}"], .token-axis-label-bottom[data-token-idx="${{tokenIdx}}"]`
                ).forEach(axisLabel => {{
                    axisLabel.dataset.hoverHighlighted = '1';
                    axisLabel.style.color = HOVER_HIGHLIGHT_RED;
                    axisLabel.style.fontWeight = 'bold';
                }});
                const wrapper = boxEl.closest('.contribution-wrapper');
                const tokenLabel = wrapper ? wrapper.querySelector('.token-label') : null;
                if (tokenLabel) {{
                    tokenLabel.dataset.hoverHighlighted = '1';
                    styleTokenLabelNum(tokenLabel, true);
                }}
            }}

            function clearBoxHoverHighlight(tokenIdx, boxEl) {{
                if (boxEl.dataset.originalBackgroundColor !== undefined) {{
                    boxEl.style.backgroundColor = boxEl.dataset.originalBackgroundColor;
                    delete boxEl.dataset.originalBackgroundColor;
                }} else if (parseInt(boxEl.dataset.tokenIdx, 10) === 0) {{
                    boxEl.style.backgroundColor = 'rgba(232, 232, 232, 0.375)';
                }} else {{
                    boxEl.style.backgroundColor = 'rgba(232, 232, 232, 0.25)';
                }}
                boxEl.querySelectorAll('[data-token-idx]').forEach(circle => {{
                    const originalOpacity = circle.dataset.originalOpacity;
                    if (originalOpacity) circle.style.opacity = originalOpacity;
                }});
                document.querySelectorAll(
                    `.token-axis-label-top[data-token-idx="${{tokenIdx}}"], .token-axis-label-bottom[data-token-idx="${{tokenIdx}}"]`
                ).forEach(axisLabel => {{
                    delete axisLabel.dataset.hoverHighlighted;
                }});
                const wrapper = boxEl.closest('.contribution-wrapper');
                const tokenLabel = wrapper ? wrapper.querySelector('.token-label') : null;
                if (tokenLabel) delete tokenLabel.dataset.hoverHighlighted;
                refreshTokenLabelColors();
                refreshTokenAxisLabelColors();
            }}

            function refreshLayerLabelFromSelection() {{
                document.querySelectorAll('.layer-label').forEach(el => el.classList.remove('layer-label-focused'));
                document.querySelectorAll('.contribution-box').forEach(box => {{
                    const layerNum = parseInt(box.dataset.layerIdx, 10);
                    if (Number.isNaN(layerNum)) return;
                    // Token selection uses .selected on the chosen column only; do not
                    // highlight every box in the clicked layer.
                    box.classList.toggle('layer-focused', focusedLayer !== null && layerNum === focusedLayer);
                }});
            }}

            function clearLayerFocusOnly() {{
                if (focusedLayer === null) return;
                focusedLayer = null;
                document.querySelectorAll('.layer-label').forEach(el => el.classList.remove('layer-label-focused'));
                document.querySelectorAll('.contribution-box').forEach(box => box.classList.remove('layer-focused'));
                document.querySelectorAll('.contribution-wrapper:not([data-is-duplicate="true"])').forEach(wrapper => {{
                    const box = wrapper.querySelector('.contribution-box');
                    const layerNum = box ? parseInt(box.dataset.layerIdx, 10) : 0;
                    const tokenIdx = parseInt(wrapper.dataset.tokenIdx, 10);
                    wrapper.style.zIndex = computeWrapperZIndex(layerNum, tokenIdx);
                }});
                refreshLayerLabelFromSelection();
            }}

            function isVizInteractiveTarget(el) {{
                return el && el.closest(VIZ_INTERACTIVE_SELECTOR);
            }}

            function resetCircleOpacitiesToDefault() {{
                document.querySelectorAll('.contribution-circle').forEach(circle => {{
                    circle.style.opacity = '1.0';
                    delete circle.dataset.originalOpacity;
                }});
            }}

            function clearTokenSelectionOnly() {{
                if (selectedToken === null) return;
                selectedToken = null;
                clickedLayer = null;
                document.querySelectorAll('.duplicate-overall-container').forEach(el => el.remove());
                document.querySelectorAll('.token-axis-label-duplicate').forEach(el => el.remove());
                for (const layer in duplicateCirclePositions) delete duplicateCirclePositions[layer];
                document.querySelectorAll('.contribution-box').forEach(box => box.classList.remove('selected'));
                document.querySelectorAll('.contribution-wrapper:not([data-is-duplicate="true"])').forEach(wrapper => {{
                    wrapper.style.opacity = '1.0';
                    const tokenIdx = parseInt(wrapper.dataset.tokenIdx, 10);
                    const box = wrapper.querySelector('.contribution-box');
                    const layerNum = box ? parseInt(box.dataset.layerIdx, 10) : 0;
                    wrapper.style.zIndex = computeWrapperZIndex(layerNum, tokenIdx);
                }});
                resetCircleOpacitiesToDefault();
                updateTokenAxisLabels();
                refreshLayerLabelFromSelection();
                refreshTokenLabelColors();
                refreshTokenAxisLabelColors();
            }}

            function applyLayerFocus(layer) {{
                if (focusedLayer === layer) {{
                    focusedLayer = null;
                }} else {{
                    clearTokenSelectionOnly();
                    focusedLayer = layer;
                }}
                refreshLayerLabelFromSelection();
                document.querySelectorAll('.contribution-wrapper:not([data-is-duplicate="true"])').forEach(wrapper => {{
                    const box = wrapper.querySelector('.contribution-box');
                    const layerNum = box ? parseInt(box.dataset.layerIdx, 10) : 0;
                    const tokenIdx = parseInt(wrapper.dataset.tokenIdx, 10);
                    wrapper.style.zIndex = computeWrapperZIndex(layerNum, tokenIdx);
                }});
                updateCirclePositionsAndDrawPaths();
            }}

            function resetVisualizationSelection() {{
                if (
                    selectedToken === null &&
                    clickedLayer === null &&
                    focusedLayer === null &&
                    !document.querySelector('.duplicate-overall-container')
                ) {{
                    return;
                }}
                selectedToken = null;
                clickedLayer = null;
                focusedLayer = null;
                document.querySelectorAll('.duplicate-overall-container').forEach(el => el.remove());
                document.querySelectorAll('.token-axis-label-duplicate').forEach(el => el.remove());
                for (const layer in duplicateCirclePositions) delete duplicateCirclePositions[layer];
                document.querySelectorAll('.contribution-box').forEach(box => {{
                    box.classList.remove('selected');
                    box.classList.remove('layer-focused');
                }});
                document.querySelectorAll('.layer-label').forEach(el => el.classList.remove('layer-label-focused'));
                document.querySelectorAll('.contribution-wrapper:not([data-is-duplicate="true"])').forEach(wrapper => {{
                    wrapper.style.opacity = '1.0';
                    const tokenIdx = parseInt(wrapper.dataset.tokenIdx, 10);
                    const box = wrapper.querySelector('.contribution-box');
                    const layerNum = box ? parseInt(box.dataset.layerIdx, 10) : 0;
                    wrapper.style.zIndex = computeWrapperZIndex(layerNum, tokenIdx);
                }});
                resetCircleOpacitiesToDefault();
                updateTokenAxisLabels();
                refreshLayerLabelFromSelection();
                refreshTokenLabelColors();
                refreshTokenAxisLabelColors();
                updateCirclePositionsAndDrawPaths();
                window.parent.postMessage({{
                    type: 'streamlit:setComponentValue',
                    value: null
                }}, '*');
            }}

            function handleVizBackgroundClick(e) {{
                if (isVizInteractiveTarget(e.target)) return;
                resetVisualizationSelection();
            }}

            function rebuildVisualizationLayout() {{
                recalcLayoutMetrics();
                clearVisualization();
                buildVisualization(() => {{
                    updateTokenAxisLabels();
                    setTimeout(() => {{
                        if (selectedToken !== null) {{
                            updateSelectedToken(selectedToken);
                            updateDuplicateAxisLabels();
                        }}
                        updateCirclePositionsAndDrawPaths();
                        syncContainerHeight();
                        syncVizScrollCenter();
                    }}, 100);
                }});
            }}

            function recalcLayoutMetrics() {{
                numTokens = getSequenceLength();
                numTokensToShow = numTokens;
                numLayersToShow = Math.min(13, z2zData.length);
                topLayer = numLayersToShow - 1;
                circleSlotSize = CIRCLE_SLOT_PX;
                circleCenterOffset = circleSlotSize / 2;
                if (useSlotCap) {{
                    circleMaxDiameter = Math.min(38, circleSlotSize * 0.92);
                    maxCircleSizeFixed = circleMaxDiameter;
                }} else {{
                    maxCircleSizeFixed = 42;
                }}
                const leftGutter = applyContainerLeftGutter();
                const hasDuplicatePanel = !!document.querySelector('.duplicate-overall-container') || selectedToken !== null;
                container.style.paddingRight = (hasDuplicatePanel ? duplicatePanelRightMargin : 16) + 'px';
                container.style.width = measureContainerContentWidth() + 'px';
                container.style.maxWidth = 'none';
                container.style.overflowX = 'visible';
                repositionTargetAxisCaptionY();
                const estimatedBottom = estimateVisualizationBottom();
                container.style.paddingBottom = layoutContainerBottomMargin + 'px';
                container.style.minHeight = (estimatedBottom + layoutContainerBottomMargin) + 'px';
            }}

            function clearVisualization() {{
                focusedLayer = null;
                container.querySelectorAll('.contribution-wrapper, .layer-label').forEach(el => el.remove());
                document.querySelectorAll('.duplicate-overall-container').forEach(el => el.remove());
                document.querySelectorAll('.token-axis-label-top, .token-axis-label-bottom, .token-axis-label-duplicate').forEach(el => el.remove());
                pathSvg.innerHTML = '';
                topLayerTokenPositions.length = 0;
                bottomLayerTokenPositions.length = 0;
                for (const k in circlePositions) delete circlePositions[k];
                for (const k in duplicateCirclePositions) delete duplicateCirclePositions[k];
            }}

            // チェックボックスの初期設定
            const showPathsCheckbox = document.getElementById('showPathsCheckbox');
            const alignHorizontalCheckbox = document.getElementById('alignHorizontalCheckbox');
            const alignHorizontalControl = document.getElementById('alignHorizontalControl');
            
            showPathsCheckbox.checked = showPaths;
            alignHorizontalCheckbox.checked = alignHorizontal;
            alignHorizontalControl.style.display = showPaths ? 'block' : 'none';
            
            // チェックボックスの変更イベント
            showPathsCheckbox.addEventListener('change', function() {{
                showPaths = this.checked;
                alignHorizontalControl.style.display = showPaths ? 'block' : 'none';
                if (!showPaths) {{
                    alignHorizontalCheckbox.checked = false;
                    alignHorizontal = false;
                }}
                updateCirclePositionsAndDrawPaths();
            }});
            
            alignHorizontalCheckbox.addEventListener('change', function() {{
                alignHorizontal = this.checked;
                // 全てのwrapperの位置を再計算
                document.querySelectorAll('.contribution-wrapper:not([data-is-duplicate="true"])').forEach(wrapper => {{
                    const tokenIdx = parseInt(wrapper.dataset.tokenIdx);
                    const contributionBox = wrapper.querySelector('.contribution-box');
                    const layer = contributionBox ? parseInt(contributionBox.dataset.layerIdx || 0) : 0;
                    
                    // 横軸位置を再計算
                    const horizontalSpacing = getTokenHorizontalSpacing();
                    const rightOffset = tokenIdx * horizontalSpacing;
                    wrapper.style.left = rightOffset + 'px';
                    
                    // 縦軸位置を再計算（上から layer=numLayersToShow-1 が先頭）
                    const layerOffset = getLayerTop(layer);
                    // Token数が6以下の場合は縦方向のずらしを大きくする
                    const verticalSpacing = getTokenVerticalSpacing();
                    const tokenOffset = tokenIdx * verticalSpacing;
                    wrapper.style.top = (layerOffset + tokenOffset) + 'px';
                }});
                
                // 複製の位置も再計算
                document.querySelectorAll('.contribution-wrapper[data-is-duplicate="true"]').forEach(wrapper => {{
                    const tokenIdx = parseInt(wrapper.dataset.tokenIdx);
                    const duplicateVerticalSpacing = alignHorizontal ? 0 : 2;
                    // Token順を逆順にする（Token 11~0の順番）
                    const reverseTokenIdx = numTokensToShow - 1 - tokenIdx;
                    const duplicateTokenOffset = reverseTokenIdx * duplicateVerticalSpacing;
                    wrapper.style.top = duplicateTokenOffset + 'px';
                }});
                
                updateCirclePositionsAndDrawPaths();
                // 下の軸の位置も更新
                updateTokenAxisLabels();
            }});
            
            function buildVisualization(onComplete) {{
            ensureVizTooltips();
            bindCircleTooltipDelegation();
            
            // 選択状態を管理する関数（初回のみ定義）
            if (!updateSelectedToken) updateSelectedToken = function(tokenIdx) {{
                const willDeselect = selectedToken === tokenIdx;
                const hadDuplicate = !!document.querySelector('.duplicate-overall-container');

                function applySelectionChange() {{
                // 既存の複製をすべて削除
                document.querySelectorAll('.duplicate-overall-container').forEach(container => container.remove());
                // 複製用のToken軸ラベルも削除
                document.querySelectorAll('.token-axis-label-duplicate').forEach(label => label.remove());
                // 複製の円の位置情報もクリア
                for (let layer in duplicateCirclePositions) {{
                    delete duplicateCirclePositions[layer];
                }}
                
                if (selectedToken === tokenIdx) {{
                    // 同じTokenをクリックした場合は選択解除
                    selectedToken = null;
                    clickedLayer = null; // クリックしたLayerもリセット
                }} else {{
                    // 新しいTokenを選択（Layer選択は解除）
                    clearLayerFocusOnly();
                    selectedToken = tokenIdx;
                    // clickedLayerはクリックイベントで設定される
                }}
                // すべてのLayerで選択状態を更新
                document.querySelectorAll('.contribution-box').forEach(box => {{
                    const boxTokenIdx = parseInt(box.dataset.tokenIdx);
                    if (selectedToken !== null && boxTokenIdx === selectedToken) {{
                        box.classList.add('selected');
                    }} else {{
                        box.classList.remove('selected');
                    }}
                }});
                // 透過率と重ね順を再計算
                document.querySelectorAll('.contribution-wrapper:not([data-is-duplicate="true"])').forEach(wrapper => {{
                    const wrapperTokenIdx = parseInt(wrapper.dataset.tokenIdx);
                    // Tokenラベルの色を更新（クリックしたLayerのToken Labelは除外）
                    const contributionBox = wrapper.querySelector('.contribution-box');
                    const wrapperLayer = contributionBox ? parseInt(contributionBox.dataset.layerIdx) : null;
                    // token-label colors synced below via refreshTokenLabelColors()
                    if (selectedToken === null) {{
                        wrapper.style.opacity = 1.0;
                        const boxLayer = contributionBox ? parseInt(contributionBox.dataset.layerIdx, 10) : 0;
                        wrapper.style.zIndex = computeWrapperZIndex(boxLayer, wrapperTokenIdx);
                    }} else {{
                        // 選択されたTokenからの距離に応じて透過率を設定
                        const distance = Math.abs(wrapperTokenIdx - selectedToken);
                        const maxDistance = Math.max(selectedToken, numTokensToShow - 1 - selectedToken);
                        const maxOpacity = 1.0;
                        const minOpacity = 0.2;
                        const opacity = maxDistance === 0 ? maxOpacity : maxOpacity - ((distance / maxDistance) * (maxOpacity - minOpacity));
                        wrapper.style.opacity = opacity;
                        const boxLayer = contributionBox ? parseInt(contributionBox.dataset.layerIdx, 10) : 0;
                        // 選択されたTokenは最前面に
                        if (wrapperTokenIdx === selectedToken) {{
                            wrapper.style.zIndex = computeWrapperZIndex(boxLayer, wrapperTokenIdx);
                            
                            // 選択されたTokenの右側に複製を作成（もっと右に配置）
                            const originalBox = wrapper.querySelector('.contribution-box');
                            if (originalBox) {{
                                // Layer情報を取得
                                const layerIdx = parseInt(originalBox.dataset.layerIdx);
                                // 位置を計算（ラベルの幅 + gap + 円の総幅 + マージン）
                                const labelWidth = 120;
                                const gap = 10;
                                const boxPaddingLeft = 10;
                                const boxPaddingRight = 10;
                                const circleWidth = circleSlotSize;
                                const estimatedBoxWidth = labelWidth + gap + boxPaddingLeft + (numTokens * circleWidth) + boxPaddingRight;
                                const horizontalSpacing = getTokenHorizontalSpacing();
                                const rightOffset = wrapperTokenIdx * horizontalSpacing;
                                
                                // 複製全体を囲むコンテナを作成（既に存在するかチェック）
                                let duplicateOverallContainer = document.querySelector('.duplicate-overall-container');
                                if (!duplicateOverallContainer) {{
                                    duplicateOverallContainer = document.createElement('div');
                                    duplicateOverallContainer.className = 'duplicate-overall-container';
                                    duplicateOverallContainer.style.position = 'absolute';
                                    duplicateOverallContainer.style.zIndex = String(WRAPPER_Z_INDEX_BASE + numTokensToShow + 1500);
                                    
                                    // 最終位置（右側）を計算（複製元により近づける、少し右に移動）
                                    const finalLeft = rightOffset + estimatedBoxWidth + 150;
                                    
                                    // 複製コンテナの位置を計算
                                    const duplicateLayerSpacing = 10; // Layer間隔
                                    const axisLabelHeight = 50; // 軸ラベルの高さ
                                    
                                    // 最上段Layerの上端に合わせて複製パネルを配置（Display options と重ならないよう下げる）
                                    const layerAlignedTop = getLayerTop(topLayer) - layoutTopLabelOffset;
                                    const finalTop = getDuplicatePanelTop(layerAlignedTop);
                                    
                                    // 元の上側Token軸ラベルは複製の有無に影響させない（複製側の軸はduplicateAxisContainer内で表示）
                                    
                                    duplicateOverallContainer.style.left = finalLeft + 'px';
                                    duplicateOverallContainer.style.top = finalTop + 'px';
                                    
                                    // 複製全体を枠線で囲む（薄い青背景、薄い青の枠線、角を丸く、マージンに余裕）
                                    duplicateOverallContainer.style.backgroundColor = 'rgba(255, 251, 245, 0.3)'; // さらに薄い肌色の背景（透明度30%）
                                    duplicateOverallContainer.style.border = '2px solid rgba(245, 237, 224, 0.3)'; // さらに薄い肌色の枠線（透明度30%）
                                    duplicateOverallContainer.style.borderRadius = '12px'; // 角を丸く
                                    duplicateOverallContainer.style.paddingTop = '5px'; // 上のpaddingを最小限に（軸ラベルの直下に配置）
                                    duplicateOverallContainer.style.paddingLeft = '15px';
                                    duplicateOverallContainer.style.paddingRight = '15px';
                                    // 下のpaddingを自動計算：Layer数 × Layer間隔 + 余裕
                                    const bottomPadding = (numLayersToShow * duplicateLayerSpacing) + 30;
                                    duplicateOverallContainer.style.paddingBottom = bottomPadding + 'px';
                                    duplicateOverallContainer.style.overflow = 'visible'; // はみ出しを許可（見切れを防ぐため）
                                    duplicateOverallContainer.style.opacity = '1';

                                    container.appendChild(duplicateOverallContainer);
                                    finishDuplicatePanelIn(duplicateOverallContainer);
                                    
                                    // 複製用のToken軸ラベルコンテナを作成
                                    const duplicateAxisContainer = document.createElement('div');
                                    duplicateAxisContainer.className = 'duplicate-axis-container';
                                    duplicateAxisContainer.id = 'duplicateTokenAxisLabelsTop';
                                    duplicateAxisContainer.style.position = 'relative';
                                    duplicateAxisContainer.style.height = axisLabelHeight + 'px'; // 軸ラベルの高さ
                                    duplicateAxisContainer.style.opacity = '1'; // 即座に表示
                                    duplicateOverallContainer.appendChild(duplicateAxisContainer);
                                }}
                                
                                // Layerごとの複製コンポーネントコンテナを作成（既に存在するかチェック）
                                let duplicateLayerContainer = document.querySelector(`.duplicate-layer-container[data-layer-idx="${{layerIdx}}"]`);
                                if (!duplicateLayerContainer) {{
                                    duplicateLayerContainer = document.createElement('div');
                                    duplicateLayerContainer.className = 'duplicate-layer-container';
                                    duplicateLayerContainer.dataset.layerIdx = layerIdx;
                                    duplicateLayerContainer.style.position = 'relative';
                                    // 複製のLayer間隔をもっと狭める
                                    const duplicateLayerSpacing = 10; // 10px
                                    // Layerの順番を逆順にする（Layer 11~0の順番）
                                    const reverseLayerIdx = numLayersToShow - 1 - layerIdx;
                                    // 軸ラベルの直下に最初のLayerを配置（Layer 12）
                                    // Layer間の間隔は10pxで、最初のLayer（Layer 12）は軸ラベルの直下に配置
                                    duplicateLayerContainer.style.marginTop = reverseLayerIdx === 0 ? '0px' : duplicateLayerSpacing + 'px';
                                    // Layerの順番を逆順にするため、appendChildの代わりにinsertBeforeを使用
                                    const firstLayerContainer = duplicateOverallContainer.querySelector('.duplicate-layer-container');
                                    if (firstLayerContainer) {{
                                        duplicateOverallContainer.insertBefore(duplicateLayerContainer, firstLayerContainer);
                                    }} else {{
                                        duplicateOverallContainer.appendChild(duplicateLayerContainer);
                                    }}
                                }}
                                
                                // 複製用のラッパーを作成
                                const duplicateWrapper = document.createElement('div');
                                duplicateWrapper.className = 'contribution-wrapper';
                                duplicateWrapper.dataset.tokenIdx = wrapperTokenIdx;
                                duplicateWrapper.dataset.isDuplicate = 'true';
                                duplicateWrapper.style.position = 'relative';
                                duplicateWrapper.style.left = '0px';
                                // 複製の縦間隔を狭める（verticalSpacingを小さく）
                                // alignHorizontalがtrueの場合は縦軸も揃える（verticalSpacingを0にする）
                                const duplicateVerticalSpacing = alignHorizontal ? 0 : 2; // 元の5pxより小さく、または0
                                // Token順を逆順にする（Token 11~0の順番）
                                const reverseTokenIdx = numTokensToShow - 1 - wrapperTokenIdx;
                                const duplicateTokenOffset = reverseTokenIdx * duplicateVerticalSpacing;
                                duplicateWrapper.style.top = duplicateTokenOffset + 'px';
                                duplicateWrapper.style.zIndex = String(WRAPPER_Z_INDEX_BASE + numTokensToShow + 1001);
                                duplicateWrapper.style.opacity = 1.0; // 複製は透明度を下げない
                                duplicateWrapper.style.display = 'flex';
                                duplicateWrapper.style.alignItems = 'center'; // 縦方向の中央揃え
                                duplicateWrapper.style.gap = '10px'; // ラベルとボックスの間隔
                                
                                // LayerラベルとTokenラベルを縦に並べたコンテナを作成
                                const duplicateLabelsContainer = document.createElement('div');
                                duplicateLabelsContainer.style.display = 'flex';
                                duplicateLabelsContainer.style.flexDirection = 'column'; // 縦に並べる
                                duplicateLabelsContainer.style.alignItems = 'flex-start'; // 左揃え
                                duplicateLabelsContainer.style.width = '120px'; // Tokenラベルの幅と同じ
                                duplicateLabelsContainer.style.flexShrink = '0';
                                duplicateLabelsContainer.style.paddingTop = '5px'; // 縦軸を少し下に下げる
                                
                                // Layerラベル（上に表示）
                                const duplicateLayerLabel = document.createElement('div');
                                duplicateLayerLabel.className = 'layer-label layer-label-duplicate';
                                duplicateLayerLabel.textContent = `Layer ${{layerIdx}}`;
                                duplicateLayerLabel.style.fontSize = '20px';
                                duplicateLayerLabel.style.fontWeight = 'bold';
                                duplicateLayerLabel.style.color = '#333';
                                duplicateLayerLabel.style.whiteSpace = 'nowrap';
                                duplicateLayerLabel.style.marginBottom = '2px'; // Tokenラベルとの間隔
                                duplicateLabelsContainer.appendChild(duplicateLayerLabel);
                                
                                // Tokenラベル（複製ではToken番号とToken名を表示、Layerラベルの下に配置）
                                const duplicateTokenLabel = createTokenLabelElement(wrapperTokenIdx);
                                duplicateTokenLabel.style.paddingLeft = '20px'; // Layerラベルより右にずらす
                                duplicateTokenLabel.addEventListener('click', function(e) {{
                                    e.stopPropagation();
                                    clickedLayer = layerIdx;
                                    updateSelectedToken(wrapperTokenIdx);
                                }});
                                styleTokenLabelNum(duplicateTokenLabel, selectedToken !== null && wrapperTokenIdx === selectedToken);
                                duplicateLabelsContainer.appendChild(duplicateTokenLabel);
                                
                                duplicateWrapper.appendChild(duplicateLabelsContainer);
                                
                                // 貢献度ボックスの複製を作成
                                const duplicateContributionBox = originalBox.cloneNode(true);
                                duplicateContributionBox.classList.add('selected');
                                duplicateContributionBox.dataset.isDuplicate = 'true';
                                
                                // ホバー時の背景色をリセット（元の背景色に戻す）
                                duplicateContributionBox.style.backgroundColor = 'rgba(232, 232, 232, 0.25)';
                                
                                // 元の位置を取得（複製元の位置）
                                const originalBoxRect = originalBox.getBoundingClientRect();
                                const containerRect = container.getBoundingClientRect();
                                const originalBoxLeft = originalBoxRect.left - containerRect.left;
                                const originalBoxTop = originalBoxRect.top - containerRect.top;
                                
                                duplicateContributionBox.style.position = 'relative';
                                duplicateContributionBox.style.left = 'auto';
                                duplicateContributionBox.style.top = 'auto';
                                duplicateWrapper.appendChild(duplicateContributionBox);
                                
                                // クリックイベントを再設定
                                duplicateContributionBox.addEventListener('click', function(e) {{
                                    e.stopPropagation();
                                    const clickedTokenIdx = parseInt(this.dataset.tokenIdx);
                                    updateSelectedToken(clickedTokenIdx);
                                }});
                                // ホバーイベントを再設定
                                duplicateContributionBox.addEventListener('mouseenter', function() {{
                                    const layerIdx = parseInt(this.dataset.layerIdx, 10);
                                    const colTokenIdx = parseInt(this.dataset.tokenIdx, 10);
                                    globalTooltipEl.textContent = `Layer ${{layerIdx}}, Target token ${{colTokenIdx}}: ${{tokenLabelAt(colTokenIdx)}}`;
                                    const rect = this.getBoundingClientRect();
                                    globalTooltipEl.style.left = (rect.left + rect.width / 2) + 'px';
                                    globalTooltipEl.style.top = (rect.top - 5) + 'px';
                                    globalTooltipEl.style.transform = 'translate(-50%, -100%)';
                                    globalTooltipEl.style.display = 'block';
                                    applyBoxHoverHighlight(colTokenIdx, this);
                                }});
                                duplicateContributionBox.addEventListener('mouseleave', function() {{
                                    globalTooltipEl.style.display = 'none';
                                    const colTokenIdx = parseInt(this.dataset.tokenIdx, 10);
                                    clearBoxHoverHighlight(colTokenIdx, this);
                                }});
                                duplicateContributionBox.addEventListener('mousemove', function() {{
                                    const rect = this.getBoundingClientRect();
                                    globalTooltipEl.style.left = (rect.left + rect.width / 2) + 'px';
                                    globalTooltipEl.style.top = (rect.top - 5) + 'px';
                                }});
                                
                                // duplicateContributionBoxはアニメーション後に追加されるため、ここでは追加しない
                                duplicateLayerContainer.appendChild(duplicateWrapper);
                                registerDuplicateCirclePositions(duplicateContributionBox, layerIdx, wrapperTokenIdx);
                            }}
                        }} else {{
                            wrapper.style.zIndex = String(computeWrapperZIndex(boxLayer, wrapperTokenIdx));
                        }}
                    }}
                }});
                
                // 円の透過率も再計算（選択されたTokenの円は透過率を下げる）
                document.querySelectorAll('[data-token-idx]').forEach(circle => {{
                    const circleTokenIdx = parseInt(circle.dataset.tokenIdx);
                    if (selectedToken === null) {{
                        circle.style.opacity = 1.0;
                    }} else {{
                        if (circleTokenIdx === selectedToken) {{
                            // 選択されたTokenの円は周りより少し高い透過率（薄く）
                            circle.style.opacity = 0.9;
                        }} else {{
                            // 選択されたTokenからの距離に応じて透過率を設定
                            const distance = Math.abs(circleTokenIdx - selectedToken);
                            const maxDistance = Math.max(selectedToken, numTokensToShow - 1 - selectedToken);
                            const maxOpacity = 1.0;
                            const minOpacity = 0.2;
                            if (maxDistance > 0) {{
                                const opacity = maxOpacity - ((distance / maxDistance) * (maxOpacity - minOpacity));
                                circle.style.opacity = opacity;
                            }} else {{
                                circle.style.opacity = 1.0;
                            }}
                        }}
                    }}
                }});
                // Streamlitに選択されたTokenを送信
                if (selectedToken !== null) {{
                    window.parent.postMessage({{
                        type: 'streamlit:setComponentValue',
                        value: selectedToken
                    }}, '*');
                }}
                refreshTokenLabelColors();
                refreshLayerLabelFromSelection();
                // Token軸ラベルを更新（選択されたTokenを赤文字・太文字にする）
                updateTokenAxisLabels();
                // 複製のToken軸ラベルを更新
                setTimeout(() => {{
                    updateDuplicateAxisLabels();
                    // 複製のTokenラベルの色も更新
                    refreshTokenLabelColors();
                    refreshLayerLabelFromSelection();
                    // 複製パネル内のパスを含めて更新
                    updateCirclePositionsAndDrawPaths();
                }}, 50);
                requestAnimationFrame(() => {{
                    updateCirclePositionsAndDrawPaths();
                    recalcLayoutMetrics();
                    syncContainerHeight();
                }});
                }}

                if (willDeselect && hadDuplicate) {{
                    animateDuplicatePanelsOut(applySelectionChange);
                    return;
                }}
                applySelectionChange();
            }};
            
            vizBuildGeneration += 1;
            const buildToken = vizBuildGeneration;
            let nextLayer = 0;

            function buildLayerChunk() {{
                if (buildToken !== vizBuildGeneration) return;
                const chunk = document.createDocumentFragment();
                const endLayer = Math.min(nextLayer + VIZ_LAYERS_PER_FRAME, numLayersToShow);
                for (let layer = nextLayer; layer < endLayer; layer++) {{
                const layerTop = getLayerTop(layer);

                const layerLabel = document.createElement('div');
                layerLabel.className = 'layer-label';
                layerLabel.dataset.layerIdx = layer;
                layerLabel.textContent = `Layer ${{layer}}`;
                layerLabel.style.position = 'absolute';
                layerLabel.style.top = (layerTop - 25) + 'px';
                layerLabel.style.left = '0px'; // containerのpadding内に配置
                layerLabel.style.cursor = 'pointer';
                layerLabel.addEventListener('click', function(e) {{
                    e.stopPropagation();
                    applyLayerFocus(layer);
                }});
                chunk.appendChild(layerLabel);
                
                // 各Tokenについて貢献度を表示
                for (let tokenIdx = 0; tokenIdx < numTokensToShow; tokenIdx++) {{
                    // 各src_tokenからこのTokenへの貢献度を取得
                    // データ構造: z2z[layer][target_token][source_token]
                    const contributions = [];
                    for (let srcIdx = 0; srcIdx < numTokens; srcIdx++) {{
                        if (z2zData[layer] && z2zData[layer][tokenIdx] && z2zData[layer][tokenIdx][srcIdx] !== undefined) {{
                            const value = z2zData[layer][tokenIdx][srcIdx];
                            contributions.push(value);
                        }} else {{
                            contributions.push(0);
                        }}
                    }}
                    
                    // 貢献度の最小値と最大値を計算（表示判定用、円のサイズには使用しない）
                    const absContributions = contributions.map(v => Math.abs(v));
                    const values = absContributions.filter(v => v > 0);
                    if (values.length === 0) {{
                        continue;
                    }}
                    
                    const minVal = Math.min(...values);
                    const maxVal = Math.max(...values);
                    
                    // ラッパーを作成
                    const wrapper = document.createElement('div');
                    wrapper.className = 'contribution-wrapper';
                    wrapper.dataset.tokenIdx = tokenIdx;
                    // 右にずらす（均等に、ラベルの左端を基準に）
                    // alignHorizontalがtrueの場合は横軸座標を揃える（全て0にする）
                    const horizontalSpacing = getTokenHorizontalSpacing(); // 均一な間隔（約6.67px）または0
                    const rightOffset = tokenIdx * horizontalSpacing;
                    wrapper.style.left = rightOffset + 'px';
                    wrapper.style.marginLeft = '0';
                    wrapper.style.paddingLeft = '0';
                    // 下にずらす（Layerごとに150px、Tokenごとに5px、全体を50px下げる）
                    const layerOffset = getLayerTop(layer); // 各Layerごとに150px下に、全体を50px下げる
                    // alignHorizontalがtrueの場合は縦軸も揃える（verticalSpacingを0にする）
                    // Token数が6以下の場合は縦方向のずらしを大きくする
                    const verticalSpacing = getTokenVerticalSpacing();
                    const tokenOffset = tokenIdx * verticalSpacing; // Tokenごとに下にずらす
                    wrapper.style.top = (layerOffset + tokenOffset) + 'px';
                    
                    // 一番上のレイヤーのToken位置を記録（横軸ラベル用）
                    // この時点ではまだ円の位置が確定していないので、後で記録する
                    
                    wrapper.style.zIndex = computeWrapperZIndex(layer, tokenIdx);
                    // 透明度：選択されたTokenから離れるほど透過度を下げる（初期状態）
                    if (selectedToken !== null) {{
                        const distance = Math.abs(tokenIdx - selectedToken); // 選択されたTokenからの距離
                        const maxDistance = Math.max(selectedToken, numTokensToShow - 1 - selectedToken); // 最大距離
                        const maxOpacity = 1.0; // 選択されたTokenは完全不透明
                        const minOpacity = 0.2; // 最も遠いTokenは20%の透明度
                        const opacity = maxDistance === 0 ? maxOpacity : maxOpacity - ((distance / maxDistance) * (maxOpacity - minOpacity));
                        wrapper.style.opacity = opacity;
                    }} else {{
                        wrapper.style.opacity = 1.0; // 選択されていない場合は全て不透明
                    }}
                    
                    // Tokenラベル（番号 + トークン文字）
                    const tokenLabel = createTokenLabelElement(tokenIdx);
                    styleTokenLabelNum(tokenLabel, selectedToken !== null && tokenIdx === selectedToken);
                    tokenLabel.addEventListener('click', function(e) {{
                        e.stopPropagation();
                        clickedLayer = layer;
                        updateSelectedToken(tokenIdx);
                    }});
                    wrapper.appendChild(tokenLabel);
                    
                    // 貢献度ボックス
                    const contributionBox = document.createElement('div');
                    contributionBox.className = 'contribution-box';
                    contributionBox.dataset.tokenIdx = tokenIdx;
                    contributionBox.dataset.layerIdx = layer;
                    // ツールチップを設定（LayerとTokenの情報）
                    contributionBox.setAttribute('data-tooltip', `Layer ${{layer}}, Target token ${{tokenIdx}}: ${{tokenLabelAt(tokenIdx)}}`);
                    // ホバー時にツールチップを表示
                    contributionBox.addEventListener('mouseenter', function() {{
                        const colTokenIdx = parseInt(this.dataset.tokenIdx, 10);
                        const layerIdx = parseInt(this.dataset.layerIdx, 10);
                        globalTooltipEl.textContent = `Layer ${{layerIdx}}, Target token ${{colTokenIdx}}: ${{tokenLabelAt(colTokenIdx)}}`;
                        const rect = this.getBoundingClientRect();
                        globalTooltipEl.style.left = (rect.left + rect.width / 2) + 'px';
                        globalTooltipEl.style.top = (rect.top - 5) + 'px';
                        globalTooltipEl.style.transform = 'translate(-50%, -100%)';
                        globalTooltipEl.style.display = 'block';
                        applyBoxHoverHighlight(colTokenIdx, this);
                    }});
                    contributionBox.addEventListener('mouseleave', function() {{
                        globalTooltipEl.style.display = 'none';
                        const colTokenIdx = parseInt(this.dataset.tokenIdx, 10);
                        clearBoxHoverHighlight(colTokenIdx, this);
                        refreshLayerLabelFromSelection();
                    }});
                    contributionBox.addEventListener('mousemove', function() {{
                        const rect = this.getBoundingClientRect();
                        globalTooltipEl.style.left = (rect.left + rect.width / 2) + 'px';
                        globalTooltipEl.style.top = (rect.top - 5) + 'px';
                    }});
                    tokenLabel.addEventListener('mouseenter', function(e) {{
                        e.stopPropagation();
                        applyBoxHoverHighlight(tokenIdx, contributionBox);
                    }});
                    tokenLabel.addEventListener('mouseleave', function() {{
                        clearBoxHoverHighlight(tokenIdx, contributionBox);
                        refreshLayerLabelFromSelection();
                    }});
                    // 選択されたTokenの場合はselectedクラスを追加
                    if (tokenIdx === selectedToken) {{
                        contributionBox.classList.add('selected');
                    }}
                    // クリックイベントを追加
                    contributionBox.addEventListener('click', function(e) {{
                        e.stopPropagation();
                        clickedLayer = parseInt(this.dataset.layerIdx, 10);
                        updateSelectedToken(parseInt(this.dataset.tokenIdx, 10));
                    }});
                    // 一番上のToken（tokenIdx=0）だけ透過度を37.5%にする
                    if (tokenIdx === 0) {{
                        contributionBox.style.backgroundColor = 'rgba(232, 232, 232, 0.375)';
                    }}
                    
                    const bar = document.createElement('div');
                    bar.className = 'contribution-bar';
                    bar.style.display = 'flex';
                    bar.style.alignItems = 'center';
                    bar.style.gap = '0';
                    
                    // 貢献度の合計を計算（列内シェア用）
                    const totalContribution = contributions.reduce((sum, v) => sum + Math.abs(v), 0);
                    const normalizedContributions = contributions.map(v => totalContribution > 0 ? Math.abs(v) / totalContribution : 0);
                    const maxNormalizedShare = normalizedContributions.length > 0 ? Math.max(...normalizedContributions) : 0;
                    
                    // 各トークンへの貢献度を円の大きさで表現
                    contributions.forEach((value, idx) => {{
                        const circleContainer = document.createElement('div');
                        circleContainer.className = 'contribution-circle-slot';
                        circleContainer.dataset.srcIdx = idx.toString();
                        circleContainer.style.display = 'inline-flex';
                        circleContainer.style.alignItems = 'center';
                        circleContainer.style.justifyContent = 'center';
                        // コンテナのサイズはToken数に応じて可変
                        circleContainer.style.width = circleSlotSize + 'px';
                        circleContainer.style.height = circleSlotSize + 'px';
                        circleContainer.style.flexShrink = '0';
                        circleContainer.style.overflow = 'visible'; // 円がはみ出しても表示
                        circleContainer.style.position = 'relative'; // 相対位置指定
                        
                        const labelWidth = 120;
                        const gap = 10;
                        const boxPaddingLeft = 10;
                        const circlePosition = idx * circleSlotSize;
                        const circleCenter = circleCenterOffset;
                        const tokenPosition = rightOffset + labelWidth + gap + boxPaddingLeft + circlePosition + circleCenter;
                        if (layer === topLayer && tokenIdx === 0) {{
                            const existingTop = topLayerTokenPositions.find(p => p.tokenIdx === idx);
                            if (!existingTop) {{
                                topLayerTokenPositions.push({{
                                    tokenIdx: idx,
                                    tokenName: tokenLabelAt(idx),
                                    position: tokenPosition
                                }});
                            }}
                        }}
                        if (layer === 0 && tokenIdx === numTokensToShow - 1) {{
                            const existingBottom = bottomLayerTokenPositions.find(p => p.tokenIdx === idx);
                            if (!existingBottom) {{
                                bottomLayerTokenPositions.push({{
                                    tokenIdx: idx,
                                    tokenName: tokenLabelAt(idx),
                                    position: tokenPosition
                                }});
                            }}
                        }}
                        
                        if (Math.abs(value) > 0) {{
                            const shareRatio = maxNormalizedShare > 0 ? (normalizedContributions[idx] / maxNormalizedShare) : 0;
                            const normalizedSize = mapShareToDiameter(shareRatio);
                            
                            const circle = document.createElement('div');
                            circle.className = 'contribution-circle';
                            circle.dataset.shareRatio = String(shareRatio);
                            // 完全な正円にするための設定
                            circle.style.borderRadius = '50%';
                            circle.style.aspectRatio = '1 / 1';
                            circle.style.padding = '0';
                            circle.style.margin = '0';
                            circle.style.border = 'none';
                            circle.style.boxSizing = 'border-box';
                            circle.style.display = 'block';
                            circle.style.flexShrink = '0';
                            circle.style.flexGrow = '0';
                            circle.style.cursor = 'pointer';
                            circle.style.position = 'absolute';
                            applyCircleDiameter(circle, normalizedSize);
                            circle.dataset.tokenIdx = tokenIdx.toString(); // 円が属するTokenのIDを記録
                            circle.dataset.contributionTokenIdx = idx.toString(); // この円が表す貢献度のTokenのID
                            circle.dataset.layerIdx = layer.toString(); // LayerのID
                            // 自分のToken（tokenIdx == idx）の場合は色を赤にする
                            if (tokenIdx === idx) {{
                                circle.style.backgroundColor = '#F44336'; // 赤
                            }} else {{
                                circle.style.backgroundColor = '#2196F3'; // 通常の青
                            }}
                            circle.title = `Token ${{idx}}: ${{tokenLabelAt(idx)}}\\nValue: ${{value.toFixed(4)}}`;
                            circleContainer.style.cursor = 'pointer';
                            
                            // 円の位置を記録（レンダリング後に更新）
                            if (!circlePositions[layer]) {{
                                circlePositions[layer] = {{}};
                            }}
                            if (!circlePositions[layer][tokenIdx]) {{
                                circlePositions[layer][tokenIdx] = {{}};
                            }}
                            // 位置は後で更新（circleContainerに追加された後）
                            circlePositions[layer][tokenIdx][idx] = {{
                                element: circle,
                                value: value,
                                size: normalizedSize
                            }};
                            
                            const contributionPercent = totalContribution > 0 ? ((Math.abs(value) / totalContribution) * 100).toFixed(2) : '0.00';
                            circle.dataset.contributionPct = contributionPercent + '%';
                            
                            // 選択されたTokenの円は透過率を下げる（周りより少し高い透過率）
                            if (tokenIdx === selectedToken) {{
                                circle.style.opacity = 0.9; // 周りより少し高い透過率（薄く）
                            }} else {{
                                // 親のwrapperの透過率に応じて設定（初期状態）
                                const distance = Math.abs(tokenIdx - selectedToken);
                                const maxDistance = Math.max(selectedToken !== null ? selectedToken : 0, numTokensToShow - 1 - (selectedToken !== null ? selectedToken : 0));
                                const maxOpacity = 1.0;
                                const minOpacity = 0.2;
                                if (selectedToken !== null && maxDistance > 0) {{
                                    const opacity = maxOpacity - ((distance / maxDistance) * (maxOpacity - minOpacity));
                                    circle.style.opacity = opacity;
                                }} else {{
                                    circle.style.opacity = 1.0;
                                }}
                            }}
                            
                            circleContainer.appendChild(circle);
                            bar.appendChild(circleContainer);
                        }} else {{
                            // 貢献度が0でも位置を保持（空のコンテナ）
                            bar.appendChild(circleContainer);
                        }}
                    }});
                
                    contributionBox.appendChild(bar);
                    wrapper.appendChild(contributionBox);
                    chunk.appendChild(wrapper);
                }}
            }}
                container.appendChild(chunk);
                nextLayer = endLayer;
                if (nextLayer < numLayersToShow) {{
                    requestAnimationFrame(buildLayerChunk);
                }} else if (typeof onComplete === 'function') {{
                    onComplete();
                }}
            }}
            buildLayerChunk();
            }}

            // 円の位置を更新し、Layer間の線を描画する関数
            function updateCirclePositionsAndDrawPaths() {{
                if (!showPaths) {{
                    if (pathSvg) pathSvg.innerHTML = '';
                    return;
                }}
                // まず、すべての円の位置を更新（元の位置）
                for (let layer = 0; layer < numLayersToShow; layer++) {{
                    if (!circlePositions[layer]) continue;
                    for (let tokenIdx in circlePositions[layer]) {{
                        if (!circlePositions[layer][tokenIdx]) continue;
                        for (let circleIdx in circlePositions[layer][tokenIdx]) {{
                            const circleData = circlePositions[layer][tokenIdx][circleIdx];
                            const circle = circleData.element;
                            if (!circle || !circle.parentElement) continue;
                            const rect = circle.getBoundingClientRect();
                            const containerRect = container.getBoundingClientRect();
                            circleData.x = rect.left - containerRect.left + rect.width / 2;
                            circleData.y = rect.top - containerRect.top + rect.height / 2;
                        }}
                    }}
                }}
                
                // 複製の円の位置も更新
                for (let layer = 0; layer < numLayersToShow; layer++) {{
                    if (!duplicateCirclePositions[layer]) continue;
                    for (let tokenIdx in duplicateCirclePositions[layer]) {{
                        if (!duplicateCirclePositions[layer][tokenIdx]) continue;
                        for (let circleIdx in duplicateCirclePositions[layer][tokenIdx]) {{
                            const circleData = duplicateCirclePositions[layer][tokenIdx][circleIdx];
                            const circle = circleData.element;
                            if (!circle || !circle.parentElement) continue;
                            const rect = circle.getBoundingClientRect();
                            const containerRect = container.getBoundingClientRect();
                            circleData.x = rect.left - containerRect.left + rect.width / 2;
                            circleData.y = rect.top - containerRect.top + rect.height / 2;
                        }}
                    }}
                }}
                
                // 既存のパスを削除
                pathSvg.innerHTML = '';
                
                // パスの表示カット（閾値）を統一設定
                // 閾値の計算式: pathThreshold = 1.0 / (numTokensToShow * 0.75)
                // 正規化された貢献度がこの閾値以下のパスは表示しない
                // 例: numTokensToShow = 10 の場合、pathThreshold = 1.0 / (10 * 0.75) = 1.0 / 7.5 ≈ 0.133
                const pathThreshold = 1.0 / (numTokensToShow * 0.75);
                
                // 全Layerの貢献度の最大値を計算（線の太さと透明度の基準）
                let maxValue = 0;
                const considerCircleValue = (circleData) => {{
                    if (!circleData) return;
                    const value = Math.abs(circleData.value);
                    if (value > maxValue) maxValue = value;
                }};
                for (let layer = 0; layer < numLayersToShow; layer++) {{
                    if (!circlePositions[layer]) continue;
                    for (let tokenIdx in circlePositions[layer]) {{
                        if (!circlePositions[layer][tokenIdx]) continue;
                        for (let circleIdx in circlePositions[layer][tokenIdx]) {{
                            considerCircleValue(circlePositions[layer][tokenIdx][circleIdx]);
                        }}
                    }}
                    if (!duplicateCirclePositions[layer]) continue;
                    for (let tokenIdx in duplicateCirclePositions[layer]) {{
                        if (!duplicateCirclePositions[layer][tokenIdx]) continue;
                        for (let circleIdx in duplicateCirclePositions[layer][tokenIdx]) {{
                            considerCircleValue(duplicateCirclePositions[layer][tokenIdx][circleIdx]);
                        }}
                    }}
                }}
                
                // 各Layerから次のLayerへ線を描画（元の位置）
                // 前のLayerの各Tokenの長方形内の全ての円から、次のLayerのそのToken自身の赤い円に繋ぐ
                for (let layer = 0; layer < numLayersToShow - 1; layer++) {{
                    const nextLayer = layer + 1;
                    if (!circlePositions[layer] || !circlePositions[nextLayer]) continue;
                    
                    // 前のLayerの各Tokenについて、そのTokenの長方形内の全ての円を取得
                    for (let sourceTokenIdx in circlePositions[layer]) {{
                        if (!circlePositions[layer][sourceTokenIdx]) continue;
                        
                        // 次のLayerのそのToken自身の赤い円を取得（sourceTokenIdx === targetTokenIdx）
                        if (!circlePositions[nextLayer][sourceTokenIdx]) continue;
                        const targetCircle = circlePositions[nextLayer][sourceTokenIdx][sourceTokenIdx];
                        if (!hasCircleCoords(targetCircle)) continue;
                        
                        // 前のLayerのsourceTokenIdxの長方形内の全ての円（各Tokenからの貢献度円）を取得
                        for (let circleIdx in circlePositions[layer][sourceTokenIdx]) {{
                            const sourceCircle = circlePositions[layer][sourceTokenIdx][circleIdx];
                            if (!hasCircleCoords(sourceCircle)) continue;
                            
                            // 貢献度の値に基づいて線の太さを決定（細い線は現状維持、太い線だけ圧縮）
                            const contributionValue = Math.abs(sourceCircle.value);
                            const minLineWidth = 1.5;
                            const normalizedValue = maxValue > 0 ? (contributionValue / maxValue) : 0;
                            
                            // 貢献度が閾値以下なら表示しない（統一された閾値を使用）
                            if (normalizedValue <= pathThreshold) continue;
                            
                            const lineWidth = contributionToPathWidth(
                                normalizedValue, minLineWidth, PATH_WIDTH_MAX_MAIN
                            );
                            
                            const normalizedValueSquared = normalizedValue * normalizedValue; // 2乗（透明度用）
                            
                            // 貢献度の値に基づいて線の透明度を決定（2乗で計算）
                            const minOpacity = 0.2;
                            const maxOpacity = 1.0;
                            const baseOpacity = minOpacity + normalizedValueSquared * (maxOpacity - minOpacity);
                            
                            // 選択されたTokenから遠いTokenに関連するパスの透明度を下げる
                            // 長方形の透明度と同じロジックを使用
                            let finalOpacity = baseOpacity;
                            if (selectedToken !== null) {{
                                const tokenDistance = Math.abs(parseInt(sourceTokenIdx) - selectedToken);
                                const maxDistance = Math.max(selectedToken, numTokensToShow - 1 - selectedToken);
                                const maxWrapperOpacity = 1.0;
                                const minWrapperOpacity = 0.2;
                                
                                // 長方形の透明度と同じ計算（距離に基づく透明度）
                                const distanceOpacity = maxDistance === 0 ? maxWrapperOpacity : 
                                    maxWrapperOpacity - ((tokenDistance / maxDistance) * (maxWrapperOpacity - minWrapperOpacity));
                                
                                // ベースの透明度（貢献度に基づく）と距離による透明度を掛け合わせる
                                finalOpacity = baseOpacity * distanceOpacity;
                            }}
                            
                            // 選択されたTokenに関連する線は赤色、それ以外は灰色
                            const lineColor = (selectedToken !== null && parseInt(sourceTokenIdx) === selectedToken) ? '#FF0000' : '#888';
                            
                            // 線を描画
                            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                            line.setAttribute('x1', sourceCircle.x);
                            line.setAttribute('y1', sourceCircle.y);
                            line.setAttribute('x2', targetCircle.x);
                            line.setAttribute('y2', targetCircle.y);
                            line.setAttribute('stroke', lineColor);
                            line.setAttribute('stroke-width', lineWidth);
                            line.setAttribute('opacity', finalOpacity);
                            pathSvg.appendChild(line);
                        }}
                    }}
                }}
                
                // 複製の円同士も線を描画
                // 前のLayerの選択されたTokenの長方形内の全ての円から、次のLayerの選択されたToken自身の赤い円に繋ぐ
                // 複製では、選択されたTokenの長方形内に全てのTokenからの貢献度円が含まれている
                if (selectedToken !== null) {{
                    for (let layer = 0; layer < numLayersToShow - 1; layer++) {{
                        const nextLayer = layer + 1;
                        if (!duplicateCirclePositions[layer] || !duplicateCirclePositions[nextLayer]) continue;
                        
                        // 複製では、選択されたTokenの長方形だけが存在する
                        const duplicateTokenIdx = selectedToken;
                        if (!duplicateCirclePositions[layer][duplicateTokenIdx] || !duplicateCirclePositions[nextLayer][duplicateTokenIdx]) continue;
                        
                        // 次のLayerの選択されたToken自身の赤い円を取得
                        const targetCircle = duplicateCirclePositions[nextLayer][duplicateTokenIdx][duplicateTokenIdx];
                        if (!hasCircleCoords(targetCircle)) continue;
                        
                        // 前のLayerの選択されたTokenの長方形内の全ての円（各Tokenからの貢献度円）を取得
                        for (let circleIdx in duplicateCirclePositions[layer][duplicateTokenIdx]) {{
                            const sourceCircle = duplicateCirclePositions[layer][duplicateTokenIdx][circleIdx];
                            if (!hasCircleCoords(sourceCircle)) continue;
                            
                            // 貢献度の値に基づいて線の太さを決定（細い線は現状維持、太い線だけ圧縮）
                            const contributionValue = Math.abs(sourceCircle.value);
                            const minLineWidth = 1.0;
                            const normalizedValue = maxValue > 0 ? (contributionValue / maxValue) : 0;
                            
                            // 貢献度が閾値以下なら表示しない（統一された閾値を使用）
                            if (normalizedValue <= pathThreshold) continue;
                            
                            const lineWidth = contributionToPathWidth(
                                normalizedValue, minLineWidth, PATH_WIDTH_MAX_DUP
                            );
                            
                            const normalizedValueSquared = normalizedValue * normalizedValue; // 2乗（透明度用）
                            
                            // 貢献度の値に基づいて線の透明度を決定（2乗で計算）
                            const minOpacity = 0.2;
                            const maxOpacity = 1.0;
                            const baseOpacity = minOpacity + normalizedValueSquared * (maxOpacity - minOpacity);
                            
                            // 選択されたTokenから遠いTokenに関連するパスの透明度を下げる
                            let finalOpacity = baseOpacity;
                            if (selectedToken !== null) {{
                                // 複製では、選択されたToken自身なので距離は0
                                const tokenDistance = 0;
                                const maxDistance = Math.max(selectedToken, numTokensToShow - 1 - selectedToken);
                                const maxWrapperOpacity = 1.0;
                                const minWrapperOpacity = 0.2;
                                
                                // 選択されたToken自身なので、距離による透明度の影響はなし
                                const distanceOpacity = maxWrapperOpacity;
                                
                                // ベースの透明度と距離による透明度の小さい方を採用
                                finalOpacity = Math.min(baseOpacity, distanceOpacity);
                            }}
                            
                            // 選択されたTokenに関連する線は赤色
                            const lineColor = '#FF0000';
                            
                            // 線を描画
                            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                            line.setAttribute('x1', sourceCircle.x);
                            line.setAttribute('y1', sourceCircle.y);
                            line.setAttribute('x2', targetCircle.x);
                            line.setAttribute('y2', targetCircle.y);
                            line.setAttribute('stroke', lineColor);
                            line.setAttribute('stroke-width', lineWidth);
                            line.setAttribute('opacity', finalOpacity);
                            pathSvg.appendChild(line);
                        }}
                    }}
                }}
                updateTokenAxisLabels();
            }}
            
            // ウィンドウリサイズ時にも更新（スロットル）
            window.addEventListener('resize', throttle(() => {{
                repositionDuplicatePanelIfPresent();
                updateCirclePositionsAndDrawPaths();
                syncContainerHeight();
                syncVizScrollCenter();
            }}, 150));
            
            // 横軸ラベルを表示（一番上のレイヤーのToken位置を基準に）
            const tokenAxisLabelsTop = document.getElementById('tokenAxisLabelsTop');
            const tokenAxisLabelsBottom = document.getElementById('tokenAxisLabelsBottom');
            
            // 表示上の最上段Layerの上端位置を計算（元のToken軸は複製の有無に影響されない）
            // top axis position via getTopAxisPosition()
            
            // 下の軸ラベルの位置調整用の変数（一括で変更可能）
            // bottom label gap: layoutBottomLabelOffset
            
            function setupAxisLabelContainers() {{
                [tokenAxisLabelsTop, tokenAxisLabelsBottom].forEach((el) => {{
                    if (!el) return;
                    el.style.position = 'absolute';
                    el.style.top = '0';
                    el.style.left = '0';
                    el.style.width = '100%';
                    el.style.height = '0';
                    el.style.pointerEvents = 'none';
                    el.style.zIndex = String(WRAPPER_Z_INDEX_BASE - 1);
                }});
            }}
            setupAxisLabelContainers();

            function collectAxisAnchorsFromRow(layerIdx, tokenRowIdx, edge) {{
                const anchors = [];
                const box = container.querySelector(
                    `.contribution-box[data-layer-idx="${{layerIdx}}"][data-token-idx="${{tokenRowIdx}}"]`
                );
                if (!box) return anchors;
                const bar = box.querySelector('.contribution-bar');
                if (!bar) return anchors;
                const containerRect = container.getBoundingClientRect();
                Array.from(bar.querySelectorAll('.contribution-circle-slot')).forEach((slot) => {{
                    const srcIdx = parseInt(slot.dataset.srcIdx, 10);
                    if (Number.isNaN(srcIdx) || srcIdx < 0 || srcIdx >= numTokens) return;
                    const rect = slot.getBoundingClientRect();
                    const left = rect.left - containerRect.left;
                    const top = edge === 'above'
                        ? rect.top - containerRect.top - layoutTopLabelOffset
                        : rect.bottom - containerRect.top + layoutBottomLabelOffset;
                    anchors.push({{
                        tokenIdx: srcIdx,
                        tokenName: tokenLabelAt(srcIdx),
                        left,
                        top,
                    }});
                }});
                return anchors;
            }}

            function createTokenAxisLabel(tokenIdx, tokenName, options = {{}}) {{
                const {{
                    variant = 'top',
                    rotationDeg = 45,
                    left = 0,
                    top = 0,
                }} = options;
                const label = document.createElement('div');
                label.className = `token-axis-label token-axis-label-${{variant}}`;
                label.dataset.tokenIdx = tokenIdx;
                label.style.cursor = 'pointer';
                if (selectedToken !== null && tokenIdx === selectedToken) {{
                    label.style.color = '#FF0000';
                    label.style.fontWeight = 'bold';
                }} else {{
                    label.style.color = '#333';
                    label.style.fontWeight = 'normal';
                }}
                label.textContent = tokenName;
                label.style.left = left + 'px';
                label.style.top = top + 'px';
                label.style.transform = `rotate(${{rotationDeg}}deg)`;
                label.addEventListener('click', function(e) {{
                    e.stopPropagation();
                    updateSelectedToken(tokenIdx);
                }});
                return label;
            }}

            const sourceAxisCaptionOffset = 28;
            const axisCaptionEdgeMargin = 10;
            const axisCaptionLeftGutterMin = 160;
            const duplicatePanelRightMargin = 48;

            function setContainerWidthTransition(enabled) {{
                if (enabled) {{
                    container.classList.add('z2z-container--animate');
                    if (vizLayoutTransitionTimer) clearTimeout(vizLayoutTransitionTimer);
                    vizLayoutTransitionTimer = setTimeout(() => {{
                        container.classList.remove('z2z-container--animate');
                        vizLayoutTransitionTimer = null;
                    }}, VIZ_LAYOUT_TRANSITION_MS + 80);
                    return;
                }}
                if (vizLayoutTransitionTimer) {{
                    clearTimeout(vizLayoutTransitionTimer);
                    vizLayoutTransitionTimer = null;
                }}
                container.classList.remove('z2z-container--animate');
            }}

            function syncVizScrollGutter(gutterPx) {{
                if (!vizScrollEl) return;
                const gutter = typeof gutterPx === 'number'
                    ? gutterPx
                    : (parseFloat(container.style.paddingLeft) || axisCaptionLeftGutterMin);
                vizScrollEl.style.setProperty('--z2z-left-gutter', Math.max(0, Math.round(gutter)) + 'px');
            }}

            function getVizScrollMinLead() {{
                if (!vizScrollEl) return 0;
                const raw = getComputedStyle(vizScrollEl).getPropertyValue('--z2z-left-gutter').trim();
                const parsed = parseFloat(raw);
                return Number.isFinite(parsed) ? Math.max(0, parsed) : 0;
            }}

            function resetVizScrollSpacers() {{
                if (vizScrollLeadEl) {{
                    vizScrollLeadEl.style.flex = '';
                    vizScrollLeadEl.style.width = '';
                }}
                if (vizScrollTrailEl) {{
                    vizScrollTrailEl.style.flex = '';
                    vizScrollTrailEl.style.width = '';
                }}
            }}

            function measureSourceCaptionOffsetInBody() {{
                const topCap = document.getElementById('tokenAxisCaptionTop');
                if (!topCap || !vizScrollBodyEl) return null;
                const bodyRect = vizScrollBodyEl.getBoundingClientRect();
                const capRect = topCap.getBoundingClientRect();
                if (!Number.isFinite(capRect.width)) return null;
                return capRect.left + capRect.width / 2 - bodyRect.left;
            }}

            function measureSourceCaptionCenterInScroll() {{
                const topCap = document.getElementById('tokenAxisCaptionTop');
                if (!topCap || !vizScrollEl) return null;
                const scrollRect = vizScrollEl.getBoundingClientRect();
                const capRect = topCap.getBoundingClientRect();
                if (!Number.isFinite(capRect.width)) return null;
                return capRect.left + capRect.width / 2 - scrollRect.left + vizScrollEl.scrollLeft;
            }}

            function syncVizScrollCenter() {{
                if (!vizScrollEl || !vizScrollBodyEl) return;
                const captionOffset = measureSourceCaptionOffsetInBody();
                if (captionOffset === null) return;

                const minLead = getVizScrollMinLead();
                const bodyWidth = vizScrollBodyEl.offsetWidth;
                const clientWidth = vizScrollEl.clientWidth;
                if (clientWidth <= 0) return;
                const viewportCenter = clientWidth / 2;

                resetVizScrollSpacers();
                void vizScrollEl.offsetWidth;

                const overflows = (minLead + bodyWidth) > clientWidth;
                if (!overflows) {{
                    const desiredLead = Math.max(minLead, viewportCenter - captionOffset);
                    const trailWidth = Math.max(0, clientWidth - desiredLead - bodyWidth);
                    if (vizScrollLeadEl) {{
                        vizScrollLeadEl.style.flex = '0 0 auto';
                        vizScrollLeadEl.style.width = Math.round(desiredLead) + 'px';
                    }}
                    if (vizScrollTrailEl) {{
                        vizScrollTrailEl.style.flex = '0 0 auto';
                        vizScrollTrailEl.style.width = Math.round(trailWidth) + 'px';
                    }}
                    vizScrollEl.scrollLeft = 0;
                    return;
                }}

                const captionCenter = measureSourceCaptionCenterInScroll();
                if (captionCenter === null) return;
                const maxScrollLeft = Math.max(0, vizScrollEl.scrollWidth - clientWidth);
                vizScrollEl.scrollLeft = Math.max(0, Math.min(maxScrollLeft, captionCenter - viewportCenter));
            }}

            function measureLayerLabelLeftOverflow() {{
                let overflow = 0;
                const containerRect = container.getBoundingClientRect();
                container.querySelectorAll('.layer-label:not(.layer-label-duplicate)').forEach(el => {{
                    const rect = el.getBoundingClientRect();
                    overflow = Math.max(overflow, axisCaptionEdgeMargin - (rect.left - containerRect.left));
                }});
                return overflow;
            }}

            function estimateDuplicatePanelRightEdge(tokenIdx) {{
                const labelWidth = 120;
                const gap = 10;
                const boxPaddingLeft = 10;
                const boxPaddingRight = 10;
                const circleWidth = circleSlotSize;
                const estimatedBoxWidth = labelWidth + gap + boxPaddingLeft + (numTokens * circleWidth) + boxPaddingRight;
                const horizontalSpacing = getTokenHorizontalSpacing();
                const rightOffset = tokenIdx * horizontalSpacing;
                const finalLeft = rightOffset + estimatedBoxWidth + 150;
                return finalLeft + estimatedBoxWidth + 24;
            }}

            function measureContainerContentWidth() {{
                const leftGutter = parseFloat(container.style.paddingLeft) || axisCaptionLeftGutterMin;
                const rightPad = parseFloat(container.style.paddingRight) || 0;
                const tokenLabelWidth = 120;
                const boxPad = 20;
                const vizWidth = tokenLabelWidth + 10 + boxPad + numTokensToShow * circleSlotSize + 40;
                let width = leftGutter + vizWidth + rightPad;
                const duplicateOverallContainer = document.querySelector('.duplicate-overall-container');
                if (duplicateOverallContainer) {{
                    const layoutRight = duplicateOverallContainer.offsetLeft + duplicateOverallContainer.offsetWidth;
                    width = Math.max(width, layoutRight + duplicatePanelRightMargin);
                    const containerRect = container.getBoundingClientRect();
                    const dupRect = duplicateOverallContainer.getBoundingClientRect();
                    width = Math.max(width, dupRect.right - containerRect.left + duplicatePanelRightMargin);
                }} else if (selectedToken !== null) {{
                    width = Math.max(width, estimateDuplicatePanelRightEdge(selectedToken) + duplicatePanelRightMargin);
                }}
                return width;
            }}

            function animateDuplicatePanelsOut(done) {{
                const panels = Array.from(document.querySelectorAll('.duplicate-overall-container'));
                if (panels.length === 0) {{
                    if (typeof done === 'function') done();
                    return;
                }}
                setContainerWidthTransition(true);
                panels.forEach(panel => panel.classList.add('z2z-dup-leaving'));
                setTimeout(() => {{
                    panels.forEach(panel => panel.remove());
                    if (typeof done === 'function') done();
                }}, VIZ_LAYOUT_TRANSITION_MS);
            }}

            function finishDuplicatePanelIn(duplicateOverallContainer) {{
                if (!duplicateOverallContainer) return;
                duplicateOverallContainer.classList.add('z2z-dup-entering');
                setContainerWidthTransition(true);
                scheduleRecalcLayoutMetrics();
                requestAnimationFrame(() => {{
                    requestAnimationFrame(() => {{
                        duplicateOverallContainer.classList.remove('z2z-dup-entering');
                        scheduleRecalcLayoutMetrics();
                        syncContainerHeight();
                        syncVizScrollCenter();
                    }});
                }});
            }}

            function getMidLayerCenterY() {{
                const midLayer = Math.floor(numLayersToShow / 2);
                return getLayerTop(midLayer) + circleSlotSize / 2;
            }}

            function repositionTargetAxisCaptionY() {{
                const targetCap = document.getElementById('targetTokenAxisCaption');
                if (!targetCap) return;
                positionTargetAxisCaption(targetCap, getMidLayerCenterY());
            }}

            function applyContainerLeftGutter() {{
                const targetCap = document.getElementById('targetTokenAxisCaption');
                let gutter = axisCaptionLeftGutterMin;
                container.style.paddingLeft = gutter + 'px';
                if (!targetCap) {{
                    syncVizScrollGutter(gutter);
                    return gutter;
                }}

                repositionTargetAxisCaptionY();
                const captionHeight = targetCap.offsetHeight;
                const captionWidth = targetCap.offsetWidth;
                const heuristic = Math.max(
                    axisCaptionLeftGutterMin,
                    captionHeight + captionWidth + axisCaptionEdgeMargin + 8);
                gutter = Math.max(gutter, heuristic);

                for (let attempt = 0; attempt < 6; attempt += 1) {{
                    container.style.paddingLeft = gutter + 'px';
                    repositionTargetAxisCaptionY();
                    const containerRect = container.getBoundingClientRect();
                    const captionRect = targetCap.getBoundingClientRect();
                    const deficit = axisCaptionEdgeMargin - (captionRect.left - containerRect.left);
                    if (deficit <= 0.5) break;
                    gutter += Math.ceil(deficit);
                }}
                const layerOverflow = measureLayerLabelLeftOverflow();
                if (layerOverflow > 0) {{
                    gutter += Math.ceil(layerOverflow);
                    container.style.paddingLeft = gutter + 'px';
                    repositionTargetAxisCaptionY();
                }}
                syncVizScrollGutter(gutter);
                return gutter;
            }}

            function positionTargetAxisCaption(el, midLayerCenterY) {{
                if (!el) return;
                el.style.transform = 'rotate(-90deg)';
                el.style.transformOrigin = 'left center';
                const captionHeight = el.offsetHeight;
                const captionWidth = el.offsetWidth;
                el.style.left = (-captionHeight) + 'px';
                el.style.top = (midLayerCenterY - captionWidth / 2) + 'px';
            }}

            function clampCaptionLeftEdge(el, marginPx) {{
                if (!el) return;
                const containerRect = container.getBoundingClientRect();
                const rect = el.getBoundingClientRect();
                const clip = marginPx - (rect.left - containerRect.left);
                if (clip <= 0) return;
                const left = parseFloat(el.style.left) || 0;
                el.style.left = (left + clip) + 'px';
            }}

            function positionSourceAxisCaption(el, centerX, topY) {{
                if (!el) return;
                el.style.transform = 'translateX(-50%)';
                el.style.textAlign = 'center';
                el.style.left = centerX + 'px';
                el.style.top = topY + 'px';
                clampCaptionLeftEdge(el, axisCaptionEdgeMargin);
            }}

            function measureTokenAxisLabelBounds(selector) {{
                const labels = Array.from(document.querySelectorAll(selector));
                if (labels.length === 0) return null;
                const containerRect = container.getBoundingClientRect();
                let minX = Infinity;
                let maxX = -Infinity;
                let minY = Infinity;
                let maxY = -Infinity;
                labels.forEach(el => {{
                    const rect = el.getBoundingClientRect();
                    minX = Math.min(minX, rect.left - containerRect.left);
                    maxX = Math.max(maxX, rect.right - containerRect.left);
                    minY = Math.min(minY, rect.top - containerRect.top);
                    maxY = Math.max(maxY, rect.bottom - containerRect.top);
                }});
                return {{
                    centerX: (minX + maxX) / 2,
                    minY,
                    maxY,
                }};
            }}

            function updateAxisRoleCaptions() {{
                const targetCap = document.getElementById('targetTokenAxisCaption');
                const topCap = document.getElementById('tokenAxisCaptionTop');
                const bottomCap = document.getElementById('tokenAxisCaptionBottom');
                if (!targetCap || !topCap || !bottomCap) return;

                const leftGutter = applyContainerLeftGutter();
                positionTargetAxisCaption(targetCap, getMidLayerCenterY());

                const topBounds = measureTokenAxisLabelBounds('.token-axis-label-top');
                if (topBounds) {{
                    positionSourceAxisCaption(
                        topCap,
                        topBounds.centerX,
                        topBounds.minY - sourceAxisCaptionOffset);
                }} else {{
                    positionSourceAxisCaption(
                        topCap,
                        leftGutter + 80,
                        getTopAxisPosition() - sourceAxisCaptionOffset);
                }}

                const bottomBounds = measureTokenAxisLabelBounds('.token-axis-label-bottom');
                const lastRow = Math.max(0, numTokensToShow - 1);
                if (bottomBounds) {{
                    positionSourceAxisCaption(
                        bottomCap,
                        bottomBounds.centerX,
                        bottomBounds.maxY + sourceAxisCaptionOffset);
                }} else {{
                    positionSourceAxisCaption(
                        bottomCap,
                        leftGutter + 80,
                        getLayerTop(0) + lastRow * getTokenVerticalSpacing() + circleSlotSize
                            + layoutBottomLabelOffset + sourceAxisCaptionOffset);
                }}
                syncVizScrollCenter();
            }}

            function updateTokenAxisLabels() {{
                document.querySelectorAll('.token-axis-label-top, .token-axis-label-bottom').forEach((label) => label.remove());
                setupAxisLabelContainers();

                const axisRotTop = z2zLayoutRotate90 ? -90 : 45;
                const axisRotBottom = z2zLayoutRotate90 ? -90 : 45;

                const topAnchors = collectAxisAnchorsFromRow(topLayer, 0, 'above');
                const topPositions = topAnchors.length > 0 ? topAnchors : topLayerTokenPositions.map(({{ tokenIdx, tokenName, position }}) => ({{
                    tokenIdx,
                    tokenName,
                    left: position,
                    top: getTopAxisPosition(),
                }}));

                topPositions.forEach(({{ tokenIdx, tokenName, left, top }}) => {{
                    const labelTop = createTokenAxisLabel(tokenIdx, tokenName, {{
                        variant: 'top',
                        rotationDeg: axisRotTop,
                        left,
                        top,
                    }});
                    tokenAxisLabelsTop.appendChild(labelTop);
                }});

                const lastRow = numTokensToShow - 1;
                const bottomAnchors = collectAxisAnchorsFromRow(0, lastRow, 'below');
                const bottomPositions = bottomAnchors.length > 0 ? bottomAnchors : bottomLayerTokenPositions.map(({{ tokenIdx, tokenName, position }}) => ({{
                    tokenIdx,
                    tokenName,
                    left: position,
                    top: getLayerTop(0) + lastRow * getTokenVerticalSpacing() + circleSlotSize + layoutBottomLabelOffset,
                }}));

                bottomPositions.forEach(({{ tokenIdx, tokenName, left, top }}) => {{
                    const labelBottom = createTokenAxisLabel(tokenIdx, tokenName, {{
                        variant: 'bottom',
                        rotationDeg: axisRotBottom,
                        left,
                        top,
                    }});
                    tokenAxisLabelsBottom.appendChild(labelBottom);
                }});
                updateAxisRoleCaptions();
                syncContainerHeight();
            }}

            // 複製の位置にToken軸ラベルを表示する関数
            function updateDuplicateAxisLabels() {{
                // 既存の複製用ラベルを削除
                document.querySelectorAll('.token-axis-label-duplicate').forEach(label => label.remove());
                
                if (selectedToken !== null) {{
                    // 複製のToken軸ラベルコンテナを取得
                    const duplicateAxisContainer = document.getElementById('duplicateTokenAxisLabelsTop');
                    if (!duplicateAxisContainer) {{
                        return;
                    }}
                    
                    // 複製の位置を計算（元の位置から右にずらす）
                    const labelWidth = 120;
                    const gap = 10;
                    const boxPaddingLeft = 10;
                    const boxPaddingRight = 10;
                    const circleWidth = circleSlotSize;
                    const estimatedBoxWidth = labelWidth + gap + boxPaddingLeft + (numTokens * circleWidth) + boxPaddingRight;
                    const horizontalSpacing = getTokenHorizontalSpacing();
                    const rightOffset = selectedToken * horizontalSpacing;
                    const duplicateOffset = rightOffset + estimatedBoxWidth + 100; // 複製の左端位置
                    
                    // 各Tokenの円の位置を計算してラベルを表示（元の位置と同じ位置計算を使用）
                    topLayerTokenPositions.forEach(({{tokenIdx, tokenName, position}}) => {{
                        // コンテナ内での相対位置を計算
                        // 元の位置計算: rightOffset + labelWidth + gap + boxPaddingLeft + circlePosition + circleCenter
                        // 複製では、rightOffsetが0になるように相対位置を計算
                        const labelWidth = 120;
                        const gap = 10;
                        const boxPaddingLeft = 10;
                        const circlePosition = tokenIdx * circleSlotSize;
                        const circleCenter = circleCenterOffset;
                        // 複製コンテナ内での相対位置（rightOffsetを除いた部分）
                        const duplicatePosition = labelWidth + gap + boxPaddingLeft + circlePosition + circleCenter;
                        
                        // 複製用のラベルを作成
                        const duplicateLabel = document.createElement('div');
                        duplicateLabel.className = 'token-axis-label token-axis-label-duplicate';
                        duplicateLabel.dataset.tokenIdx = tokenIdx;
                        
                        // 選択されたTokenの場合は赤文字・太文字
                        if (selectedToken !== null && tokenIdx === selectedToken) {{
                            duplicateLabel.style.color = '#FF0000';
                            duplicateLabel.style.fontWeight = 'bold';
                        }} else {{
                            duplicateLabel.style.color = '#333';
                            duplicateLabel.style.fontWeight = 'normal';
                        }}
                        
                        duplicateLabel.textContent = tokenName;
                        duplicateLabel.style.position = 'absolute';
                        duplicateLabel.style.left = duplicatePosition + 'px';
                        duplicateLabel.style.top = '0px'; // コンテナ内での相対位置（コンテナの上端）
                        duplicateLabel.style.transform = z2zLayoutRotate90
                            ? 'translateX(-50%) rotate(-90deg)'
                            : 'translateX(-50%) rotate(45deg)';
                        duplicateLabel.style.transformOrigin = 'center bottom';
                        duplicateLabel.style.opacity = '1'; // 即座に表示
                        duplicateLabel.style.cursor = 'pointer'; // クリック可能であることを示す
                        
                        // クリックイベントを追加
                        duplicateLabel.addEventListener('click', function(e) {{
                            e.stopPropagation();
                            updateSelectedToken(tokenIdx);
                        }});
                        duplicateAxisContainer.appendChild(duplicateLabel);
                    }});
                }}
            }}
            
{switch_sample_js}
            function bindLayoutSliders() {{
                const sliders = [
                    ['tokenOffsetX', 'tokenOffsetXVal', v => layoutTokenOffsetX = parseFloat(v)],
                    ['tokenOffsetY', 'tokenOffsetYVal', v => layoutTokenOffsetY = parseFloat(v)],
                    ['layerSpacing', 'layerSpacingVal', v => layoutLayerSpacing = parseFloat(v)],
                ];
                sliders.forEach(([id, valId, setter]) => {{
                    const el = document.getElementById(id);
                    const valEl = document.getElementById(valId);
                    if (!el) return;
                    el.addEventListener('input', () => {{
                        setter(el.value);
                        if (valEl) valEl.textContent = el.value;
                        rebuildVisualizationLayout();
                    }});
                }});
            }}
            bindLayoutSliders();
            bindCircleSizeControls();
            bindDisplayOptionsPanelToggle();

            container.addEventListener('click', handleVizBackgroundClick);
            const layoutRoot = document.getElementById('z2z-layout-root');
            if (layoutRoot) layoutRoot.addEventListener('click', handleVizBackgroundClick);

            function runInitialVisualization() {{
                const vizRoot = document.getElementById('z2z-layout-root');
                if (vizRoot) vizRoot.classList.add('z2z-demo-loading');
                const boot = () => {{
                    recalcLayoutMetrics();
                    buildVisualization(() => {{
                        updateTokenAxisLabels();
                        requestAnimationFrame(() => {{
                            updateCirclePositionsAndDrawPaths();
                            syncContainerHeight();
                            syncVizScrollCenter();
                            if (vizRoot) vizRoot.classList.remove('z2z-demo-loading');
                        }});
                    }});
                }};
                if (typeof ensureDemoLoaded === 'function' && (!z2zData || !tokens)) {{
                    ensureDemoLoaded(currentSampleId, currentSourceId).then(demo => {{
                        if (!demo) {{
                            if (vizRoot) vizRoot.classList.remove('z2z-demo-loading');
                            return;
                        }}
                        z2zData = demo.z2zData;
                        tokens = demo.tokens;
                        boot();
                    }}).catch(() => {{
                        if (vizRoot) vizRoot.classList.remove('z2z-demo-loading');
                    }});
                }} else {{
                    scheduleDeferred(boot, 200);
                }}
            }}
            runInitialVisualization();

            (function initEmbedBridge() {{
                const queryEmbed = new URLSearchParams(window.location.search).get('embed') === '1';
                if (queryEmbed && !isEmbedPage) {{
                    document.documentElement.classList.add('lig-iframe-embed');
                    document.body.classList.add('lig-demo-embed');
                    const banner = document.querySelector('.lig-hero-banner');
                    if (banner) banner.style.display = 'none';
                    const panel = document.getElementById('displayOptionsPanel');
                    const toggle = document.getElementById('displayOptionsToggle');
                    const body = document.getElementById('displayOptionsBody');
                    if (panel) {{
                        panel.classList.add('is-collapsed');
                        panel.classList.remove('is-expanded');
                    }}
                    if (toggle) toggle.setAttribute('aria-expanded', 'false');
                    if (body) body.setAttribute('inert', '');
                }}
                if (!isEmbedPage && !queryEmbed) return;
                if (window.parent === window) return;
                const scheduleSync = () => requestAnimationFrame(syncContainerHeight);
                window.addEventListener('resize', scheduleSync);
                window.addEventListener('load', scheduleSync);
                if (typeof MutationObserver !== 'undefined') {{
                    const obs = new MutationObserver(scheduleSync);
                    obs.observe(document.body, {{ childList: true, subtree: true, attributes: true }});
                }}
                syncContainerHeight();
                setTimeout(syncContainerHeight, 250);
                setTimeout(syncContainerHeight, 1000);
            }})();
        </script>
        {site_particles_scripts}
    </body>
    </html>
    """
    return html



def render_z2z_html(
    z2z_data: List[List[List[float]]],
    tokens: List[str],
    *,
    title: str = "Layer-wise token contributions (z2z IG)",
    description: Optional[str] = None,
    layout_mode: str = "normal",
    circle_max_style: str = "slot_cap",
    target_token: Optional[int] = None,
    num_layers: int = 12,
) -> str:
    """Build standalone interactive HTML for z2z contribution maps."""
    return create_interactive_visualization(
        z2z_data=z2z_data,
        tokens=tokens,
        layer=0,
        target_token=target_token,
        num_layers=num_layers,
        show_paths=False,
        align_horizontal=False,
        title=title,
        description=description,
        layout_mode=layout_mode,
        circle_max_style=circle_max_style,
    )


def render_z2z_multi_html(
    demo_matrix: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    demo_sources: List[Dict[str, Any]],
    demo_sample_labels: Dict[str, str],
    initial_sample_id: Optional[str] = None,
    initial_source_id: Optional[str] = None,
    title: str = "",
    layout_mode: str = "normal",
    circle_max_style: str = "slot_cap",
    target_token: Optional[int] = None,
    num_layers: int = 12,
    show_site_banner: bool = True,
    embed_mode: bool = False,
) -> str:
    """Build standalone HTML with embedded sample×source data and client-side switch."""
    if not demo_matrix:
        raise ValueError("demo_matrix must not be empty")
    initial_sample_id = initial_sample_id or next(iter(demo_matrix))
    initial_source_id = initial_source_id or demo_sources[0]["id"]
    first = demo_matrix[initial_sample_id][initial_source_id]
    init_z2z = first.get("z2z_data", [])
    init_tokens = first.get("tokens", [])
    return create_interactive_visualization(
        z2z_data=init_z2z,
        tokens=init_tokens,
        layer=0,
        target_token=target_token,
        num_layers=num_layers,
        show_paths=False,
        align_horizontal=False,
        title=title,
        description=None,
        layout_mode=layout_mode,
        circle_max_style=circle_max_style,
        demo_matrix=demo_matrix,
        demo_sources=demo_sources,
        demo_sample_labels=demo_sample_labels,
        initial_sample_id=initial_sample_id,
        initial_source_id=initial_source_id,
        show_site_banner=show_site_banner and not embed_mode,
        embed_mode=embed_mode,
    )
