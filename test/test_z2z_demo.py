"""The initial demo must render before any lazy data fetch."""

import json
import re

from lig.viz.z2z_contribution import render_z2z_multi_html


def test_initial_demo_keeps_inline_data_when_chunk_url_is_present():
    matrix = [[[1.0, 0.25], [0.5, 1.0]]]
    tokens = ["a", "b"]
    page = render_z2z_multi_html(
        {
            "initial": {
                "zero": {
                    "data_url": "demo_data/initial.json",
                    "z2z_data": matrix,
                    "tokens": tokens,
                },
            },
            "other": {"zero": {"data_url": "demo_data/other.json"}},
        },
        demo_sources=[{"id": "zero", "label": "Zero"}],
        demo_sample_labels={"initial": "Initial", "other": "Other"},
        initial_sample_id="initial",
        initial_source_id="zero",
    )
    payload = json.loads(re.search(r"const demoMatrix = (.*);", page).group(1))
    assert payload["initial"]["zero"]["z2zData"] == matrix
    assert payload["initial"]["zero"]["tokens"] == tokens
    assert payload["other"]["zero"]["dataUrl"] == "demo_data/other.json"
    assert "z2zData" not in payload["other"]["zero"]
