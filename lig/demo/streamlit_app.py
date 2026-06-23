"""Streamlit demo for paper z2z visualization."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from lig.viz.z2z_contribution import load_demo_payload, render_z2z_html

DEMO_JSON = Path(__file__).resolve().parents[2] / "examples/paper_demo/lig_z2z_zero.json"


def main() -> None:
    st.set_page_config(page_title="LIG z2z demo", layout="wide")
    st.title("LIG — layer-wise token contributions (paper example)")

    path = st.sidebar.text_input("JSON path", str(DEMO_JSON))
    layout = st.sidebar.radio("Layout", ["90° rotate (paper)", "Normal"], index=0)
    layout_mode = "rotate_90_cw" if layout.startswith("90") else "normal"

    data_path = Path(path)
    if not data_path.exists():
        st.error(f"File not found: {data_path}")
        return

    z2z_data, tokens, text = load_demo_payload(str(data_path))
    st.caption(text)
    html = render_z2z_html(
        z2z_data,
        tokens,
        title="LIG z2z (Zero baseline)",
        description=f"<p>{text}</p>",
        layout_mode=layout_mode,
    )
    components.html(html, height=900, scrolling=True)

    with st.expander("Raw JSON preview"):
        st.json(json.loads(data_path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
