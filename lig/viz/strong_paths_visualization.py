"""
強いパス可視化のHTML版
"""

import html as html_module
import json
from typing import Dict, List, Optional

import numpy as np


def create_strong_paths_html(
    path_info_list: List[Dict],
    tokens: List[str],
    att_threshold: float = 0.2,
    mlp_threshold: float = 0.2,
    num_heads: int = 12,
    same_token_threshold_factor: float = 1.0,
    punctuation_att_threshold: float = None,
    lrp_overlay: Optional[Dict] = None,
    path_weight_mode: str = "ig",
    enable_node_click_lrp: bool = True,
    render_empty: bool = False,
    title: str = "BERT 強いパス可視化（Strong Paths Visualization）",
    description: str = None,
    input_texts: Optional[List[str]] = None,
) -> str:
    """
    強いパスをHTML/JavaScriptでインタラクティブに可視化する

    Args:
        path_info_list: 経路ステップのリスト
        tokens: トークンリスト
        att_threshold: ATT閾値
        mlp_threshold: MLP閾値
        num_heads: ヘッド数
        same_token_threshold_factor: 同一トークンATTの閾値倍率
        punctuation_att_threshold: 約物トークン経由ATTの閾値
        lrp_overlay: LRPオーバーレイデータ
        path_weight_mode: パス線の重み付けモード（"ig" or "ig_times_lrp"）
        title: HTMLのタイトル
        description: HTMLの説明文
        input_texts: 入力文のリスト（複数可）

    Returns:
        HTML文字列
    """
    if not path_info_list:
        if not render_empty:
            return "<div>No data</div>"
        steps = []
    else:
        steps = path_info_list
        if isinstance(steps, tuple):
            steps = steps[0]

    # データをJSON形式に変換
    steps_json = json.dumps(steps)
    tokens_json = json.dumps(tokens)

    # HTMLエスケープ処理（f-string内で使うため事前に処理）
    escaped_title = html_module.escape(title)
    escaped_description = html_module.escape(description) if description else None
    escaped_input_texts = (
        [html_module.escape(text) for text in input_texts] if input_texts else None
    )

    # LRPオーバーレイが無い場合、IGの貢献度からノードのAttributionを計算
    if not lrp_overlay and steps:
        # zノードのAttribution: そのzノードから出るすべてのATTステップのinfluenceの合計
        z_attr_layers = {}
        u_attr_layers = {}

        # 各層のzノードのAttributionを計算
        for layer in range(12):  # layer 0-11 (z12は別途計算)
            z_attr = np.zeros(len(tokens))
            for step in steps:
                step_type = str(step.get("type", "")).lower()
                step_layer = step.get("layer", 0)
                step_token = step.get("token", 0)
                influence = float(step.get("influence", 0.0))

                # ATTステップ: z(layer) → u(layer, head)
                if step_type in ("attn", "att", "attention") and step_layer == layer:
                    if 0 <= step_token < len(tokens):
                        z_attr[step_token] += influence

            if np.any(z_attr > 0):
                z_attr_layers[layer] = z_attr

        # z12のAttribution: z12に入るすべてのMLPステップ（u(11, h) → z(12)）のinfluenceの合計
        z12_attr = np.zeros(len(tokens))
        for step in steps:
            step_type = str(step.get("type", "")).lower()
            step_layer = step.get("layer", 0)
            step_token = step.get("token", 0)
            influence = float(step.get("influence", 0.0))

            # MLPステップ: u(11, head) → z(12)
            if step_type in ("mlp", "mlp_step", "mlp_final") and step_layer == 11:
                if 0 <= step_token < len(tokens):
                    z12_attr[step_token] += influence

        # z12をz_attr_layersに追加
        if np.any(z12_attr > 0):
            z_attr_layers[12] = z12_attr

        # 各層のuノードのAttributionを計算
        for layer in range(12):  # layer 0-11
            u_attr = np.zeros((len(tokens), num_heads))
            for step in steps:
                step_type = str(step.get("type", "")).lower()
                step_layer = step.get("layer", 0)
                step_token = step.get(
                    "token", 0
                )  # MLPステップではtokenがuノードのtoken
                step_head = step.get("head", 0)
                influence = float(step.get("influence", 0.0))

                # MLPステップ: u(layer, head) → z(layer+1)
                if (
                    step_type in ("mlp", "mlp_step", "mlp_final")
                    and step_layer == layer
                ):
                    if 0 <= step_token < len(tokens) and 0 <= step_head < num_heads:
                        u_attr[step_token, step_head] += influence

            if np.any(u_attr > 0):
                u_attr_layers[layer] = u_attr

        # final_output_z (z12) はz12のAttributionを使用
        final_output_z = z12_attr

        # 正規化（最大値で割る）
        if len(z_attr_layers) > 0:
            max_z = max(
                np.max(arr)
                for arr in z_attr_layers.values()
                if isinstance(arr, np.ndarray)
            )
            if max_z > 0:
                for layer in z_attr_layers:
                    if isinstance(z_attr_layers[layer], np.ndarray):
                        z_attr_layers[layer] = z_attr_layers[layer] / max_z
                final_output_z = (
                    final_output_z / max_z
                    if np.max(final_output_z) > 0
                    else final_output_z
                )

        if len(u_attr_layers) > 0:
            max_u = max(
                np.max(arr)
                for arr in u_attr_layers.values()
                if isinstance(arr, np.ndarray)
            )
            if max_u > 0:
                for layer in u_attr_layers:
                    if isinstance(u_attr_layers[layer], np.ndarray):
                        u_attr_layers[layer] = u_attr_layers[layer] / max_u

        # lrp_overlay形式に変換
        lrp_overlay = {
            "z_attr_layers": {k: v for k, v in z_attr_layers.items()},
            "u_attr_layers": {k: v for k, v in u_attr_layers.items()},
            "final_output_z": final_output_z,
            "z_threshold": 0.1,
            "u_threshold": 0.1,
            "target_token_idx": 0,
            "show_z_nodes": True,
            "show_u_nodes": True,
            "color_by_attr": True,
        }

    # LRPオーバーレイデータの変換
    lrp_data_json = "null"
    z_threshold_display = 0.1
    u_threshold_display = 0.1
    if lrp_overlay:
        z_threshold_display = lrp_overlay.get("z_threshold", 0.1)
        u_threshold_display = lrp_overlay.get("u_threshold", 0.1)
        lrp_data = {
            "z_attr_layers": {
                str(k): v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in lrp_overlay.get("z_attr_layers", {}).items()
            },
            "u_attr_layers": {
                str(k): v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in lrp_overlay.get("u_attr_layers", {}).items()
            },
            "final_output_z": (
                lrp_overlay.get("final_output_z").tolist()
                if isinstance(lrp_overlay.get("final_output_z"), np.ndarray)
                else lrp_overlay.get("final_output_z")
            ),
            "z_threshold": z_threshold_display,
            "u_threshold": u_threshold_display,
            "target_token_idx": lrp_overlay.get(
                "target_token_idx", 0
            ),  # ターゲットトークンインデックスを追加
        }
        lrp_data_json = json.dumps(lrp_data)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{escaped_title}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #ffffff; /* 全面白背景 */
                overflow: auto;
            }}
            h1 {{
                text-align: center;  /* タイトルを中央寄せ */
                margin: 16px 0 0 0;
            }}
            .container {{
                position: relative;
                width: 100%;
                min-height: auto;   /* 固定高さを廃止し可変に */
                height: auto;       /* コンテンツに追従 */
                background-color: #ffffff; /* 枠・角丸なし */
                border: none;
                border-radius: 0;
                padding: 0;
                overflow: auto;     /* 必要に応じてスクロール */
            }}
            .plot-area {{
                position: relative;
                margin-left: 150px; /* 90度回転版：左側にLayerラベル */
                margin-right: 50px;
                margin-top: 150px; /* 90度回転版：上側にTokenラベル */
                margin-bottom: 80px;
                border: 2px solid #000;  /* 枠線を復活 */
                overflow: visible;  /* ノードが軸からはみ出しても表示されるように */
                background-color: transparent; /* 背景を透明にして、レイヤーセクションの背景色を表示 */
            }}
            .token-row {{
                position: absolute;
                left: 0;
                width: 100%;
                background-color: transparent; /* 背景を透明にして、レイヤーセクションの背景（ATT/MLP）を表示 */
                box-sizing: border-box;
                z-index: 1; /* レイヤーセクションの上に配置 */
            }}
            .token-label {{
                position: absolute;
                top: -140px; /* 90度回転版：上側にTokenラベル */
                left: 50%;
                transform: translateX(-50%);
                transform-origin: center center;
                font-weight: bold;
                font-size: 12px;
                white-space: nowrap;
                padding: 2px 5px;
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 4px;
                z-index: 10;
                width: auto;
                text-align: center;
                display: inline-block;
            }}
            .head-label {{
                position: absolute;
                top: -80px; /* 90度回転版：上側にHeadラベル */
                font-size: 10px;
                color: #666;
                font-weight: bold;
                z-index: 10;
                width: 60px;
                text-align: center;
                transform: rotate(-90deg); /* Headラベルを90度回転 */
                transform-origin: center center;
            }}
            .head-divider {{
                stroke: #444;
                stroke-width: 0.5;
                stroke-dasharray: 2,2;
                opacity: 0.6;
            }}
            .node-circle {{
                fill: blue;
                stroke: black;
                stroke-width: 1.5;
                cursor: pointer;
                pointer-events: all;  /* ノード全体でマウスイベントを受け取る */
            }}
            .node-circle:hover {{
                stroke-width: 2.5;
            }}
            .path-line {{
                fill: none;
                cursor: pointer;
                pointer-events: stroke;
            }}
            .path-line:hover {{
                opacity: 1;
            }}
            .tooltip {{
                position: fixed;  /* 単体HTML用。Streamlit iframe 内では JS が absolute に切替 */
                background-color: rgba(0, 0, 0, 0.9);
                color: white;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 12px;
                pointer-events: none;
                z-index: 10000;
                display: none;
            }}
            .x-axis-label {{
                position: absolute;
                left: -30px; /* 90度回転版：左側にLayerラベル */
                font-size: 11px;
                font-weight: bold;
                text-align: center;
                transform: translateY(-50%) rotate(-90deg);
                transform-origin: center center;
            }}
            .section-label {{
                position: absolute;
                left: -20px; /* 90度回転版：左側にATT/MLPラベル */
                font-size: 10px;
                color: #666;
                text-align: center;
                transform: translateY(-50%) rotate(-90deg);
                transform-origin: center center;
            }}
            .layer-label {{
                position: absolute;
                left: -80px; /* 90度回転版：左側にLayerラベル（さらに左に移動） */
                font-size: 10px;
                font-weight: bold;
                text-align: center;
                transform: translateY(-50%) rotate(-90deg);
                transform-origin: center center;
            }}
            .settings-panel {{
                display: flex;
                gap: 20px;
                margin: 16px auto;
                padding: 20px;
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 6px;
                max-width: 1400px;
                font-size: 14px;
            }}
            .settings-panel h2 {{
                margin: 0 0 12px 0;
                font-size: 16px;
                font-weight: bold;
                color: #333;
            }}
            .settings-left {{
                flex: 1;
                min-width: 0;
            }}
            .settings-right {{
                flex: 1;
                min-width: 0;
                padding-left: 20px;
                border-left: 2px solid #ddd;
            }}
            .settings-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 16px;
                margin-top: 8px;
            }}
            .visualization-info {{
                background-color: white;
                padding: 16px;
                border-radius: 4px;
                border: 1px solid #ccc;
            }}
            .visualization-info h3 {{
                margin: 0 0 12px 0;
                font-size: 15px;
                font-weight: bold;
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 8px;
            }}
            .visualization-info-item {{
                margin: 12px 0;
            }}
            .visualization-info-label {{
                font-weight: bold;
                color: #555;
                font-size: 13px;
                margin-bottom: 4px;
            }}
            .visualization-info-content {{
                color: #2c3e50;
                font-size: 14px;
                line-height: 1.6;
                padding: 8px;
                background-color: #f8f9fa;
                border-radius: 4px;
                border-left: 3px solid #3498db;
            }}
            .visualization-info-text {{
                color: #2c3e50;
                font-size: 13px;
                line-height: 1.6;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            .visualization-explanation-container {{
                display: flex;
                flex-wrap: nowrap;
                gap: 16px;
                margin-top: 20px;
                overflow-x: auto;
            }}
            .visualization-explanation-box {{
                flex: 0 0 600px;
                min-width: 600px;
                background-color: white;
                padding: 16px;
                border-radius: 4px;
                border: 1px solid #ccc;
            }}
            .visualization-explanation-box h3 {{
                margin: 0 0 12px 0;
                font-size: 15px;
                font-weight: bold;
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 8px;
            }}
            .setting-item {{
                display: flex;
                flex-direction: column;
                padding: 10px 12px;
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 4px;
                position: relative;
                cursor: help;
            }}
            .setting-item.has-help:hover {{
                border-color: #3498db;
                box-shadow: 0 2px 4px rgba(52, 152, 219, 0.2);
            }}
            .setting-row {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 6px;
            }}
            .setting-label {{
                font-weight: bold;
                color: #555;
                font-size: 13px;
            }}
            .setting-value {{
                font-family: 'Courier New', monospace;
                color: #0066cc;
                font-weight: bold;
                font-size: 13px;
                min-width: 60px;
                text-align: right;
            }}
            .setting-slider {{
                width: 100%;
                margin-top: 4px;
            }}
            .help-tooltip {{
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                margin-bottom: 8px;
                padding: 8px 12px;
                background-color: #2c3e50;
                color: white;
                border-radius: 4px;
                font-size: 12px;
                opacity: 0;
                pointer-events: none;
                transition: opacity 0.2s;
                z-index: 1000;
                max-width: 300px;
                white-space: normal;
                line-height: 1.4;
            }}
            .setting-item.has-help:hover .help-tooltip {{
                opacity: 1;
            }}
            .help-tooltip::after {{
                content: '';
                position: absolute;
                top: 100%;
                left: 50%;
                transform: translateX(-50%);
                border: 6px solid transparent;
                border-top-color: #2c3e50;
            }}
            .info-panel {{
                margin: 16px auto;
                padding: 16px 20px;
                background-color: #e8f4f8;
                border: 1px solid #b3d9e6;
                border-radius: 6px;
                max-width: 1000px;
            }}
            .info-panel h2 {{
                margin: 0 0 12px 0;
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
            }}
            .info-panel p {{
                margin: 8px 0;
                color: #34495e;
                line-height: 1.6;
            }}
            .input-texts-section {{
                margin-top: 16px;
                padding-top: 16px;
                border-top: 2px solid #b3d9e6;
            }}
            .input-text-item {{
                margin: 12px 0;
                padding: 12px;
                background-color: white;
                border: 1px solid #d0e8f2;
                border-radius: 4px;
                border-left: 4px solid #3498db;
            }}
            .input-text-label {{
                font-weight: bold;
                color: #2980b9;
                margin-bottom: 6px;
                font-size: 14px;
            }}
            .input-text-content {{
                color: #2c3e50;
                font-size: 15px;
                line-height: 1.5;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
        </style>
    </head>
    <body>
        <h1>{escaped_title}</h1>
        {f'<div class="info-panel"><h2>説明</h2><p>{escaped_description}</p></div>' if escaped_description else ''}
        {f'<div class="info-panel"><h2>入力文</h2><div class="input-texts-section">' + ''.join([f'<div class="input-text-item"><div class="input-text-label">入力文 {i+1}:</div><div class="input-text-content">{text}</div></div>' for i, text in enumerate(escaped_input_texts)]) + '</div></div>' if escaped_input_texts else ''}
        <div class="settings-panel">
            <div class="settings-left">
                <h2>設定</h2>
                <div style="margin-bottom: 16px; padding: 12px; background-color: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; font-size: 13px; line-height: 1.6;">
                    <strong>💡 閾値とは：</strong>閾値は、可視化に表示する情報をフィルタリングするための基準値です。影響度（influence）やAttribution値が閾値以上のパスやノードのみが表示されます。閾値を上げると表示されるパスが減り、より重要な情報のみが表示されます。閾値を下げるとより多くのパスが表示されますが、ノイズが増える可能性があります。
                </div>
                <div class="settings-grid">
                <div class="setting-item has-help">
                    <span class="help-tooltip">Attentionの強いパス判定に用いる閾値。影響度（influence）がこの値以上のATTステップ（z(l) → u(l,h)）のみが可視化に表示されます。値が大きいほど、より重要なパスのみが表示されます。</span>
                    <div class="setting-row">
                        <span class="setting-label">強いパスの閾値（ATT）:</span>
                        <span class="setting-value" id="attThresholdDisplay">{att_threshold:.3f}</span>
                    </div>
                    <input type="range" min="0" max="1" step="0.01" value="{att_threshold}" 
                           class="setting-slider" id="attThresholdSlider">
                </div>
                <div class="setting-item has-help">
                    <span class="help-tooltip">MLPの強いパス判定に用いる閾値。影響度（influence）がこの値以上のMLPステップ（u(l,h) → z(l+1)）のみが可視化に表示されます。値が大きいほど、より重要なパスのみが表示されます。</span>
                    <div class="setting-row">
                        <span class="setting-label">強いパスの閾値（MLP）:</span>
                        <span class="setting-value" id="mlpThresholdDisplay">{mlp_threshold:.3f}</span>
                    </div>
                    <input type="range" min="0" max="1" step="0.01" value="{mlp_threshold}" 
                           class="setting-slider" id="mlpThresholdSlider">
                </div>
                <div class="setting-item has-help">
                    <span class="help-tooltip">zノード（各層の出力）の表示に用いるしきい値。LRPで計算されたAttribution値がこの値以上のzノードのみが強調表示されます。zノードは各層の最終的な出力を表します。</span>
                    <div class="setting-row">
                        <span class="setting-label">zノードしきい値:</span>
                        <span class="setting-value" id="zThresholdDisplay">{z_threshold_display:.3f}</span>
                    </div>
                    <input type="range" min="0" max="1" step="0.001" value="{z_threshold_display}" 
                           class="setting-slider" id="zThresholdSlider">
                </div>
                <div class="setting-item has-help">
                    <span class="help-tooltip">uノード（Attention後の各ヘッドの出力）の表示に用いるしきい値。LRPで計算されたAttribution値がこの値以上のuノードのみが強調表示されます。uノードは各ヘッドのAttention後の状態を表します。</span>
                    <div class="setting-row">
                        <span class="setting-label">uノードしきい値:</span>
                        <span class="setting-value" id="uThresholdDisplay">{u_threshold_display:.3f}</span>
                    </div>
                    <input type="range" min="0" max="1" step="0.001" value="{u_threshold_display}" 
                           class="setting-slider" id="uThresholdSlider">
                </div>
                <div class="setting-item has-help">
                    <span class="help-tooltip">z_i(l)→u_i(l,h) の同一トークン内ATTにのみ適用する独立した閾値倍率。同一トークン内のAttention（自分自身へのAttention）の場合、ATT閾値にこの倍率を掛けた値が使用されます。通常は1.0以上に設定して、同一トークン内のAttentionをより厳しくフィルタリングします。</span>
                    <div class="setting-row">
                        <span class="setting-label">同一トークンATTの閾値:</span>
                        <span class="setting-value" id="sameTokenThresholdDisplay">{same_token_threshold_factor:.3f}</span>
                    </div>
                    <input type="range" min="0" max="10" step="0.1" value="{same_token_threshold_factor}" 
                           class="setting-slider" id="sameTokenThresholdSlider">
                </div>
                <div class="setting-item has-help">
                    <span class="help-tooltip">[CLS], [SEP], , . ? ! などの約物トークンが関与するATTにのみ適用する独立した閾値。ソーストークンまたはターゲットトークンのどちらかが約物の場合にこの閾値が使用されます。</span>
                    <div class="setting-row">
                        <span class="setting-label">約物トークン経由ATTの閾値:</span>
                        <span class="setting-value" id="punctuationThresholdDisplay">{(f'{punctuation_att_threshold:.3f}' if punctuation_att_threshold is not None else 'N/A')}</span>
                    </div>
                    <input type="range" min="0" max="1" step="0.01" value="{punctuation_att_threshold if punctuation_att_threshold is not None else 0.8}" 
                           class="setting-slider" id="punctuationThresholdSlider" 
                           {'disabled' if punctuation_att_threshold is None else ''}>
                </div>
                </div>
            </div>
            <div class="settings-right">
                <h2>可視化情報</h2>
                <div class="visualization-explanation-container">
                    <div class="visualization-explanation-box">
                        <h3>この可視化について</h3>
                        <div class="visualization-info-item">
                            <div class="visualization-info-label">可視化の種類</div>
                            <div class="visualization-info-content">BERT 強いパス可視化（Strong Paths Visualization）</div>
                        </div>
                        <div class="visualization-info-item">
                            <div class="visualization-info-label">説明</div>
                            <div class="visualization-info-text">{escaped_description if escaped_description else 'BERTモデル内の重要な情報伝達経路（強いパス）を可視化します。各層におけるMLP機構とAttention機構の情報伝達を、タイムライン形式で表示します。'}</div>
                        </div>
                        {f'<div class="visualization-info-item"><div class="visualization-info-label">入力文</div><div class="visualization-info-content">' + '<br>'.join([f'<strong>入力文 {i+1}:</strong> {text}' for i, text in enumerate(escaped_input_texts)]) + '</div></div>' if escaped_input_texts else ''}
                        <div class="visualization-info-item">
                            <div class="visualization-info-label">トークン数</div>
                            <div class="visualization-info-content">{len(tokens)}個</div>
                        </div>
                        <div class="visualization-info-item">
                            <div class="visualization-info-label">トークンリスト</div>
                            <div class="visualization-info-content" style="max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 12px;">{' '.join([f'{i}:{tok}' for i, tok in enumerate(tokens)])}</div>
                        </div>
                    </div>
                    <div class="visualization-explanation-box">
                        <h3>ノード強調（LRP Attribution）</h3>
                        <div class="visualization-info-item">
                            <div class="visualization-info-label">概要</div>
                            <div class="visualization-info-text">ノード（円形のマーカー）の色の濃さと透明度は、Layer-wise Relevance Propagation (LRP) で計算されたAttribution値を表します。</div>
                        </div>
                        <div class="visualization-info-item">
                            <div class="visualization-info-label">zノード</div>
                            <div class="visualization-info-text">各層の最終出力を表す青色の円。Attribution値が高いほど濃い青色で表示されます。</div>
                        </div>
                        <div class="visualization-info-item">
                            <div class="visualization-info-label">uノード</div>
                            <div class="visualization-info-text">各Attentionヘッドの出力を表す青色の円。Attribution値が高いほど濃い青色で表示されます。</div>
                        </div>
                        <div class="visualization-info-item">
                            <div class="visualization-info-label">表示方法</div>
                            <div class="visualization-info-text">Attribution値がしきい値以上のノードは強調表示され、しきい値未満のノードは極薄で表示されます（ホバーやクリックは可能です）。</div>
                        </div>
                        <div class="visualization-info-item">
                            <div class="visualization-info-label">計算方法</div>
                            <div class="visualization-info-text">LRP情報が利用できない場合は、Integrated Gradients (IG) の貢献度から自動計算されます。</div>
                        </div>
                    </div>
                    <div class="visualization-explanation-box">
                        <h3>パスの濃さと太さ</h3>
                        <div class="visualization-info-item">
                            <div class="visualization-info-label">概要</div>
                            <div class="visualization-info-text">パス（線）の視覚的特徴は、情報伝達の重要度を表します。</div>
                        </div>
                        <div class="visualization-info-item">
                            <div class="visualization-info-label">線の太さ</div>
                            <div class="visualization-info-text">Integrated Gradients (IG) で計算された影響度（influence）に比例します。影響度が大きいパスほど太い線で表示されます。</div>
                        </div>
                        <div class="visualization-info-item">
                            <div class="visualization-info-label">線の濃さ（透明度）</div>
                            <div class="visualization-info-text">影響度とLRP Attributionの組み合わせ（IG×LRP）に比例します。重要度が高いパスほど濃く（不透明に）表示されます。</div>
                        </div>
                        <div class="visualization-info-item">
                            <div class="visualization-info-label">MLPパス</div>
                            <div class="visualization-info-text">u(l,h) → z(l+1) の情報伝達を表す線。同じトークン内での情報処理を可視化します。</div>
                        </div>
                        <div class="visualization-info-item">
                            <div class="visualization-info-label">ATTパス</div>
                            <div class="visualization-info-text">z(l) → u(l,h) の情報伝達を表す線。異なるトークン間のAttention関係を可視化します。</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="container" id="container">
            <div class="plot-area" id="plotArea">
                <svg id="svg" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: auto; z-index: 5; overflow: visible;"></svg>
            </div>
        </div>
        <div class="tooltip" id="tooltip"></div>
        
        <script>
            const steps = {steps_json};
            const tokens = {tokens_json};
            const numHeads = {num_heads};
            const attThreshold = {att_threshold};
            const mlpThreshold = {mlp_threshold};
            const sameTokenThresholdFactor = {same_token_threshold_factor};
            const punctuationAttThreshold = {punctuation_att_threshold if punctuation_att_threshold is not None else 'null'};
            const pathWeightMode = "{path_weight_mode}";
            const lrpData = {lrp_data_json};
            
            // ツールチップの位置を調整する関数（画面外に出ないように）
            function setTooltipPosition(tooltip, event) {{
                const offset = 10;
                const padding = 5;
                
                // ツールチップを一時的に表示してサイズを取得
                tooltip.style.visibility = 'hidden';
                tooltip.style.display = 'block';
                const tooltipWidth = tooltip.offsetWidth;
                const tooltipHeight = tooltip.offsetHeight;
                tooltip.style.visibility = '';
                
                const inIframe = window.parent && window.parent !== window;
                // 単体HTML: fixed + client（ページ内スクロール対応）
                // Streamlit iframe: absolute + page（iframe全高展開・親スクロール対応）
                tooltip.style.position = inIframe ? 'absolute' : 'fixed';
                
                let pointerX, pointerY, boundLeft, boundTop, boundRight, boundBottom;
                if (inIframe) {{
                    pointerX = event.pageX;
                    pointerY = event.pageY;
                    const docRect = document.documentElement.getBoundingClientRect();
                    const scrollX = window.pageXOffset || document.documentElement.scrollLeft || 0;
                    const scrollY = window.pageYOffset || document.documentElement.scrollTop || 0;
                    const parentW = window.parent.innerWidth;
                    const parentH = window.parent.innerHeight;
                    const visibleLeft = Math.max(0, -docRect.left);
                    const visibleTop = Math.max(0, -docRect.top);
                    const visibleWidth = Math.min(docRect.right, parentW) - Math.max(docRect.left, 0);
                    const visibleHeight = Math.min(docRect.bottom, parentH) - Math.max(docRect.top, 0);
                    boundLeft = scrollX + visibleLeft + padding;
                    boundTop = scrollY + visibleTop + padding;
                    boundRight = scrollX + visibleLeft + visibleWidth - padding;
                    boundBottom = scrollY + visibleTop + visibleHeight - padding;
                }} else {{
                    pointerX = event.clientX;
                    pointerY = event.clientY;
                    boundLeft = padding;
                    boundTop = padding;
                    boundRight = window.innerWidth - padding;
                    boundBottom = window.innerHeight - padding;
                }}
                
                let x = pointerX + offset;
                let y = pointerY + offset;
                
                if (x + tooltipWidth > boundRight) {{
                    x = pointerX - tooltipWidth - offset;
                }}
                if (y + tooltipHeight > boundBottom) {{
                    y = pointerY - tooltipHeight - offset;
                }}
                if (x < boundLeft) {{
                    x = pointerX + offset;
                }}
                if (y < boundTop) {{
                    y = pointerY + offset;
                }}
                
                tooltip.style.left = x + 'px';
                tooltip.style.top = y + 'px';
            }}
            
            function hideTooltip() {{
                const tooltip = document.getElementById('tooltip');
                if (tooltip) tooltip.style.display = 'none';
            }}
            
            // ---- クリック起点で簡易LRPを再計算するための準備 ----
            // IGパス値をRelevanceとして扱い、クリックノードのRelevance=1を
            // 入力側へ正規化分配する簡易逆伝播（近似LRP）。
            // 背景はそのまま、ノードとパスのみ再描画する。
            const attIncomingByLayer = {{}};
            const mlpIncomingByLayer = {{}};
            
            // 現在選択されているノードを追跡（初期状態に戻すため）
            // selectedNode: オブジェクト {{nodeType: 'z'|'u', layer: number, token: number, head: number}} または null
            let selectedNode = null;
            
            // destKey helpers
            function uKey(layer, token, head) {{ return `u|${{layer}}|${{token}}|${{head}}`; }}
            function zKey(layer, token) {{ return `z|${{layer}}|${{token}}`; }}
            
            // ビルド: 各ノードへの「逆向き」入力エッジリストを構築
            steps.forEach((step, idx) => {{
                const sType = (step.type || '').toLowerCase();
                const layer = step.layer || 0;
                const token = step.token || 0;
                const head = step.head || step.mlp_head || 0;
                const influence = step.influence || 0;
                const targetToken = step.target_token !== undefined ? step.target_token : token;
                
                if (sType.indexOf('att') !== -1) {{
                    // ATT: z(l, token) -> u(l, targetToken, head)
                    const dest = uKey(layer, targetToken, head);
                    const src = zKey(layer, token);
                    if (!attIncomingByLayer[layer]) attIncomingByLayer[layer] = [];
                    attIncomingByLayer[layer].push({{ dest, src, w: influence, idx }});
                }} else if (sType.indexOf('mlp') !== -1) {{
                    // MLP: u(l, token, head) -> z(l+1, token)
                    const destLayer = layer + 1;
                    const dest = zKey(destLayer, token);
                    const src = uKey(layer, token, head);
                    if (!mlpIncomingByLayer[destLayer]) mlpIncomingByLayer[destLayer] = [];
                    mlpIncomingByLayer[destLayer].push({{ dest, src, w: influence, idx }});
                }}
            }});
            
            // 簡易逆伝播（クリックノード起点）
            function computeRelevanceFromNode(nodeType, layer, token, head) {{
                const relZ = {{}};  // key: z|l|t
                const relU = {{}};  // key: u|l|t|h
                const edgeRel = new Array(steps.length).fill(0);
                
                // 開始ノードの初期化
                if (nodeType === 'z') {{
                    relZ[zKey(layer, token)] = 1.0;
                }} else {{
                    relU[uKey(layer, token, head)] = 1.0;
                }}
                
                // uノードから開始された場合は、まず同じ層のATT逆伝播を行う
                if (nodeType === 'u') {{
                    const attIn = attIncomingByLayer[layer];
                    if (attIn && attIn.length > 0) {{
                        const destMap = {{}};
                        attIn.forEach(e => {{
                            if (!destMap[e.dest]) destMap[e.dest] = [];
                            destMap[e.dest].push(e);
                        }});
                        Object.keys(destMap).forEach(dest => {{
                            const destRel = relU[dest] || 0;
                            if (destRel <= 0) return;
                            const edges = destMap[dest];
                            const sumW = edges.reduce((s, e) => s + Math.max(e.w, 0), 0);
                            if (sumW <= 0) return;
                            edges.forEach(e => {{
                                const share = destRel * (Math.max(e.w, 0) / sumW);
                                relZ[e.src] = (relZ[e.src] || 0) + share;
                                edgeRel[e.idx] = share;
                            }});
                        }});
                    }}
                }}
                
                // layer (uノード開始ならそのlayer) から 1 まで逆順に伝播
                const startL = nodeType === 'z' ? layer : layer;
                for (let l = startL; l >= 1; l--) {{
                    // 1. z(l) -> u(l-1) MLP逆向き伝播
                    const mlpIn = mlpIncomingByLayer[l];
                    if (mlpIn && mlpIn.length > 0) {{
                        const destMap = {{}};
                        mlpIn.forEach(e => {{
                            if (!destMap[e.dest]) destMap[e.dest] = [];
                            destMap[e.dest].push(e);
                        }});
                        Object.keys(destMap).forEach(dest => {{
                            const destRel = relZ[dest] || 0;
                            if (destRel <= 0) return;
                            const edges = destMap[dest];
                            const sumW = edges.reduce((s, e) => s + Math.max(e.w, 0), 0);
                            if (sumW <= 0) return;
                            edges.forEach(e => {{
                                const share = destRel * (Math.max(e.w, 0) / sumW);
                                relU[e.src] = (relU[e.src] || 0) + share;
                                edgeRel[e.idx] = share;
                            }});
                        }});
                    }}
                    
                    // 2. u(l-1) -> z(l-1) ATT逆向き伝播
                    const attIn = attIncomingByLayer[l - 1];
                    if (attIn && attIn.length > 0) {{
                        const destMap = {{}};
                        attIn.forEach(e => {{
                            if (!destMap[e.dest]) destMap[e.dest] = [];
                            destMap[e.dest].push(e);
                        }});
                        Object.keys(destMap).forEach(dest => {{
                            const destRel = relU[dest] || 0;
                            if (destRel <= 0) return;
                            const edges = destMap[dest];
                            const sumW = edges.reduce((s, e) => s + Math.max(e.w, 0), 0);
                            if (sumW <= 0) return;
                            edges.forEach(e => {{
                                const share = destRel * (Math.max(e.w, 0) / sumW);
                                relZ[e.src] = (relZ[e.src] || 0) + share;
                                edgeRel[e.idx] = share;
                            }});
                        }});
                    }}
                }}
                
                return {{ relZ, relU, edgeRel }};
            }}
            

            // 初期状態に戻す関数
            function resetToInitialState() {{
                selectedNode = null;
                // ページをリロードして初期状態に戻す
                location.reload();
            }}
            
            // 既存描画をクリアし、背景を残したままノード/パスのみ再描画
            function redrawWithRelevance(relZ, relU, edgeRel) {{
                // 既存のノード・パスを削除（背景は残す）
                const oldNodes = svg.querySelectorAll('.node-circle');
                oldNodes.forEach(n => n.remove());
                const oldPaths = svg.querySelectorAll('.path-line');
                oldPaths.forEach(p => p.remove());
                
                // z/u のレイヤー別配列を生成（表示のため）
                const zAttrLayers = {{}};
                const uAttrLayers = {{}};
                for (let l = 0; l <= 12; l++) {{
                    const zArr = new Array(tokens.length).fill(0);
                    zAttrLayers[l] = zArr;
                }}
                for (let l = 0; l <= 11; l++) {{
                    const uArr = [];
                    for (let t = 0; t < tokens.length; t++) {{
                        uArr.push(new Array(numHeads).fill(0));
                    }}
                    uAttrLayers[l] = uArr;
                }}
                
                Object.entries(relZ).forEach(([k, v]) => {{
                    const [, lStr, tStr] = k.split('|');
                    const l = parseInt(lStr);
                    const t = parseInt(tStr);
                    if (zAttrLayers[l] && t < zAttrLayers[l].length) zAttrLayers[l][t] = v;
                }});
                Object.entries(relU).forEach(([k, v]) => {{
                    const [, lStr, tStr, hStr] = k.split('|');
                    const l = parseInt(lStr);
                    const t = parseInt(tStr);
                    const h = parseInt(hStr);
                    if (uAttrLayers[l] && t < uAttrLayers[l].length && h < numHeads) {{
                        uAttrLayers[l][t][h] = v;
                    }}
                }});
                
                // 最大値を算出
                let zMax = 0, uMax = 0;
                Object.values(zAttrLayers).forEach(arr => arr.forEach(v => {{ if (v > zMax) zMax = v; }}));
                Object.values(uAttrLayers).forEach(arr => arr.forEach(tok => tok.forEach(v => {{ if (v > uMax) uMax = v; }})));
                const zThreshold = 0.0;
                const uThreshold = 0.0;
                
                // ノード再描画（既存ロジックを再利用）
                const zNodeRadius = 4;  // zノードの半径
                const uNodeRadius = 2.5;  // uノードの半径（12倍あるので小さく）
                const eps = 1e-9;
                function getAlphaFromAttr(attr, threshold, maxAttr) {{
                    if (maxAttr <= threshold + eps) {{
                        return 0.4;
                    }}
                    const norm = (attr - threshold) / Math.max(eps, (maxAttr - threshold));
                    return Math.min(1.0, Math.max(0.2, norm));
                }}
                function getColorIntensityFromAttr(attr, threshold, maxAttr) {{
                    if (maxAttr <= threshold + eps) {{
                        return 0.3;
                    }}
                    const norm = (attr - threshold) / Math.max(eps, (maxAttr - threshold));
                    return Math.min(1.0, Math.max(0.1, norm));
                }}
                
                // z ノード（90度回転版）
                for (let l = 0; l <= 12; l++) {{
                    const arr = zAttrLayers[l];
                    const yPos = yPositions['z' + l];
                    if (yPos === undefined) continue;
                    for (let t = 0; t < Math.min(arr.length, tokens.length); t++) {{
                        const attr = arr[t] || 0;
                        const xPos = t * totalSpacing + tokenWidth / 2;
                        const alpha = getAlphaFromAttr(attr, zThreshold, zMax);
                        const colorIntensity = getColorIntensityFromAttr(attr, zThreshold, zMax);
                        const circle = document.createElementNS(ns, 'circle');
                        circle.setAttribute('cx', xPos);
                        circle.setAttribute('cy', yPos);
                        circle.setAttribute('r', zNodeRadius);
                        const r = Math.floor(100 + (255 - 100) * (1.0 - colorIntensity));
                        const g = Math.floor(150 + (255 - 150) * (1.0 - colorIntensity));
                        const b = Math.floor(200 + (255 - 200) * (1.0 - colorIntensity));
                        circle.setAttribute('fill', `rgb(${{r}}, ${{g}}, ${{b}})`);
                        circle.setAttribute('opacity', attr >= zThreshold ? alpha : 0.01);
                        circle.setAttribute('stroke', 'black');
                        circle.setAttribute('stroke-width', '1.5');
                        circle.setAttribute('class', 'node-circle clickable-node');
                        circle.setAttribute('data-node-type', 'z');
                        circle.setAttribute('data-layer', l);
                        circle.setAttribute('data-token', t);
                        circle.setAttribute('data-head', '-1');
                        circle.style.cursor = 'pointer';
                        circle.addEventListener('mouseenter', function(e) {{
                            tooltip.innerHTML =
                                '<strong>zノード</strong><br>' +
                                `Layer: ${{l}}<br>` +
                                `Token: ${{t}} (${{tokens[t]}})<br>` +
                                `Attribution: ${{attr.toFixed(6)}}`;
                            tooltip.style.display = 'block';
                            setTooltipPosition(tooltip, e);
                        }});
                        circle.addEventListener('mouseleave', function() {{
                            tooltip.style.display = 'none';
                        }});
                        circle.addEventListener('click', function(e) {{
                            e.stopPropagation();
                            const nodeKey = {{ nodeType: 'z', layer: l, token: t, head: -1 }};
                            // 同じノードが再度クリックされたら初期状態に戻す
                            if (selectedNode && 
                                selectedNode.nodeType === nodeKey.nodeType &&
                                selectedNode.layer === nodeKey.layer &&
                                selectedNode.token === nodeKey.token &&
                                selectedNode.head === nodeKey.head) {{
                                resetToInitialState();
                                return;
                            }}
                            selectedNode = nodeKey;
                            const res = computeRelevanceFromNode('z', l, t, -1);
                            redrawWithRelevance(res.relZ, res.relU, res.edgeRel);
                        }});
                        svg.appendChild(circle);
                    }}
                }}
                
                // u ノード（90度回転版）
                for (let l = 0; l <= 11; l++) {{
                    const arr = uAttrLayers[l];
                    const yPos = yPositions['u' + l];
                    if (yPos === undefined) continue;
                    for (let t = 0; t < arr.length; t++) {{
                        for (let h = 0; h < Math.min(numHeads, arr[t].length); h++) {{
                            const attr = arr[t][h] || 0;
                            const xPos = getHeadCenterXPosition(t, h);
                            const alpha = getAlphaFromAttr(attr, uThreshold, uMax);
                            const colorIntensity = getColorIntensityFromAttr(attr, uThreshold, uMax);
                            const circle = document.createElementNS(ns, 'circle');
                            circle.setAttribute('cx', xPos);
                            circle.setAttribute('cy', yPos);
                            circle.setAttribute('r', uNodeRadius);  // uノードは12倍あるので小さく
                            const r = Math.floor(100 + (255 - 100) * (1.0 - colorIntensity));
                            const g = Math.floor(150 + (255 - 150) * (1.0 - colorIntensity));
                            const b = Math.floor(200 + (255 - 200) * (1.0 - colorIntensity));
                            circle.setAttribute('fill', `rgb(${{r}}, ${{g}}, ${{b}})`);
                            circle.setAttribute('opacity', attr >= uThreshold ? alpha : 0.01);
                            circle.setAttribute('stroke', 'black');
                            circle.setAttribute('stroke-width', '1.5');
                            circle.setAttribute('class', 'node-circle clickable-node');
                            circle.setAttribute('data-node-type', 'u');
                            circle.setAttribute('data-layer', l);
                            circle.setAttribute('data-token', t);
                            circle.setAttribute('data-head', h);
                            circle.style.cursor = 'pointer';
                            circle.addEventListener('mouseenter', function(e) {{
                                tooltip.innerHTML =
                                    '<strong>uノード</strong><br>' +
                                `Layer: ${{l}}<br>` +
                                `Token: ${{t}} (${{tokens[t]}})<br>` +
                                `Head: ${{h}}<br>` +
                                `Attribution: ${{attr.toFixed(6)}}`;
                                tooltip.style.display = 'block';
                                setTooltipPosition(tooltip, e);
                            }});
                            circle.addEventListener('mouseleave', function() {{
                                tooltip.style.display = 'none';
                            }});
                            circle.addEventListener('click', function(e) {{
                                e.stopPropagation();
                                const nodeKey = {{ nodeType: 'u', layer: l, token: t, head: h }};
                                // 同じノードが再度クリックされたら初期状態に戻す
                                if (selectedNode && 
                                    selectedNode.nodeType === nodeKey.nodeType &&
                                    selectedNode.layer === nodeKey.layer &&
                                    selectedNode.token === nodeKey.token &&
                                    selectedNode.head === nodeKey.head) {{
                                    resetToInitialState();
                                    return;
                                }}
                                selectedNode = nodeKey;
                                const res = computeRelevanceFromNode('u', l, t, h);
                                redrawWithRelevance(res.relZ, res.relU, res.edgeRel);
                            }});
                            svg.appendChild(circle);
                        }}
                    }}
                }}
                
                // パス再描画（新方式：IG×LRPノード重みと同じロジック）
                // ノードクリック時に計算したrelZ, relUをLRPノード重みとして使用
                let maxEffMlp = 0.0;
                let maxEffAtt = 0.0;
                const maxEffMlpByLayer = Array(12).fill(0.0);
                const maxEffAttByLayer = Array(12).fill(0.0);
                
                // 事前に最大値を計算
                for (let i = 0; i < steps.length; i++) {{
                    const step = steps[i];
                    const stepType = (step.type || '').toLowerCase();
                    const layer = step.layer || 0;
                    const token = step.token || 0;
                    const head = step.head || step.mlp_head || 0;
                    const targetToken = step.target_token !== undefined ? step.target_token : token;
                    const influence = step.influence || 0;
                    
                    let eff = influence;
                    if (stepType.indexOf('mlp') !== -1) {{
                        // MLP: u(l,h) → z(l+1)
                        // z(l+1)のLRPノード重みを使用
                        const nextLayer = layer + 1;
                        let lrpWeight = 0.0;
                        if (nextLayer <= 12 && zAttrLayers[nextLayer] && token < zAttrLayers[nextLayer].length) {{
                            lrpWeight = zAttrLayers[nextLayer][token] || 0.0;
                        }}
                        lrpWeight = Math.max(0.0, lrpWeight);
                        eff = influence * lrpWeight;
                    }} else if (stepType.indexOf('att') !== -1 || stepType.indexOf('attn') !== -1) {{
                        // ATT: z(l) → u(l,h)
                        // u(l, targetToken, head)のLRPノード重みを使用
                        let lrpWeight = 0.0;
                        if (uAttrLayers[layer] && targetToken < uAttrLayers[layer].length && head < uAttrLayers[layer][targetToken].length) {{
                            lrpWeight = uAttrLayers[layer][targetToken][head] || 0.0;
                        }}
                        lrpWeight = Math.max(0.0, lrpWeight);
                        eff = influence * lrpWeight;
                    }}
                    
                    if (stepType.indexOf('mlp') !== -1) {{
                        if (eff > maxEffMlp) maxEffMlp = eff;
                        if (layer >= 0 && layer < 12 && eff > maxEffMlpByLayer[layer]) {{
                            maxEffMlpByLayer[layer] = eff;
                        }}
                    }} else if (stepType.indexOf('att') !== -1 || stepType.indexOf('attn') !== -1) {{
                        if (eff > maxEffAtt) maxEffAtt = eff;
                        if (layer >= 0 && layer < 12 && eff > maxEffAttByLayer[layer]) {{
                            maxEffAttByLayer[layer] = eff;
                        }}
                    }}
                }}
                const minEffDenom = 1e-6;
                
                for (let i = 0; i < steps.length; i++) {{
                    const step = steps[i];
                    const stepType = (step.type || '').toLowerCase();
                    const layer = step.layer || 0;
                    const token = step.token || 0;
                    const head = step.head || step.mlp_head || 0;
                    const targetToken = step.target_token !== undefined ? step.target_token : token;
                    const influence = step.influence || 0;
                    
                    // 新方式：IG×LRPノード重みと同じロジック
                    let effVal = influence;
                    if (stepType.indexOf('mlp') !== -1) {{
                        // MLP: u(l,h) → z(l+1)
                        // z(l+1)のLRPノード重みを使用
                        const nextLayer = layer + 1;
                        let lrpWeight = 0.0;
                        if (nextLayer <= 12 && zAttrLayers[nextLayer] && token < zAttrLayers[nextLayer].length) {{
                            lrpWeight = zAttrLayers[nextLayer][token] || 0.0;
                        }}
                        lrpWeight = Math.max(0.0, lrpWeight);
                        if (lrpWeight <= 1e-12) {{
                            continue;  // LRP重みが0の場合はスキップ（新方式と同じ）
                        }}
                        effVal = influence * lrpWeight;
                    }} else if (stepType.indexOf('att') !== -1 || stepType.indexOf('attn') !== -1) {{
                        // ATT: z(l) → u(l,h)
                        // u(l, targetToken, head)のLRPノード重みを使用
                        let lrpWeight = 0.0;
                        if (uAttrLayers[layer] && targetToken < uAttrLayers[layer].length && head < uAttrLayers[layer][targetToken].length) {{
                            lrpWeight = uAttrLayers[layer][targetToken][head] || 0.0;
                        }}
                        lrpWeight = Math.max(0.0, lrpWeight);
                        if (lrpWeight <= 1e-12) {{
                            continue;  // LRP重みが0の場合はスキップ（新方式と同じ）
                        }}
                        effVal = influence * lrpWeight;
                    }}
                    
                    // 閾値チェック（新方式と同じく元のinfluenceでチェック）
                    let shouldDraw = false;
                    if (stepType.indexOf('mlp') !== -1) {{
                        shouldDraw = influence >= mlpThreshold;
                    }} else if (stepType.indexOf('att') !== -1) {{
                        const isSameToken = token === targetToken;
                        const punctTokens = ['[CLS]', '[SEP]', ',', '.', '?', '!'];
                        const isPunctuation = punctTokens.indexOf(tokens[token]) !== -1;
                        if (isSameToken && sameTokenThresholdFactor !== 1.0) {{
                            shouldDraw = influence >= (attThreshold * sameTokenThresholdFactor);
                        }} else if (isPunctuation && punctuationAttThreshold !== null) {{
                            shouldDraw = influence >= punctuationAttThreshold;
                        }} else {{
                            shouldDraw = influence >= attThreshold;
                        }}
                    }}
                    if (!shouldDraw) continue;
                    
                    let x0, y0, x1, y1;
                    if (stepType.indexOf('mlp') !== -1) {{
                        // MLP: u(l,h) → z(l+1)（90度回転版）
                        x0 = getHeadCenterXPosition(token, head);
                        y0 = yPositions['u' + layer];
                        x1 = token * totalSpacing + tokenWidth / 2;
                        y1 = yPositions['z' + (layer + 1)];
                    }} else if (stepType.indexOf('att') !== -1) {{
                        // ATT: z(l) → u(l,h)（90度回転版）
                        x0 = token * totalSpacing + tokenWidth / 2;
                        y0 = yPositions['z' + layer];
                        x1 = getHeadCenterXPosition(targetToken, head);
                        y1 = yPositions['u' + layer];
                    }} else {{
                        continue;
                    }}
                    
                    const isMlp = stepType.indexOf('mlp') !== -1;
                    const layerMax = isMlp
                        ? (maxEffMlpByLayer[layer] > 0 ? maxEffMlpByLayer[layer] : maxEffMlp)
                        : (maxEffAttByLayer[layer] > 0 ? maxEffAttByLayer[layer] : maxEffAtt);
                    const maxEff = layerMax > 0 ? layerMax : (isMlp ? maxEffMlp : maxEffAtt);
                    const scaleDenom = Math.max(minEffDenom, maxEff > 0 ? maxEff : 1.0);
                    let norm = Math.max(0, Math.min(1.0, effVal / scaleDenom));
                    norm = Math.pow(norm, 0.35);
                    norm = norm * norm * norm;
                    
                    const minOpacity = 0.002; // 下限をさらに下げてコントラスト強化
                    const maxOpacity = 1.0;
                    const opacity = minOpacity + norm * (maxOpacity - minOpacity);
                    
                    const minStrokeWidth = 0.005; // 下限をさらに細くして差を強調
                    const maxStrokeWidth = 3.0;
                    const strokeWidth = minStrokeWidth + norm * (maxStrokeWidth - minStrokeWidth);
                    
                    const color = '#ff8c00';
                    const path = document.createElementNS(ns, 'path');
                    path.setAttribute('d', 'M ' + x0 + ' ' + y0 + ' L ' + x1 + ' ' + y1);
                    path.setAttribute('stroke', color);
                    path.setAttribute('stroke-width', strokeWidth);
                    path.setAttribute('opacity', opacity);
                    path.setAttribute('class', 'path-line');
                    path.setAttribute('pointer-events', 'stroke');
                    
                    path.addEventListener('mouseenter', function(e) {{
                        let pathDescription = '';
                        let influencePercent = (influence * 100).toFixed(2) + '%';
                        let effPercent = (effVal * 100).toFixed(2) + '%';
                        if (stepType.indexOf('mlp') !== -1) {{
                            pathDescription = `u(${{layer}}, ${{head}}) → z(${{layer + 1}})<br>` +
                                `Token: ${{token}} (${{tokens[token]}})<br>` +
                                `Head: ${{head}}<br>` +
                                `Layer: ${{layer}} → ${{layer + 1}}`;
                        }} else if (stepType.indexOf('att') !== -1) {{
                            pathDescription = `z(${{layer}}) → u(${{layer}}, ${{head}})<br>` +
                                `Source Token: ${{token}} (${{tokens[token]}})<br>` +
                                `Target Token: ${{targetToken}} (${{tokens[targetToken]}})<br>` +
                                `Head: ${{head}}<br>` +
                                `Layer: ${{layer}}`;
                        }} else {{
                            pathDescription = `Type: ${{stepType}}<br>` +
                                `Layer: ${{layer}}<br>` +
                                `Token: ${{token}} (${{tokens[token]}})<br>` +
                                `Head: ${{head}}`;
                        }}
                        tooltip.innerHTML = 
                            '<strong>パス情報</strong><br>' +
                            pathDescription + '<br>' +
                            '<strong>IG影響度: ' + influencePercent + '</strong><br>' +
                            '(Raw: ' + influence.toFixed(6) + ')<br>' +
                            '<strong>IG×LRP: ' + effPercent + '</strong><br>' +
                            '(Raw: ' + effVal.toFixed(6) + ')<br>' +
                            `太さ: ${{strokeWidth.toFixed(2)}} px<br>` +
                            `濃さ: ${{(opacity * 100).toFixed(1)}}%`;
                        tooltip.style.display = 'block';
                        setTooltipPosition(tooltip, e);
                    }});
                    path.addEventListener('mouseleave', function() {{
                        tooltip.style.display = 'none';
                    }});
                    
                    svg.appendChild(path);
                }}
            }}
            
            const container = document.getElementById('container');
            const plotArea = document.getElementById('plotArea');
            const svg = document.getElementById('svg');
            const tooltip = document.getElementById('tooltip');
            
            // タイムライン座標の設定
            // 並び: z0, u0, z1, u1, ..., z11, u11, z12
            const indexSeq = [["z", 0]];
            for (let l = 0; l < 12; l++) {{
                indexSeq.push(["u", l]);
                indexSeq.push(["z", l + 1]);
            }}
            const nPositions = indexSeq.length;
            // 90度左回転版：Tokenが横軸、Layerが縦軸
            const layerHeight = 80;  // Layer間の縦の間隔（元のlayerWidth）
            const totalHeight = (nPositions - 1) * layerHeight;  // 元のtotalWidth
            const tokenWidth = 100;  // Tokenの横のエリアの長さ（元のtokenHeight）
            const tokenHorizontalSpacing = 20;  // Token間の横軸の距離（元のtokenVerticalSpacing）
            // Head間の間隔: Tokenの横幅をHead数-1で均等分割
            const headSpacing = tokenWidth / (numHeads - 1);  // Head間の間隔（Tokenの横幅を均等分割）
            const totalSpacing = tokenWidth + tokenHorizontalSpacing;
            // 最後のTokenの右には余白を設けない
            const plotWidth = tokens.length * tokenWidth + (tokens.length - 1) * tokenHorizontalSpacing;  // 元のplotHeight
            
            // ヘッド/uノードの横座標計算関数（90度回転版）
            // Tokenの左端から headIdx * headSpacing の位置がuノードの座標
            function getHeadXPosition(tokenIdx, headIdx) {{
                return tokenIdx * totalSpacing + headIdx * headSpacing;
            }}
            function getHeadCenterXPosition(tokenIdx, headIdx) {{
                return getHeadXPosition(tokenIdx, headIdx);
            }}
            
            // プロットエリアのサイズを設定（90度回転版）
            plotArea.style.width = plotWidth + 'px';
            plotArea.style.height = totalHeight + 'px';
            plotArea.style.position = 'relative';
            svg.setAttribute('width', plotWidth);
            svg.setAttribute('height', totalHeight);
            
            // SVG名前空間を定義
            const ns = "http://www.w3.org/2000/svg";
            
            // 座標マッピング: z0, u0, z1, u1, ..., z12 のy座標（90度回転版：Layerが縦軸、下がLayer0、上がLayer11）
            // z0は下（totalHeight）、u0は上から2番目、z1は上から3番目、...（反転）
            const yPositions = {{}};
            for (let idx = 0; idx < nPositions; idx++) {{
                const [kind, layer] = indexSeq[idx];
                const key = kind + layer;
                // 下から上に配置（反転）：totalHeight - idx * layerHeight
                yPositions[key] = totalHeight - idx * layerHeight;
            }}
            
            // 背景をSVGのrect要素で描画
            // まず、レイヤーセクション：z->u（ATT区間）は青、u->z（MLP区間）は赤（最背面）
            const backgroundRects = [];
            for (let idx = 0; idx < nPositions - 1; idx++) {{
                const [kind, layer] = indexSeq[idx];
                const [nextKind, nextLayer] = indexSeq[idx + 1];
                const yPos = yPositions[kind + layer];
                const nextYPos = yPositions[nextKind + nextLayer];
                const sectionHeight = Math.abs(nextYPos - yPos);
                const sectionTop = Math.min(yPos, nextYPos);
                
                const rect = document.createElementNS(ns, 'rect');
                rect.setAttribute('x', '0');
                rect.setAttribute('y', sectionTop.toString());
                rect.setAttribute('width', plotWidth.toString());
                rect.setAttribute('height', sectionHeight.toString());
                
                if (kind === 'z' && nextKind === 'u') {{
                    // z(l) → u(l): ATT 区間 → 青
                    rect.setAttribute('fill', 'rgba(217, 230, 255, 0.4)');
                }} else if (kind === 'u' && nextKind === 'z') {{
                    // u(l) → z(l+1): MLP 区間 → 赤
                    rect.setAttribute('fill', 'rgba(255, 217, 217, 0.4)');
                }} else {{
                    rect.setAttribute('fill', 'transparent');
                }}
                
                rect.setAttribute('style', 'pointer-events: none;'); // マウスイベントを受け取らない
                backgroundRects.push(rect);
            }}
            
            // レイヤーセクションの背景を最背面に配置
            for (let i = 0; i < backgroundRects.length; i++) {{
                svg.insertBefore(backgroundRects[i], svg.firstChild);
            }}
            
            // トークン列とヘッド分割を描画（90度回転版：Tokenが横軸）
            for (let tokenIdx = 0; tokenIdx < tokens.length; tokenIdx++) {{
                const token = tokens[tokenIdx];
                const xPos = tokenIdx * totalSpacing;
                const xCenter = xPos + tokenWidth / 2;
                
                // トークン列（90度回転版：縦に伸びる）
                const column = document.createElement('div');
                column.className = 'token-row';
                column.style.left = xPos + 'px';
                column.style.width = tokenWidth + 'px';
                column.style.height = totalHeight + 'px';
                plotArea.appendChild(column);
                
                // Headの区間の灰色背景をSVGのrect要素で描画（レイヤーセクションの背景の上に配置）
                for (let h = 0; h < numHeads; h++) {{
                    const headX = getHeadXPosition(tokenIdx, h);
                    const nextHeadX = h < numHeads - 1 ? getHeadXPosition(tokenIdx, h + 1) : xPos + tokenWidth;
                    const headWidth = nextHeadX - headX;
                    
                    const headRect = document.createElementNS(ns, 'rect');
                    headRect.setAttribute('x', headX.toString());
                    headRect.setAttribute('y', '0');
                    headRect.setAttribute('width', headWidth.toString());
                    headRect.setAttribute('height', totalHeight.toString());
                    headRect.setAttribute('fill', 'rgba(240, 240, 240, 0.6)'); // 薄い灰色
                    headRect.setAttribute('style', 'pointer-events: none;'); // マウスイベントを受け取らない
                    // レイヤーセクションの背景の後に追加（上に表示される）
                    svg.appendChild(headRect);
                }}
                
                // トークンラベル（図の外、上側）
                const label = document.createElement('div');
                label.className = 'token-label';
                label.textContent = tokenIdx + ':' + token;
                label.style.left = xCenter + 'px';
                label.style.top = '-140px';
                label.style.transform = 'translateX(-50%)';
                plotArea.appendChild(label);
                
                // ヘッドラベルと点線（図の外、上側）
                for (let h = 0; h < numHeads; h++) {{
                    // ヘッドの位置を計算（90度回転版：横座標）
                    const headX = getHeadXPosition(tokenIdx, h);
                    const headCenterX = getHeadCenterXPosition(tokenIdx, h);
                    
                    // ヘッドラベル（絶対位置で配置）
                    const headLabel = document.createElement('div');
                    headLabel.className = 'head-label';
                    headLabel.style.position = 'absolute';
                    headLabel.style.left = (headCenterX - 20) + 'px';
                    headLabel.style.top = '-80px';
                    headLabel.textContent = 'H' + h;
                    plotArea.appendChild(headLabel);
                    
                    // ヘッド分割線（点線）をSVGで描画（縦線）
                    const divider = document.createElementNS(ns, 'line');
                    divider.setAttribute('x1', headX);
                    divider.setAttribute('y1', '0');
                    divider.setAttribute('x2', headX);
                    divider.setAttribute('y2', totalHeight);
                    divider.setAttribute('class', 'head-divider');
                    svg.appendChild(divider);
                }}
                
                // トークン列の左境界線（点線）をSVGで描画
                const leftDivider = document.createElementNS(ns, 'line');
                leftDivider.setAttribute('x1', xPos);
                leftDivider.setAttribute('y1', '0');
                leftDivider.setAttribute('x2', xPos);
                leftDivider.setAttribute('y2', totalHeight);
                leftDivider.setAttribute('class', 'head-divider');
                svg.appendChild(leftDivider);
                
                // トークン列の右境界線（点線）をSVGで描画
                const rightDivider = document.createElementNS(ns, 'line');
                rightDivider.setAttribute('x1', xPos + tokenWidth);
                rightDivider.setAttribute('y1', '0');
                rightDivider.setAttribute('x2', xPos + tokenWidth);
                rightDivider.setAttribute('y2', totalHeight);
                rightDivider.setAttribute('class', 'head-divider');
                svg.appendChild(rightDivider);
            }}
            
            // 縦軸ラベル（z0, u0, z1, u1, ..., z12）を描画（90度回転版：Layerが縦軸）
            for (let idx = 0; idx < nPositions; idx++) {{
                const [kind, layer] = indexSeq[idx];
                const yPos = yPositions[kind + layer];
                const label = kind + layer;
                
                const axisLabel = document.createElement('div');
                axisLabel.className = 'x-axis-label';
                axisLabel.style.top = yPos + 'px';
                axisLabel.style.left = '-30px';
                axisLabel.style.transform = 'translateY(-50%) rotate(-90deg)';
                axisLabel.textContent = label;
                plotArea.appendChild(axisLabel);
            }}
            
            // ATT/MLPラベルとLayerラベルを描画（90度回転版：Layerが縦軸）
            for (let idx = 0; idx < nPositions - 1; idx++) {{
                const [kind, layer] = indexSeq[idx];
                const [nextKind, nextLayer] = indexSeq[idx + 1];
                const yPos = yPositions[kind + layer];
                const nextYPos = yPositions[nextKind + nextLayer];
                const midY = (yPos + nextYPos) / 2;  // 区間の中央
                
                if (kind === 'z' && nextKind === 'u') {{
                    // z(l) → u(l): ATT 区間の中央にATTラベル
                    const typeLabel = document.createElement('div');
                    typeLabel.className = 'section-label';
                    typeLabel.style.top = midY + 'px';
                    typeLabel.style.left = '-20px';
                    typeLabel.style.transform = 'translateY(-50%) rotate(-90deg)';
                    typeLabel.textContent = 'ATT';
                    plotArea.appendChild(typeLabel);
                }} else if (kind === 'u' && nextKind === 'z') {{
                    // u(l) → z(l+1): MLP 区間の中央にMLPラベル
                    const typeLabel = document.createElement('div');
                    typeLabel.className = 'section-label';
                    typeLabel.style.top = midY + 'px';
                    typeLabel.style.left = '-20px';
                    typeLabel.style.transform = 'translateY(-50%) rotate(-90deg)';
                    typeLabel.textContent = 'MLP';
                    plotArea.appendChild(typeLabel);
                }}
                
                // Layerラベルはuの位置に配置（90度回転版）
                if (kind === 'u') {{
                    const layerLabel = document.createElement('div');
                    layerLabel.className = 'layer-label';
                    layerLabel.style.top = yPos + 'px';
                    layerLabel.style.left = '-60px'; /* さらに左に移動 */
                    layerLabel.style.transform = 'translateY(-50%) rotate(-90deg)';
                    layerLabel.textContent = 'Layer' + layer; /* Layer0形式に変更 */
                    layerLabel.style.cursor = 'pointer';
                    
                    // Layerラベルにホバーイベントを追加
                    layerLabel.addEventListener('mouseenter', function(e) {{
                        // このLayerに関連するパス数をカウント
                        let mlpCount = 0;
                        let attCount = 0;
                        for (let i = 0; i < steps.length; i++) {{
                            const s = steps[i];
                            const sType = (s.type || '').toLowerCase();
                            const sLayer = s.layer || 0;
                            
                            if (sLayer === layer) {{
                                if (sType.indexOf('mlp') !== -1) {{
                                    const sInfluence = s.influence || 0;
                                    if (sInfluence >= mlpThreshold) mlpCount++;
                                }} else if (sType.indexOf('att') !== -1 || sType.indexOf('attn') !== -1) {{
                                    const sInfluence = s.influence || 0;
                                    if (sInfluence >= attThreshold) attCount++;
                                }}
                            }}
                        }}
                        
                        tooltip.innerHTML = 
                            '<strong>Layer ' + layer + ' 情報</strong><br>' +
                            'MLPパス数: ' + mlpCount + '<br>' +
                            'ATTパス数: ' + attCount + '<br>' +
                            '合計パス数: ' + (mlpCount + attCount);
                        tooltip.style.display = 'block';
                        setTooltipPosition(tooltip, e);
                    }});
                    layerLabel.addEventListener('mouseleave', function() {{
                        tooltip.style.display = 'none';
                    }});
                    
                    plotArea.appendChild(layerLabel);
                }}
            }}
            
            // 事前に「実効影響度」の最大値を求める（太さ・濃さのコントラスト用）
            let maxEffMlp = 0.0;
            let maxEffAtt = 0.0;
            const maxEffMlpByLayer = Array(12).fill(0.0);
            const maxEffAttByLayer = Array(12).fill(0.0);
            for (let i = 0; i < steps.length; i++) {{
                const step = steps[i];
                const stepType = (step.type || '').toLowerCase();
                const layer = step.layer || 0;
                const token = step.token || 0;
                const head = step.head || step.mlp_head || 0;
                const influence = step.influence || 0;
                const targetToken = step.target_token !== undefined ? step.target_token : token;

                let eff = influence;
                if (pathWeightMode === "ig_times_lrp" && lrpData) {{
                    const zAttrLayers = lrpData.z_attr_layers || {{}};
                    const uAttrLayers = lrpData.u_attr_layers || {{}};
                    const finalOutputZ = lrpData.final_output_z;
                    if (stepType.indexOf('mlp') !== -1) {{
                        let lrpWeight = 0.0;
                        const nextLayer = layer + 1;
                        if (nextLayer <= 11) {{
                            const zAttrs = zAttrLayers[nextLayer];
                            if (Array.isArray(zAttrs) && token < zAttrs.length) {{
                                lrpWeight = parseFloat(zAttrs[token]) || 0.0;
                            }}
                        }} else if (finalOutputZ && Array.isArray(finalOutputZ) && token < finalOutputZ.length) {{
                            lrpWeight = parseFloat(finalOutputZ[token]) || 0.0;
                        }}
                        lrpWeight = Math.max(0.0, lrpWeight);
                        // LRP重みが小さい場合でも、元のinfluenceで閾値チェックするためcontinueしない
                        eff = influence * Math.max(lrpWeight, 0.0);
                    }} else if (stepType.indexOf('att') !== -1 || stepType.indexOf('attn') !== -1) {{
                        let lrpWeight = 0.0;
                        const uAttrs = uAttrLayers[layer];
                        if (uAttrs && Array.isArray(uAttrs) && targetToken < uAttrs.length) {{
                            const tokenAttrs = uAttrs[targetToken];
                            if (Array.isArray(tokenAttrs) && head < tokenAttrs.length) {{
                                lrpWeight = parseFloat(tokenAttrs[head]) || 0.0;
                            }}
                        }}
                        lrpWeight = Math.max(0.0, lrpWeight);
                        // LRP重みが小さい場合でも、元のinfluenceで閾値チェックするためcontinueしない
                        eff = influence * Math.max(lrpWeight, 0.0);
                    }}
                }}
                if (stepType.indexOf('mlp') !== -1) {{
                    if (eff > maxEffMlp) maxEffMlp = eff;
                    if (layer >= 0 && layer < 12 && eff > maxEffMlpByLayer[layer]) {{
                        maxEffMlpByLayer[layer] = eff;
                    }}
                }} else if (stepType.indexOf('att') !== -1 || stepType.indexOf('attn') !== -1) {{
                    if (eff > maxEffAtt) maxEffAtt = eff;
                    if (layer >= 0 && layer < 12 && eff > maxEffAttByLayer[layer]) {{
                        maxEffAttByLayer[layer] = eff;
                    }}
                }}
            }}
            // 安全な下限
            const minEffDenom = 1e-6;

            // パスを再描画する関数
            function redrawPaths() {{
                // 既存のパスを削除
                const oldPaths = svg.querySelectorAll('.path-line');
                oldPaths.forEach(p => p.remove());
                
                // 現在の閾値を取得
                const currentAttThreshold = parseFloat(document.getElementById('attThresholdSlider').value);
                const currentMlpThreshold = parseFloat(document.getElementById('mlpThresholdSlider').value);
                const currentSameTokenFactor = parseFloat(document.getElementById('sameTokenThresholdSlider').value);
                const punctSlider = document.getElementById('punctuationThresholdSlider');
                const currentPunctuationThreshold = punctSlider.disabled ? null : parseFloat(punctSlider.value);
                
                // 強いパスを描画
                for (let i = 0; i < steps.length; i++) {{
                const step = steps[i];
                const stepType = (step.type || '').toLowerCase();
                const layer = step.layer || 0;
                const token = step.token || 0;
                const head = step.head || step.mlp_head || 0;
                const influence = step.influence || 0;
                const targetToken = step.target_token !== undefined ? step.target_token : token;
                
                // 閾値チェック
                // サンプル選択モードでは、MLP/ATTパスは既に正規化されている（合計1）
                // そのため、閾値は0.0〜1.0の範囲で設定されている
                let shouldDraw = false;
                if (stepType.indexOf('mlp') !== -1) {{
                    // MLP: 正規化後の値（0〜1）で閾値チェック
                    shouldDraw = influence >= currentMlpThreshold;
                }} else if (stepType.indexOf('att') !== -1 || stepType.indexOf('attn') !== -1) {{
                    // ATTの場合は追加の閾値チェック
                    const isSameToken = token === targetToken;
                    const punctTokens = ['[CLS]', '[SEP]', ',', '.', '?', '!'];
                    const isPunctuation = punctTokens.indexOf(tokens[token]) !== -1 || punctTokens.indexOf(tokens[targetToken]) !== -1;
                    
                    if (isSameToken && currentSameTokenFactor !== 1.0) {{
                        shouldDraw = influence >= (currentAttThreshold * currentSameTokenFactor);
                    }} else if (isPunctuation && currentPunctuationThreshold !== null) {{
                        shouldDraw = influence >= currentPunctuationThreshold;
                    }} else {{
                        shouldDraw = influence >= currentAttThreshold;
                    }}
                }}
                
                if (!shouldDraw) continue;
                
                // 座標を計算（90度回転版：Tokenが横軸、Layerが縦軸）
                let x0, y0, x1, y1;
                
                if (stepType.indexOf('mlp') !== -1) {{
                    // MLP: u(l,h) → z(l+1)（90度回転版）
                    x0 = getHeadCenterXPosition(token, head);
                    y0 = yPositions['u' + layer];
                    x1 = token * totalSpacing + tokenWidth / 2;
                    y1 = yPositions['z' + (layer + 1)];
                }} else if (stepType.indexOf('att') !== -1 || stepType.indexOf('attn') !== -1) {{
                    // ATT: z(l) → u(l,h)（90度回転版）
                    x0 = token * totalSpacing + tokenWidth / 2;
                    y0 = yPositions['z' + layer];
                    x1 = getHeadCenterXPosition(targetToken, head);
                    y1 = yPositions['u' + layer];
                }} else {{
                    continue;
                }}
                
                // 線の色と太さを決定（影響度に基づく）
                // 新方式（IG×LRP）の場合、LRPノード重みを掛ける
                let effectiveInfluence = influence;

                if (pathWeightMode === "ig_times_lrp" && lrpData) {{
                    const zAttrLayers = lrpData.z_attr_layers || {{}};
                    const uAttrLayers = lrpData.u_attr_layers || {{}};
                    const finalOutputZ = lrpData.final_output_z;
                    
                    if (stepType.indexOf('mlp') !== -1) {{
                        // MLP: u(l,h) → z(l+1)
                        let lrpWeight = 0.0;
                        const nextLayer = layer + 1;
                        if (nextLayer <= 11) {{
                            const zAttrs = zAttrLayers[nextLayer];
                            if (Array.isArray(zAttrs) && token < zAttrs.length) {{
                                lrpWeight = parseFloat(zAttrs[token]) || 0.0;
                            }}
                        }} else if (finalOutputZ && Array.isArray(finalOutputZ) && token < finalOutputZ.length) {{
                            lrpWeight = parseFloat(finalOutputZ[token]) || 0.0;
                        }}
                        lrpWeight = Math.max(0.0, lrpWeight);
                        if (lrpWeight <= 1e-12) {{
                            continue;
                        }}
                        effectiveInfluence = influence * lrpWeight;
                    }} else if (stepType.indexOf('att') !== -1 || stepType.indexOf('attn') !== -1) {{
                        // ATT: z(l) → u(l,h)
                        let lrpWeight = 0.0;
                        const uAttrs = uAttrLayers[layer];
                        if (uAttrs && Array.isArray(uAttrs) && targetToken < uAttrs.length) {{
                            const tokenAttrs = uAttrs[targetToken];
                            if (Array.isArray(tokenAttrs) && head < tokenAttrs.length) {{
                                lrpWeight = parseFloat(tokenAttrs[head]) || 0.0;
                            }}
                        }}
                        lrpWeight = Math.max(0.0, lrpWeight);
                        if (lrpWeight <= 1e-12) {{
                            continue;
                        }}
                        effectiveInfluence = influence * lrpWeight;
                    }}
                }}
                
                // 閾値に対して線の濃さを正規化。最大実効値があればそれを使ってスケーリング
                const threshold = stepType.indexOf('mlp') !== -1 ? mlpThreshold : attThreshold;
                const isMlp = stepType.indexOf('mlp') !== -1;
                const layerMax = isMlp
                    ? (maxEffMlpByLayer[layer] > 0 ? maxEffMlpByLayer[layer] : maxEffMlp)
                    : (maxEffAttByLayer[layer] > 0 ? maxEffAttByLayer[layer] : maxEffAtt);
                const globalMax = isMlp ? maxEffMlp : maxEffAtt;
                const maxEff = layerMax > 0 ? layerMax : globalMax;
                const scaleDenom = Math.max(minEffDenom, maxEff > 0 ? maxEff : 1.0);
                let norm = Math.max(0, Math.min(1.0, effectiveInfluence / scaleDenom));  // 0〜1に正規化
                
                // 非線形マッピングで差を強調（0.35乗で中間を持ち上げ、さらに3乗で広げる）
                norm = Math.pow(norm, 0.35);
                norm = norm * norm * norm;
                
                // 不透明度: 影響度に基づいて0.01〜1.0の範囲で設定（より広いレンジ）
                const minOpacity = 0.002; // 下限をさらに下げてコントラスト強化
                const maxOpacity = 1.0;
                const opacity = minOpacity + norm * (maxOpacity - minOpacity);
                
                // 線の太さ: 影響度に基づいて0.005〜3.0の範囲で設定（さらに細く）
                const minStrokeWidth = 0.005; // 下限をより細く
                const maxStrokeWidth = 3.0;
                const strokeWidth = minStrokeWidth + norm * (maxStrokeWidth - minStrokeWidth);
                
                const color = '#ff8c00';  // 全パス共通色（オレンジ）
                
                // SVGパスを描画（クリック動作は無効化）
                const path = document.createElementNS(ns, 'path');
                path.setAttribute('d', 'M ' + x0 + ' ' + y0 + ' L ' + x1 + ' ' + y1);
                path.setAttribute('stroke', color);
                path.setAttribute('stroke-width', strokeWidth);
                path.setAttribute('opacity', opacity);
                path.setAttribute('class', 'path-line');
                path.setAttribute('pointer-events', 'stroke');  // ストローク部分のみマウスイベントを受け取る
                
                // ツールチップ（パスの詳細情報）
                path.addEventListener('mouseenter', function(e) {{
                    // パスの種類に応じて表示内容を変更
                    let pathDescription = '';
                    let influencePercent = (influence * 100).toFixed(2) + '%';
                    
                    if (stepType.indexOf('mlp') !== -1) {{
                        // MLP: u(l,h) → z(l+1)
                        pathDescription = `u(${{layer}}, ${{head}}) → z(${{layer + 1}})<br>` +
                            `Token: ${{token}} (${{tokens[token]}})<br>` +
                            `Head: ${{head}}<br>` +
                            `Layer: ${{layer}} → ${{layer + 1}}`;
                    }} else if (stepType.indexOf('att') !== -1 || stepType.indexOf('attn') !== -1) {{
                        // ATT: z(l) → u(l,h)
                        pathDescription = `z(${{layer}}) → u(${{layer}}, ${{head}})<br>` +
                            `Source Token: ${{token}} (${{tokens[token]}})<br>` +
                            `Target Token: ${{targetToken}} (${{tokens[targetToken]}})<br>` +
                            `Head: ${{head}}<br>` +
                            `Layer: ${{layer}}`;
                    }} else {{
                        pathDescription = `Type: ${{stepType}}<br>` +
                            `Layer: ${{layer}}<br>` +
                            `Token: ${{token}} (${{tokens[token]}})<br>` +
                            `Head: ${{head}}`;
                    }}
                    
                    tooltip.innerHTML = 
                        '<strong>パス情報</strong><br>' +
                        pathDescription + '<br>' +
                        '<strong>影響度: ' + influencePercent + '</strong><br>' +
                        '(Raw: ' + influence.toFixed(6) + ')<br>' +
                        `太さ: ${{strokeWidth.toFixed(2)}} px<br>` +
                        `濃さ: ${{(opacity * 100).toFixed(1)}}%`;
                    tooltip.style.display = 'block';
                    setTooltipPosition(tooltip, e);
                }});
                path.addEventListener('mouseleave', function() {{
                    tooltip.style.display = 'none';
                }});
                
                svg.appendChild(path);
                }}
            }}
            
            // 初期描画
            redrawPaths();
            
            // スライダーのイベントハンドラ
            document.getElementById('attThresholdSlider').addEventListener('input', function(e) {{
                const value = parseFloat(e.target.value);
                document.getElementById('attThresholdDisplay').textContent = value.toFixed(3);
                redrawPaths();
            }});
            
            document.getElementById('mlpThresholdSlider').addEventListener('input', function(e) {{
                const value = parseFloat(e.target.value);
                document.getElementById('mlpThresholdDisplay').textContent = value.toFixed(3);
                redrawPaths();
            }});
            
            document.getElementById('zThresholdSlider').addEventListener('input', function(e) {{
                const value = parseFloat(e.target.value);
                document.getElementById('zThresholdDisplay').textContent = value.toFixed(3);
                // zノードの再描画はLRPオーバーレイ部分で行う必要があるが、簡易的にスキップ
            }});
            
            document.getElementById('uThresholdSlider').addEventListener('input', function(e) {{
                const value = parseFloat(e.target.value);
                document.getElementById('uThresholdDisplay').textContent = value.toFixed(3);
                // uノードの再描画はLRPオーバーレイ部分で行う必要があるが、簡易的にスキップ
            }});
            
            document.getElementById('sameTokenThresholdSlider').addEventListener('input', function(e) {{
                const value = parseFloat(e.target.value);
                document.getElementById('sameTokenThresholdDisplay').textContent = value.toFixed(3);
                redrawPaths();
            }});
            
            const punctSlider = document.getElementById('punctuationThresholdSlider');
            if (punctSlider && !punctSlider.disabled) {{
                punctSlider.addEventListener('input', function(e) {{
                    const value = parseFloat(e.target.value);
                    document.getElementById('punctuationThresholdDisplay').textContent = value.toFixed(3);
                    redrawPaths();
                }});
            }}
            
            // LRPオーバーレイ: ノード強調（丸）を描画
            if (lrpData && (lrpData.z_attr_layers || lrpData.final_output_z) && lrpData.u_attr_layers) {{
                const zAttrLayers = lrpData.z_attr_layers || {{}};
                const uAttrLayers = lrpData.u_attr_layers;
                const finalOutputZ = lrpData.final_output_z;  // z12のAttribution（最終出力）
                const zThreshold = lrpData.z_threshold !== undefined ? lrpData.z_threshold : 0.1;
                const uThreshold = lrpData.u_threshold !== undefined ? lrpData.u_threshold : 0.1;
                const zNodeRadius = 4;  // zノードの半径
                const uNodeRadius = 2.5;  // uノードの半径（12倍あるので小さく）
                const eps = 1e-9;
                
                // Attribution値の最大値を計算（正規化用）
                let zMax = 0;
                let uMax = 0;
                
                // z12Attrsを先に定義（zMax計算とz12ノード描画の両方で使用）
                let z12Attrs = null;
                if (zAttrLayers['12'] && Array.isArray(zAttrLayers['12'])) {{
                    z12Attrs = zAttrLayers['12'];
                }} else if (finalOutputZ && Array.isArray(finalOutputZ)) {{
                    z12Attrs = finalOutputZ;
                }}
                
                // zノードの最大値を取得（layer_0からlayer_12まで、z12も含む）
                for (let layerKey in zAttrLayers) {{
                    const zAttrs = zAttrLayers[layerKey];
                    if (Array.isArray(zAttrs)) {{
                        for (let i = 0; i < zAttrs.length; i++) {{
                            const val = parseFloat(zAttrs[i]) || 0;
                            if (val > zMax) zMax = val;
                        }}
                    }}
                }}
                
                // final_output_z（z12）の最大値も考慮
                if (finalOutputZ && Array.isArray(finalOutputZ)) {{
                    for (let i = 0; i < finalOutputZ.length; i++) {{
                        const val = parseFloat(finalOutputZ[i]) || 0;
                        if (val > zMax) zMax = val;
                    }}
                }}
                
                // z12Attrsがnullの場合（デフォルトで1.0が設定される場合）も考慮
                // z12ノードがデフォルトで1.0の場合、zMaxの計算に1.0を含める
                if (z12Attrs === null) {{
                    if (1.0 > zMax) zMax = 1.0;
                }}
                
                // uノードの最大値を取得
                for (let layerKey in uAttrLayers) {{
                    const uAttrs = uAttrLayers[layerKey];
                    if (Array.isArray(uAttrs)) {{
                        for (let i = 0; i < uAttrs.length; i++) {{
                            const tokenAttrs = uAttrs[i];
                            if (Array.isArray(tokenAttrs)) {{
                                for (let j = 0; j < tokenAttrs.length; j++) {{
                                    const val = parseFloat(tokenAttrs[j]) || 0;
                                    if (val > uMax) uMax = val;
                                }}
                            }}
                        }}
                    }}
                }}
                
                // Attribution値から不透明度を計算する関数
                function getAlphaFromAttr(attr, threshold, maxAttr) {{
                    if (maxAttr <= threshold + eps) {{
                        return 0.4;  // 全て同値の場合の退避値
                    }}
                    const norm = (attr - threshold) / Math.max(eps, (maxAttr - threshold));
                    return Math.min(1.0, Math.max(0.2, norm));  // 0.2〜1.0の範囲で不透明度を設定
                }}
                
                // Attribution値から色の濃さ（RGB値）を計算する関数
                function getColorIntensityFromAttr(attr, threshold, maxAttr) {{
                    if (maxAttr <= threshold + eps) {{
                        return 0.3;  // 全て同値の場合の退避値
                    }}
                    const norm = (attr - threshold) / Math.max(eps, (maxAttr - threshold));
                    return Math.min(1.0, Math.max(0.1, norm));  // 0.1〜1.0の範囲で色の濃さを設定
                }}
                
                // zノードの強調（各レイヤー、各トークン）
                const targetTokenIdx = lrpData.target_token_idx !== undefined ? lrpData.target_token_idx : 0;
                
                // まず、z12（最終出力層）を描画
                // final_output_zまたはz_attr_layers[12]から取得、なければ貢献度1で表示
                // z12のy座標（90度回転版：Layerが縦軸）
                const yPosZ12 = yPositions['z12'];
                if (yPosZ12 !== undefined) {{
                    // z12Attrsは既にzMax計算時に定義されている
                    
                    // すべてのトークンに対してz12ノードを描画
                    for (let tokenIdx = 0; tokenIdx < tokens.length; tokenIdx++) {{
                        // z12Attrsが存在する場合はその値を使用、なければ貢献度1
                        const attr = z12Attrs ? (parseFloat(z12Attrs[tokenIdx]) || 0) : 1.0;
                        
                        // z12のすべてのノードを表示（しきい値以上のものは強調表示）（90度回転版）
                        
                        const xPos = tokenIdx * totalSpacing + tokenWidth / 2;
                        const yPos = yPosZ12;
                        // zMaxが0の場合やz12Attrsが存在しない場合でも表示できるようにする
                        // 貢献度1の場合は最大値として扱う
                        const effectiveZMax = (zMax > 0) ? zMax : 1.0;
                        const alpha = getAlphaFromAttr(attr, zThreshold, effectiveZMax);
                        const colorIntensity = getColorIntensityFromAttr(attr, zThreshold, effectiveZMax);
                        const circle = document.createElementNS(ns, 'circle');
                        circle.setAttribute('cx', xPos);
                        circle.setAttribute('cy', yPos);
                        circle.setAttribute('r', zNodeRadius);
                        circle.setAttribute('class', 'node-circle');
                        // 色の濃さを反映（薄い青から濃い青へ）
                        const r = Math.floor(100 + (255 - 100) * (1.0 - colorIntensity));  // 100〜255の範囲
                        const g = Math.floor(150 + (255 - 150) * (1.0 - colorIntensity));  // 150〜255の範囲
                        const b = Math.floor(200 + (255 - 200) * (1.0 - colorIntensity));  // 200〜255の範囲（薄い青）
                        circle.setAttribute('fill', `rgb(${{r}}, ${{g}}, ${{b}})`);
                        // しきい値未満でも極薄で描画しホバー・クリック可能にする
                        const displayOpacity = attr >= zThreshold ? alpha : 0.01;
                        circle.setAttribute('opacity', displayOpacity);
                        circle.setAttribute('stroke', 'black');
                        circle.setAttribute('stroke-width', '1.5');
                        circle.setAttribute('class', 'node-circle clickable-node');
                        circle.setAttribute('data-node-type', 'z');
                        circle.setAttribute('data-layer', '12');  // z12として保存
                        circle.setAttribute('data-token', tokenIdx);
                        circle.setAttribute('data-head', '-1'); // zノードはheadなし
                        circle.style.cursor = 'pointer';
                        
                        // ホバー: ノード情報を表示
                        circle.addEventListener('mouseenter', function(e) {{
                            tooltip.innerHTML =
                                '<strong>zノード</strong><br>' +
                                `Layer: 12<br>` +
                                `Token: ${{tokenIdx}} (${{tokens[tokenIdx]}})<br>` +
                                `Attribution: ${{attr.toFixed(6)}}`;
                            tooltip.style.display = 'block';
                            setTooltipPosition(tooltip, e);
                        }});
                        circle.addEventListener('mouseleave', function() {{
                            tooltip.style.display = 'none';
                        }});
                        
                        // クリック: 簡易逆伝播で再描画
                        circle.addEventListener('click', function(e) {{
                            e.stopPropagation();
                            const res = computeRelevanceFromNode('z', 12, tokenIdx, -1);
                            redrawWithRelevance(res.relZ, res.relU, res.edgeRel);
                        }});
                        
                        svg.appendChild(circle);
                    }}
                }}
                
                // 次に、z0からz11までを描画（layer_0からlayer_11まで）
                for (let layerKey in zAttrLayers) {{
                    const layer = parseInt(layerKey);
                    // z12は既に描画済みなのでスキップ
                    if (layer === 12) continue;
                    const zAttrs = zAttrLayers[layerKey];
                    if (!Array.isArray(zAttrs)) continue;
                    
                    // LRP計算では、layer 0のzはz0、layer 11のzはz11に対応（90度回転版）
                    const yPos = yPositions['z' + layer];
                    if (yPos === undefined) {{
                        console.warn('zノードの座標が見つかりません: layer=' + layer + ', key=z' + layer);
                        continue;
                    }}
                    
                    for (let tokenIdx = 0; tokenIdx < Math.min(zAttrs.length, tokens.length); tokenIdx++) {{
                        const attr = parseFloat(zAttrs[tokenIdx]) || 0;
                        
                        const xPos = tokenIdx * totalSpacing + tokenWidth / 2;
                        const alpha = getAlphaFromAttr(attr, zThreshold, zMax);
                        const colorIntensity = getColorIntensityFromAttr(attr, zThreshold, zMax);
                        const circle = document.createElementNS(ns, 'circle');
                        circle.setAttribute('cx', xPos);
                        circle.setAttribute('cy', yPos);
                        circle.setAttribute('r', zNodeRadius);
                        circle.setAttribute('class', 'node-circle');
                        // 色の濃さを反映（薄い青から濃い青へ）
                        const r = Math.floor(100 + (255 - 100) * (1.0 - colorIntensity));  // 100〜255の範囲
                        const g = Math.floor(150 + (255 - 150) * (1.0 - colorIntensity));  // 150〜255の範囲
                        const b = Math.floor(200 + (255 - 200) * (1.0 - colorIntensity));  // 200〜255の範囲（薄い青）
                        circle.setAttribute('fill', `rgb(${{r}}, ${{g}}, ${{b}})`);
                        // しきい値未満でも極薄で描画しホバー可能にする
                        const displayOpacity = attr >= zThreshold ? alpha : 0.01;
                        circle.setAttribute('opacity', displayOpacity);
                        circle.setAttribute('stroke', 'black');
                        circle.setAttribute('stroke-width', '1.5');
                        circle.setAttribute('class', 'node-circle clickable-node');
                        circle.setAttribute('data-node-type', 'z');
                        circle.setAttribute('data-layer', layer);
                        circle.setAttribute('data-token', tokenIdx);
                        circle.setAttribute('data-head', '-1'); // zノードはheadなし
                        circle.style.cursor = 'pointer';
                        
                        // ホバー: ノード情報を表示
                        circle.addEventListener('mouseenter', function(e) {{
                            tooltip.innerHTML =
                                '<strong>zノード</strong><br>' +
                                `Layer: ${{layer}}<br>` +
                                `Token: ${{tokenIdx}} (${{tokens[tokenIdx]}})<br>` +
                                `Attribution: ${{attr.toFixed(6)}}`;
                            tooltip.style.display = 'block';
                            setTooltipPosition(tooltip, e);
                        }});
                        circle.addEventListener('mouseleave', function() {{
                            tooltip.style.display = 'none';
                        }});
                        
                        // クリック: 簡易逆伝播で再描画
                        circle.addEventListener('click', function(e) {{
                            e.stopPropagation();
                            const nodeKey = {{ nodeType: 'z', layer: layer, token: tokenIdx, head: -1 }};
                            // 同じノードが再度クリックされたら初期状態に戻す
                            if (selectedNode && 
                                selectedNode.nodeType === nodeKey.nodeType &&
                                selectedNode.layer === nodeKey.layer &&
                                selectedNode.token === nodeKey.token &&
                                selectedNode.head === nodeKey.head) {{
                                resetToInitialState();
                                return;
                            }}
                            selectedNode = nodeKey;
                            const res = computeRelevanceFromNode('z', layer, tokenIdx, -1);
                            redrawWithRelevance(res.relZ, res.relU, res.edgeRel);
                        }});
                        
                        svg.appendChild(circle);
                    }}
                }}
                
                // uノードの強調（各レイヤー、各トークン、各ヘッド）
                for (let layerKey in uAttrLayers) {{
                    const layer = parseInt(layerKey);
                    const uAttrs = uAttrLayers[layerKey];
                    if (!Array.isArray(uAttrs)) continue;
                    
                    const yPos = yPositions['u' + layer];
                    if (yPos === undefined) continue;
                    
                    for (let tokenIdx = 0; tokenIdx < Math.min(uAttrs.length, tokens.length); tokenIdx++) {{
                        const tokenAttrs = uAttrs[tokenIdx];
                        if (!Array.isArray(tokenAttrs)) continue;
                        
                        for (let headIdx = 0; headIdx < Math.min(tokenAttrs.length, numHeads); headIdx++) {{
                            const attr = parseFloat(tokenAttrs[headIdx]) || 0;
                            const xPos = getHeadCenterXPosition(tokenIdx, headIdx);
                            const alpha = getAlphaFromAttr(attr, uThreshold, uMax);
                            const colorIntensity = getColorIntensityFromAttr(attr, uThreshold, uMax);
                            const circle = document.createElementNS(ns, 'circle');
                            circle.setAttribute('cx', xPos);
                            circle.setAttribute('cy', yPos);
                            circle.setAttribute('r', uNodeRadius);  // uノードは12倍あるので小さく
                            circle.setAttribute('class', 'node-circle');
                            // 色の濃さを反映（薄い青から濃い青へ）
                            const r = Math.floor(100 + (255 - 100) * (1.0 - colorIntensity));  // 100〜255の範囲
                            const g = Math.floor(150 + (255 - 150) * (1.0 - colorIntensity));  // 150〜255の範囲
                            const b = Math.floor(200 + (255 - 200) * (1.0 - colorIntensity));  // 200〜255の範囲（薄い青）
                            circle.setAttribute('fill', `rgb(${{r}}, ${{g}}, ${{b}})`);
                            // しきい値未満でも極薄で表示し、ホバー/クリックを許可
                            const displayOpacity = attr >= uThreshold ? alpha : 0.01;
                            circle.setAttribute('opacity', displayOpacity);
                            circle.setAttribute('stroke', 'black');
                            circle.setAttribute('stroke-width', '1.5');
                            circle.setAttribute('class', 'node-circle clickable-node');
                            circle.setAttribute('data-node-type', 'u');
                            circle.setAttribute('data-layer', layer);
                            circle.setAttribute('data-token', tokenIdx);
                            circle.setAttribute('data-head', headIdx);
                            circle.style.cursor = 'pointer';
                            
                            // ホバー: ノード情報を表示（しきい値未満でも極薄で描画してホバー可能に）
                            circle.addEventListener('mouseenter', function(e) {{
                                tooltip.innerHTML =
                                    '<strong>uノード</strong><br>' +
                                    `Layer: ${{layer}}<br>` +
                                    `Token: ${{tokenIdx}} (${{tokens[tokenIdx]}})<br>` +
                                    `Head: ${{headIdx}}<br>` +
                                    `Attribution: ${{attr.toFixed(6)}}`;
                                tooltip.style.display = 'block';
                                setTooltipPosition(tooltip, e);
                            }});
                            circle.addEventListener('mouseleave', function() {{
                                tooltip.style.display = 'none';
                            }});
                            
                                // クリック: 簡易逆伝播で再描画
                                circle.addEventListener('click', function(e) {{
                                    e.stopPropagation();
                                    const nodeKey = {{ nodeType: 'u', layer: layer, token: tokenIdx, head: headIdx }};
                                    // 同じノードが再度クリックされたら初期状態に戻す
                                    if (selectedNode && 
                                        selectedNode.nodeType === nodeKey.nodeType &&
                                        selectedNode.layer === nodeKey.layer &&
                                        selectedNode.token === nodeKey.token &&
                                        selectedNode.head === nodeKey.head) {{
                                        resetToInitialState();
                                        return;
                                    }}
                                    selectedNode = nodeKey;
                                    const res = computeRelevanceFromNode('u', layer, tokenIdx, headIdx);
                                    redrawWithRelevance(res.relZ, res.relU, res.edgeRel);
                                }});
                            
                            svg.appendChild(circle);
                        }}
                    }}
                }}
            }}
            
            // 関係ないところをクリックしたら初期状態に戻す
            plotArea.addEventListener('click', function(e) {{
                // ノードやパスをクリックした場合は何もしない（イベント伝播が止まっている）
                if (e.target.classList.contains('node-circle') || 
                    e.target.classList.contains('path-line') ||
                    e.target.closest('.node-circle') ||
                    e.target.closest('.path-line')) {{
                    return;
                }}
                // ノードが選択されている場合のみ初期状態に戻す
                if (selectedNode !== null) {{
                    resetToInitialState();
                }}
            }});
            
            // Streamlit iframe: 親ページスクロール時にツールチップを非表示
            if (window.parent && window.parent !== window) {{
                try {{
                    window.parent.document.addEventListener('scroll', hideTooltip, {{ capture: true, passive: true }});
                    window.parent.addEventListener('resize', hideTooltip, {{ passive: true }});
                }} catch (err) {{
                    // 同一オリジンでない場合はスキップ
                }}
            }}
            
            // Streamlitのiframe高さを動的に調整
            function updateIframeHeight() {{
                const body = document.body;
                const html = document.documentElement;
                const height = Math.max(
                    body.scrollHeight,
                    body.offsetHeight,
                    html.clientHeight,
                    html.scrollHeight,
                    html.offsetHeight
                );
                
                // 親ウィンドウ（Streamlitのiframe）に高さを通知
                if (window.parent && window.parent !== window) {{
                    window.parent.postMessage({{
                        type: 'streamlit:setFrameHeight',
                        height: height
                    }}, '*');
                }}
            }}
            
            // 初期化時に高さを設定
            updateIframeHeight();
            
            // リサイズやコンテンツ変更時に高さを更新
            window.addEventListener('resize', updateIframeHeight);
            
            // MutationObserverでDOMの変更を監視
            const observer = new MutationObserver(updateIframeHeight);
            observer.observe(document.body, {{
                childList: true,
                subtree: true,
                attributes: true,
                attributeFilter: ['style', 'class']
            }});
            
            // 描画完了後に高さを更新
            setTimeout(updateIframeHeight, 100);
            setTimeout(updateIframeHeight, 500);
            setTimeout(updateIframeHeight, 1000);
        </script>
    </body>
    </html>
    """

    return html
