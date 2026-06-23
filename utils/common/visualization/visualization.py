# visualization.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tempfile
import base64
import os
from typing import List, Tuple, Dict, Optional
from scipy.interpolate import PchipInterpolator  # スプライン補間用に追加

def visualize_graph(
    nodes_left: List[str], 
    nodes_right: List[str], 
    edges: List[Tuple[int, int, float]], 
    title: str, 
    width: int = 400, 
    node_spacing: int = 10, 
    node_size: int = 15
) -> str:
    """
    グラフ可視化の共通関数
    
    Args:
        nodes_left: 左側のノードラベル
        nodes_right: 右側のノードラベル  
        edges: (from_idx, to_idx, weight)のタプルリスト
        title: グラフタイトル
        width: グラフの幅
        node_spacing: ノード間隔
        node_size: ノードサイズ
    """
    G = nx.DiGraph()
    
    # 左側ノード追加
    for i, label in enumerate(nodes_left):
        G.add_node(i, label=label, pos=(0, -i * node_spacing))
        
    # 右側ノード追加  
    for i, label in enumerate(nodes_right):
        G.add_node(len(nodes_left) + i, label=label, pos=(width, -i * node_spacing))

    # エッジ追加
    for src, dst, weight in edges:
        if weight > 0.01: # 閾値以上のみ表示
            alpha = min(weight * 2, 1.0)
            G.add_edge(src, len(nodes_left) + dst, weight=weight, width=weight * 5, color=(0,0,1,alpha))
    
    # 描画
    plt.figure(figsize=(8,6))
    pos = nx.get_node_attributes(G, 'pos')
    
    # ノード描画
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_size*50)
    
    # エッジ描画
    edges_list = G.edges()
    if edges_list:
        weights = [G[u][v]['width'] for (u,v) in edges_list]
        colors = [G[u][v]['color'] for (u,v) in edges_list]
        nx.draw_networkx_edges(G, pos, width=weights, edge_color=colors, arrows=True)
    
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.title(title)
    plt.axis('off')
    
    # SVG出力
    tmp = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    plt.savefig(tmp.name, format='svg', bbox_inches='tight')
    plt.close()
    with open(tmp.name, 'r') as f:
        svg = f.read()
    os.remove(tmp.name)
    return f"data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}"

def plot_head_graph(
    mat: np.ndarray, 
    layer_idx: int, 
    token_idx: int, 
    threshold: float=0.2, 
    scale: float=15
) -> str:
    """
    H×H影響度行列を可視化
    
    Args:
        mat: H×H 寄与度行列 (0–1 正規化済)
        layer_idx: レイヤーインデックス
        token_idx: トークンインデックス
        threshold: エッジ表示の閾値
        scale: エッジの太さスケール
    """
    H = mat.shape[0]
    G = nx.DiGraph()
    
    # 左側ノード (L-1層のヘッド出力)
    for i in range(H):
        G.add_node(f"L{i}", pos=(0, -i*30), label=f"Head{i}")
    
    # 右側ノード (L層のヘッド入力)
    for j in range(H):
        G.add_node(f"R{j}", pos=(300, -j*30), label=f"Head{j}")
    
    # 閾値以上のエッジを追加
    for i in range(H):
        for j in range(H):
            w = mat[i,j]
            if w >= threshold:
                G.add_edge(f"L{i}", f"R{j}", weight=w)
    
    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_node_attributes(G, 'label')
    plt.figure(figsize=(10,8))
    plt.title(f"Head Influence Matrix - Layer {layer_idx}, Token {token_idx}")
    
    # ノード描画
    nx.draw_networkx_nodes(G, pos, nodelist=[f"L{i}" for i in range(H)], node_color='lightblue', node_size=300)
    nx.draw_networkx_nodes(G, pos, nodelist=[f"R{j}" for j in range(H)], node_color='lightgreen', node_size=300)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # エッジ描画
    for u,v,data in G.edges(data=True):
        w = data['weight']
        rgba=(1.0,0.0,0.0,min(w*2,1.0))
        nx.draw_networkx_edges(G,pos,edgelist=[(u,v)],width=w*scale,edge_color=[rgba],arrows=False)
    
    plt.axis('off')
    
    # SVG出力
    tmp = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    plt.savefig(tmp.name, format='svg', bbox_inches='tight')
    plt.close()
    with open(tmp.name, 'r') as f:
        svg = f.read()
    os.remove(tmp.name)
    return f"data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}"

def draw_smooth_edge(ax, start_pos, end_pos, start_color, end_color):
    """
    2つのノード間に滑らかな曲線を描画する共通関数
    
    Args:
        ax: matplotlib axes
        start_pos: 開始位置 (x, y)
        end_pos: 終了位置 (x, y)
        start_color: 開始色
        end_color: 終了色
    """
    # 始点と終点の間の距離を計算
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    
    # PCHIP補間のための制御点を生成（パラメトリック補間）
    # 水平方向のオフセット量を設定
    h_offset = abs(dx) * 0.2  # 横軸距離の20%をオフセットとして使用
    
    # 線の方向を判定（左から右か、右から左か）
    going_right = dx > 0
    
    # 5つの制御点を設定
    # 線の方向に応じて制御点の位置を調整
    mid_x = start_pos[0] + dx * 0.5
    mid_y = start_pos[1] + dy * 0.5
    
    if going_right:  # 左から右への線
        # 1. 始点
        # 2. 始点より横軸が少し右で縦軸が同じ点
        # 3. 中間点
        # 4. 終点より横軸が少し左で縦軸が同じ点
        # 5. 終点
        start_control_x = start_pos[0] + h_offset
        end_control_x = end_pos[0] - h_offset
    else:  # 右から左への線
        # 1. 始点
        # 2. 始点より横軸が少し左で縦軸が同じ点
        # 3. 中間点
        # 4. 終点より横軸が少し右で縦軸が同じ点
        # 5. 終点
        start_control_x = start_pos[0] - h_offset
        end_control_x = end_pos[0] + h_offset
    
    # パラメータt（0から1まで）を使用してパラメトリック補間
    t_control = np.array([0.0, 0.2, 0.5, 0.8, 1.0])  # パラメータ値（厳密に増加）
    x_control = np.array([
        start_pos[0],                    # 始点
        start_control_x,                 # 始点近くの制御点
        mid_x,                           # 中間点
        end_control_x,                   # 終点近くの制御点
        end_pos[0]                       # 終点
    ])
    y_control = np.array([
        start_pos[1],                    # 始点
        start_pos[1],                    # 始点と同じ高さ
        mid_y,                           # 中間点
        end_pos[1],                      # 終点と同じ高さ
        end_pos[1]                       # 終点
    ])
    
    # PCHIP補間を使用（パラメトリック）
    # tパラメータに基づいてX、Y座標をそれぞれ補間
    pchip_x = PchipInterpolator(t_control, x_control)
    pchip_y = PchipInterpolator(t_control, y_control)
    
    # 補間点の生成
    t_smooth = np.linspace(0.0, 1.0, 50)
    x_smooth = pchip_x(t_smooth)
    y_smooth = pchip_y(t_smooth)
    
    # 色のグラデーションを生成して各セグメントを描画
    for i in range(len(x_smooth) - 1):
        t = i / (len(x_smooth) - 1)
        current_color = tuple(start_color[j] * (1-t) + end_color[j] * t for j in range(3))
        ax.plot([x_smooth[i], x_smooth[i+1]], 
               [y_smooth[i], y_smooth[i+1]], 
               color=current_color,
               linewidth=2,
               alpha=0.7,
               zorder=1)

def plot_important_path(
    path_info: List[Dict], 
    tokens: List[str], 
    num_heads: int,
    target_token_idx: int = None
) -> plt.Figure:
    """
    重要経路を可視化（ノード位置で滑らかに曲がる曲線表示）
    
    Args:
        path_info: 経路情報のリスト（各層でのトークン、ヘッド、影響度などの情報）
        tokens: トークンリスト（入力テキストのトークン）
        num_heads: ヘッド数（BERTのアテンションヘッド数）
    
    Returns:
        matplotlib Figure: 可視化されたグラフ
    """
    print(f"🔧 plot_important_path: 開始")
    print(f"  - path_info: {len(path_info)}ステップ")
    print(f"  - tokens: {tokens}")
    print(f"  - target_token_idx: {target_token_idx}")
    # グラフ作成
    G = nx.DiGraph()
    nodes = []
    edges = []
    
    # トークン領域の設定
    token_height = 2.0  # 各トークン領域の高さ
    token_spacing = 0.5  # トークン領域間の空白
    total_spacing = token_height + token_spacing  # 1トークン分の合計高さ
    
    # ヘッドごとの分割線の設定
    num_heads = 12  # BERTのヘッド数
    head_height = token_height / num_heads  # 1ヘッド分の高さ
    
    # トークンの位置マッピングを作成（領域間の空白を考慮）
    token_positions = {token: idx * total_spacing for idx, token in enumerate(tokens)}
    
    # ヘッドごとの色を設定
    colors = plt.cm.coolwarm(np.linspace(0, 1, num_heads))
    head_colors = {h: colors[h] for h in range(num_heads)}
    node_colors = []
    
    # ノードとエッジを追加
    for i, info in enumerate(path_info):
        layer = info["layer"]
        token = tokens[info['token']]
        token_text = f"L{layer}\n{token}"
        nodes.append((token_text, {"layer": layer, "token": token}))
        
        # 各レイヤー自身のMLPの値に基づいて色を設定
        mlp_head = info.get('mlp_head', info.get('head', 0))  # mlp_headがない場合はheadを使用
        if mlp_head == '-':  # 入力層の場合
            node_colors.append('lightgray')
        else:
            try:
                head_idx = int(mlp_head)
                node_colors.append(head_colors[head_idx])
            except (ValueError, TypeError):
                node_colors.append('lightgray')
        
        if i > 0:
            next_info = path_info[i-1]
            # エッジの色を設定（グラデーション用）
            edges.append((
                f"L{layer}\n{token}",
                f"L{next_info['layer']}\n{tokens[next_info['token']]}",
                {
                    "color": node_colors[-1],  # 現在のノードの色
                    "next_color": node_colors[-2] if i > 1 else node_colors[-1]  # 次のノードの色
                }
            ))

    # Targetノードを追加（target_token_idxが指定されている場合）
    if target_token_idx is not None and 0 <= target_token_idx < len(tokens):
        target_token = tokens[target_token_idx]
        target_node_text = f"Target\n{target_token}"
        
        # 最後のレイヤーのノードを見つける
        max_layer = max(info["layer"] for info in path_info)
        last_layer_node = None
        last_layer_color = 'lightgray'
        
        # 最後のレイヤーのノードを探す
        for i, info in enumerate(path_info):
            if info["layer"] == max_layer:
                last_layer_node = f"L{info['layer']}\n{tokens[info['token']]}"
                last_layer_color = node_colors[i]
                break
        
        # Targetノードを追加
        nodes.append((target_node_text, {"layer": max_layer + 1, "token": target_token}))
        node_colors.append(last_layer_color)  # 最後のレイヤーと同じ色
        
        # 最後のレイヤーからTargetノードへのエッジを追加
        if last_layer_node:
            edges.append((
                last_layer_node,
                target_node_text,
                {
                    "color": last_layer_color,
                    "next_color": last_layer_color
                }
            ))

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # ノードの位置を設定
    pos = {}
    for node in G.nodes():
        layer = G.nodes[node]["layer"]
        token = G.nodes[node]["token"]
        
        # Targetノードの場合は特別な処理
        if node.startswith("Target\n"):
            if target_token_idx is not None:
                # Targetノードは選択されたトークンの位置に配置
                base_y = -token_positions[tokens[target_token_idx]]
                max_layer = max(info["layer"] for info in path_info)
                pos[node] = ((max_layer + 1) * 5, base_y)
            continue
        
        # 通常のノードの位置設定
        base_y = -token_positions[token]
        
        # path_infoから該当するノードのヘッド情報を取得
        node_info = next((info for info in path_info if 
                        info["layer"] == layer and 
                        tokens[info["token"]] == token), None)
        
        if node_info is not None:
            # ヘッド情報の取得と変換
            mlp_head = node_info.get('mlp_head', node_info.get('head', 0))  # mlp_headがない場合はheadを使用
            try:
                head_idx = int(mlp_head)  # 数値に変換（L0も含めて）
                # ヘッド領域内の中央（上から head 0 → head 11。分割線と一致）
                head_y = base_y + token_height / 2 - (head_idx + 0.5) * head_height
                pos[node] = (layer * 5, head_y)
            except (ValueError, TypeError):
                head_idx = 0  # 変換に失敗した場合は0を使用
                pos[node] = (layer*5, base_y)
        else:
            # ヘッド情報が見つからない場合はトークンの中央に配置
            pos[node] = (layer*5, base_y)

    # プロット作成
    # グラフのサイズをトークン領域の高さに応じて動的に調整
    total_height = len(tokens) * total_spacing
    fig_width = 14  # 幅は固定
    fig_height = max(8, total_height * 0.5)  # スケーリング係数を0.8から0.5に変更して縦方向をコンパクトに
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # トークン領域の背景と境界線を描画
    for idx, token in enumerate(tokens):
        y_pos = -idx * total_spacing
        
        # 背景色を描画（薄いグレー）
        ax.axhspan(
            y_pos - token_height/2,  # 下の境界
            y_pos + token_height/2,  # 上の境界
            xmin=0,
            xmax=1,
            color='lightgray',
            alpha=0.3
        )
        
        # トークン領域の境界線を描画
        ax.axhline(y=y_pos - token_height/2, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=y_pos + token_height/2, color='gray', linestyle='--', alpha=0.5)
        
        # ヘッドごとの分割線を描画（上から Head0 → Head11）
        for h in range(num_heads):
            head_y = y_pos + token_height / 2 - h * head_height
            ax.axhline(
                y=head_y,
                xmin=0,
                xmax=1,
                color='gray',
                linestyle=':',
                alpha=0.3,
                linewidth=0.5
            )
            
            if h < num_heads:
                next_head_y = y_pos + token_height / 2 - (h + 1) * head_height
                mid_y = (head_y + next_head_y) / 2
                
                ax.text(
                    -0.1,
                    mid_y,
                    f"Head{h}",
                    fontsize=8,
                    verticalalignment='center',
                    horizontalalignment='right',
                    color='gray',
                    alpha=0.7
                )
        
        # トークンラベルを描画
        ax.text(
            -4.5,
            y_pos,  # 領域の中央に配置
            f"{idx}: {token}",
            fontsize=14,
            verticalalignment='center',
            horizontalalignment='right',
            rotation=270,  # 時計回りに90度回転
            bbox=dict(
                facecolor='white',
                alpha=0.7,
                edgecolor='gray',
                boxstyle='round,pad=0.2'
            )
        )

    # 凡例の固定位置を設定
    max_layer = max(info["layer"] for info in path_info)
    legend_x = max_layer * 5 + 6
    legend_y = 0
    legend_width = 3
    legend_height = 0.4
    legend_spacing = 0.5
    
    # 凡例の背景を描画
    legend_bg = plt.Rectangle(
        (legend_x, legend_y - (num_heads-1)*legend_spacing - legend_height),
        legend_width + 2,
        num_heads * legend_spacing + legend_height,
        facecolor='white',
        alpha=0.8,
        zorder=0
    )
    ax.add_patch(legend_bg)
    
    # 各ヘッドの色見本とラベルを描画（フォントサイズを2倍に）
    for h in range(num_heads):
        rect = plt.Rectangle(
            (legend_x, legend_y - h*legend_spacing),
            legend_width * 0.3,
            legend_height,
            facecolor=head_colors[h],
            zorder=1
        )
        ax.add_patch(rect)
        ax.text(
            legend_x + legend_width * 0.4,
            legend_y - h*legend_spacing + legend_height/2,
            f"Head {h}",
            horizontalalignment='left',
            verticalalignment='center',
            fontsize=16,  # 8から16に変更
            color='black',
            zorder=1
        )
    
    # グラフ本体を描画
    # ノードを点として描画（この部分を削除）
    # for node, pos_xy in pos.items():
    #     layer = G.nodes[node]["layer"]
    #     x, y = pos_xy
    #     ax.plot(x, y, 
    #             color=node_colors[nodes.index((node, G.nodes[node]))],
    #             marker='o',
    #             markersize=8,
    #             zorder=2)

    # エッジを滑らかなグラデーションで描画
    for (u, v, data) in G.edges(data=True):
        start_pos = pos[u]
        end_pos = pos[v]
        
        draw_smooth_edge(ax, start_pos, end_pos, data['color'], data['next_color'])

    # タイトルと表示範囲の設定
    plt.title("Token Influence Propagation Path", pad=20, fontsize=20)
    
    # 表示範囲を調整
    ax.set_xlim(0, legend_x + legend_width + 3)  # 左端を-3から-1.5に変更して原点に近づける
    ax.set_ylim(-total_height + token_height/2, token_height/2)
        
    # 横軸にレイヤー番号を表示（Targetも含める）
    max_layer = max(info["layer"] for info in path_info)
    layer_positions = [layer * 5 for layer in range(max_layer + 1)]
    layer_labels = [f"Layer {layer}" for layer in range(max_layer + 1)]
    
    # Targetノードがある場合は追加
    if target_token_idx is not None:
        layer_positions.append((max_layer + 1) * 5)
        layer_labels.append("Target")
    
    ax.set_xticks(layer_positions)
    ax.set_xticklabels(layer_labels, fontsize=10)
    
    # 縦軸は非表示
    ax.set_ylabel("")
    ax.tick_params(labelleft=False)
    
    # レイアウトを調整
    plt.tight_layout()
    
    print(f"✅ plot_important_path: 完了")
    return fig