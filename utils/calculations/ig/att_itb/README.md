# ATT ITB（Attention Input Token Baseline）

Attention 機構に対する **Input Token Baseline (ITB)** に基づく Integrated Gradients の実装です。

## 定義

- **寄与を見たい出力トークン**を $j$ と固定する。
- **ITB**では、入力の線形補間を
  - $z_k^{(l)}(a; j) = z_j^{(l)} + a(z_k^{(l)} - z_j^{(l)})$
  とおく。つまりベースラインは「出力トークン $j$ の入力表現 $z_j^{(l)}$」を全位置に置いたもの。
- この補間に対する IG により、各入力トークン $k$ の「出力 $u_j^{(l,h)}$ への寄与」が得られる。
- 自己トークン $j$ の寄与は定義により 0。

論文では、Attention 機構として貢献度を測る方法としては ITB が最も理にかなっているとして、ATT 側は ITB で議論する（`BERT_IG_baselin_paper/IBIS/sections/04_method.tex` 参照）。

## 命題：ATTITBa=0 基準の補完不変性

**略号**: **Braw** = ITB strict（自己項=0）、**B0** = ITB-zeroRatio（Zero-baseline ratio）、**Bmap** = ITB-mapRatio（Attention map ratio）。Layer 側の zeroRatio 参照は **LIG(B0)** と表記する。

**命題**（F 基準の補完不変性）: 層 $l$・ヘッド $h$・出力トークン $j$ を固定し、ATT 経路の ITB 自己項処理を $r \in \{\mathrm{Braw},\ \mathrm{B0},\ \mathrm{Bmap}\}$（実装キー: raw / zeroRatio / mapRatio）とする。このとき端点 $a=0$ で MLP に継承される基準

$$u_j^{(l,h)}(0) = \mathrm{ATT}_j^{(l,h)}\bigl(\{z_k^{(l)}(0;j)\}_k\bigr) = \mathrm{ATT}_j^{(l,h)}\bigl(\{z_j^{(l)}\}_k\bigr)$$

は $r$ に依存しない。したがって $\mathrm{IG}_{h,j}^{\mathrm{MLP},\mathrm{ATTITBa}=0}$ は Braw／B0／Bmap のいずれの下でも同一の写像である。

**証明**: 比率補完（zeroRatio／mapRatio）は積分の**後**にのみ作用する。自己スコア $\mathrm{IG}_{j,j,h}^{\mathrm{ATT,ITB}}$ を比率推定で置き換え、完全性を保つよう $\{\mathrm{IG}_{i,j,h}\}_i$ を再スケーリングするだけで、補間経路 $z_k^{(l)}(a;j) = z_j^{(l)} + a(z_k^{(l)} - z_j^{(l)})$・写像 $\mathrm{ATT}_j^{(l,h)}$・端点値 $u_j^{(l,h)}(0), u_j^{(l,h)}(1)$ を変更しない。ATTITBa=0 の MLP 帰属は ATT 経路に $u_j^{(l,h)}(0)$ を通じてのみ依存し、$a=0$ ではどの $r$ でも全トークンが $z_j^{(l)}$ に退化するから、主張が従う。∎

**実装上の帰結**: `get_mlp_baseline_att_itb_eq_zero` の 1 種類で足り、ATT を ITB／ITB-zeroRatio／ITB-mapRatio のどれと組み合わせても MLP キャッシュは共通である（`att_itb_a0` / `att_itb_zr_a0` / `att_itb_map_a0` はラベル違いで数値は同一）。合成 $L_2$ の差はすべて ATT 側スコアに起因する。BERT・GPT-2 の合成診断でも $L_2$ の厳密一致を数値確認済み。論文の対応箇所: Neural Networks 版 `main.tex` の Proposition（F 基準の補完不変性）、ICONIP 版 §ベースライン定義の同段落。

## 実装の対応関係

- 既存の `utils.calculations.ig.attention` では、`baseline_method="self_input_token"` かつ `input_type="z"` の z→u IG が **ATT ITB** と同一である。
- 本パッケージ `att_itb` は、その組み合わせを固定したラッパーを提供し、「ATT ITB」という名前で利用できるようにする。

## API

- `compute_att_itb_multi_layer`: 複数レイヤー・単一出力トークン $j$ の ATT ITB を計算。
- `compute_att_itb_multi_layer_multi_token`: 複数レイヤー×複数出力トークンの ATT ITB を一括計算。
- `ATT_ITB_BASELINE_METHOD`: `"self_input_token"`（内部で attention_ig に渡す値）。
- `ATT_ITB_INPUT_TYPE`: `"z"`（入力埋め込み経路）。

## 参照

- 理論メモ: `docs/theory/paper/2.transformerのLRPについて.md`（**ATTITBa=0**：ATT の ITB で得た $a=0$ 出力を MLP の基準にする）
- 論文: `BERT_IG_baselin_paper/IBIS/sections/04_method.tex`（Attention 側の ITB、入力補間の式）
