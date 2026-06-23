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

## 実装の対応関係

- 既存の `utils.calculations.ig.attention` では、`baseline_method="self_input_token"` かつ `input_type="z"` の z→u IG が **ATT ITB** と同一である。
- 本パッケージ `att_itb` は、その組み合わせを固定したラッパーを提供し、「ATT ITB」という名前で利用できるようにする。

## API

- `compute_att_itb_multi_layer`: 複数レイヤー・単一出力トークン $j$ の ATT ITB を計算。
- `compute_att_itb_multi_layer_multi_token`: 複数レイヤー×複数出力トークンの ATT ITB を一括計算。
- `ATT_ITB_BASELINE_METHOD`: `"self_input_token"`（内部で attention_ig に渡す値）。
- `ATT_ITB_INPUT_TYPE`: `"z"`（入力埋め込み経路）。

## 参照

- 理論メモ: `theory/paper/2.transformerのLRPについて.md`（**ATTITBa=0**：ATT の ITB で得た $a=0$ 出力を MLP の基準にする）
- 論文: `BERT_IG_baselin_paper/IBIS/sections/04_method.tex`（Attention 側の ITB、入力補間の式）
