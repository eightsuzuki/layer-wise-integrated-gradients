"""BERT の ATT IG が本文 Eq.~(ig-att) の定義どおりであることを固定する。

2026-08-25 に 3 つの逸脱が見つかった:

1. `_extract_target_output` が `attention_output[0, ...]` とバッチ添字を 0 で固定
   していた。Captum は IG の補間ステップをバッチとして一度に渡すので、経路上の
   1 点しか見ておらず、他のステップに対する勾配が恒等的に 0 だった。
2. 次元方向を `torch.norm` で潰していた。Eq.~(ig-att) は「和」で、ノルムだと
   符号が消え、完全性 sum_i IG = g(1) - g(0) も成り立たない。
3. z^(l) に位置埋め込みを足し直し、層 0..l-1 を再通過してから層 l の attention を
   取っていた。Eq.~(ig-att) が微分するのは ATT^(l) だけ。

いずれも「もっともらしい数値」を出し続けていた（全層でほぼ一様、正規化エントロピー
0.99）。ここでは性質を書く — 実装の内部状態ではなく、定義との一致と完全性を見る。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("captum")
from captum.attr import IntegratedGradients  # noqa: E402

SENTENCE = "the quick brown fox jumps over the lazy dog .".split()


@pytest.fixture(scope="module")
def bert():
    from transformers import BertModel, BertTokenizerFast

    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device), tok, torch.device(device)


def _hidden(model, tok, device):
    enc = tok(SENTENCE, is_split_into_words=True, return_tensors="pt").to(device)
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    return enc, hs


def _reference_ig(model, hs, layer, token, head, baseline, steps):
    """本文 Eq.~(ig-att) をそのまま書いたもの。ATT^(l) のみ・次元は和。"""
    nh = model.config.num_attention_heads
    hd = model.config.hidden_size // nh
    z = hs[layer].detach()
    attn = model.encoder.layer[layer].attention.self

    def head_out(x):
        c = attn(x)[0]
        return c.view(c.size(0), c.size(1), nh, hd)[:, token, head, :]

    if baseline == "zero":
        base = torch.zeros_like(z)
    else:
        base = z.clone()
        base[:, :] = z[:, token : token + 1, :]
    b = head_out(base).detach()

    def f(x):
        return torch.norm(head_out(x) - b, dim=-1)

    attributions = IntegratedGradients(f).attribute(
        inputs=z.float(), baselines=base.float(), n_steps=steps,
        method="riemann_trapezoid",
    )
    with torch.no_grad():
        delta = float(f(z.float())) - float(f(base.float()))
    return attributions.sum(-1).squeeze(0).detach().cpu().numpy(), delta


def _production_ig(model, enc, hs, layer, token, head, baseline, steps):
    from utils.calculations.ig.attention.attention_ig import (
        compute_attention_ig_global_analysis_multi_layer as run,
    )

    inputs = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "token_type_ids": enc.get("token_type_ids"),
    }
    out = run(
        bert_model=model, inputs=inputs, layer_indices=[layer],
        target_token_idx=token, target_head_idx=head, num_steps=steps,
        cached_hidden_states=hs, baseline_method=baseline, input_type="z",
        use_direct_computation=False,
    )
    return np.asarray(out[layer]["ig_values"], dtype=np.float64)


@pytest.mark.parametrize("baseline", ["zero", "self_input_token"])
@pytest.mark.parametrize("layer", [0, 5, 11])
def test_matches_equation_ig_att(bert, baseline, layer):
    """本番の値が、定義をそのまま書いた参照実装と一致する。"""
    model, tok, device = bert
    enc, hs = _hidden(model, tok, device)
    token, head = 3, 0
    prod = _production_ig(model, enc, hs, layer, token, head, baseline, 32)
    ref, _ = _reference_ig(model, hs, layer, token, head, baseline, 32)
    n = min(len(prod), len(ref))
    scale = max(np.abs(ref[:n]).max(), 1e-8)
    assert np.abs(prod[:n] - ref[:n]).max() / scale < 1e-2


@pytest.mark.parametrize("baseline", ["zero", "self_input_token"])
def test_completeness_improves_with_steps(bert, baseline):
    """完全性が steps を増やすと縮む。

    IG が経路積分になっていれば成り立つ。バッチ添字を 0 に固定していた頃は
    steps を増やしても縮まなかった（1 点しか見ていないため）。
    """
    model, tok, device = bert
    enc, hs = _hidden(model, tok, device)
    layer, token, head = 5, 3, 0
    _, delta = _reference_ig(model, hs, layer, token, head, baseline, 32)
    err = []
    for steps in (32, 256):
        vals, d = _reference_ig(model, hs, layer, token, head, baseline, steps)
        err.append(abs(vals.sum() - d) / max(abs(d), 1e-8))
    assert err[1] < err[0] / 2, f"完全性が改善しない: {err}"


def test_target_output_keeps_batch_dimension(bert):
    """Captum が渡す補間ステップのバッチ次元が落ちない。"""
    from utils.calculations.ig.attention.attention_models import AttentionModel

    model, tok, device = bert
    enc, hs = _hidden(model, tok, device)
    am = AttentionModel(
        bert_model=model, layer_idx=5, target_token_idx=3, target_head_idx=0
    )
    batched = hs[5].repeat(7, 1, 1)
    out = am._compute_attention_output_vector(
        batched, enc["attention_mask"].float(), enc.get("token_type_ids")
    )
    assert out.shape[0] == 7, f"バッチ次元が落ちている: {tuple(out.shape)}"


def test_att_ig_is_single_layer(bert):
    """層 l の ATT だけを微分している。

    下位層を通り直していれば、層 l-1 の重みを変えても結果が変わってしまう。
    """
    model, tok, device = bert
    enc, hs = _hidden(model, tok, device)
    layer, token, head = 5, 3, 0
    before = _production_ig(model, enc, hs, layer, token, head, "zero", 32)
    dense = model.encoder.layer[layer - 1].output.dense
    with torch.no_grad():
        saved = dense.weight.clone()
        dense.weight.mul_(1.5)
    try:
        after = _production_ig(model, enc, hs, layer, token, head, "zero", 32)
    finally:
        with torch.no_grad():
            dense.weight.copy_(saved)
    n = min(len(before), len(after))
    assert np.allclose(before[:n], after[:n], atol=1e-5), (
        "下位層の重みを変えたら ATT IG が変わった = 下位層を通り直している"
    )
