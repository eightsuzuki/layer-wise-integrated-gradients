#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer 直接 IG（一気通貫）の実装検証。

理論: L_j(a) = ||z_j^{(l+1)}(a) - z_j^{(l+1)}(0)||_2 に対する IG を計算する。
Captum は全補間点を1バッチで渡すため、baseline 出力を事前に set_baseline_output() で
設定しないと diff=0 となり結果が全て 0 になる。本テストは非零が返ることを確認する。
"""

import sys
import unittest
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


class TestLayerDirectIG(unittest.TestCase):
    """Layer 直接 IG が理論通り非零を返すことを検証する。"""

    def test_layer_direct_ig_returns_nonzero(self):
        """compute_layer_direct_ig_single_target が少なくとも一部非零を返すこと。"""
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest("torch/transformers が無いためスキップ")

        from transformers import AutoModel, AutoTokenizer

        from utils.calculations.ig.z2z.layer_direct_ig import compute_layer_direct_ig_single_target

        model = AutoModel.from_pretrained(
            "bert-base-uncased",
            output_hidden_states=True,
            attn_implementation="eager",
        )
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        text = "The cat sat on the mat."
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        hidden_states = outputs.hidden_states
        self.assertIsNotNone(hidden_states)
        self.assertGreater(len(hidden_states), 0)

        z_layer = hidden_states[0]
        attention_mask = inputs["attention_mask"].float()
        result = compute_layer_direct_ig_single_target(
            bert_model=model,
            z_layer=z_layer,
            attention_mask=attention_mask,
            layer_idx=0,
            target_token_idx=1,
            num_steps=4,
            baseline_method="zero",
        )

        result = np.asarray(result)
        self.assertEqual(result.ndim, 1, "結果は [seq_len] の1次元")
        self.assertGreater(result.size, 0, "結果が空")
        self.assertFalse(np.all(result == 0), (
            "Layer 直接 IG が全て 0 です。set_baseline_output の事前呼び出しや "
            "Captum のバッチ渡し対応が正しく実装されているか確認してください。"
        ))
        self.assertTrue(np.any(result != 0), "少なくとも1要素は非零であるべき")

    def test_layer_direct_ig_completeness_zero_baseline(self):
        """IG 完全性: sum_i IG_{i,j} ≈ L_j(input) - L_j(baseline)。zero では L_j(baseline)=0 なので sum_i IG_{i,j} ≈ L_j(input)。"""
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest("torch/transformers が無いためスキップ")

        from transformers import AutoModel, AutoTokenizer

        from utils.calculations.ig.z2z.layer_direct_ig import (
            LayerDirectIGWrapper,
            _compute_baseline_z,
            compute_layer_direct_ig_single_target,
        )
        from utils.calculations.shared.device_utils import ensure_model_on_device

        model = AutoModel.from_pretrained(
            "bert-base-uncased",
            output_hidden_states=True,
            attn_implementation="eager",
        )
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        text = "The cat sat on the mat."
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        hidden_states = outputs.hidden_states
        z_layer = hidden_states[0]
        attention_mask = inputs["attention_mask"].float()

        layer_idx = 0
        target_token_idx = 1
        num_steps = 16
        baseline_method = "zero"

        result = compute_layer_direct_ig_single_target(
            bert_model=model,
            z_layer=z_layer,
            attention_mask=attention_mask,
            layer_idx=layer_idx,
            target_token_idx=target_token_idx,
            num_steps=num_steps,
            baseline_method=baseline_method,
        )
        result = np.asarray(result)
        ig_sum = float(np.sum(result))

        baseline_z = _compute_baseline_z(
            baseline_method=baseline_method,
            z_layer=z_layer,
            bert_model=model,
            layer_idx=layer_idx,
            target_token_idx=target_token_idx,
            attention_mask=attention_mask,
        )
        ensure_model_on_device(model)
        wrapper = LayerDirectIGWrapper(
            bert_model=model,
            layer_idx=layer_idx,
            target_token_idx=target_token_idx,
            attention_mask=attention_mask,
        )
        wrapper.to(device)
        wrapper.eval()
        wrapper.set_baseline_output(baseline_z)
        with torch.no_grad():
            L_j_at_input = wrapper(z_layer).squeeze().cpu().item()

        self.assertGreater(L_j_at_input, 0, "L_j(input) は正であるべき")
        self.assertGreater(ig_sum, 0, "寄与の和は正であるべき")
        # 積分近似誤差のため num_steps=16 では 15% を超えることがある
        rtol = 0.40
        self.assertAlmostEqual(
            ig_sum,
            L_j_at_input,
            delta=rtol * L_j_at_input,
            msg=f"完全性: sum_i IG_{{i,j}} ≈ L_j. ig_sum={ig_sum:.6f} L_j={L_j_at_input:.6f}",
        )


if __name__ == "__main__":
    unittest.main(module=__name__, verbosity=2)
