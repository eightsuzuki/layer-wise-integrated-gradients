#!/usr/bin/env python3
"""
ATT ITB「生」(direct_zero) の理論検証: 自己項 IG_{j,j}^{ATT,input} が 0 に近いか確認する。

理論 (BERT_IG_baselin_paper/IBIS sections/04_method.tex § self Input Token Baseline):
  z_i(0) = z_j のとき、自己トークン j は (z_j - z_j) = 0 より IG_{j,j}^{ATT,input} = 0 となる。
  これはサブワード単位の主張。

注意: キャッシュの attns は語単位に集約済み（_build_word_level_matrix）。
  語が複数サブワードからなる場合、語の「自己項」= 同一語内の全サブワード間の寄与の和となり、
  サブワード同士の交差項は 0 でないため、語単位では自己項が 0 に近くならないことがある。
  したがって FAIL が出ても、サブワード単位の実装が理論通りなら問題ない場合がある。
attns 軸: [L, H, S, T] = [層, ヘッド, ソース語, ターゲット語]。自己項は attns[l,h,j,j]。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_attns(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    arr = np.asarray(data.get("attns"), dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(f"attns must be 4D, got {arr.shape}")
    return arr


def verify_self_terms_zero(attns: np.ndarray, atol: float = 1e-5, rtol: float = 1e-4) -> dict:
    """自己項 attns[:, :, j, j] が 0 に近いかチェック。"""
    L, H, S, T = attns.shape
    n = min(S, T)
    diag = np.array([attns[:, :, j, j] for j in range(n)])  # [T, L, H]
    abs_max = np.abs(diag).max()
    abs_mean = np.abs(diag).mean()
    col_sums = attns[:, :, :n, :n].sum(axis=2)  # [L, H, T]
    col_sum_norms = np.abs(col_sums).max()
    # 相対誤差: 自己項の絶対値 / 列和の典型スケール
    scale = max(col_sum_norms, 1e-12)
    rel = abs_max / scale
    ok = abs_max <= atol or rel <= rtol
    return {
        "abs_max_self": float(abs_max),
        "abs_mean_self": float(abs_mean),
        "col_sum_scale": float(scale),
        "rel_self_vs_scale": float(rel),
        "passed": bool(ok),
        "atol": atol,
        "rtol": rtol,
    }


def main():
    parser = argparse.ArgumentParser(description="ITB 生の自己項が 0 か検証（IBIS 理論）")
    parser.add_argument(
        "direct_zero_dir",
        type=Path,
        help="ATT ITB direct_zero キャッシュ（..._baseline_self_input_token_direct_zero）",
    )
    parser.add_argument("--atol", type=float, default=1e-5, help="絶対許容誤差")
    parser.add_argument("--rtol", type=float, default=1e-4, help="相対許容誤差（列和に対する比）")
    parser.add_argument("--max-samples", type=int, default=5, help="検証するサンプル数")
    args = parser.parse_args()

    direct_zero_dir = args.direct_zero_dir
    if not direct_zero_dir.is_dir():
        print(f"Error: not a directory: {direct_zero_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(direct_zero_dir.glob("sample_*.json"))[: args.max_samples]
    if not files:
        print(f"No sample_*.json in {direct_zero_dir}", file=sys.stderr)
        sys.exit(1)

    print("理論: ITB ではサブワード単位で自己項 IG_{j,j}^{ATT,input} = 0（IBIS 04_method.tex）")
    print("注意: キャッシュは語単位集約のため、語が複数サブワードのときは自己項が 0 でないことがある。")
    print(f"検証: attns[l,h,j,j]（語単位）が atol={args.atol} または rel<={args.rtol} で 0 に近いか\n")

    all_passed = True
    for f in files:
        try:
            attns = load_attns(f)
            r = verify_self_terms_zero(attns, atol=args.atol, rtol=args.rtol)
            status = "OK" if r["passed"] else "FAIL"
            if not r["passed"]:
                all_passed = False
            print(f"  {f.name}: {status}  abs_max_self={r['abs_max_self']:.2e}  rel={r['rel_self_vs_scale']:.2e}")
        except Exception as e:
            print(f"  {f.name}: ERROR {e}")
            all_passed = False

    print()
    if all_passed:
        print("結論: 全サンプルで語単位の自己項は 0 に近いです。")
    else:
        print(
            "結論: 語単位の自己項が 0 から外れています。"
            " 理論はサブワード単位のため、語集約後は同一語内の交差項で非零になり得ます。"
            " サブワード単位の実装は utils/calculations/ig/attention/core/baseline_computation.py の self_input_token を参照。"
        )
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
