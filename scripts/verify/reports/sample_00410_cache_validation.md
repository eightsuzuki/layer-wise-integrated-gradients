# sample_00410 cache validation

- Generated: 2026-06-21T16:39:34.417364+00:00
- Sample: PTB dev #410
- Cache root: `/home/data/eight/bert_token_embedding_visualization/cache/ptb_ig_analysis`
- Recompute layer IG: `False`

## Layer-direct z2z (3 methods)

| method | status | check | shape | max_abs_diff | max_rel_diff | recompute_s |
|--------|--------|-------|-------|--------------|--------------|-------------|
| zero | OK | exists_only | [12, 41, 41] |  |  |  |
| itb | OK | exists_only | [12, 41, 41] |  |  |  |
| itb_zero_ratio | PASS | derived_from_itb_zero | [12, 41, 41] | 0.0 | 0.0 |  |

## Composed z2z (8 combinations)

| suffix | status | check | shape | max_abs_diff | max_rel_diff |
|--------|--------|-------|-------|--------------|--------------|
| ATT_zero_MLP_zero | PASS | recompose_att_mlp | [12, 35, 35] | 0.0 | 0.0 |
| ATT_zero_MLP_ATTITBa0 | PASS | recompose_att_mlp | [12, 35, 35] | 0.0 | 0.0 |
| ATT_ITB_raw_MLP_zero | PASS | recompose_att_mlp | [12, 35, 35] | 0.0 | 0.0 |
| ATT_ITB_raw_MLP_ATTITBa0 | PASS | recompose_att_mlp | [12, 35, 35] | 0.0 | 0.0 |
| ATT_ITB_map_MLP_zero | PASS | recompose_att_mlp | [12, 35, 35] | 0.0 | 0.0 |
| ATT_ITB_map_MLP_ATTITBa0 | PASS | recompose_att_mlp | [12, 35, 35] | 0.0 | 0.0 |
| ATT_ITB_zero_base_ratio_MLP_zero | PASS | recompose_att_mlp | [12, 35, 35] | 0.0 | 0.0 |
| ATT_ITB_zero_base_ratio_MLP_ATTITBa0 | PASS | recompose_att_mlp | [12, 35, 35] | 0.0 | 0.0 |

## Summary

- Layer-direct failures/missing: 0 / 3
- Composed failures/missing: 0 / 8
