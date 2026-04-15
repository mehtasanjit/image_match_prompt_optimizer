# Aggressive Precision Metric (Match-Only)

## Weights
| Case | Score |
|------|-------|
| TP (Match → Match) | **+1** |
| FP (Match → Not Match) | **-9** |
| FN (Not Match → Match) | **-0.5** |
| TN (Not Match → Not Match) | **0** |

## Intent
Target **~90% Match precision** — maximum precision push.

- **FP:FN ratio = 18:1** — extremely precision-favored
- Breakeven: model needs >90% of Match predictions to be correct
- FN=-0.5 is very weak — optimizer may learn "never predict Match" to game the metric

## When to Use
- When precision is the absolute top priority and recall can drop significantly
- **Warning:** High risk of recall collapse — monitor recall closely
- Best paired with a recall floor check outside the metric

## Risk
The weak FN penalty means the optimizer can score well by simply never predicting Match (all TN=0, no FP penalties). Use the "guarded" variant if this is a concern.

## Response Key
`match_score`

## Deploy
```bash
gcloud functions deploy <function_name> \
  --gen2 --runtime python312 --region us-central1 \
  --source custom_metric_weighted_match_only_aggressive \
  --entry-point product_match_weighted_metric \
  --trigger-http --allow-unauthenticated