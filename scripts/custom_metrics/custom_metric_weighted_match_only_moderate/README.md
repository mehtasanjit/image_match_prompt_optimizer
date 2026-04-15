# Moderate Precision Metric (Match-Only)

## Weights
| Case | Score |
|------|-------|
| TP (Match → Match) | **+1** |
| FP (Match → Not Match) | **-7** |
| FN (Not Match → Match) | **-1** |
| TN (Not Match → Not Match) | **0** |

## Intent
Target **~87% Match precision** with mild recall pressure.

- **FP:FN ratio = 7:1** — strongly precision-favored
- Breakeven: model needs >87.5% of Match predictions to be correct
- FN=-1 keeps a small penalty for missed matches, preventing total recall collapse

## When to Use
- Default precision-focused metric
- Safe choice — unlikely to cause "never predict Match" gaming
- Good starting point before trying more aggressive variants

## Response Key
`match_score`

## Deploy
```bash
gcloud functions deploy <function_name> \
  --gen2 --runtime python312 --region us-central1 \
  --source custom_metric_weighted_match_only_moderate \
  --entry-point product_match_weighted_metric \
  --trigger-http --allow-unauthenticated