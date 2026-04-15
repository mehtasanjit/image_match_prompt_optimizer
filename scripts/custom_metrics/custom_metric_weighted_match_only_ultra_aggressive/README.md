# Ultra Aggressive Precision Metric (Match-Only)

## Weights
| Case | Score |
|------|-------|
| TP (Match → Match) | **+1** |
| FP (Match → Not Match) | **-25** |
| FN (Not Match → Match) | **-0.25** |
| TN (Not Match → Not Match) | **0** |

## Intent
Target **~95% Match precision**. FP:FN ratio = 100:1. Maximum precision ceiling — near-zero tolerance for false Match predictions.

## When to Use
- Experimental only — understanding theoretical precision ceiling
- Very likely to produce near-zero recall

## Response Key
`match_score`

## Entry Point
`product_match_weighted_metric`

## Deploy
```bash
gcloud functions deploy <function_name> \
  --gen2 --runtime python312 --region us-central1 \
  --source external/custom_metrics/custom_metric_weighted_match_only_ultra_aggressive \
  --entry-point product_match_weighted_metric \
  --trigger-http --allow-unauthenticated