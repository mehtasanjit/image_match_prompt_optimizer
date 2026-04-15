# Binary Match Accuracy Metric (Unweighted)

## Weights
| Case | Score |
|------|-------|
| TP (Match → Match) | **1.0** |
| FP (Match → Not Match) | **0.0** |
| FN (Not Match → Match) | **0.0** |
| TN (Not Match → Not Match) | **1.0** |

## Intent
Binary accuracy for Match classification. Positive class = Match. Simple 0/1 scoring with no precision/recall bias.

## When to Use
- Baseline Match evaluation before GEPA optimization
- Comparing against weighted metrics to understand bias impact

## Response Key
`match_score`

## Entry Point
`product_match_custom_metric`

## Deploy
```bash
gcloud functions deploy <function_name> \
  --gen2 --runtime python312 --region us-central1 \
  --source external/custom_metrics/custom_metric_match_only \
  --entry-point product_match_custom_metric \
  --trigger-http --allow-unauthenticated