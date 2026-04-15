# Balanced F1 Metric (Mismatch-Only)

## Weights
| Case | Score |
|------|-------|
| TP (Mismatch → Mismatch) | **+2** |
| FP (Mismatch → Not Mismatch) | **-3** |
| FN (Not Mismatch → Mismatch) | **-3** |
| TN (Not Mismatch → Not Mismatch) | **0** |

## Intent
Target **balanced F1** (~60% breakeven). FP:FN ratio = 1:1. Equal penalty for false positives and false negatives.

## When to Use
- Initial exploration for Mismatch classification
- Understanding baseline precision-recall frontier

## Response Key
`mismatch_score`

## Entry Point
`product_mismatch_weighted_metric`

## Deploy
```bash
gcloud functions deploy <function_name> \
  --gen2 --runtime python312 --region us-central1 \
  --source external/custom_metrics/custom_metric_weighted_mismatch_only_balanced \
  --entry-point product_mismatch_weighted_metric \
  --trigger-http --allow-unauthenticated