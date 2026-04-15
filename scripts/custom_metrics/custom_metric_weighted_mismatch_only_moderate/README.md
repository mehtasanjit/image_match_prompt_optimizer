# Moderate Precision Metric (Mismatch-Only)

## Weights
| Case | Score |
|------|-------|
| TP (Mismatch → Mismatch) | **+1** |
| FP (Mismatch → Not Mismatch) | **-7** |
| FN (Not Mismatch → Mismatch) | **-1** |
| TN (Not Mismatch → Not Mismatch) | **0** |

## Intent
Target **~87% Mismatch precision**. FP:FN ratio = 7:1. Good balance between precision and recall.

## When to Use
- Default choice for most categories in Stage 2 (Mismatch classification)
- When ~87% precision is acceptable and you want decent recall

## Response Key
`mismatch_score`

## Entry Point
`product_mismatch_weighted_metric`

## Deploy
```bash
gcloud functions deploy <function_name> \
  --gen2 --runtime python312 --region us-central1 \
  --source external/custom_metrics/custom_metric_weighted_mismatch_only_moderate \
  --entry-point product_mismatch_weighted_metric \
  --trigger-http --allow-unauthenticated