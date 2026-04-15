# Guarded Precision Metric (Mismatch-Only)

## Weights
| Case | Score |
|------|-------|
| TP (Mismatch → Mismatch) | **+1** |
| FP (Mismatch → Not Mismatch) | **-9** |
| FN (Not Mismatch → Mismatch) | **-2** |
| TN (Not Mismatch → Not Mismatch) | **+0.5** |

## Intent
Target **~90% Mismatch precision** with a **strong recall floor**. FP:FN ratio = 4.5:1.

- FN=-2 prevents recall collapse
- TN=+0.5 rewards correct non-mismatch predictions

## When to Use
- **Recommended** for production Mismatch classification
- Best balance of precision and recall protection

## Response Key
`mismatch_score`

## Entry Point
`product_mismatch_weighted_metric`

## Deploy
```bash
gcloud functions deploy <function_name> \
  --gen2 --runtime python312 --region us-central1 \
  --source external/custom_metrics/custom_metric_weighted_mismatch_only_guarded \
  --entry-point product_mismatch_weighted_metric \
  --trigger-http --allow-unauthenticated