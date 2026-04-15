# Highly Aggressive Precision Metric (Mismatch-Only)

## Weights
| Case | Score |
|------|-------|
| TP (Mismatch → Mismatch) | **+1** |
| FP (Mismatch → Not Mismatch) | **-15** |
| FN (Not Mismatch → Mismatch) | **-0.25** |
| TN (Not Mismatch → Not Mismatch) | **0** |

## Intent
Target **~93% Mismatch precision**. FP:FN ratio = 60:1. Extreme FP penalty with near-negligible FN penalty.

## When to Use
- Experimental — understanding precision ceiling for Mismatch
- Very likely to produce near-zero recall

## Response Key
`mismatch_score`

## Entry Point
`product_mismatch_weighted_metric`

## Deploy
```bash
gcloud functions deploy <function_name> \
  --gen2 --runtime python312 --region us-central1 \
  --source external/custom_metrics/custom_metric_weighted_mismatch_only_highly_aggressive \
  --entry-point product_mismatch_weighted_metric \
  --trigger-http --allow-unauthenticated