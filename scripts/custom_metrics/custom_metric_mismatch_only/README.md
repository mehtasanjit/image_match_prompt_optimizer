# Binary Mismatch Accuracy Metric (Unweighted)

## Weights
| Case | Score |
|------|-------|
| TP (Mismatch → Mismatch) | **1.0** |
| FP (Mismatch → Not Mismatch) | **0.0** |
| FN (Not Mismatch → Mismatch) | **0.0** |
| TN (Not Mismatch → Not Mismatch) | **1.0** |

## Intent
Binary accuracy for Mismatch classification. Positive class = Mismatch. Simple 0/1 scoring with no precision/recall bias.

## When to Use
- Baseline Mismatch evaluation before GEPA optimization
- Stage 2 benchmark runs

## Response Key
`mismatch_score`

## Entry Point
`product_mismatch_custom_metric`

## Deploy
```bash
gcloud functions deploy <function_name> \
  --gen2 --runtime python312 --region us-central1 \
  --source external/custom_metrics/custom_metric_mismatch_only \
  --entry-point product_mismatch_custom_metric \
  --trigger-http --allow-unauthenticated