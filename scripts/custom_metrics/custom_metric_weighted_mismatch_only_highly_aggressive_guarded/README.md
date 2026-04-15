# Highly Aggressive Guarded Precision Metric (Mismatch-Only)

## Weights
| Case | Score |
|------|-------|
| TP (Mismatch → Mismatch) | **+1** |
| FP (Mismatch → Not Mismatch) | **-15** |
| FN (Not Mismatch → Mismatch) | **-3** |
| TN (Not Mismatch → Not Mismatch) | **+0.5** |

## Intent
Target **~93% Mismatch precision** with a **recall floor**. FP:FN ratio = 5:1.

- FP=-15 drives high precision
- FN=-3 prevents recall collapse
- TN=+0.5 rewards correct non-mismatch predictions

## When to Use
- When you need highest Mismatch precision AND cannot tolerate recall collapse
- Use with higher iteration counts (18-20) for convergence

## Response Key
`mismatch_score`

## Entry Point
`product_mismatch_weighted_metric`

## Deploy
```bash
gcloud functions deploy <function_name> \
  --gen2 --runtime python312 --region us-central1 \
  --source external/custom_metrics/custom_metric_weighted_mismatch_only_highly_aggressive_guarded \
  --entry-point product_mismatch_weighted_metric \
  --trigger-http --allow-unauthenticated