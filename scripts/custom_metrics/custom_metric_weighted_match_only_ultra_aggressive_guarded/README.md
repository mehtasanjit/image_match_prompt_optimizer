# Ultra Aggressive Guarded Precision Metric (Match-Only)

## Weights
| Case | Score |
|------|-------|
| TP (Match → Match) | **+1** |
| FP (Match → Not Match) | **-25** |
| FN (Not Match → Match) | **-4** |
| TN (Not Match → Not Match) | **+0.5** |

## Intent
Target **~95% Match precision** with a **strong recall floor**. FP:FN ratio = 6.25:1.

- FP=-25 drives extreme precision
- FN=-4 is the strongest recall guard of any variant — prevents recall collapse
- TN=+0.5 rewards correct rejections

## When to Use
- When you need maximum precision AND cannot tolerate recall collapse
- Safety-critical applications

## Response Key
`match_score`

## Entry Point
`product_match_weighted_metric`

## Deploy
```bash
gcloud functions deploy <function_name> \
  --gen2 --runtime python312 --region us-central1 \
  --source external/custom_metrics/custom_metric_weighted_match_only_ultra_aggressive_guarded \
  --entry-point product_match_weighted_metric \
  --trigger-http --allow-unauthenticated