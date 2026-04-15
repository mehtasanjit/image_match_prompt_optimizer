# Guarded Precision Metric (Match-Only)

## Weights
| Case | Score |
|------|-------|
| TP (Match → Match) | **+1** |
| FP (Match → Not Match) | **-9** |
| FN (Not Match → Match) | **-2** |
| TN (Not Match → Not Match) | **+0.5** |

## Intent
Target **~90% Match precision** with a **strong recall floor** to prevent gaming.

- **FP:FN ratio = 4.5:1** — precision-favored but with meaningful recall pressure
- Breakeven: model needs >90% of Match predictions to be correct
- FN=-2 creates a real cost for missed matches — optimizer can't just "never predict Match"
- TN=+0.5 rewards correct rejections, giving positive signal for conservative-but-correct behavior

## When to Use
- **Recommended** precision metric — balances aggressive precision with recall protection
- Best for production where you need 85-90% precision AND ≥50% recall
- The TN reward prevents the optimizer from seeing "predict nothing" as equally good as "predict correctly"

## Design Rationale
The FP=-9 drives precision above 90%. The FN=-2 prevents recall from collapsing below ~50%. The TN=+0.5 is a small positive signal that rewards the model for correctly identifying non-matches, rather than treating all non-Match predictions as zero-value.

## Response Key
`match_score`

## Deploy
```bash
gcloud functions deploy <function_name> \
  --gen2 --runtime python312 --region us-central1 \
  --source custom_metric_weighted_match_only_guarded \
  --entry-point product_match_weighted_metric \
  --trigger-http --allow-unauthenticated