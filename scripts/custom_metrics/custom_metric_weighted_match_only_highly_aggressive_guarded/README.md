# Highly Aggressive + Guarded Precision Metric (Match-Only)

## Weights
| Case | Score |
|------|-------|
| TP (Match → Match) | **+1** |
| FP (Match → Not Match) | **-15** |
| FN (Not Match → Match) | **-3** |
| TN (Not Match → Not Match) | **+0.5** |

## Intent
Target **~93%+ Match precision** with **strong recall floor** — combines extreme FP penalty with meaningful FN cost.

- **FP:FN ratio = 5:1** — precision-favored but with real recall pressure
- Breakeven: model needs >93.75% of Match predictions to be correct
- FN=-3 creates significant cost for missed matches — prevents "never predict Match" gaming
- TN=+0.5 rewards correct rejections

## When to Use
- **Best of both worlds** — pushes for 93%+ precision while maintaining ≥40% recall
- Use when highly_aggressive alone collapses recall too much
- Recommended for production where both high precision AND reasonable recall matter

## Comparison
| Metric | FP | FN | TN | Precision Target | Recall Risk |
|--------|----|----|-----|-----------------|-------------|
| aggressive | -9 | -0.5 | 0 | ~86% | Low |
| highly_aggressive | -15 | -0.25 | 0 | ~93% | **High** (collapse risk) |
| **highly_aggressive_guarded** | **-15** | **-3** | **+0.5** | ~93% | **Protected** |

## Response Key
`match_score`

## Deploy
```bash
gcloud functions deploy <function_name> \
  --runtime python312 --region us-central1 \
  --source custom_metric_weighted_match_only_highly_aggressive_guarded \
  --entry-point product_match_weighted_metric \
  --trigger-http --allow-unauthenticated