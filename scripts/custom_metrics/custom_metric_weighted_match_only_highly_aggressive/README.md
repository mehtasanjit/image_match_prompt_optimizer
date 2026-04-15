# Highly Aggressive Precision Metric (Match-Only)

## Weights
| Case | Score |
|------|-------|
| TP (Match → Match) | **+1** |
| FP (Match → Not Match) | **-15** |
| FN (Not Match → Match) | **-0.25** |
| TN (Not Match → Not Match) | **0** |

## Intent
Target **~93%+ Match precision** — maximum precision push beyond "aggressive" (~86%).

- **FP:FN ratio = 60:1** — extreme precision bias
- Breakeven: model needs >93.75% of Match predictions to be correct
- FN=-0.25 is near-zero — optimizer has almost no recall pressure
- Each single FP costs 15 TPs to recover from

## When to Use
- When aggressive (FP=-9) plateaued at ~86% and you need to push higher
- Expect significant recall drop — the optimizer will only predict Match for the most obvious cases
- **High risk of recall collapse** — use only if precision >90% is non-negotiable

## Progression
| Metric | FP | Achieved Precision |
|--------|----|--------------------|
| aggressive | -9 | ~86% |
| **highly_aggressive** | **-15** | target ~93%+ |

## Response Key
`match_score`

## Deploy
```bash
gcloud functions deploy <function_name> \
  --runtime python312 --region us-central1 \
  --source custom_metric_weighted_match_only_highly_aggressive \
  --entry-point product_match_weighted_metric \
  --trigger-http --allow-unauthenticated