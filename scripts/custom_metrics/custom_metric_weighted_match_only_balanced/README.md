# Balanced Metric (Match-Only)

## Weights
| Case | Score |
|------|-------|
| TP (Match → Match) | **+2** |
| FP (Match → Not Match) | **-3** |
| FN (Not Match → Match) | **-3** |
| TN (Not Match → Not Match) | **0** |

## Intent
**F1-optimized baseline** — equal penalty for precision and recall errors.

- **FP:FN ratio = 1:1** — no precision/recall preference
- Breakeven: model needs >60% of Match predictions to be correct
- TP=+2 gives stronger positive signal than other variants

## When to Use
- Baseline comparison metric — run alongside precision-focused metrics to see how much recall you sacrifice for precision
- When you want the optimizer to find the natural precision/recall balance without bias
- Good for understanding the model's "natural" operating point

## Comparison Role
This metric exists to answer: "How much precision do we gain from the precision-focused metrics vs the natural F1-optimal point?" Compare results from this metric against moderate/aggressive/guarded to quantify the tradeoff.

## Response Key
`match_score`

## Deploy
```bash
gcloud functions deploy <function_name> \
  --gen2 --runtime python312 --region us-central1 \
  --source custom_metric_weighted_match_only_balanced \
  --entry-point product_match_weighted_metric \
  --trigger-http --allow-unauthenticated