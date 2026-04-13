# Custom Metrics for Product Image Matching

## Overview

Custom metrics are deployed as **GCP Cloud Functions** and used as scorers during GEPA prompt optimization. Each function receives a model prediction and ground truth, computes a weighted score based on the confusion matrix quadrant, and returns the score.

For the design rationale behind these cost functions — including how to choose weights, the FP:FN ratio, and the "gaming" problem — see **[cost_function_design.md](cost_function_design.md)**.

There are two families:
- **Match metrics**: Positive class = `Match` (Stage 1 — "Is this a confirmed match?")
- **Mismatch metrics**: Positive class = `Mismatch` (Stage 2 — "Is this a confirmed mismatch?")

Each family has **1 unweighted + 8 (match) or 6 (mismatch) weighted** variants.

---

## Confusion Matrix

**Match metrics (Stage 1):**
| | GT = Match | GT ≠ Match |
|---|---|---|
| **Pred = Match** | TP | FP (false match — dangerous) |
| **Pred ≠ Match** | FN (missed match) | TN |

**Mismatch metrics (Stage 2):**
| | GT = Mismatch | GT ≠ Mismatch |
|---|---|---|
| **Pred = Mismatch** | TP | FP (false mismatch alarm) |
| **Pred ≠ Mismatch** | FN (missed mismatch) | TN |

---

## Match Metrics

### Unweighted

| Property | Value |
|---|---|
| **Variant** | Unweighted (binary accuracy) |
| **CF Deploy Name** | `img_match_metric_match_only_unweighted` |
| **Entry Point** | `product_match_custom_metric` |
| **Score Key** | `match_score` |
| **Weights** | TP=1, FP=0, FN=0, TN=1 |
| **Source** | `custom_metrics/custom_metric_match_only` |
| **Use Case** | Baseline evaluation before GEPA optimization |

### Weighted

All weighted match metrics share:
- **Entry Point**: `product_match_weighted_metric`
- **Score Key**: `match_score`

| Variant | CF Deploy Name | TP | FP | FN | TN | FP:FN | Precision Target |
|---|---|---|---|---|---|---|---|
| **Balanced** | `img_match_metric_match_only_balanced` | +2 | -3 | -3 | 0 | 1:1 | ~60% (F1) |
| **Moderate** | `img_match_metric_match_only_moderate` | +1 | -7 | -1 | 0 | 7:1 | ~87% |
| **Aggressive** | `img_match_metric_match_only_aggressive` | +1 | -9 | -0.5 | 0 | 18:1 | ~90% |
| **Guarded** ★ | `img_match_metric_match_only_guarded` | +1 | -9 | -2 | +0.5 | 4.5:1 | ~90% + recall floor |
| **Highly Aggressive** | `img_match_metric_match_only_highly_aggressive` | +1 | -15 | -0.25 | 0 | 60:1 | ~93% |
| **Highly Agg. Guarded** | `img_match_metric_match_only_hi_agg_guarded` | +1 | -15 | -3 | +0.5 | 5:1 | ~93% + recall floor |
| **Ultra Aggressive** | `img_match_metric_match_only_ultra_aggressive` | +1 | -25 | -0.25 | 0 | 100:1 | ~95% |
| **Ultra Agg. Guarded** | `img_match_metric_match_only_ultra_agg_guarded` | +1 | -25 | -4 | +0.5 | 6.25:1 | ~95% + recall floor |

★ **Recommended default for production** — best precision stability across data splits.

---

## Mismatch Metrics

### Unweighted

| Property | Value |
|---|---|
| **Variant** | Unweighted (binary accuracy) |
| **CF Deploy Name** | `img_match_metric_mismatch_only_unweighted` |
| **Entry Point** | `product_mismatch_custom_metric` |
| **Score Key** | `mismatch_score` |
| **Weights** | TP=1, FP=0, FN=0, TN=1 |
| **Source** | `custom_metrics/custom_metric_mismatch_only` |
| **Use Case** | Baseline mismatch evaluation |

### Weighted

All weighted mismatch metrics share:
- **Entry Point**: `product_mismatch_weighted_metric`
- **Score Key**: `mismatch_score`

| Variant | CF Deploy Name | TP | FP | FN | TN | FP:FN | Precision Target |
|---|---|---|---|---|---|---|---|
| **Balanced** | `img_match_metric_mismatch_only_balanced` | +2 | -3 | -3 | 0 | 1:1 | ~60% (F1) |
| **Moderate** | `img_match_metric_mismatch_only_moderate` | +1 | -7 | -1 | 0 | 7:1 | ~87% |
| **Aggressive** | `img_match_metric_mismatch_only_aggressive` | +1 | -9 | -0.5 | 0 | 18:1 | ~90% |
| **Guarded** ★ | `img_match_metric_mismatch_only_guarded` | +1 | -9 | -2 | +0.5 | 4.5:1 | ~90% + recall floor |
| **Highly Aggressive** | `img_match_metric_mismatch_only_highly_aggressive` | +1 | -15 | -0.25 | 0 | 60:1 | ~93% |
| **Highly Agg. Guarded** | `img_match_metric_mismatch_only_hi_agg_guarded` | +1 | -15 | -3 | +0.5 | 5:1 | ~93% + recall floor |

★ **Recommended default for production.**

---

## Design Principles

1. **TP**: Reward for correct positive predictions. +1 or +2.
2. **FP**: Penalty for false alarms. Primary precision lever — higher = more precision-focused.
3. **FN**: Penalty for missed positives. Lower = optimizer allowed to miss some.
4. **TN**: Reward for correct rejections. Non-zero in "guarded" variants prevents gaming by never predicting the positive class.

### The FP:FN Ratio

Determines the theoretical precision breakeven:

```
P(positive | features) > FP_weight / (FP_weight + FN_weight)
```

| FP:FN Ratio | Theoretical Precision Target |
|---|---|
| 1:1 | ~50% (balanced) |
| 7:1 | ~87% |
| 18:1 | ~90% |
| 60:1 | ~93%+ |
| 100:1 | ~95%+ |

### Guarded vs Non-Guarded

Non-guarded variants (TN=0) allow the optimizer to game the metric by never predicting the positive class — achieving "perfect precision" with zero recall.

Guarded variants add:
- **FN penalty** (e.g., -2 or -3) — creates real cost for missed positives
- **TN reward** (+0.5) — gives positive signal for correct rejections, not just zero

This forces the optimizer to actually learn the classification task rather than exploiting the "predict nothing" shortcut.

---

## Deployment

### Match metrics
```bash
bash external/scripts/deploy_match_only_metrics.sh <PROJECT_ID> <REGION>
```

### Mismatch metrics
```bash
bash external/scripts/deploy_mismatch_only_metrics.sh <PROJECT_ID> <REGION>
```

### Selection Guide

| Scenario | Recommended Metric |
|---|---|
| Initial exploration | `balanced` |
| Default production (Match) | `guarded` |
| Default production (Mismatch) | `guarded` or `moderate` |
| High-precision production | `highly_aggressive_guarded` |
| Maximum precision (experimental) | `ultra_aggressive` |
| Understanding precision ceiling | `highly_aggressive` or `ultra_aggressive` |

### Grid Search

Typical GEPA grid search grids over 2-4 CF variants × 2-3 iteration counts:

```bash
# Match:
--eval_cf_names img_match_metric_match_only_guarded,img_match_metric_match_only_moderate
--num_iterations 12,16

# Mismatch:
--eval_cf_names img_match_metric_mismatch_only_guarded,img_match_metric_mismatch_only_moderate
--num_iterations 12,16