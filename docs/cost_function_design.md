# Cost Function Design: Customizing for Precision-Recall Requirements

## Why Custom Cost Functions?

Standard accuracy metrics treat all errors equally. In image match classification, errors are **not equal**:

- A **false Match** (FP) can lead to incorrect downstream decisions — far more costly than missing a valid match
- A **false Mismatch** (FN) results in a missed detection but is recoverable through manual review

Custom cost functions encode this asymmetry directly into the optimization objective, steering the prompt optimizer (GEPA) toward prompts that prioritize precision over recall.

---

## How Cost Functions Work

Each cost function assigns a numeric score to each prediction based on the confusion matrix quadrant:

| Quadrant | Meaning | Typical Score |
|---|---|---|
| **TP** (True Positive) | Correctly predicted positive class | Reward (+1 to +2) |
| **FP** (False Positive) | Incorrectly predicted positive class | Penalty (-3 to -25) |
| **FN** (False Negative) | Missed a positive case | Penalty (-0.25 to -4) |
| **TN** (True Negative) | Correctly rejected | Neutral (0) or small reward (+0.5) |

GEPA sums these scores across all training items per iteration. The critic model sees the total score and adjusts the prompt to maximize it.

---

## The FP:FN Ratio — The Precision-Recall Knob

The ratio of FP penalty to FN penalty determines where the optimizer settles on the precision-recall curve:

```
Theoretical precision threshold = |FP| / (|FP| + |FN|)
```

| FP:FN Ratio | Precision Target | Behavior |
|---|---|---|
| 1:1 | ~50% | Balanced F1 — equal weight on precision and recall |
| 7:1 | ~87% | Moderate — good starting point for most use cases |
| 18:1 | ~90% | Aggressive — strong precision push, recall may drop |
| 60:1 | ~93%+ | Highly aggressive — near-maximum precision, high recall collapse risk |
| 100:1 | ~95%+ | Ultra aggressive — experimental, expect very low recall |

---

## The "Gaming" Problem and Guarded Variants

With high FP:FN ratios and TN=0, the optimizer discovers a degenerate shortcut: **never predict the positive class**. This achieves perfect precision (no FPs because no positive predictions) with zero recall.

**Guarded variants** close this loophole with two mechanisms:

1. **Meaningful FN penalty** (e.g., -2 or -3): Creates real cost for missed positives. The optimizer can't avoid all positive predictions without paying a significant score penalty.

2. **TN reward** (+0.5): Gives the optimizer a positive signal for correct rejections. Without this, rejecting everything scores the same as rejecting nothing (both score 0 for TN). With TN=+0.5, the optimizer is rewarded for actually learning to discriminate.

### Example: Aggressive vs Guarded

| Variant | TP | FP | FN | TN | What happens |
|---|---|---|---|---|---|
| Aggressive | +1 | -9 | -0.5 | 0 | Optimizer learns "never predict Match" → 100% precision, 0% recall |
| **Guarded** | +1 | -9 | **-2** | **+0.5** | Optimizer must predict Match sometimes → 85-90% precision, 50-70% recall |

---

## Designing a New Cost Function

### Step 1: Define your precision target

| If you need... | Set FP:FN ratio to... |
|---|---|
| ~85% precision | 5:1 to 7:1 |
| ~90% precision | 9:1 to 18:1 |
| ~93%+ precision | 15:1 to 60:1 |
| ~95%+ precision | 25:1 to 100:1 |

### Step 2: Decide if you need recall protection

- **Without guarding** (TN=0, low FN): Maximum precision ceiling but risk of recall collapse. Use for experimental runs.
- **With guarding** (TN=+0.5, FN=-2 to -4): Precision ceiling may be slightly lower but recall is protected. Use for production.

### Step 3: Set the weights

```python
# Example: ~90% precision with recall floor
WEIGHTS = {
    "TP": +1,    # Reward correct positive predictions
    "FP": -9,    # Heavy penalty for false positives
    "FN": -2,    # Meaningful penalty for missed positives
    "TN": +0.5,  # Small reward for correct rejections
}
```

### Step 4: Implement

Each cost function is a GCP Cloud Function with the same interface:

**Input** (HTTP POST JSON):
```json
{
  "response": "{\"product_match\": \"Match\", \"reason\": \"...\"}",
  "target": "Match"
}
```

**Output**:
```json
{
  "product_match_weighted_score_match_only": 1.0
}
```

The function:
1. Extracts the prediction from `response` (parses JSON, reads `product_match`)
2. Compares against `target` (ground truth)
3. Determines the confusion matrix quadrant (TP/FP/FN/TN)
4. Returns the corresponding weight as the score

---

## Practical Observations

From our experiments across multiple product categories:

| Finding | Detail |
|---|---|
| **Guarded consistently outperforms aggressive** | The recall floor prevents degenerate solutions. Guarded CF achieved 86% precision where aggressive achieved 65%. |
| **The effective FP:FN ratio matters more than individual weights** | TP=+1, FP=-9, FN=-2 performs similarly to TP=+2, FP=-18, FN=-4 — it's the ratio that shapes the curve. |
| **TN=+0.5 is the sweet spot** | TN=0 allows gaming. TN=+1.0 over-rewards rejection. +0.5 provides just enough positive signal. |
| **Higher than 100:1 FP:FN shows no benefit** | Beyond ~95% precision target, the optimizer can't distinguish signal from noise in the cost function. |
| **Start with guarded, not aggressive** | Guarded is the recommended default. Switch to aggressive only if guarded's precision ceiling is proven insufficient for your use case. |

---

## Quick Reference

### Recommended Defaults

| Use Case | Recommended Variant | Weights |
|---|---|---|
| First optimization run | Guarded | TP=+1, FP=-9, FN=-2, TN=+0.5 |
| Need higher precision | Highly Aggressive Guarded | TP=+1, FP=-15, FN=-3, TN=+0.5 |
| Exploring precision ceiling | Highly Aggressive (unguarded) | TP=+1, FP=-15, FN=-0.25, TN=0 |
| Balanced P/R baseline | Balanced | TP=+2, FP=-3, FN=-3, TN=0 |

---

## Pre-built Metrics

This repository includes **16 pre-built cost functions** (9 match + 7 mismatch) ready to deploy as Cloud Functions. Each implements the interface described above with a specific weight configuration.

See **[metrics.md](metrics.md)** for the full catalog of pre-built metrics, including:
- Cloud Function deploy names
- Entry points and score keys
- Weight tables for each variant
- Deployment scripts
