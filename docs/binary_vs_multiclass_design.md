# Design Choice: Binary Decomposition vs 3-Class Classification

## The Decision

We decompose the 3-class classification (Match / Mismatch / Inconclusive) into **two independent binary classifiers** run in sequence, rather than using a single prompt to predict all three classes simultaneously.

**Our architecture:**
```
Image Pair → Match Prompt (Match vs Not_Match)
                 ├── Match → DONE (confirmed match)
                 └── Not_Match → Mismatch Prompt (Mismatch vs Not_Mismatch)
                                      ├── Mismatch → DONE (confirmed mismatch)
                                      └── Not_Mismatch → Inconclusive
```

**The alternative (rejected):**
```
Image Pair → Single Prompt → Match / Mismatch / Inconclusive
```

---

## Why Binary Decomposition?

### 1. Independent Precision Control Per Class

The core problem requirement is **different precision targets for different classes**:
- Match predictions need >90% precision (false matches are very costly)
- Mismatch predictions need >85% precision (false mismatch declarations are also costly)
- Inconclusive is the "safe fallback" — no precision requirement

With a single 3-class prompt, the cost function must simultaneously optimize precision for Match AND Mismatch. These objectives can conflict: making the prompt more conservative about Match may make it over-predict Mismatch, and vice versa.

Binary decomposition lets each stage have its **own cost function, own optimization run, and own precision target** — they don't interfere with each other.

### 2. Better Optimization Signal

GEPA (the prompt optimizer) works by reflecting on failures and adjusting the prompt. With 3 classes, the critic model sees a mix of failure types:
- False Match on a Mismatch item
- False Match on an Inconclusive item
- False Mismatch on a Match item
- False Mismatch on an Inconclusive item
- False Inconclusive on a Match item
- ... (6 possible error types)

The critic must diagnose and fix all simultaneously. This dilutes the optimization signal.

With binary decomposition, each optimizer sees **only 2 error types** (FP and FN for its specific class). The reflection is focused and actionable: "the prompt is predicting Match when it shouldn't — add guardrails for X."

This is consistent with findings in ML literature: binary classifiers in one-vs-rest configurations often outperform multi-class models when per-class optimization is needed (Rifkin & Klautau, 2004; Allwein et al., 2000).

### 3. Composability and Modularity

Binary prompts are independently:
- **Optimizable**: Run GEPA separately for Match and Mismatch, in parallel if needed
- **Evaluable**: Each prompt has its own precision/recall/F1 metrics
- **Deployable**: Update the Match prompt without touching the Mismatch prompt
- **Testable**: A/B test one stage without affecting the other

A single 3-class prompt creates tight coupling — any change affects all three classes.

### 4. The Inconclusive Class is Not a "Real" Class

Inconclusive is fundamentally different from Match and Mismatch:
- Match and Mismatch are **definitive determinations** about product identity
- Inconclusive is an **epistemic state** — "the evidence is insufficient to decide"

Treating Inconclusive as a third class forces the model to learn when to "give up," which is a meta-cognitive task fundamentally different from visual comparison. In the binary decomposition, Inconclusive emerges naturally as the residual: "not confidently a Match AND not confidently a Mismatch."

This avoids training the model to explicitly predict "I don't know" — which is notoriously hard to optimize for and tends to be either over-used (model becomes lazy) or under-used (model forces a decision on ambiguous cases).

### 5. Asymmetric Cost Structure Maps Naturally to Binary

The weighted cost functions (guarded, aggressive, etc.) are designed for binary confusion matrices. A 3-class cost function would require a 3×3 cost matrix with 9 parameters — much harder to design, tune, and reason about.

Binary decomposition lets us use well-understood 2×2 cost matrices with only 4 parameters each, where the FP:FN ratio has a clear, interpretable meaning.

---

## Tradeoffs and Limitations

### Additional Latency
Two sequential model calls instead of one. In practice, the second call only fires for ~40-60% of items (those that aren't Match), so average latency increase is ~50%, not 100%.

### Inconsistency Risk
The two prompts are optimized independently. In rare cases, a prompt pair may behave inconsistently (e.g., the Match prompt says "not enough evidence" while the Mismatch prompt confidently says "mismatch" on the same ambiguous image). This is mitigated by the sequential architecture — the Mismatch prompt only sees items the Match prompt already rejected.

### Double Optimization Effort
Two GEPA runs per category instead of one. Partially offset by each run being simpler (binary) and potentially converging faster.

---

## Precedent in ML Literature

The binary decomposition approach is well-established:

- **One-vs-Rest (OvR)** is the standard multi-class strategy for SVMs and logistic regression, preferred when per-class tuning is needed (Rifkin & Klautau, "In Defense of One-Vs-All Classification," JMLR 2004)
- **Error-Correcting Output Codes (ECOC)** decompose multi-class problems into multiple binary sub-problems for robustness (Dietterich & Bakiri, 1995)
- **Cascaded classifiers** (sequential binary decisions) are used in object detection (Viola-Jones) where early rejection is valuable — analogous to our Match-first architecture
- **Hierarchical classification** in NLP often outperforms flat multi-class when classes have different granularity or different error costs

The key insight from this literature: **when classes have asymmetric costs and different optimization requirements, decomposition into focused binary classifiers outperforms monolithic multi-class approaches.**

---

## Flow Variants

### Match-First (Default)
Best when Match items are the majority or when confirming matches early reduces downstream work.
```
Match? → Yes → DONE
         No  → Mismatch? → Yes → DONE
                            No  → Inconclusive
```

### Mismatch-First
Best when Mismatch items are the majority or mismatch detection is the primary use case.
```
Mismatch? → Yes → DONE
             No  → Match? → Yes → DONE
                             No  → Inconclusive
```

Both variants are supported via the `--first_step` flag in the multi-step pipeline.