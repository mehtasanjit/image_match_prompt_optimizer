# Problem Statement: Automated Prompt Optimization for Image Match Classification

## Overview

This framework addresses the problem of **product image comparison**: given a **reference image** (e.g., from a product catalog or listing) and a **candidate image** (a photo taken under real-world conditions, such as a customer photo, warehouse capture, or field image), determine whether both images depict the same product.

The comparison produces one of three outcomes:

- **Match** — the candidate image shows the same product as the reference image
- **Mismatch** — the candidate image shows a different product
- **Inconclusive** — image quality, occlusion, or missing visual information prevents a definitive determination

This applies broadly to any product domain — electronics, apparel, accessories, appliances, etc. — where visual verification of product identity is needed.

The system prompt that drives this classification must be carefully tuned per product category, and each new model version can degrade a previously working prompt. This project automates that prompt optimization process.

## The Core Challenge

The classification task is deceptively hard because:

1. **Visual similarity ≠ product identity.** Two different product variants may look nearly identical, differing only in subtle features (texture, color shade, component shape). The model must learn category-specific visual heuristics.

2. **User images are noisy.** Products may be photographed in packaging, poor lighting, unusual angles, or with partial occlusion. The model must distinguish "image is unclear" (→ Inconclusive) from "product is clearly different" (→ Mismatch).

3. **Precision is asymmetric.** A false Match (claiming a product matches when it doesn't) has much higher cost than a false Mismatch. The optimization must heavily penalize false positives while maintaining reasonable recall.

4. **Prompts are fragile across model versions.** A prompt optimized for one model version may degrade significantly on the next, requiring re-optimization across all deployed categories.

## What We're Optimizing

The system prompt instructs the model on:
- How to describe and compare products across the two images
- Category-specific directives (e.g., check strap design, pattern layout, material texture)
- When to classify as Inconclusive vs making a definitive call
- Output format (structured JSON with reasoning)

The optimization seeks to find prompt instructions that maximize **Match precision (>90%)** while maintaining **Match recall (>80%)** across unseen test data.

## Why Manual Tuning Fails

- **Doesn't scale:** Many product categories × prompt tuning per category × re-tuning per model version = unsustainable manual effort
- **Non-obvious tradeoffs:** Small prompt changes cause large precision swings; the precision-recall Pareto frontier is hard to navigate manually
- **No systematic evaluation:** Without automated metrics and grid search, prompt engineers rely on spot-checking rather than statistical validation

## The Framework: Automatable and Replicable

The central design goal is not just to optimize prompts for initial categories, but to create a **repeatable, automated pipeline** that can be applied to any new product category with minimal human intervention.

The framework:
1. Takes a base system prompt and labeled evaluation data (reference image, user image, ground truth label) as input
2. Uses prompt optimization (MLflow GEPA) to iteratively refine the prompt instructions
3. Evaluates against custom weighted metrics that encode the precision-first objective
4. Runs grid search across multiple cost function variants and iteration counts
5. Selects the best prompt based on validation set performance with train/test consistency checks
6. Produces per-category optimized prompts that can be rapidly re-optimized when model versions change

**Replicability across categories** is achieved by:
- A **category-agnostic base prompt** that works as the starting point — the optimizer adds category-specific refinements
- **Configurable sampling and metric configs** (JSON files) that parameterize the pipeline without code changes
- **Grid search over cost functions** that automatically finds the right precision-recall tradeoff for each category's error distribution
- **Standardized evaluation scripts** that produce comparable metrics (precision, recall, F1, confusion matrix) across all categories

**Replicability across model versions** is achieved by:
- Re-running the same pipeline with the new model version — the optimizer adapts the prompt to the new model's behavior
- No manual prompt re-engineering required; the same base prompt + labeled data + pipeline produces an updated optimized prompt

## Dataset

- Multiple product categories, ~400 labeled samples per category
- Each sample: reference image + user image + ground truth label (Match / Mismatch / Inconclusive)
- Data split into **train / validation / test** using stratified sampling to ensure balanced label representation

## Success Criteria

| Metric | Target |
|--------|--------|
| Match Precision | >90% |
| Match Recall | >80% |
| Precision stability across splits (max−min) | <0.10 |
| Time to optimize a new category | <1 day (automated) |