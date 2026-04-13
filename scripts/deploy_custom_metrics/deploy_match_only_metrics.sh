#!/bin/bash
# Deploy all match-only metric Cloud Functions to GCP.
# Prefix: img_match = Image Matching
#
# Usage:
#   bash external/scripts/deploy_match_only_metrics.sh [PROJECT_ID] [REGION]
#
# Defaults:
#   PROJECT_ID = <your-project-id>
#   REGION     = us-central1

set -euo pipefail

PROJECT="${1:-<your-project-id>}"
REGION="${2:-us-central1}"
RUNTIME="python312"
ENTRY_POINT="product_match_weighted_metric"

# Base directory — resolves to external/ regardless of where the script is called from
BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

declare -A METRICS
METRICS[moderate]="custom_metrics/custom_metric_weighted_match_only_moderate"
METRICS[aggressive]="custom_metrics/custom_metric_weighted_match_only_aggressive"
METRICS[guarded]="custom_metrics/custom_metric_weighted_match_only_guarded"
METRICS[balanced]="custom_metrics/custom_metric_weighted_match_only_balanced"
METRICS[highly_aggressive]="custom_metrics/custom_metric_weighted_match_only_highly_aggressive"
METRICS[highly_aggressive_guarded]="custom_metrics/custom_metric_weighted_match_only_highly_aggressive_guarded"
METRICS[ultra_aggressive]="custom_metrics/custom_metric_weighted_match_only_ultra_aggressive"
METRICS[ultra_aggressive_guarded]="custom_metrics/custom_metric_weighted_match_only_ultra_aggressive_guarded"

# Unweighted match metric
UNWEIGHTED_SOURCE="custom_metrics/custom_metric_match_only"
UNWEIGHTED_ENTRY="product_match_custom_metric"

echo "=== Deploying 9 match-only metrics (1 unweighted + 8 weighted) ==="
echo "Project: ${PROJECT}"
echo "Region:  ${REGION}"
echo ""

# Deploy unweighted metric first
echo "--- Deploying: metric_match_only_unweighted ---"
echo "  Source: ${BASE_DIR}/${UNWEIGHTED_SOURCE}"
gcloud functions deploy "img_match_metric_match_only_unweighted" \
    --quiet \
    --runtime "${RUNTIME}" \
    --region "${REGION}" \
    --project "${PROJECT}" \
    --source "${BASE_DIR}/${UNWEIGHTED_SOURCE}" \
    --entry-point "${UNWEIGHTED_ENTRY}" \
    --trigger-http \
    --allow-unauthenticated \
    --memory 256Mi \
    --timeout 30s
echo "  ✓ Deployed: img_match_metric_match_only_unweighted"
echo ""

# Deploy weighted metrics
for KEY in moderate aggressive guarded balanced highly_aggressive highly_aggressive_guarded ultra_aggressive ultra_aggressive_guarded; do
    SOURCE_DIR="${BASE_DIR}/${METRICS[$KEY]}"
    FUNC_NAME="img_match_metric_match_only_${KEY}"

    # Shorten names that exceed 63 char limit
    if [ "$KEY" = "highly_aggressive_guarded" ]; then
        FUNC_NAME="img_match_metric_match_only_hi_agg_guarded"
    fi
    if [ "$KEY" = "ultra_aggressive_guarded" ]; then
        FUNC_NAME="img_match_metric_match_only_ultra_agg_guarded"
    fi

    echo "--- Deploying: ${FUNC_NAME} ---"
    echo "  Source: ${SOURCE_DIR}"

    gcloud functions deploy "${FUNC_NAME}" \
        --quiet \
        --runtime "${RUNTIME}" \
        --region "${REGION}" \
        --project "${PROJECT}" \
        --source "${SOURCE_DIR}" \
        --entry-point "${ENTRY_POINT}" \
        --trigger-http \
        --allow-unauthenticated \
        --memory 256Mi \
        --timeout 30s

    echo "  ✓ Deployed: ${FUNC_NAME}"
    echo ""
done

echo "=== All 9 match-only metrics deployed ==="
echo ""
echo "Cloud Function names:"
echo "  - img_match_metric_match_only_unweighted (score key: match_score)"
for KEY in moderate aggressive guarded balanced highly_aggressive highly_aggressive_guarded ultra_aggressive ultra_aggressive_guarded; do
    if [ "$KEY" = "highly_aggressive_guarded" ]; then
        echo "  - img_match_metric_match_only_hi_agg_guarded"
    elif [ "$KEY" = "ultra_aggressive_guarded" ]; then
        echo "  - img_match_metric_match_only_ultra_agg_guarded"
    else
        echo "  - img_match_metric_match_only_${KEY}"
    fi
done
echo ""
echo "All weighted return key: match_score"
