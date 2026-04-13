#!/bin/bash
# Deploy all mismatch-only metric Cloud Functions to GCP.
# Prefix: img_match = Image Matching
#
# Usage:
#   bash external/scripts/deploy_mismatch_only_metrics.sh [PROJECT_ID] [REGION]
#
# Defaults:
#   PROJECT_ID = <your-project-id>
#   REGION     = us-central1

set -euo pipefail

PROJECT="${1:-<your-project-id>}"
REGION="${2:-us-central1}"
RUNTIME="python312"
ENTRY_POINT="product_mismatch_weighted_metric"

# Base directory — resolves to external/ regardless of where the script is called from
BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

declare -A METRICS
METRICS[balanced]="custom_metrics/custom_metric_weighted_mismatch_only_balanced"
METRICS[moderate]="custom_metrics/custom_metric_weighted_mismatch_only_moderate"
METRICS[aggressive]="custom_metrics/custom_metric_weighted_mismatch_only_aggressive"
METRICS[guarded]="custom_metrics/custom_metric_weighted_mismatch_only_guarded"
METRICS[highly_aggressive]="custom_metrics/custom_metric_weighted_mismatch_only_highly_aggressive"
METRICS[highly_aggressive_guarded]="custom_metrics/custom_metric_weighted_mismatch_only_highly_aggressive_guarded"

# Unweighted mismatch metric
UNWEIGHTED_SOURCE="custom_metrics/custom_metric_mismatch_only"
UNWEIGHTED_ENTRY="product_mismatch_custom_metric"

echo "=== Deploying 7 mismatch-only metrics (1 unweighted + 6 weighted) ==="
echo "Project: ${PROJECT}"
echo "Region:  ${REGION}"
echo ""

# Deploy unweighted metric first
echo "--- Deploying: metric_mismatch_only_unweighted ---"
echo "  Source: ${BASE_DIR}/${UNWEIGHTED_SOURCE}"
gcloud functions deploy "img_match_metric_mismatch_only_unweighted" \
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
echo "  ✓ Deployed: img_match_metric_mismatch_only_unweighted"
echo ""

# Deploy weighted metrics
for KEY in balanced moderate aggressive guarded highly_aggressive highly_aggressive_guarded; do
    SOURCE_DIR="${BASE_DIR}/${METRICS[$KEY]}"
    # Shorten highly_aggressive_guarded to fit 63 char limit
    if [ "$KEY" = "highly_aggressive_guarded" ]; then
        FUNC_NAME="img_match_metric_mismatch_only_hi_agg_guarded"
    else
        FUNC_NAME="img_match_metric_mismatch_only_${KEY}"
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

echo "=== All 7 mismatch metrics deployed ==="
echo ""
echo "Cloud Function names:"
echo "  - img_match_metric_mismatch_only_unweighted (score key: mismatch_score)"
for KEY in balanced moderate aggressive guarded highly_aggressive highly_aggressive_guarded; do
    if [ "$KEY" = "highly_aggressive_guarded" ]; then
        echo "  - img_match_metric_mismatch_only_hi_agg_guarded"
    else
        echo "  - img_match_metric_mismatch_only_${KEY}"
    fi
done
echo ""
echo "All weighted return key: mismatch_score"
