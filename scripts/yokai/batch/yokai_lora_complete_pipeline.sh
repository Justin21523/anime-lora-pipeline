#!/bin/bash
################################################################################
# Yokai Watch Complete LoRA Training Pipeline
#
# Automates the entire workflow from raw episodes to ready-to-train datasets:
#
# Stage 1: Segmentation (from process_yokai_full_optimized.sh)
# Stage 2: Character Clustering (from process_yokai_full_optimized.sh)
# Stage 3: Cluster Analysis & Quality Assessment
# Stage 4: Data Augmentation (for small clusters)
# Stage 5: Caption Generation
# Stage 6: Training Data Preparation
# Stage 7: Validation
# Stage 8: Background LoRA Preparation (optional)
#
# Usage:
#   ./yokai_lora_complete_pipeline.sh
#
# Environment Variables (optional overrides):
#   INPUT_DIR - Episodes directory (default: /home/b0979/yokai_input_fast)
#   OUTPUT_BASE - Base output directory (default: /mnt/c/.../yokai_lora_pipeline)
#   SKIP_STAGES - Comma-separated stages to skip (e.g., "1,2" to skip segmentation and clustering)
#   MIN_CLUSTER_SIZE - Minimum cluster size for training (default: 15)
#   AUGMENT_SMALL_CLUSTERS - Enable augmentation for small clusters (default: true)
#   PREPARE_BACKGROUNDS - Prepare background LoRA data (default: false)
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration (with defaults)
INPUT_DIR="${INPUT_DIR:-/home/b0979/yokai_input_fast}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai_lora_pipeline}"
CONDA_ENV="${CONDA_ENV:-blip2-env}"

# Pipeline settings
SKIP_STAGES="${SKIP_STAGES:-}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-15}"
AUGMENT_SMALL_CLUSTERS="${AUGMENT_SMALL_CLUSTERS:-true}"
PREPARE_BACKGROUNDS="${PREPARE_BACKGROUNDS:-false}"

# Create timestamped output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

# Log file
LOG_FILE="${OUTPUT_DIR}/pipeline.log"

# Helper function for logging
log() {
    echo -e "${BLUE}[$(date +"%Y-%m-%d %H:%M:%S")]${NC} $1" | tee -a "${LOG_FILE}"
}

log_success() {
    echo -e "${GREEN}[$(date +"%Y-%m-%d %H:%M:%S")] ✓ $1${NC}" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[$(date +"%Y-%m-%d %H:%M:%S")] ✗ $1${NC}" | tee -a "${LOG_FILE}"
}

log_warning() {
    echo -e "${YELLOW}[$(date +"%Y-%m-%d %H:%M:%S")] ⚠ $1${NC}" | tee -a "${LOG_FILE}"
}

# Check if stage should be skipped
should_skip_stage() {
    local stage=$1
    if [[ ",${SKIP_STAGES}," == *",${stage},"* ]]; then
        return 0  # Skip
    else
        return 1  # Don't skip
    fi
}

# Banner
echo ""
echo "================================================================================"
echo "                 YOKAI WATCH COMPLETE LORA TRAINING PIPELINE"
echo "================================================================================"
echo ""
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo ""
echo "Pipeline Settings:"
echo "  Min cluster size: ${MIN_CLUSTER_SIZE}"
echo "  Augment small clusters: ${AUGMENT_SMALL_CLUSTERS}"
echo "  Prepare backgrounds: ${PREPARE_BACKGROUNDS}"
if [ -n "${SKIP_STAGES}" ]; then
    echo "  Skip stages: ${SKIP_STAGES}"
fi
echo ""
echo "Log file: ${LOG_FILE}"
echo "================================================================================"
echo ""

# Save configuration
cat > "${OUTPUT_DIR}/pipeline_config.txt" <<EOF
Yokai Watch LoRA Pipeline Configuration
Generated: $(date)

Input Directory: ${INPUT_DIR}
Output Directory: ${OUTPUT_DIR}
Conda Environment: ${CONDA_ENV}

Pipeline Settings:
  Min Cluster Size: ${MIN_CLUSTER_SIZE}
  Augment Small Clusters: ${AUGMENT_SMALL_CLUSTERS}
  Prepare Backgrounds: ${PREPARE_BACKGROUNDS}
  Skip Stages: ${SKIP_STAGES}
EOF

# Stage tracking
CURRENT_STAGE=0
TOTAL_STAGES=8
if [ "${PREPARE_BACKGROUNDS}" != "true" ]; then
    TOTAL_STAGES=7
fi

################################################################################
# STAGE 1 & 2: SEGMENTATION AND CLUSTERING
################################################################################
CURRENT_STAGE=1

if should_skip_stage "1,2"; then
    log_warning "Skipping Stages 1-2: Segmentation & Clustering"

    # User must provide existing layered frames
    LAYERED_FRAMES_DIR="${LAYERED_FRAMES_DIR:-${OUTPUT_DIR}/layered_frames}"
    CLUSTERS_DIR="${CLUSTERS_DIR:-${OUTPUT_DIR}/character_clusters}"

    if [ ! -d "${LAYERED_FRAMES_DIR}" ] || [ ! -d "${CLUSTERS_DIR}" ]; then
        log_error "Stages 1-2 skipped but no existing data found"
        log_error "Please set LAYERED_FRAMES_DIR and CLUSTERS_DIR"
        exit 1
    fi
else
    log "Stage 1-2/${TOTAL_STAGES}: Running Segmentation & Clustering"
    log "This may take several hours for full dataset..."

    # Run the optimized processing script
    SEGMENTATION_OUTPUT="${OUTPUT_DIR}/stage1_2_segmentation_clustering"

    # Export for the processing script
    export INPUT_DIR
    export OUTPUT_DIR="${SEGMENTATION_OUTPUT}"

    ./scripts/batch/process_yokai_full_optimized.sh 2>&1 | tee -a "${LOG_FILE}"

    if [ $? -eq 0 ]; then
        log_success "Stage 1-2 complete"
        LAYERED_FRAMES_DIR="${SEGMENTATION_OUTPUT}/layered_frames"
        CLUSTERS_DIR="${SEGMENTATION_OUTPUT}/character_clusters"
    else
        log_error "Stage 1-2 failed"
        exit 1
    fi
fi

################################################################################
# STAGE 3: CLUSTER ANALYSIS
################################################################################
CURRENT_STAGE=3

if should_skip_stage "3"; then
    log_warning "Skipping Stage 3: Cluster Analysis"
    CLUSTER_ANALYSIS="${CLUSTER_ANALYSIS:-${CLUSTERS_DIR}/cluster_analysis.json}"
else
    log "Stage 3/${TOTAL_STAGES}: Analyzing Character Clusters"

    CLUSTER_ANALYSIS="${OUTPUT_DIR}/stage3_cluster_analysis.json"

    conda run -n ${CONDA_ENV} python3 scripts/tools/analyze_yokai_clusters.py \
        "${CLUSTERS_DIR}" \
        --output-json "${CLUSTER_ANALYSIS}" \
        --output-html "${OUTPUT_DIR}/cluster_analysis.html" \
        2>&1 | tee -a "${LOG_FILE}"

    if [ $? -eq 0 ]; then
        log_success "Stage 3 complete"
        log "Analysis report: ${OUTPUT_DIR}/cluster_analysis.html"
    else
        log_error "Stage 3 failed"
        exit 1
    fi
fi

################################################################################
# STAGE 4: DATA AUGMENTATION
################################################################################
CURRENT_STAGE=4

if [ "${AUGMENT_SMALL_CLUSTERS}" != "true" ] || should_skip_stage "4"; then
    log_warning "Skipping Stage 4: Data Augmentation"
    AUGMENTED_CLUSTERS_DIR="${CLUSTERS_DIR}"
else
    log "Stage 4/${TOTAL_STAGES}: Augmenting Small Clusters"

    AUGMENTED_CLUSTERS_DIR="${OUTPUT_DIR}/stage4_augmented_clusters"

    conda run -n ${CONDA_ENV} python3 scripts/tools/augment_small_clusters.py \
        "${CLUSTERS_DIR}" \
        --output-dir "${AUGMENTED_CLUSTERS_DIR}" \
        --max-original 30 \
        --min-original 5 \
        --aug-intensity medium \
        2>&1 | tee -a "${LOG_FILE}"

    if [ $? -eq 0 ]; then
        log_success "Stage 4 complete"
    else
        log_error "Stage 4 failed"
        exit 1
    fi
fi

################################################################################
# STAGE 5: CAPTION GENERATION
################################################################################
CURRENT_STAGE=5

if should_skip_stage "5"; then
    log_warning "Skipping Stage 5: Caption Generation"
else
    log "Stage 5/${TOTAL_STAGES}: Generating Captions with BLIP2"
    log "This may take 30-60 minutes depending on cluster count..."

    conda run -n ${CONDA_ENV} python3 scripts/tools/batch_generate_captions_yokai.py \
        "${AUGMENTED_CLUSTERS_DIR}" \
        --cluster-analysis "${CLUSTER_ANALYSIS}" \
        --model "Salesforce/blip2-opt-6.7b" \
        --device cuda \
        --batch-size 8 \
        2>&1 | tee -a "${LOG_FILE}"

    if [ $? -eq 0 ]; then
        log_success "Stage 5 complete"
    else
        log_error "Stage 5 failed"
        exit 1
    fi
fi

################################################################################
# STAGE 6: TRAINING DATA PREPARATION
################################################################################
CURRENT_STAGE=6

if should_skip_stage "6"; then
    log_warning "Skipping Stage 6: Training Data Preparation"
else
    log "Stage 6/${TOTAL_STAGES}: Preparing Training Data"

    TRAINING_DATA_DIR="${OUTPUT_DIR}/training_data"

    # Filter clusters by minimum size
    SELECTED_CLUSTERS=$(find "${AUGMENTED_CLUSTERS_DIR}" -mindepth 1 -maxdepth 1 -type d -name "cluster_*" | while read cluster_dir; do
        num_images=$(find "$cluster_dir" -name "*.png" | wc -l)
        if [ $num_images -ge ${MIN_CLUSTER_SIZE} ]; then
            basename "$cluster_dir"
        fi
    done)

    if [ -z "${SELECTED_CLUSTERS}" ]; then
        log_error "No clusters meet minimum size requirement (${MIN_CLUSTER_SIZE} images)"
        exit 1
    fi

    log "Selected $(echo "${SELECTED_CLUSTERS}" | wc -w) clusters for training"

    conda run -n ${CONDA_ENV} python3 scripts/tools/prepare_yokai_lora_training.py \
        "${AUGMENTED_CLUSTERS_DIR}" \
        --output-dir "${TRAINING_DATA_DIR}" \
        --cluster-analysis "${CLUSTER_ANALYSIS}" \
        --validation-split 0.1 \
        2>&1 | tee -a "${LOG_FILE}"

    if [ $? -eq 0 ]; then
        log_success "Stage 6 complete"
        log "Training data: ${TRAINING_DATA_DIR}"
    else
        log_error "Stage 6 failed"
        exit 1
    fi
fi

################################################################################
# STAGE 7: VALIDATION
################################################################################
CURRENT_STAGE=7

if should_skip_stage "7"; then
    log_warning "Skipping Stage 7: Validation"
else
    log "Stage 7/${TOTAL_STAGES}: Validating Training Data"

    VALIDATION_REPORT="${OUTPUT_DIR}/validation_report.json"

    conda run -n ${CONDA_ENV} python3 scripts/tools/validate_yokai_training_data.py \
        "${TRAINING_DATA_DIR}" \
        --output-report "${VALIDATION_REPORT}" \
        --min-resolution 512 \
        2>&1 | tee -a "${LOG_FILE}"

    if [ $? -eq 0 ]; then
        log_success "Stage 7 complete"
        log "Validation report: ${VALIDATION_REPORT}"
    else
        log_warning "Stage 7 validation found issues - please review"
    fi
fi

################################################################################
# STAGE 8: BACKGROUND LORA PREPARATION (OPTIONAL)
################################################################################
CURRENT_STAGE=8

if [ "${PREPARE_BACKGROUNDS}" != "true" ] || should_skip_stage "8"; then
    log_warning "Skipping Stage 8: Background LoRA Preparation"
else
    log "Stage 8/${TOTAL_STAGES}: Preparing Background LoRA Data"

    BACKGROUNDS_DIR="${LAYERED_FRAMES_DIR}/background"
    BACKGROUND_TRAINING_DIR="${OUTPUT_DIR}/background_training_data"

    if [ ! -d "${BACKGROUNDS_DIR}" ]; then
        log_warning "Background directory not found: ${BACKGROUNDS_DIR}"
    else
        conda run -n ${CONDA_ENV} python3 scripts/tools/prepare_background_lora.py \
            "${BACKGROUNDS_DIR}" \
            --output-dir "${BACKGROUND_TRAINING_DIR}" \
            --repeat-count 10 \
            --max-backgrounds 500 \
            --similarity-threshold 0.95 \
            2>&1 | tee -a "${LOG_FILE}"

        if [ $? -eq 0 ]; then
            log_success "Stage 8 complete"
            log "Background training data: ${BACKGROUND_TRAINING_DIR}"
        else
            log_error "Stage 8 failed"
        fi
    fi
fi

################################################################################
# PIPELINE COMPLETE
################################################################################

echo ""
echo "================================================================================"
echo "                         PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
log_success "All stages completed successfully"
echo ""
echo "📊 Summary:"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Training data: ${TRAINING_DATA_DIR}"
if [ "${PREPARE_BACKGROUNDS}" == "true" ]; then
    echo "  Background data: ${BACKGROUND_TRAINING_DIR}"
fi
echo "  Log file: ${LOG_FILE}"
echo ""
echo "📁 Next Steps:"
echo ""
echo "1. Review cluster analysis:"
echo "   Open: ${OUTPUT_DIR}/cluster_analysis.html"
echo ""
echo "2. Optionally select specific characters:"
echo "   python3 scripts/tools/interactive_character_selector.py \\"
echo "     ${AUGMENTED_CLUSTERS_DIR} \\"
echo "     --analysis ${CLUSTER_ANALYSIS}"
echo ""
echo "3. Start LoRA training for selected characters:"
echo "   See training configs in: ${TRAINING_DATA_DIR}/configs/"
echo ""
echo "   Example:"
echo "   cd /path/to/kohya_ss"
echo "   accelerate launch train_network.py \\"
echo "     --config_file ${TRAINING_DATA_DIR}/configs/char_000_config.toml"
echo ""
echo "4. Review validation report:"
echo "   cat ${VALIDATION_REPORT}"
echo ""
echo "================================================================================"
echo ""

# Save summary
cat > "${OUTPUT_DIR}/PIPELINE_SUMMARY.txt" <<EOF
Yokai Watch LoRA Pipeline Summary
==================================

Completed: $(date)
Duration: ${SECONDS} seconds

Output Directory: ${OUTPUT_DIR}

Stage Results:
  1-2. Segmentation & Clustering: $([ -d "${CLUSTERS_DIR}" ] && echo "✓" || echo "✗")
  3. Cluster Analysis: $([ -f "${CLUSTER_ANALYSIS}" ] && echo "✓" || echo "✗")
  4. Data Augmentation: $([ -d "${AUGMENTED_CLUSTERS_DIR}" ] && echo "✓" || echo "✗")
  5. Caption Generation: ✓
  6. Training Data Preparation: $([ -d "${TRAINING_DATA_DIR}" ] && echo "✓" || echo "✗")
  7. Validation: $([ -f "${VALIDATION_REPORT}" ] && echo "✓" || echo "✗")
  8. Background Preparation: $([ "${PREPARE_BACKGROUNDS}" == "true" ] && echo "✓" || echo "-")

Key Outputs:
  - Training Data: ${TRAINING_DATA_DIR}
  - Cluster Analysis: ${OUTPUT_DIR}/cluster_analysis.html
  - Validation Report: ${VALIDATION_REPORT}
  - Log File: ${LOG_FILE}

Next Steps:
  1. Review cluster analysis HTML
  2. (Optional) Use interactive selector to choose characters
  3. Start LoRA training with configs in training_data/configs/
  4. Review validation report for any issues

For detailed documentation, see:
  docs/YOKAI_LORA_TRAINING_GUIDE.md
EOF

log_success "Pipeline summary saved: ${OUTPUT_DIR}/PIPELINE_SUMMARY.txt"

exit 0
