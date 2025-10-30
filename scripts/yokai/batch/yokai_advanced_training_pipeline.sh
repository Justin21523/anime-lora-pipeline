#!/bin/bash
################################################################################
# Yokai Watch Advanced Training Pipeline
#
# Integrated pipeline for all advanced training features:
# 1. Summon scene detection (visual + audio)
# 2. Action sequence extraction (AnimateDiff)
# 3. Special effects organization
# 4. Style classification (AI + manual)
# 5. Multi-concept LoRA preparation
# 6. ControlNet preprocessing
#
# Usage:
#   ./scripts/batch/yokai_advanced_training_pipeline.sh
#
# Configuration:
#   Edit the variables below to customize processing
################################################################################

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print with color
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

################################################################################
# CONFIGURATION
################################################################################

# Input directories
EPISODES_DIR="${EPISODES_DIR:-/home/b0979/yokai_input_fast}"
CHARACTER_CLUSTERS_DIR="${CHARACTER_CLUSTERS_DIR:-/mnt/c/AI_LLM_projects/ai_warehouse/training_data/yokai-watch/character_clusters}"
LAYERED_FRAMES_DIR="${LAYERED_FRAMES_DIR:-/mnt/c/AI_LLM_projects/ai_warehouse/cache/yokai-watch/layered_frames}"
BACKGROUND_DIR="${LAYERED_FRAMES_DIR}/background"

# Output directories
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/c/AI_LLM_projects/ai_warehouse/training_data/yokai-watch/advanced}"
SUMMON_SCENES_DIR="${OUTPUT_BASE}/summon_scenes"
ACTION_SEQUENCES_DIR="${OUTPUT_BASE}/action_sequences"
ORGANIZED_EFFECTS_DIR="${OUTPUT_BASE}/organized_effects"
STYLE_TAXONOMY_FILE="${OUTPUT_BASE}/yokai_taxonomy.json"
MULTI_CONCEPT_DIR="${OUTPUT_BASE}/multi_concept_training"
CONTROLNET_DIR="${OUTPUT_BASE}/controlnet_datasets"

# Feature toggles (set to "true" or "false")
ENABLE_SUMMON_DETECTION="${ENABLE_SUMMON_DETECTION:-true}"
ENABLE_ACTION_SEQUENCES="${ENABLE_ACTION_SEQUENCES:-true}"
ENABLE_EFFECTS_ORGANIZATION="${ENABLE_EFFECTS_ORGANIZATION:-true}"
ENABLE_STYLE_CLASSIFICATION="${ENABLE_STYLE_CLASSIFICATION:-true}"
ENABLE_MULTI_CONCEPT="${ENABLE_MULTI_CONCEPT:-true}"
ENABLE_CONTROLNET="${ENABLE_CONTROLNET:-true}"

# Processing parameters
SUMMON_MIN_SCORE="${SUMMON_MIN_SCORE:-60.0}"
SUMMON_EXTRACT_MODE="${SUMMON_EXTRACT_MODE:-sample}"  # key/all/sample
ACTION_LENGTHS="${ACTION_LENGTHS:-16 32}"  # Space-separated sequence lengths
STYLE_THRESHOLD="${STYLE_THRESHOLD:-0.3}"
STYLE_SAMPLE_SIZE="${STYLE_SAMPLE_SIZE:-10}"
STYLE_INTERACTIVE="${STYLE_INTERACTIVE:-true}"  # Interactive review
CONTROLNET_TYPES="${CONTROLNET_TYPES:-canny depth openpose lineart segmentation}"

# Device
DEVICE="${DEVICE:-cuda}"

# Conda environment
CONDA_ENV="${CONDA_ENV:-blip2-env}"

################################################################################
# HELPER FUNCTIONS
################################################################################

check_dependencies() {
    print_info "Checking dependencies..."

    # Check conda environment
    if ! conda env list | grep -q "^${CONDA_ENV} "; then
        print_error "Conda environment '${CONDA_ENV}' not found"
        exit 1
    fi

    # Check input directories
    if [ ! -d "$EPISODES_DIR" ] && [ "$ENABLE_SUMMON_DETECTION" = "true" -o "$ENABLE_ACTION_SEQUENCES" = "true" ]; then
        print_warning "Episodes directory not found: $EPISODES_DIR"
        print_warning "Disabling summon detection and action sequences"
        ENABLE_SUMMON_DETECTION="false"
        ENABLE_ACTION_SEQUENCES="false"
    fi

    if [ ! -d "$CHARACTER_CLUSTERS_DIR" ]; then
        print_warning "Character clusters directory not found: $CHARACTER_CLUSTERS_DIR"
        print_warning "Disabling style classification and multi-concept training"
        ENABLE_STYLE_CLASSIFICATION="false"
        ENABLE_MULTI_CONCEPT="false"
    fi

    print_success "Dependencies checked"
}

create_output_dirs() {
    print_info "Creating output directories..."
    mkdir -p "$OUTPUT_BASE"
    [ "$ENABLE_SUMMON_DETECTION" = "true" ] && mkdir -p "$SUMMON_SCENES_DIR"
    [ "$ENABLE_ACTION_SEQUENCES" = "true" ] && mkdir -p "$ACTION_SEQUENCES_DIR"
    [ "$ENABLE_EFFECTS_ORGANIZATION" = "true" ] && mkdir -p "$ORGANIZED_EFFECTS_DIR"
    [ "$ENABLE_MULTI_CONCEPT" = "true" ] && mkdir -p "$MULTI_CONCEPT_DIR"
    [ "$ENABLE_CONTROLNET" = "true" ] && mkdir -p "$CONTROLNET_DIR"
    print_success "Output directories created"
}

################################################################################
# STAGE 1: SUMMON SCENE DETECTION
################################################################################

run_summon_detection() {
    if [ "$ENABLE_SUMMON_DETECTION" != "true" ]; then
        print_info "Summon detection disabled, skipping..."
        return 0
    fi

    print_info "=========================================="
    print_info "STAGE 1: Summon Scene Detection"
    print_info "=========================================="

    conda run -n "$CONDA_ENV" python3 scripts/tools/yokai_summon_scene_detector.py \
        "$EPISODES_DIR" \
        --output-dir "$SUMMON_SCENES_DIR" \
        --extract-mode "$SUMMON_EXTRACT_MODE" \
        --min-score "$SUMMON_MIN_SCORE" \
        --device "$DEVICE"

    print_success "Summon scene detection complete"
    echo ""
}

################################################################################
# STAGE 2: ACTION SEQUENCE EXTRACTION
################################################################################

run_action_extraction() {
    if [ "$ENABLE_ACTION_SEQUENCES" != "true" ]; then
        print_info "Action sequence extraction disabled, skipping..."
        return 0
    fi

    print_info "=========================================="
    print_info "STAGE 2: Action Sequence Extraction"
    print_info "=========================================="

    conda run -n "$CONDA_ENV" python3 scripts/tools/action_sequence_extractor.py \
        "$EPISODES_DIR" \
        --output-dir "$ACTION_SEQUENCES_DIR" \
        --lengths $ACTION_LENGTHS \
        --format animatediff \
        --device "$DEVICE"

    print_success "Action sequence extraction complete"
    echo ""
}

################################################################################
# STAGE 3: SPECIAL EFFECTS ORGANIZATION
################################################################################

run_effects_organization() {
    if [ "$ENABLE_EFFECTS_ORGANIZATION" != "true" ]; then
        print_info "Effects organization disabled, skipping..."
        return 0
    fi

    if [ ! -d "$SUMMON_SCENES_DIR" ]; then
        print_warning "Summon scenes directory not found, skipping effects organization"
        return 0
    fi

    print_info "=========================================="
    print_info "STAGE 3: Special Effects Organization"
    print_info "=========================================="

    conda run -n "$CONDA_ENV" python3 scripts/tools/special_effects_organizer.py \
        "$SUMMON_SCENES_DIR" \
        --output-dir "$ORGANIZED_EFFECTS_DIR" \
        --separate-layers \
        --device "$DEVICE"

    print_success "Effects organization complete"
    echo ""
}

################################################################################
# STAGE 4: STYLE CLASSIFICATION
################################################################################

run_style_classification() {
    if [ "$ENABLE_STYLE_CLASSIFICATION" != "true" ]; then
        print_info "Style classification disabled, skipping..."
        return 0
    fi

    print_info "=========================================="
    print_info "STAGE 4: Style Classification"
    print_info "=========================================="

    INTERACTIVE_FLAG=""
    if [ "$STYLE_INTERACTIVE" != "true" ]; then
        INTERACTIVE_FLAG="--no-interactive"
    fi

    conda run -n "$CONDA_ENV" python3 scripts/tools/yokai_style_classifier.py \
        "$CHARACTER_CLUSTERS_DIR" \
        --output-json "$STYLE_TAXONOMY_FILE" \
        --threshold "$STYLE_THRESHOLD" \
        --sample-size "$STYLE_SAMPLE_SIZE" \
        $INTERACTIVE_FLAG \
        --device "$DEVICE"

    print_success "Style classification complete"
    echo ""
}

################################################################################
# STAGE 5: MULTI-CONCEPT LORA PREPARATION
################################################################################

run_multi_concept_prep() {
    if [ "$ENABLE_MULTI_CONCEPT" != "true" ]; then
        print_info "Multi-concept LoRA preparation disabled, skipping..."
        return 0
    fi

    if [ ! -f "$STYLE_TAXONOMY_FILE" ]; then
        print_warning "Style taxonomy file not found, skipping multi-concept preparation"
        print_warning "Run style classification first or disable multi-concept preparation"
        return 0
    fi

    print_info "=========================================="
    print_info "STAGE 5: Multi-Concept LoRA Preparation"
    print_info "=========================================="

    # Check if groups file exists
    GROUPS_FILE="${OUTPUT_BASE}/concept_groups.json"
    if [ ! -f "$GROUPS_FILE" ]; then
        print_info "Creating default concept groups..."
        cat > "$GROUPS_FILE" <<EOF
[
  {
    "name": "cat_type_yokai",
    "category": "appearance",
    "values": ["animal_cat"]
  },
  {
    "name": "dog_type_yokai",
    "category": "appearance",
    "values": ["animal_dog"]
  },
  {
    "name": "cute_yokai",
    "category": "style",
    "values": ["cute"]
  },
  {
    "name": "fire_attribute",
    "category": "attribute",
    "values": ["fire"]
  },
  {
    "name": "water_attribute",
    "category": "attribute",
    "values": ["water"]
  }
]
EOF
        print_info "Default groups created at: $GROUPS_FILE"
        print_info "Edit this file to customize concept groups"
    fi

    conda run -n "$CONDA_ENV" python3 scripts/tools/multi_concept_lora_preparer.py \
        "$CHARACTER_CLUSTERS_DIR" \
        --taxonomy "$STYLE_TAXONOMY_FILE" \
        --output-dir "$MULTI_CONCEPT_DIR" \
        --groups "$GROUPS_FILE" \
        --training-type concept \
        --device "$DEVICE"

    print_success "Multi-concept LoRA preparation complete"
    echo ""
}

################################################################################
# STAGE 6: CONTROLNET PREPROCESSING
################################################################################

run_controlnet_prep() {
    if [ "$ENABLE_CONTROLNET" != "true" ]; then
        print_info "ControlNet preprocessing disabled, skipping..."
        return 0
    fi

    print_info "=========================================="
    print_info "STAGE 6: ControlNet Preprocessing"
    print_info "=========================================="

    # Process each character cluster
    cluster_count=0
    for cluster_dir in "$CHARACTER_CLUSTERS_DIR"/cluster_*; do
        if [ ! -d "$cluster_dir" ]; then
            continue
        fi

        cluster_name=$(basename "$cluster_dir")
        output_dir="$CONTROLNET_DIR/$cluster_name"

        print_info "Processing $cluster_name..."

        # Check if background directory exists
        BG_FLAG=""
        if [ -d "$BACKGROUND_DIR" ]; then
            BG_FLAG="--background-dir $BACKGROUND_DIR"
        fi

        conda run -n "$CONDA_ENV" python3 scripts/tools/controlnet_complete_pipeline.py \
            "$cluster_dir" \
            --output-dir "$output_dir" \
            --control-types $CONTROLNET_TYPES \
            $BG_FLAG \
            --device "$DEVICE"

        cluster_count=$((cluster_count + 1))
    done

    print_success "ControlNet preprocessing complete ($cluster_count clusters processed)"
    echo ""
}

################################################################################
# SUMMARY AND REPORT
################################################################################

generate_summary() {
    print_info "=========================================="
    print_info "PIPELINE SUMMARY"
    print_info "=========================================="

    SUMMARY_FILE="${OUTPUT_BASE}/pipeline_summary.txt"

    {
        echo "Yokai Watch Advanced Training Pipeline"
        echo "Completed: $(date)"
        echo ""
        echo "=========================================="
        echo "CONFIGURATION"
        echo "=========================================="
        echo "Input directories:"
        echo "  Episodes: $EPISODES_DIR"
        echo "  Character clusters: $CHARACTER_CLUSTERS_DIR"
        echo "  Background: $BACKGROUND_DIR"
        echo ""
        echo "Output directory: $OUTPUT_BASE"
        echo ""
        echo "Enabled stages:"
        echo "  Summon detection: $ENABLE_SUMMON_DETECTION"
        echo "  Action sequences: $ENABLE_ACTION_SEQUENCES"
        echo "  Effects organization: $ENABLE_EFFECTS_ORGANIZATION"
        echo "  Style classification: $ENABLE_STYLE_CLASSIFICATION"
        echo "  Multi-concept prep: $ENABLE_MULTI_CONCEPT"
        echo "  ControlNet prep: $ENABLE_CONTROLNET"
        echo ""
        echo "=========================================="
        echo "OUTPUT LOCATIONS"
        echo "=========================================="

        if [ "$ENABLE_SUMMON_DETECTION" = "true" ] && [ -d "$SUMMON_SCENES_DIR" ]; then
            scene_count=$(find "$SUMMON_SCENES_DIR" -name "scene_*" -type d 2>/dev/null | wc -l || echo "0")
            echo "Summon scenes: $SUMMON_SCENES_DIR"
            echo "  Detected scenes: $scene_count"
        fi

        if [ "$ENABLE_ACTION_SEQUENCES" = "true" ] && [ -d "$ACTION_SEQUENCES_DIR" ]; then
            seq_count=$(find "$ACTION_SEQUENCES_DIR" -name "*_seq*" -type d 2>/dev/null | wc -l || echo "0")
            echo "Action sequences: $ACTION_SEQUENCES_DIR"
            echo "  Extracted sequences: $seq_count"
        fi

        if [ "$ENABLE_EFFECTS_ORGANIZATION" = "true" ] && [ -d "$ORGANIZED_EFFECTS_DIR" ]; then
            echo "Organized effects: $ORGANIZED_EFFECTS_DIR"
        fi

        if [ "$ENABLE_STYLE_CLASSIFICATION" = "true" ] && [ -f "$STYLE_TAXONOMY_FILE" ]; then
            echo "Style taxonomy: $STYLE_TAXONOMY_FILE"
        fi

        if [ "$ENABLE_MULTI_CONCEPT" = "true" ] && [ -d "$MULTI_CONCEPT_DIR" ]; then
            concept_count=$(find "$MULTI_CONCEPT_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l || echo "0")
            echo "Multi-concept training: $MULTI_CONCEPT_DIR"
            echo "  Prepared concepts: $concept_count"
        fi

        if [ "$ENABLE_CONTROLNET" = "true" ] && [ -d "$CONTROLNET_DIR" ]; then
            cn_cluster_count=$(find "$CONTROLNET_DIR" -name "cluster_*" -type d 2>/dev/null | wc -l || echo "0")
            echo "ControlNet datasets: $CONTROLNET_DIR"
            echo "  Processed clusters: $cn_cluster_count"
        fi

        echo ""
        echo "=========================================="
        echo "NEXT STEPS"
        echo "=========================================="
        echo ""
        echo "1. Review generated datasets"
        echo "2. For multi-concept training, run:"
        echo "   accelerate launch train_network.py --config_file <concept_config.toml>"
        echo ""
        echo "3. For individual character training, use existing tools:"
        echo "   python3 scripts/tools/prepare_yokai_lora_training.py"
        echo ""
        echo "4. For ControlNet training, refer to ControlNet documentation"
        echo ""
        echo "5. See docs/ADVANCED_FEATURES_QUICK_START.md for detailed workflows"
        echo ""

    } | tee "$SUMMARY_FILE"

    print_success "Summary saved to: $SUMMARY_FILE"
}

################################################################################
# MAIN EXECUTION
################################################################################

main() {
    print_info "=========================================="
    print_info "YOKAI WATCH ADVANCED TRAINING PIPELINE"
    print_info "=========================================="
    echo ""

    # Check dependencies
    check_dependencies

    # Create output directories
    create_output_dirs

    # Run stages
    run_summon_detection
    run_action_extraction
    run_effects_organization
    run_style_classification
    run_multi_concept_prep
    run_controlnet_prep

    # Generate summary
    generate_summary

    print_success "=========================================="
    print_success "PIPELINE COMPLETE"
    print_success "=========================================="
}

# Run main function
main "$@"
