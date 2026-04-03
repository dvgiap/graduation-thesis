#!/bin/bash
#
# Automated training runner for ACWI vs Baseline comparison.
#
# Usage:
#   bash run_all_training.sh                  # Run everything (Phase 1-4)
#   bash run_all_training.sh --phase 1        # Minimum viable: none+icm, 3 seeds, DoorKey-8x8
#   bash run_all_training.sh --phase 2        # Expand to 5 seeds
#   bash run_all_training.sh --phase 3        # Add count+ride
#   bash run_all_training.sh --phase 4        # Add remaining environments
#
# Each run checks if its CSV log already has >= 450 lines (near-complete).
# If so, it skips that run. Delete the CSV to force re-training.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHASE="${1:-all}"  # default: run all phases

# Parse --phase argument
if [[ "$1" == "--phase" ]]; then
    PHASE="$2"
fi

# ---- Configuration per phase ----

case "$PHASE" in
    1)
        ENVS=("MiniGrid-DoorKey-8x8-v0")
        METHODS=("none" "icm")
        SEED_START=1
        SEED_END=3
        ;;
    2)
        ENVS=("MiniGrid-DoorKey-8x8-v0")
        METHODS=("none" "icm")
        SEED_START=1
        SEED_END=5
        ;;
    3)
        ENVS=("MiniGrid-DoorKey-8x8-v0")
        METHODS=("none" "icm" "count" "ride")
        SEED_START=1
        SEED_END=5
        ;;
    4|all)
        ENVS=(
            "MiniGrid-DoorKey-8x8-v0"
            "MiniGrid-Empty-16x16-v0"
            "MiniGrid-RedBlueDoors-8x8-v0"
            "MiniGrid-UnlockPickup-v0"
        )
        METHODS=("none" "icm" "count" "ride")
        SEED_START=1
        SEED_END=5
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Usage: bash run_all_training.sh [--phase 1|2|3|4|all]"
        exit 1
        ;;
esac

MAX_STEPS=1000000
MIN_LINES=450  # A complete 1M-step run produces ~500 CSV lines

# Suffix mapping
declare -A SUFFIX_MAP
SUFFIX_MAP[none]=""
SUFFIX_MAP[icm]="_ICM"
SUFFIX_MAP[count]="_COUNT"
SUFFIX_MAP[ride]="_RIDE"

# ---- Helper ----

is_complete() {
    local csv_path="$1"
    if [[ ! -f "$csv_path" ]]; then
        return 1  # file does not exist
    fi
    local line_count
    line_count=$(wc -l < "$csv_path")
    if (( line_count >= MIN_LINES )); then
        return 0  # complete
    fi
    return 1  # incomplete
}

run_training() {
    local variant_dir="$1"  # ppo-curiosity or ppo-acwi-curiosity
    local variant_label="$2"
    local method="$3"
    local env="$4"
    local seed="$5"

    local suffix="${SUFFIX_MAP[$method]}"
    local csv_path="${SCRIPT_DIR}/${variant_dir}/logs/${env}/PPO${suffix}_${env}_seed_${seed}.csv"

    if is_complete "$csv_path"; then
        echo "[SKIP] ${variant_label} | ${method} | ${env} | seed ${seed} (already complete)"
        return
    fi

    echo ""
    echo "========================================================================"
    echo "[RUN]  ${variant_label} | ${method} | ${env} | seed ${seed}"
    echo "========================================================================"
    local start_time
    start_time=$(date +%s)

    (
        cd "${SCRIPT_DIR}/${variant_dir}"
        python train.py \
            --method "$method" \
            --env "$env" \
            --seed_start "$seed" \
            --seed_end "$seed" \
            --max_steps "$MAX_STEPS"
    )

    local end_time
    end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    echo "[DONE] ${variant_label} | ${method} | ${env} | seed ${seed} | ${elapsed}s"
}

# ---- Main loop ----

echo "============================================"
echo "  Training Runner — Phase: ${PHASE}"
echo "  Environments: ${ENVS[*]}"
echo "  Methods: ${METHODS[*]}"
echo "  Seeds: ${SEED_START}-${SEED_END}"
echo "  Max steps: ${MAX_STEPS}"
echo "============================================"
echo ""

TOTAL_START=$(date +%s)

for env in "${ENVS[@]}"; do
    for method in "${METHODS[@]}"; do
        for seed in $(seq "$SEED_START" "$SEED_END"); do
            # Run baseline
            run_training "ppo-curiosity" "Baseline" "$method" "$env" "$seed"
            # Run ACWI
            run_training "ppo-acwi-curiosity" "ACWI" "$method" "$env" "$seed"
        done
    done
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))

echo ""
echo "============================================"
echo "  All training complete!"
echo "  Total time: ${TOTAL_ELAPSED}s"
echo ""
echo "  Generate comparison figures:"
echo "    python compare_results.py --env MiniGrid-DoorKey-8x8-v0"
echo "============================================"
