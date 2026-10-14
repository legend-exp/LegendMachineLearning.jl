#!/bin/bash -l
# ============================================================================
# SBATCH script: ML-based LAr Veto — HPO on Raven-GPU (NVIDIA A100)
# Runs: process_hpo only (Hyperband search around train_model — needs GPU)
# Usage: cd <project-root> && sbatch run_hpo.sh
# ============================================================================
#SBATCH -o generated/logs/hpo.out.%j
#SBATCH -e generated/logs/hpo.err.%j
#SBATCH -D .
#SBATCH -J mlar_hpo

#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#SBATCH --mail-type=none
#SBATCH --time=12:00:00

# ── Resolve project root ────────────────────────────────────────────────────
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# ── Parse LEGEND_DATA_CONFIG from config/data_config.yaml ───────────────────
DATA_CONFIG_FILE="$PROJECT_DIR/config/data_config.yaml"
LEGEND_DATA_CONFIG="$(grep '^legend_data_config:' "$DATA_CONFIG_FILE" | sed 's/^legend_data_config:[[:space:]]*//')"
export LEGEND_DATA_CONFIG

# ── Log directory (ensure it exists) ────────────────────────────────────────
mkdir -p "$PROJECT_DIR/generated/logs"

# ── Environment setup ───────────────────────────────────────────────────────
module purge
module load gcc/14
# NOTE: Do NOT load cuda module — CUDA.jl ships its own runtime via artifacts.

export SKIP_PKG_SETUP=1
export JULIA_NUM_PRECOMPILE_TASKS=1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "============================================"
echo "Job:        $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Date:       $(date)"
echo "Project:    $PROJECT_DIR"
echo "Data cfg:   $LEGEND_DATA_CONFIG"
echo "============================================"

echo ""
echo "=== GPU Diagnostics ==="
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "======================="
echo ""

cd "$PROJECT_DIR"

# ── Julia version pin (1.12 + Zygote has pathological compilation; use 1.11) ─
JULIA_CMD="julia +1.11"
echo "Julia command: $JULIA_CMD (pinned to 1.11.x for Zygote compatibility)"

# ── Purge stale CUDA caches (may have been compiled on login node without GPU) ─
JULIA_MAJOR_MINOR="$($JULIA_CMD -e 'print("v$(VERSION.major).$(VERSION.minor)")')"
echo "Removing stale CUDA compiled caches (Julia $JULIA_MAJOR_MINOR)..."
rm -rf ~/.julia/compiled/"$JULIA_MAJOR_MINOR"/CUDA_Runtime_jll/
rm -rf ~/.julia/compiled/"$JULIA_MAJOR_MINOR"/CUDA/
rm -rf ~/.julia/compiled/"$JULIA_MAJOR_MINOR"/CUDA_Runtime_Discovery/
echo "  Done — CUDA will recompile on this GPU node."

# ── Ensure CUDA.jl is compiled on GPU node (picks up driver) ────────────────
echo "Checking CUDA on GPU node..."
$JULIA_CMD --project="$PROJECT_DIR" -e '
    using CUDA
    println("CUDA.functional() = ", CUDA.functional())
    if CUDA.functional()
        println("GPU: ", CUDA.name(CUDA.device()))
    else
        println("WARNING: CUDA not functional — HPO will use CPU (very slow)")
    end
' 2>&1 || true

# ── Run HPO (GPU + Hyperband + Zygote AD) ──────────────────────────────────
# --only=process_hpo: skip extraction/balancing/normalization/training
# Optional 1st positional arg: a single ML group to run (overrides
# `datasets.active_group:` from processing_config.yaml). Lets you launch one
# job per group on separate GPU nodes:
#     sbatch -J mlar_hpo_g1 run_hpo.sh mlgroup001
#     sbatch -J mlar_hpo_g2 run_hpo.sh mlgroup002
GROUP_ARG=""
if [[ -n "$1" ]]; then
    GROUP_ARG="--group $1"
    echo "Group override (CLI): $1"
fi
stdbuf -oL -eL $JULIA_CMD --project="$PROJECT_DIR" \
    main.jl \
    -c config/processing_config.yaml \
    --only=process_hpo \
    $GROUP_ARG

echo "============================================"
echo "Finished:   $(date)"
echo "Exit code:  $?"
echo "============================================"
