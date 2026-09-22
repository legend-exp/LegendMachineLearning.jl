#!/bin/bash -l
# ============================================================================
# SBATCH script: ML-based LAr Veto — Per-HPGe SiPM attribution heatmap (GPU)
# Runs: process_attribution only (Integrated Gradients via Zygote on GPU)
# Usage: sbatch run_attribution.sh [<group>]   — or — bash run_attribution.sh [<group>]
# ============================================================================
#SBATCH -o generated/logs/attribution.out.%j
#SBATCH -e generated/logs/attribution.err.%j
#SBATCH -D .
#SBATCH -J mlar_attr

#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#SBATCH --mail-type=none
#SBATCH --time=01:00:00

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

mkdir -p "$PROJECT_DIR/generated/logs"

# ── Environment setup ───────────────────────────────────────────────────────
module purge
module load gcc/14
# Do NOT load cuda module — CUDA.jl ships its own runtime via artifacts.

export SKIP_PKG_SETUP=1
export JULIA_NUM_PRECOMPILE_TASKS=1

NTHR=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$NTHR
export MKL_NUM_THREADS=$NTHR
export JULIA_NUM_THREADS=$NTHR

echo "============================================"
echo "Job:        ${SLURM_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "Date:       $(date)"
echo "Project:    $PROJECT_DIR"
echo "CPUs:       $NTHR"
echo "============================================"

echo ""
echo "=== GPU diagnostics ==="
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "======================="

cd "$PROJECT_DIR"

# Julia 1.11 — 1.12 + Zygote has pathological compile times with custom Lux
# container layers. Same pin as run_training.sh.
JULIA_CMD="julia +1.11"
echo "Julia command: $JULIA_CMD (pinned to 1.11.x for Zygote compatibility)"

# ── Purge stale CUDA caches (may have been compiled on login node without GPU)
JULIA_MAJOR_MINOR="$($JULIA_CMD -e 'print("v$(VERSION.major).$(VERSION.minor)")')"
echo "Removing stale CUDA compiled caches (Julia $JULIA_MAJOR_MINOR)..."
rm -rf ~/.julia/compiled/"$JULIA_MAJOR_MINOR"/CUDA_Runtime_jll/
rm -rf ~/.julia/compiled/"$JULIA_MAJOR_MINOR"/CUDA/
rm -rf ~/.julia/compiled/"$JULIA_MAJOR_MINOR"/CUDA_Runtime_Discovery/
echo "  Done — CUDA will recompile on this GPU node."

# ── Quick CUDA functional check ─────────────────────────────────────────────
echo "Checking CUDA on GPU node..."
$JULIA_CMD --project="$PROJECT_DIR" -e '
    using CUDA
    println("CUDA.functional() = ", CUDA.functional())
    if CUDA.functional()
        println("GPU: ", CUDA.name(CUDA.device()))
    else
        println("WARNING: CUDA not functional — will fall back to CPU")
    end
' 2>&1 || true

# ── Optional positional arg: single ML group override ──────────────────────
GROUP_ARG=""
if [[ -n "$1" ]]; then
    GROUP_ARG="--group $1"
    echo "Group override (CLI): $1"
fi

stdbuf -oL -eL $JULIA_CMD --project="$PROJECT_DIR" \
    --threads=$NTHR \
    main.jl \
    -c config/processing_config.yaml \
    --only=process_attribution \
    $GROUP_ARG

echo "============================================"
echo "Finished:   $(date)"
echo "Exit code:  $?"
echo "============================================"
