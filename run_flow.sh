#!/bin/bash -l
# ============================================================================
# SBATCH script: ML-based LAr Veto — Data Processing Pipeline (CPU-only)
# Runs: geometry → extraction → preparation → normalization
# Usage: cd <project-root> && sbatch run_flow.sh
# ============================================================================
#SBATCH -o generated/logs/flow.out.%j
#SBATCH -e generated/logs/flow.err.%j
#SBATCH -D .
#SBATCH -J mlar_flow

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72          # Julia manages parallelism internally via addprocs()
#SBATCH --mem=234GB                 # ~7 workers × ~30GB each + master
#SBATCH --mail-type=none
#SBATCH --time=5:00:00
#
# Resource layout (one ML group = 6 runs in ml_groupings.yaml):
#   Extraction:  6 workers × 11 threads + master ≈ 67 CPUs  (via Distributed.addprocs)
#                heap-hint per worker ≈ 234GB / 7 ≈ 33 GB
#   Other steps: master only, 72 threads = 72 CPUs  (rest idle)

# ── Resolve project root ────────────────────────────────────────────────────
# Under SLURM, BASH_SOURCE points to /var/spool/slurmd/... (node-local copy),
# so use SLURM_SUBMIT_DIR instead. For direct `bash run_flow.sh`, use BASH_SOURCE.
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

export SKIP_PKG_SETUP=1
export JULIA_NUM_PRECOMPILE_TASKS=1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-72}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-72}
export JULIA_NUM_THREADS=72 # Master threads (workers get threads_per_worker from config)

echo "============================================"
echo "Job:        $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Date:       $(date)"
echo "Project:    $PROJECT_DIR"
echo "Data cfg:   $LEGEND_DATA_CONFIG"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "============================================"

cd "$PROJECT_DIR"

# ── Run data processing flow ────────────────────────────────────────────────
# Which processors run is controlled entirely by `processors.<name>.enabled:`
# in config/processing_config.yaml. Run on a CPU node — the GPU-bound steps
# (process_training, process_prediction) only run if you explicitly enabled
# them and you're OK with CPU fallback (training is impractically slow on CPU;
# prediction works but is slower). For real GPU training use run_training.sh.
# Optional 1st positional arg: a single ML group to process (overrides
# `datasets.active_group:` from processing_config.yaml). Lets you parallelise
# across several nodes:
#     sbatch -J mlar_flow_g1 run_flow.sh mlgroup001
#     sbatch -J mlar_flow_g2 run_flow.sh mlgroup002
GROUP_ARG=""
if [[ -n "$1" ]]; then
    GROUP_ARG="--group $1"
    echo "Group override (CLI): $1"
fi
stdbuf -oL -eL julia +1.11 --project="$PROJECT_DIR" \
    --threads=${JULIA_NUM_THREADS:-4} \
    main.jl \
    -c config/processing_config.yaml \
    $GROUP_ARG

echo "============================================"
echo "Finished:   $(date)"
echo "Exit code:  $?"
echo "============================================"
