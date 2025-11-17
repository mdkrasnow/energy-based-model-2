#!/bin/bash
#SBATCH -J phase1_statistical_viability      # Job name
#SBATCH -p gpu_test                          # Partition (use gpu for real runs)
#SBATCH --account=ydu_lab                    # Your lab account
#SBATCH --gres=gpu:1                         # 1 GPU
#SBATCH -c 16                                # 16 CPU cores
#SBATCH -t 00-01:00:00                       # 1 hours
#SBATCH --mem=64G                            # 64 GB RAM
#SBATCH -o phase1_viability_%j.out           # STDOUT file
#SBATCH -e phase1_viability_%j.err           # STDERR file
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkrasnow@college.harvard.edu

echo "=============================================="
echo "  Phase 1 Statistical Viability Job Started"
echo "=============================================="
echo "Date:          $(date)"
echo "Node:          $(hostname)"
echo "Job ID:        $SLURM_JOB_ID"
echo "Submit Dir:    $SLURM_SUBMIT_DIR"
echo "SCRATCH:       $SCRATCH"
echo "=============================================="

# ------------------------------------------------------------------------------
# 1. Configure correct FASRC Scratch path
# ------------------------------------------------------------------------------

LAB_NAME="ydu_lab"                              # MUST match your lab account
LAB_SCRATCH_ROOT="$SCRATCH/${LAB_NAME}/Lab/$USER"
JOB_SCRATCH="${LAB_SCRATCH_ROOT}/phase1_${SLURM_JOB_ID}"

echo "Lab scratch root: $LAB_SCRATCH_ROOT"
echo "Job scratch dir : $JOB_SCRATCH"

# Create your personal scratch root if missing
mkdir -p "$LAB_SCRATCH_ROOT" || {
    echo "ERROR: Cannot create $LAB_SCRATCH_ROOT"
    exit 1
}

# Create a per-job scratch workspace
mkdir -p "$JOB_SCRATCH" || {
    echo "ERROR: Cannot create $JOB_SCRATCH"
    exit 1
}

cd "$JOB_SCRATCH" || {
    echo "ERROR: cd to JOB_SCRATCH failed"
    exit 1
}

echo "Now working in scratch: $(pwd)"

# ------------------------------------------------------------------------------
# 2. Copy Python driver script from home → scratch
# ------------------------------------------------------------------------------

# Must exist in your home directory:
/bin/cp "$SLURM_SUBMIT_DIR/phase1_statistical_viability.py" "$JOB_SCRATCH"/

# ------------------------------------------------------------------------------
# 3. Modules & Python environment
# ------------------------------------------------------------------------------

module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01

export PATH="$HOME/.local/bin:$PATH"

echo "Installing dependencies to ~/.local ..."
pip install --user -q torch torchvision einops accelerate tqdm \
    tabulate matplotlib numpy pandas ema-pytorch \
    ipdb seaborn scikit-learn
echo "Dependencies installed successfully."

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ------------------------------------------------------------------------------
# 4. Prepare results directory on scratch
# ------------------------------------------------------------------------------

RESULTS_DIR="$JOB_SCRATCH/phase1_results"
mkdir -p "$RESULTS_DIR"

echo "Results will be written to: $RESULTS_DIR"
echo "Starting Python job…"

# ------------------------------------------------------------------------------
# 5. Run Phase 1 Statistical Viability Test
# ------------------------------------------------------------------------------

python phase1_statistical_viability.py \
    --phase1 \
    --base-dir "$RESULTS_DIR" \
    --dataset addition

JOB_EXIT=$?
echo "Python finished with exit code: $JOB_EXIT"

# ------------------------------------------------------------------------------
# 6. Copy results back to home directory for safekeeping
# ------------------------------------------------------------------------------

FINAL_RESULTS_DIR="$SLURM_SUBMIT_DIR/phase1_results_${SLURM_JOB_ID}"
mkdir -p "$FINAL_RESULTS_DIR"

echo "Syncing results back to: $FINAL_RESULTS_DIR"
rsync -a "$RESULTS_DIR/" "$FINAL_RESULTS_DIR/"

echo "=============================================="
echo "  Job Finished at: $(date)"
echo "  Final Exit Code: $JOB_EXIT"
echo "=============================================="

exit $JOB_EXIT
