#!/bin/bash
#SBATCH -J sweep_energy_hnm                  # Job name
#SBATCH -p gpu_test                          # GPU test partition
#SBATCH --account=ydu_lab                    # Your lab account
#SBATCH --gres=gpu:1                         # 1 GPU
#SBATCH -c 16                                # 16 CPU cores
#SBATCH -t 00-12:00:00                       # 12 hours
#SBATCH --mem=64G                            # 64 GB RAM
#SBATCH -o sweep_energy_hnm_%j.out           # Output file
#SBATCH -e sweep_energy_hnm_%j.err           # Error file
#SBATCH --mail-type=END,FAIL                 # Email when done or failed
#SBATCH --mail-user=mkrasnow@college.harvard.edu

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
sacctmgr list associations user=$USER format=account%20,partition%20
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"

# Load necessary modules
module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01

# Create a per-job scratch directory and move into it
# Try multiple scratch directory options in order of preference
SCRATCH_OPTIONS=(
    "/n/holyscratch01/$USER"
    "/scratch/$USER" 
    "/tmp"
)

echo "Attempting to find suitable scratch directory..."
SCRATCH_BASE=""
for scratch_path in "${SCRATCH_OPTIONS[@]}"; do
    echo "  Trying: $scratch_path"
    if [ -d "$scratch_path" ] && [ -w "$scratch_path" ]; then
        SCRATCH_BASE="$scratch_path"
        echo "  ✓ Found writable scratch at: $SCRATCH_BASE"
        break
    elif mkdir -p "$scratch_path" 2>/dev/null; then
        SCRATCH_BASE="$scratch_path"
        echo "  ✓ Created scratch directory at: $SCRATCH_BASE"
        break
    else
        echo "  ✗ Cannot use: $scratch_path"
    fi
done

if [ -z "$SCRATCH_BASE" ]; then
    echo "ERROR: Could not find or create any scratch directory!"
    echo "Tried: ${SCRATCH_OPTIONS[*]}"
    exit 1
fi

# Create job-specific subdirectory
export SCRATCH_JOB_DIR="$SCRATCH_BASE/sweep_energy_hnm_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_JOB_DIR"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create job directory: $SCRATCH_JOB_DIR"
    exit 1
fi

cd "$SCRATCH_JOB_DIR"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to change to job directory: $SCRATCH_JOB_DIR"
    exit 1
fi

echo "Working directory: $(pwd)"

# Add local bin to PATH for installed scripts
export PATH="$HOME/.local/bin:$PATH"

# Install required packages BEFORE running the script
echo "Installing dependencies..."
pip install --user -q torch torchvision einops accelerate tqdm \
    tabulate matplotlib numpy pandas ema-pytorch \
    ipdb seaborn scikit-learn

echo "Dependencies installed successfully"

# Set experiment directory to scratch storage (avoid NFS issues)
# Use the same scratch base that we validated above
export ENERGY_HNM_EXPERIMENT_DIR="$SCRATCH_BASE/experiments_energy_hnm"
echo "Experiment directory: $ENERGY_HNM_EXPERIMENT_DIR"

# Ensure experiment directory exists
mkdir -p "$ENERGY_HNM_EXPERIMENT_DIR"
if [ $? -ne 0 ]; then
    echo "WARNING: Could not create experiment directory: $ENERGY_HNM_EXPERIMENT_DIR"
    echo "Python script will attempt to create it with robust error handling"
fi

# Ensure torch._dynamo uses a valid debug directory
export TORCH_COMPILE_DEBUG_DIR="$SCRATCH_JOB_DIR"

# Run the energy HNM hyperparameter sweep from scratch
python /n/home03/mkrasnow/sweep_energy_hnm_hyperparameters.py

echo "Job finished at: $(date)"