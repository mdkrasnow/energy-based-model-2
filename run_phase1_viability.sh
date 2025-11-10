#!/bin/bash
#SBATCH -J phase1_statistical_viability      # Job name
#SBATCH -p gpu_test                          # H100 partition
#SBATCH --account=ydu_lab                    # Your lab account
#SBATCH --gres=gpu:1                         # 1 H100 GPU
#SBATCH -c 16                                # 16 CPU cores
#SBATCH -t 00-12:00:00                       # 12 hours
#SBATCH --mem=64G                            # 64 GB RAM
#SBATCH -o phase1_viability_%j.out           # Output file
#SBATCH -e phase1_viability_%j.err           # Error file
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkrasnow@college.harvard.edu

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"

# Always start from the submit directory (your home in this case)
cd "$SLURM_SUBMIT_DIR" || { echo "Failed to cd to $SLURM_SUBMIT_DIR"; exit 1; }
echo "PWD at job start: $(pwd)"

# Define a canonical results dir in home and ensure it exists
RESULTS_DIR="$SLURM_SUBMIT_DIR/phase1_results"
mkdir -p "$RESULTS_DIR"
echo "Using RESULTS_DIR=$RESULTS_DIR"

sacctmgr list associations user=$USER format=account%20,partition%20
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"

module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01

export PATH="$HOME/.local/bin:$PATH"

echo "Installing dependencies..."
pip install --user -q torch torchvision einops accelerate tqdm \
    tabulate matplotlib numpy pandas ema-pytorch \
    ipdb seaborn scikit-learn
echo "Dependencies installed successfully"

echo "Starting Phase 1 Statistical Viability Testing..."

# IMPORTANT: pass the absolute path as base-dir
python phase1_statistical_viability.py \
  --phase1 \
  --base-dir "$RESULTS_DIR" \
  --dataset addition

echo "Job finished at: $(date)"
