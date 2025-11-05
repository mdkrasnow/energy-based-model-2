#!/bin/bash
#SBATCH -J anm_hyperparam_sweep          # Job name
#SBATCH -p gpu_test                           # H100 partition
#SBATCH --account=ydu_lab                # Your lab account
#SBATCH --gres=gpu:1                     # 1 H100 GPU
#SBATCH -c 16                            # 16 CPU cores
#SBATCH -t 00-06:00:00                    # 6 hours 
#SBATCH --mem=64G                        # 64 GB RAM
#SBATCH -o sweep_%j.out                  # Output file
#SBATCH -e sweep_%j.err                  # Error file
#SBATCH --mail-type=END,FAIL             # Email when done or failed
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

# Add local bin to PATH for installed scripts
export PATH="$HOME/.local/bin:$PATH"

# Install required packages BEFORE running the script
echo "Installing dependencies..."
pip install --user -q torch torchvision einops accelerate tqdm \
    tabulate matplotlib numpy pandas ema-pytorch \
    ipdb seaborn scikit-learn

echo "Dependencies installed successfully"

# Run the hyperparameter sweep
echo "========================================================================"
echo "Starting ANM Hyperparameter Sweep"
echo "Each config trains for 20,000 steps (~12 minutes)"
echo "========================================================================"

python smoke_test_with_diagnostics.py --sweep --dataset inverse --force

echo "========================================================================"
echo "Sweep complete! Check experiments/hyperparameter_sweep_inverse_20000steps.json"
echo "for detailed results and recommendations."
echo "========================================================================"

echo "Job finished at: $(date)"