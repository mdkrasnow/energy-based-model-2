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

# Add local bin to PATH for installed scripts
export PATH="$HOME/.local/bin:$PATH"

# Install required packages BEFORE running the script
echo "Installing dependencies..."
pip install --user -q torch torchvision einops accelerate tqdm \
    tabulate matplotlib numpy pandas ema-pytorch \
    ipdb seaborn scikit-learn

echo "Dependencies installed successfully"

# Set experiment directory to scratch storage (avoid NFS issues)
export ENERGY_HNM_EXPERIMENT_DIR="/n/holyscratch01/mkrasnow/experiments_energy_hnm"

# Run the energy HNM hyperparameter sweep
python sweep_energy_hnm_hyperparameters.py

echo "Job finished at: $(date)"