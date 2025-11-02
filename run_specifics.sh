#!/bin/bash
#SBATCH -J evaluate_specific_configs        # Job name
#SBATCH -p gpu_test                         # H100 partition
#SBATCH --account=ydu_lab                   # Your lab account
#SBATCH --gres=gpu:1                        # 1 H100 GPU
#SBATCH -c 16                               # 16 CPU cores
#SBATCH -t 00-12:00:00                      # 12 hours (50k iterations takes longer)
#SBATCH --mem=64G                           # 64 GB RAM
#SBATCH -o specific_configs_%j.out          # Output file
#SBATCH -e specific_configs_%j.err          # Error file
#SBATCH --mail-type=END,FAIL                # Email when done or failed
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

# Run the script (remove --sweep flag, keep --force if you want to retrain)
python evaluate_specific_configs.py --force

echo "Job finished at: $(date)"