#!/bin/bash
#SBATCH -J analyze_distance_penalty           # Job name
#SBATCH -p gpu_test                           # GPU test partition
#SBATCH --account=ydu_lab                     # Your lab account
#SBATCH --gres=gpu:1                          # 1 GPU
#SBATCH -c 16                                 # 16 CPU cores
#SBATCH -t 00-12:00:00                        # 12 hours
#SBATCH --mem=64G                             # 64 GB RAM
#SBATCH -o analyze_distance_penalty_%j.out    # Slurm stdout file
#SBATCH -e analyze_distance_penalty_%j.err    # Slurm stderr file
#SBATCH --mail-type=END,FAIL                  # Email when done or failed
#SBATCH --mail-user=mkrasnow@college.harvard.edu

# Fail fast, catch unbound vars, and make pipelines fail on the first error.
set -euo pipefail
# Echo commands for easier debugging (goes to Slurm -o).
set -x

# Ensure we see full Python tracebacks and unbuffered output.
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

# Make GPU visibility and torch diagnostics predictable.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Keep BLAS libraries from oversubscribing threads.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# -------------------------------
# Choose a writable scratch base
# -------------------------------
SCRATCH_CANDIDATES=()
# If cluster exposes $HOLYSCRATCH, prefer that.
if [ -n "${HOLYSCRATCH:-}" ]; then
  SCRATCH_CANDIDATES+=("${HOLYSCRATCH}/${USER}")
fi
# Common FAS RC pattern: /n/holyscratch01/<account>/<user>
SCRATCH_CANDIDATES+=("/n/holyscratch01/${SLURM_ACCOUNT:-ydu_lab}/${USER}")
SCRATCH_CANDIDATES+=("/n/holyscratch01/${SLURM_JOB_ACCOUNT:-${SLURM_ACCOUNT:-ydu_lab}}/${USER}")
# Generic node-local or shared scratch fallbacks.
SCRATCH_CANDIDATES+=("/scratch/${USER}")
SCRATCH_CANDIDATES+=("${TMPDIR:-}")
# Last resort: a scratch folder under HOME (NFS; slower but writable)
SCRATCH_CANDIDATES+=("${HOME}/scratch")

SCRATCH_BASE=""
for base in "${SCRATCH_CANDIDATES[@]}"; do
  [ -n "${base}" ] || continue
  if mkdir -p "${base}" 2>/dev/null; then
    if [ -w "${base}" ]; then
      SCRATCH_BASE="${base}"
      break
    fi
  fi
done

if [ -z "${SCRATCH_BASE}" ]; then
  echo "WARNING: No writable scratch found; falling back to \$HOME" >&2
  SCRATCH_BASE="${HOME}"
  mkdir -p "${SCRATCH_BASE}"
fi

JOB_WORKDIR="${SCRATCH_BASE}/analyze_distance_penalty.${SLURM_JOB_ID}"
mkdir -p "${JOB_WORKDIR}"
cd "${JOB_WORKDIR}"

# Create separate logs that capture FULL stdout/stderr in addition to Slurm's -o/-e.
JOB_STDOUT="${JOB_WORKDIR}/run_${SLURM_JOB_ID}.stdout.log"
JOB_STDERR="${JOB_WORKDIR}/run_${SLURM_JOB_ID}.stderr.log"

# Duplicate all subsequent stdout/stderr to our tee'd logs (Slurm still captures -o/-e).
exec > >(tee -a "${JOB_STDOUT}") 2> >(tee -a "${JOB_STDERR}" >&2)

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Scratch base: ${SCRATCH_BASE}"
echo "Workdir: ${JOB_WORKDIR}"
# Association info (account/partition)
sacctmgr list associations user="${USER}" format=account%20,partition%20 || true
# GPU visibility as seen by the process
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# Load necessary modules
module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01

# Add local bin to PATH for installed scripts
export PATH="$HOME/.local/bin:$PATH"

# Print versions to logs for reproducibility
python --version
pip --version
nvidia-smi || true

# Install required packages BEFORE running the script
echo "Installing dependencies..."
pip install --user -q torch torchvision einops accelerate tqdm tabulate matplotlib numpy pandas ema-pytorch ipdb seaborn scikit-learn scipy
echo "Dependencies installed successfully"

# Point the experiment directory to job-local scratch storage
export DISTANCE_PENALTY_EXPERIMENT_DIR="${JOB_WORKDIR}/distance_penalty_experiments"
mkdir -p "${DISTANCE_PENALTY_EXPERIMENT_DIR}"

# Echo environment summary
echo "Environment summary:"
echo "  DISTANCE_PENALTY_EXPERIMENT_DIR=${DISTANCE_PENALTY_EXPERIMENT_DIR}"
echo "  PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER}"
echo "  PYTHONUNBUFFERED=${PYTHONUNBUFFERED}"

# Run the distance penalty analysis with full stderr captured.
# Use the original submission directory for the script path if available.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
echo "Launching ${SCRIPT_DIR}/analyze_distance_penalty_effects.py --run-sweep"

python -u "${SCRIPT_DIR}/analyze_distance_penalty_effects.py" --run-sweep \
  2> >(tee -a "${JOB_WORKDIR}/run_${SLURM_JOB_ID}.combined.stderr.log" >&2) \
  | tee -a "${JOB_WORKDIR}/run_${SLURM_JOB_ID}.combined.stdout.log"

# Record Slurm accounting with ReqTRES (ReqGRES deprecated)
sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName%30,State,ExitCode,MaxRSS,MaxVMSize,Elapsed,ReqTRES,AllocTRES || true

echo "Job finished at: $(date)"
