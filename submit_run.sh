#!/bin/bash
set -euo pipefail

USER="mkrasnow"
HOST="login.rc.fas.harvard.edu"
REMOTE="${USER}@${HOST}"

# Control socket path for connection sharing
CTRL_PATH="$HOME/.ssh/ctl_%h_%p_%r"

echo "Opening master SSH connection to ${REMOTE}..."
ssh -MNf \
  -o ControlMaster=yes \
  -o ControlPath="${CTRL_PATH}" \
  -o ControlPersist=600 \
  "${REMOTE}"

echo "Running SCP transfers using shared connection..."

# Copy Python scripts
scp -o ControlPath="${CTRL_PATH}" analyze_distance_penalty_effects.py     "${REMOTE}:~/analyze_distance_penalty_effects.py"
scp -o ControlPath="${CTRL_PATH}" statistical_validation_anm.py           "${REMOTE}:~/statistical_validation_anm.py"
scp -o ControlPath="${CTRL_PATH}" sweep_energy_hnm_hyperparameters.py     "${REMOTE}:~/sweep_energy_hnm_hyperparameters.py"
scp -o ControlPath="${CTRL_PATH}" train.py                                "${REMOTE}:~/train.py"

# Copy all run scripts
scp -o ControlPath="${CTRL_PATH}" run_statistical_validation_anm.sh       "${REMOTE}:~/run_statistical_validation_anm.sh"
scp -o ControlPath="${CTRL_PATH}" run_sweep_energy_hnm.sh                 "${REMOTE}:~/run_sweep_energy_hnm.sh"
scp -o ControlPath="${CTRL_PATH}" run_analyze_distance_penalty.sh         "${REMOTE}:~/run_analyze_distance_penalty.sh"
scp -o ControlPath="${CTRL_PATH}" run_diagnostics.sh                      "${REMOTE}:~/run_diagnostics.sh"
scp -o ControlPath="${CTRL_PATH}" run_specifics.sh                        "${REMOTE}:~/run_specifics.sh"
scp -o ControlPath="${CTRL_PATH}" run_hyperparameter_sweep.sh             "${REMOTE}:~/run_hyperparameter_sweep.sh"

# Copy diffusion library
scp -o ControlPath="${CTRL_PATH}" -r diffusion_lib                        "${REMOTE}:~/"

echo "All files copied. Submitting all jobs to SLURM in parallel..."

# Submit all jobs in parallel using sbatch
ssh -o ControlPath="${CTRL_PATH}" "${REMOTE}" "chmod +x run_*.sh && \
  sbatch run_statistical_validation_anm.sh && \
  sbatch run_sweep_energy_hnm.sh && \
  sbatch run_analyze_distance_penalty.sh && \
  echo 'All jobs submitted. Check with: squeue -u mkrasnow'"

echo "Closing master SSH connection..."
ssh -O exit -o ControlPath="${CTRL_PATH}" "${REMOTE}"

echo "All files copied and scripts executed."