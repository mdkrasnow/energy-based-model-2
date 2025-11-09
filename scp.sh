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

scp -o ControlPath="${CTRL_PATH}" analyze_distance_penalty_effects.py     "${REMOTE}:~/analyze_distance_penalty_effects.py"
scp -o ControlPath="${CTRL_PATH}" statistical_validation_anm.py           "${REMOTE}:~/statistical_validation_anm.py"
scp -o ControlPath="${CTRL_PATH}" run_statistical_validation_anm.sh       "${REMOTE}:~/run_statistical_validation_anm.sh"
scp -o ControlPath="${CTRL_PATH}" run_sweep_energy_hnm.sh                 "${REMOTE}:~/run_sweep_energy_hnm.sh"
scp -o ControlPath="${CTRL_PATH}" sweep_energy_hnm_hyperparameters.py     "${REMOTE}:~/sweep_energy_hnm_hyperparameters.py"
scp -o ControlPath="${CTRL_PATH}" run_analyze_distance_penalty.sh         "${REMOTE}:~/run_analyze_distance_penalty.sh"

echo "Closing master SSH connection..."
ssh -O exit -o ControlPath="${CTRL_PATH}" "${REMOTE}"

echo "All files copied."
