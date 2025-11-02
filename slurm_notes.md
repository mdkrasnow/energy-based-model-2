# FASRC Cluster Cheatsheet

## 🔐 Connect to Cluster
```bash
ssh mkrasnow@login.rc.fas.harvard.edu
# Enter password + OpenAuth token
```

Running:

RUN THIS BEFORE SSHing ONTO THE SERVER: Update file: `scp run_specifics.sh mkrasnow@login.rc.fas.harvard.edu:~/run_specifics.sh`

Submit with `sbatch run_specifics.sh`
Check status: `squeue -u mkrasnow`
Monitor the output file: `tail -f specific_configs_42933839.out`
Check for GPU utilization: `squeue -u $USER`
Get error logs: `cat specific_configs_42933839.err`

## 📁 File Transfer

### Upload to Cluster
```bash
# Single file
scp myfile.py mkrasnow@login.rc.fas.harvard.edu:~/

# Entire directory
scp -r my_folder/ mkrasnow@login.rc.fas.harvard.edu:~/

# To specific path
scp myfile.py mkrasnow@login.rc.fas.harvard.edu:/n/netscratch/ydu_lab/mkrasnow/
```

### Download from Cluster
```bash
# Single file
scp mkrasnow@login.rc.fas.harvard.edu:~/results.csv ./

# Entire directory
scp -r mkrasnow@login.rc.fas.harvard.edu:~/energy-based-model/ ./

# Specific job output
scp mkrasnow@login.rc.fas.harvard.edu:~/smoke_test_*.out ./
```

## 🚀 Job Submission & Monitoring

### Submit Job
```bash
sbatch run_script.sh
# Returns: Submitted batch job 12345678
```

### Check Job Status
```bash
# Your jobs
squeue -u mkrasnow

# Specific job
squeue -j 12345678

# Detailed job info
scontrol show job 42605289
```

### Watch Job Output (Real-time)
```bash
# While running
tail -f smoke_test_12345678.out
tail -f smoke_test_12345678.err

# Ctrl+C to stop watching
```

### Cancel Job
```bash
scancel 12345678              # Cancel specific job
scancel -u mkrasnow           # Cancel ALL your jobs
```

### Check Completed Jobs
```bash
# Recent jobs
sacct

# Specific job with details
sacct -j 12345678 --format=JobID,Elapsed,State,MaxRSS,AllocCPUs,TotalCPU

# Jobs from last week
sacct --starttime=2025-10-14
```

## 🖥️ Check Available Resources

### Your Partitions
```bash
spart                         # List all accessible partitions
```

### GPU Partitions
```bash
scontrol show partition gpu
scontrol show partition kempner_h100
```

### Your Account Info
```bash
sacctmgr list associations user=$USER format=account%20,partition%20
```

## 📦 Modules (Software)

### Load Software
```bash
module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01
```

### Search for Modules
```bash
module avail                  # All available
module avail python           # Search for python
```

### Check Loaded Modules
```bash
module list
```

### Unload Modules
```bash
module unload python
module purge                  # Unload all
```

## 📂 Important Directories

```bash
~                             # Home (100GB, backed up)
/n/netscratch/ydu_lab/mkrasnow # Scratch (fast, 90-day retention)
```

### Navigate & Create Directories
```bash
cd /n/netscratch/ydu_lab/mkrasnow
mkdir my_experiment
ls -lh                        # List files with sizes
du -sh *                      # Check folder sizes
```

## 📝 Quick File Editing on Cluster

```bash
nano myfile.py                # Edit file
# Ctrl+O to save, Ctrl+X to exit

cat myfile.py                 # View file
less myfile.py                # View file (scrollable, q to quit)
head -20 myfile.py            # First 20 lines
tail -50 output.log           # Last 50 lines
```

## 🔧 Common SLURM Script Template

```bash
#!/bin/bash
#SBATCH -J job_name
#SBATCH -p gpu                           # or kempner_requeue
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 2-00:00:00                    # D-HH:MM:SS
#SBATCH --mem=64G
#SBATCH -o output_%j.out
#SBATCH -e error_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkrasnow@college.harvard.edu

module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01

python my_script.py
```

## ⚡ Quick Tips

### Test Before Submitting
```bash
sbatch --test-only run_script.sh         # Validate script
```

### Interactive Session (for debugging)
```bash
salloc -p gpu_test --gres=gpu:1 -c 4 --mem=16G -t 2:00:00
# Get interactive shell on GPU node for 2 hours
```

### Check Disk Usage
```bash
df -h ~                                   # Home directory usage
du -sh ~/energy-based-model              # Folder size
```

### Kill All Your Jobs (Emergency)
```bash
scancel -u mkrasnow
```

## 📊 Your Specific Setup

**Account:** `ydu_lab`  
**Email:** `mkrasnow@college.harvard.edu`  
**Main Partitions:** `gpu`, `gpu_requeue`, `kempner_requeue`

---

**💡 Pro Tip:** Bookmark this! Save it as `cluster_cheatsheet.md` in your home directory:
```bash
nano ~/cluster_cheatsheet.md
# Paste this content, Ctrl+O, Ctrl+X
```