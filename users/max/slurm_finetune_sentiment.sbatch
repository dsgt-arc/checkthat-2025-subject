#!/bin/bash
#SBATCH -J FineTuneSentimentBERT                # Job name
#SBATCH --account=paceship-dsgt_clef2025        # charge account
#SBATCH -N1 --ntasks-per-node=4                 # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=16G                       # Memory per core/
#SBATCH -t5:30:00                               # Duration
#SBATCH -qembers                                # QOS Name (Jobs finishing in less than an hour -> ember)
#SBATCH --output=Report-%j.log                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=dbang6@gatech.edu           # E-mail address for notifications

set -ue

# Change working directory
PACKAGE=~/p-dsgt_clef2025-0/checkthat-2025-subject
cd $PACKAGE
echo "[$(date)] Set SLURM_SUBMIT_DIR $PACKAGE"

# Activate Conda environment
CONDA_ENV=checkthat-subjectivity                   # Conda Env for Job
module load anaconda3                           # Load module dependencies

#conda env create -f environment.yml
conda activate $CONDA_ENV
echo "[$(date)] Activated Conda environment $CONDA_ENV"

# Run your Python script
conda run -n $CONDA_ENV python subjectivity/02-finetune-encoder.py --config_path config/run_config_sentimentbert.yml