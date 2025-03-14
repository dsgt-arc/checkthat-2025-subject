#!/bin/bash
#SBATCH -J ClassifySentences                  # Job name
#SBATCH --account=paceship-dsgt_clef2025      # Charge account
#SBATCH -N1 --ntasks-per-node=4               # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=16G                     # Memory per core
#SBATCH -t5:30:00                             # Duration (hh:mm:ss)
#SBATCH -qembers                              # QOS Name
#SBATCH --output=Report-%j.log                # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL            # Mail preferences
#SBATCH --mail-user=dbang6@gatech.edu         # E-mail address for notifications

set -ue

# Set the working directory
PACKAGE=~/p-dsgt_clef2025-0/checkthat-2025-subject
cd $PACKAGE
echo "[$(date)] Set SLURM_SUBMIT_DIR $PACKAGE"

# Activate Conda environment
CONDA_ENV=checkthat-subjectivity
module load anaconda3

# Activate the Conda environment
conda activate $CONDA_ENV
echo "[$(date)] Activated Conda environment $CONDA_ENV"

# Run `run-classify-openai.py`
conda run -n $CONDA_ENV python run-classify-openai.py --config_path config/open-ai.yml --force
