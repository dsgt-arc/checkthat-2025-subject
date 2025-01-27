#!/bin/bash
#SBATCH -J ClaimEmbeddingClustering             # Job name
#SBATCH --account=paceship-dsgt_clef2025        # charge account
#SBATCH -N1 --ntasks-per-node=4                 # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=4G                        # Memory per core
#SBATCH -t1:00:00                               # Duration 1h
#SBATCH -qembers                                # QOS Name (Jobs finishing in less than an hour -> ember)
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=mheil7@gatech.edu           # E-mail address for notifications
set -ue

# Change working directory
PACKAGE=~/p-dsgt_clef2025-0/checkthat-2025-subject
cd $PACKAGE 
echo "[$(date)] Set SLURM_SUBMIT_DIR $PACKAGE" >> Report-$SLURM_JOB_ID.out

# Create and/or activate Conda environment
CONDA_ENV=checkthat-subjectivity                # Conda Env for Job
module load anaconda3                           # Load module dependencies
#conda env create -f environment.yml >> Report-$SLURM_JOB_ID.out 2>&1  # Redirect stdout and stderr to output file
conda activate $CONDA_ENV >> Report-$SLURM_JOB_ID.out 2>&1             # Redirect stdout and stderr to output file
echo "[$(date)] Activated Conda environment $CONDA_ENV" >> Report-$SLURM_JOB_ID.out

# Run your Python script
conda run -n $CONDA_ENV python users/max/00-generate.py
