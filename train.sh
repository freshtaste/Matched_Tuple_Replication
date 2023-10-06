#!/bin/bash

#SBATCH --job-name=sim5
#SNATCH --mem=32G
#SBATCH --account=phd
#SBATCH --partition=standard
#SBATCH --output=sim.out
#SBATCH --error=sim.err
#SBATCH --time=3-12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=60
#SBATCH --mail-type=all
#SBATCH --mail-user=jliu32@chicagobooth.edu


echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"

module purge
module load anaconda/2021.05
module load R/3.6/3.6.2
source /apps/Anaconda/anaconda3-2021.05/etc/profile.d/conda.sh
conda activate hanabi

python main.py