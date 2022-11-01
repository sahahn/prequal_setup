#!/bin/sh

#SBATCH --job-name=run_prequal_pipeline
#SBATCH --partition=bluemoon
#SBATCH --time=30:00:00

#SBATCH --mem=24G

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

module load singularity/3.7.1

export LC_ALL=C

