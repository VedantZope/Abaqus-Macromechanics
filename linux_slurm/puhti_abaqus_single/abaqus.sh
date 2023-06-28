#!/bin/bash -l
#SBATCH --job-name=abaqusTest1
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=25
#SBATCH --time=00:15:00
#SBATCH --partition=test
#SBATCH --account=project_2007935
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vedant.zope@aalto.fi

# This script runs in parallel Abaqus example e1 on Puhti server using 10 cores.


unset SLURM_GTIDS

module load abaqus/2022

cd $PWD

# abq2022 job=ndb50.inp fetch

mkdir tmp_$SLURM_JOB_ID

abq2022 job=ndb50 input=ndb50.inp cpus=$SLURM_NTASKS -verbose 2 standard_parallel=all scratch=tmp_$SLURM_JOB_ID interactive

# run postprocess.py after the simulation completes
abq2022 cae noGUI=postprocess.py
