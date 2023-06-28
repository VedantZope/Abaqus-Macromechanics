#!/bin/bash -l
# Author: Xuan Binh
#SBATCH --job-name=abaqusArray
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --partition=test
#SBATCH --account=project_2007935
#SBATCH --mail-type=ALL
#SBATCH --mail-user=binh.nguyen@aalto.fi

unset SLURM_GTIDS
module purge
module load abaqus/2022	

### Change to the work directory
fullpath=$(sed -n 1p linux_slurm/array_file.txt) 
cd ${fullpath}

CPUS_TOTAL=$(( $SLURM_NTASKS*$SLURM_CPUS_PER_TASK ))

mkdir tmp_$SLURM_JOB_ID

abq2022 job=geometry.inp input=geometry.inp cpus=$CPUS_TOTAL -verbose 2 standard_parallel=all scratch=tmp_$SLURM_JOB_ID interactive

# run postprocess.py after the simulation completes
abq2022 cae noGUI=postprocess.py