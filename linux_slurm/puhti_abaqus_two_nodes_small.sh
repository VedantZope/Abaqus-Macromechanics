#!/bin/bash -l
# Author: Xuan Binh
#SBATCH --job-name=abaqus_TwoNodesSmall
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --nodes=2
#SBATCH --ntasks=80
#SBATCH --time=06:00:00
#SBATCH --partition=small
#SBATCH --account=project_2007935
#SBATCH --mail-type=ALL
#SBATCH --mail-user=binh.nguyen@aalto.fi

unset SLURM_GTIDS
module purge
module load abaqus/2022

### Change to the work directory
fullpath=$(sed -n 1p linux_slurm/array_file.txt) 
cd ${fullpath}

# env-file ++++++++++++++++++++++++++++++++++++

if [[ -f abaqus_v6.env ]]; then
    mv abaqus_v6.env abaqus_v6.env_previous
fi

# FOR ABAQUS RUNS ON > 1 NODE, MPI IMPLEMENTATION  IS IMPI !
echo "mp_mpi_implementation = IMPI" >> abaqus_v6.env
echo "mp_rsh_command = 'ssh -n -l %U %H %C'" >> abaqus_v6.env

NODE_LIST=$(scontrol show hostname ${SLURM_NODELIST} | sort -u)

mp_host_list="["
for host in ${NODE_LIST}; do
    mp_host_list="${mp_host_list}['$host', ${SLURM_CPUS_ON_NODE}],"
done

mp_host_list=$(echo ${mp_host_list} | sed -e "s/,$/]/")

echo "mp_host_list=${mp_host_list}"  >> abaqus_v6.env

# env-file ++++++++++++++++++++++++++++++++++++

mkdir tmp_$SLURM_JOB_ID

abq2022 job=geometry cpus=$SLURM_NTASKS scratch=tmp_$SLURM_JOB_ID -verbose standard_parallel=all mp_mode=mpi interactive

# run postprocess.py after the simulation completes
abq2022 cae noGUI=postprocess.py
