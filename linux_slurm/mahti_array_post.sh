#!/bin/bash -l
# created: Oct 19, 2022 22:22 PM
# author: xuanbinh
#SBATCH --account=project_2004956
#SBATCH --partition=medium
#SBATCH --time=02:00:00
#SBACTH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --hint=multithread
#SBATCH -J CPparameter_test
#SBATCH -e CPparameter_test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=binh.nguyen@aalto.fi

### Since postprocessing does not used DAMASK_spectral, this script cannot make use of MPI. 

### Prevent stack overflow for large models, especially when using openMP
ulimit -s unlimited 

### Enabling environments
source /projappl/project_2004956/damask2/damask_env.txt
source /projappl/project_2004956/damask2/damask-2.0.3/env/DAMASK.sh
PATH=$PATH:/projappl/project_2004956/damask2/damask-2.0.3/bin:/projappl/project_2004956/damask2/damask-2.0.3/processing/post

### Arguments from project code
material=$1
curvetype=$2

fullpath=$(sed -n ${SLURM_ARRAY_TASK_ID}p linux_slurm/array_${curvetype}_file.txt) 

### Change to the current working directory
cd ${fullpath}

### Postprocessing results from the spectralOut file
postResults.py --cr texture,f,p --time ${material}_tensionX.spectralOut

### Change to the current working directory of postProc
cd ${fullpath}/postProc

### Adding additional data from postprocessed results
addCauchy.py ${material}_tensionX.txt
addStrainTensors.py --left --logarithmic ${material}_tensionX.txt
addMises.py -s Cauchy ${material}_tensionX.txt
addMises.py -e 'ln(V)' ${material}_tensionX.txt