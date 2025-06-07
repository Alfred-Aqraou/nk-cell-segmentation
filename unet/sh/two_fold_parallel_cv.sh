#!/bin/sh
#BSUB -q gpua100
#BSUB -J two_fold_parallel_cv
#BSUB -n 16
#BSUB -gpu "num=1:mode=shared"
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"

#BSUB -u mathias@eksworld.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo gpuout/unet/two_fold_parallel_cv-%J.out
#BSUB -eo gpuout/unet/two_fold_parallel_cv-%J.err

nvidia-smi
# Load the cuda module
module load cuda/12.8.1

source /work3/s234843/init_project.sh
python unet/scripts/two_fold_parallel_cv.py -w 16
