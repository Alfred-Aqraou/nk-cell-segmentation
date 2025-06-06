#!/bin/sh
#BSUB -q gpuv100
#BSUB -J 02466_Project
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
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
#BSUB -oo gpuout/%J.out
#BSUB -eo gpuout/%J.err

nvidia-smi
# Load the cuda module
module load cuda/12.8.1

source /work3/s234843/init_project.sh
python unet_test.py
