#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J 02466_Project
### -- ask for number of cores (default: 1), 4 cores per gpu, so 16 for 4 gpus --
#BSUB -n 16
### -- Select the resources: 4 gpu in exclusive process mode --
#BSUB -gpu "num=4:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u mathias@eksworld.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpuout/%J.out
#BSUB -e gpuout/%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/12.6

# /appl/cuda/12.4.1/samples/bin/x86_64/linux/release/deviceQuery
source /work3/s234843/bin/miniconda3/bin/activate
conda activate 02466
python /work3/s234843/02466-Project/unet_test.py
