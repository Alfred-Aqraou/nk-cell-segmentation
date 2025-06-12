#!/bin/sh
#BSUB -q gpua40
#BSUB -J two_fold_parallel_cv
#BSUB -n 8
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
#BSUB -oo gpuout/unet/%P_%J.out
#BSUB -eo gpuout/unet/%P_%J.err

nvidia-smi
# Load the cuda module
module load cuda/12.8.1

source /work3/s234843/init_project.sh
python unet/scripts/two_fold_parallel_cv.py \
	--workers 8 \
    --epochs 100 \
    --encoders "resnet50" "resnet18" \
    --loss "iou" "dice" "combined" \
    --batchsizes 8 16 \
    --freeze "True" "False" \
    --optimizers "adam" \
    --lr 0.0005 0.001 0.0001 \
    --resize 256 256 \
    --nsplits 3
