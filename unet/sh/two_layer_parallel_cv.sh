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
#BSUB -oo gpuout/unet/parallel_%J.out
#BSUB -eo gpuout/unet/parallel_%J.err

# /work3/s234843/02466-Project/trained_models/old/dino/70queb8d/dino-70queb8d-ep=399.ckpt
# /work3/s234843/02466-Project/trained_models/dino/fdnzv5c2/dino-fdnzv5c2-ep=399.ckpt

nvidia-smi
# Load the cuda module
module load cuda/12.8.1

source /work3/s234843/init_project.sh
python unet/scripts/two_layer_parallel_cv.py \
	--workers 8 \
    --epochs 100 \
    --encoders "resnet18" "resnet50" \
    --loss "iou" "dice" "combined" \
    --batchsizes 8 16 \
    --freeze "False" \
    --optimizers "adam" \
    --lr 0.001 0.0001 0.0005 \
    --resize 256 \
    --nsplits 3
