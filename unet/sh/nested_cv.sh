#!/bin/sh
#BSUB -q gpua100
#BSUB -J two_layer
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
#BSUB -oo gpuout/unet/%J.out
#BSUB -eo gpuout/unet/%J.err

# /work3/s234843/02466-Project/trained_models/old/dino/70queb8d/dino-70queb8d-ep=399.ckpt
# /work3/s234843/02466-Project/trained_models/dino/fdnzv5c2/dino-fdnzv5c2-ep=399.ckpt
# /work3/s234843/02466-Project/trained_models/dino/55ikrsci/DINO_W 512-55ikrsci-ep=399.ckpt
# /work3/s234843/02466-Project/trained_models/dino/1z5gwu1c/DINO_BW 512-1z5gwu1c-ep=399.ckpt


nvidia-smi
# Load the cuda module
module load cuda/12.8.1

source /work3/s234843/init_project.sh
python unet/scripts/nested_CV.py \
    --workers 8 \
    --epochs 100 \
    --encoders "/work3/s234843/02466-Project/trained_models/dino/1z5gwu1c/DINO_BW 512-1z5gwu1c-ep=399.ckpt" \
    --loss "iou" "combined" \
    --batchsizes 4 8 \
    --freeze "True" "False" \
    --optimizers "adam" \
    --lr 0.001 0.0005 \
    --resize 512 \
    --outer 3 \
    --inner 3
