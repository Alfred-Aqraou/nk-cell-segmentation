#!/bin/bash
#BSUB -J train_dino_parallel
#BSUB -q gpua40
#BSUB -n 4
#BSUB -gpu "num=1:mode=shared"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -oo gpuout/dino/train_dino_parallel_%J.out
#BSUB -eo gpuout/dino/train_dino_parallel_%J.err

module load cuda/12.6

# Activate conda environment
source /work3/s234843/init_dinov2.sh

# Run DINO training with distributed setup
PYTHONPATH=. python -m torch.distributed.run \
  --nproc_per_node=4 \
  dinov2-fork/dinov2/run/train/train.py \
  --nodes 1 \
  --config-file dinov2-fork/dinov2/configs/train/vits14.yaml \
  --output-dir exports/dino/vits14 \
  train.dataset_path="ImageNet:split=TRAIN:root=/work3/s234843/02466-Project/dataset/fluorescent/without_masks_224"
