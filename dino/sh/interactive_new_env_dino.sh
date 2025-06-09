#!/bin/bash

module load cuda/11.7
module swap cuda/11.7

source /work3/s234843/init_new_dinov2.sh

# Run DINO training with distributed setup
PYTHONPATH=. python -m torch.distributed.run \
  --nproc_per_node=2 \
  dinov2-fork/dinov2/run/train/train.py \
  --nodes 1 \
  --config-file dinov2-fork/dinov2/configs/train/vits14.yaml \
  --output-dir exports/dino/vits14 \
  train.dataset_path="ImageNet:split=TRAIN:root=/work3/s234843/02466-Project/dataset/fluorescent/without_masks_224"
