#!/bin/bash

module load cuda/12.8.1
module swap cuda/12.8.1

source /work3/s234843/init_project.sh

python solo-learn/main_pretrain.py \
    --config-path scripts/pretrain/custom/ \
    --config-name dino.yaml