#!/usr/bin/env bash
set -e

python newtrain.py \
  --data-path ./data/chestxray14 \
  --adj-file ./graph/Xray_adj.pkl \
  --word-emb ./graph/word14.pkl \
  --mae-ckpt "${MAE_CKPT:-}" \
  --densenet-ckpt "${DENSENET_CKPT:-}" \
  --save-model-path ./outputs/checkpoints/chestxray/ \
  --device-ids ${CUDA_DEVICE_IDS:-0}
