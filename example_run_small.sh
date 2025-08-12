#!/bin/bash

# Example training and benchmarking for small/medium datasets
# Your data: 60K database vectors, 20K queries, 768D

echo "=== Training Asteria Model ==="
python scripts/train_asteria.py \
  --train_embeddings train.npy \
  --dim 768 \
  --epochs 5 \
  --batch_size 1024 \
  --raw_bits 32 \
  --code_bits 32 \
  --m_vantages 48 \
  --rank 48 \
  --blocks 12 \
  --steps_per_epoch 300 \
  --save_model asteria_model.pt

echo "=== Building Index ==="
python scripts/build_index.py \
  --model asteria_model.pt \
  --db_embeddings db.npy \
  --out index_state.pt

echo "=== Debug Index Stats ==="
python scripts/debug_index_stats.py \
  --model asteria_model.pt \
  --embeddings db.npy \
  --device cuda

echo "=== Benchmarking with GPU Acceleration ==="
python benchmarks/bench_search.py \
  --model asteria_model.pt \
  --index index_state.pt \
  --queries queries.npy \
  --k 10 \
  --hamming_radius 2 \
  --report recall \
  --max_eval_queries 5000 \
  --device cuda

echo "=== Full Recall Evaluation (all queries) ==="
python benchmarks/bench_search.py \
  --model asteria_model.pt \
  --index index_state.pt \
  --queries queries.npy \
  --k 10 \
  --hamming_radius 2 \
  --report recall \
  --max_eval_queries -1 \
  --device cuda
