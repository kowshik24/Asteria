#!/bin/bash

# Example training and benchmarking for small/medium datasets
# Your data: 60K database vectors, 20K queries, 768D

echo "=== Generating test data (if not already present) ==="
if [ ! -f "train.npy" ] || [ ! -f "db.npy" ] || [ ! -f "queries.npy" ]; then
    echo "Generating synthetic data..."
    python3 -c "
import numpy as np
N = 200000
D = 768
emb = np.random.randn(N, D).astype('float32')
emb /= np.linalg.norm(emb, axis=1, keepdims=True)+1e-9
np.save('train.npy', emb[:120000])
np.save('db.npy', emb[120000:180000])
np.save('queries.npy', emb[180000:])
print('Data generated: train.npy (120K), db.npy (60K), queries.npy (20K)')
"
else
    echo "Data files already exist, skipping generation."
fi

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

# Check if model was trained successfully
if [ ! -f "asteria_model.pt" ]; then
    echo "Error: Failed to create asteria_model.pt. Exiting."
    exit 1
fi

echo "=== Building Index ==="
python scripts/build_index.py \
  --model asteria_model.pt \
  --db_embeddings db.npy \
  --out index_state.pt

# Check if index was created successfully
if [ ! -f "index_state.pt" ]; then
    echo "Error: Failed to create index_state.pt. Exiting."
    exit 1
fi

echo "=== Debug Index Stats ==="
python scripts/debug_index_stats.py \
  --model asteria_model.pt \
  --embeddings db.npy \
  --device cpu

echo "=== Benchmarking with GPU Acceleration ==="
python benchmarks/bench_search.py \
  --model asteria_model.pt \
  --index_state index_state.pt \
  --queries queries.npy \
  --k 10 \
  --max_radius 2 \
  --target_mult 8 \
  --sample_queries 5000 \
  --device cuda

echo "=== Full Recall Evaluation (all queries) ==="
python benchmarks/bench_search.py \
  --model asteria_model.pt \
  --index_state index_state.pt \
  --queries queries.npy \
  --k 10 \
  --max_radius 2 \
  --target_mult 8 \
  --sample_queries 20000 \
  --device cuda

echo "=== Pipeline complete! ==="
