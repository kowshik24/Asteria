# Running the Prototype (Training + Index + Benchmarks)

## 1. Environment

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch numpy tqdm faiss-cpu  # faiss optional for baseline
```

(If CUDA, install the appropriate torch wheel.)

## 2. Prepare / Obtain Embeddings

For quick tests, create synthetic:

```python
import numpy as np
N = 200000
D = 768
emb = np.random.randn(N, D).astype('float32')
emb /= np.linalg.norm(emb, axis=1, keepdims=True)+1e-9
np.save("train.npy", emb[:120000])
np.save("db.npy", emb[120000:180000])
np.save("queries.npy", emb[180000:])
```

(Optionally extract real CLIP or DINO embeddings before training.)

## 3. Train Model

```bash
python scripts/train_asteria.py \
  --train_embeddings train.npy \
  --dim 768 \
  --epochs 5 \
  --batch_size 1024 \
  --save_model asteria_model.pt
```

Expect console prints with loss components. Increase epochs for better code balance.

## 4. Build Index

```bash
python scripts/build_index.py \
  --model asteria_model.pt \
  --db_embeddings db.npy \
  --out index_state.pt
```

## 5. Benchmark Search

```bash
python benchmarks/bench_search.py \
  --model asteria_model.pt \
  --index index_state.pt \
  --queries queries.npy \
  --k 10 \
  --hamming_radius 1 \
  --report recall
```

Outputs QPS and (if selected) approximate recall vs an internal brute force.

## 6. Synthetic Speed Smoke Test

```bash
python benchmarks/synthetic_speed.py
```

## 7. Run Tests

```bash
pip install pytest
pytest -q
```

## 8. Next Steps / Improvements

- Implement advanced ECC decoding (syndrome table) to robustly correct bit flips.
- Multi-probe ordering: precompute bit flip priority using margins (projection magnitudes).
- Replace per-query brute-force inside candidate bucket with LRSQ bound pruning:
  - Precompute dequantized low-rank vectors or use int8 dot products.
- Add CAPRG local graph after buckets to refine candidates.
- GPU kernels: fuse BOR + vantage dot products.
- Memory mapping large buckets to support > 100M embeddings.

## 9. Notes

This prototype focuses on clarity and modularity, not peak performance:

- Hamming neighbor enumeration is naive; for large code length keep radius small.
- ECC here is placeholder linear encoding (no decoding cost trade-off).
- Loss function is simplified; add explicit angular margins and code entropy regularization for stronger theoretical claims.
