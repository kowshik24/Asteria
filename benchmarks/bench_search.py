"""
Measures recall / QPS vs simple brute-force or FAISS (if available)
"""
"""
Benchmark search speed & (optional) recall vs brute force.

Usage:
  python benchmarks/bench_search.py --model asteria_model.pt --index index_state.pt \
     --queries queries.npy --k 10 --hamming_radius 1 --report recall
"""

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import time
import numpy as np
import torch
from asteria.bor import ButterflyRotation
from asteria.ecvh import ECVH
from asteria.lrsq import LRSQ
from asteria.index_cpu import AsteriaIndexCPU

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--index", type=str, required=True)
    ap.add_argument("--queries", type=str, required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--hamming_radius", type=int, default=1)
    ap.add_argument("--report", type=str, choices=["none","recall"], default="none")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max_eval_queries", type=int, default=1000, help="Max queries for recall evaluation (use -1 for all)")
    return ap.parse_args()

def brute_force_gpu(db, q, k, device='cuda'):
    """GPU-accelerated brute force search"""
    if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return brute_force_cpu(db, q, k)
    
    # Convert to torch tensors on GPU
    db_gpu = torch.tensor(db, dtype=torch.float32, device=device)
    q_gpu = torch.tensor(q, dtype=torch.float32, device=device)
    
    # Adaptive batch size based on GPU memory and data size
    # For Colab T4 (16GB), we can handle larger batches
    if q.shape[0] <= 5000:
        batch_size = min(2000, q.shape[0])  # Larger batch for smaller query sets
    else:
        batch_size = min(1000, q.shape[0])  # Conservative for large query sets
    
    n_queries = q.shape[0]
    top_idx = torch.zeros((n_queries, k), dtype=torch.int64, device='cpu')
    
    print(f"Running GPU brute force with batch size {batch_size}...")
    print(f"GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    for i in range(0, n_queries, batch_size):
        end_i = min(i + batch_size, n_queries)
        batch_q = q_gpu[i:end_i]
        
        # Compute similarities: batch_q @ db_gpu.T
        sims = torch.mm(batch_q, db_gpu.T)
        
        # Get top-k indices
        _, idx = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)
        top_idx[i:end_i] = idx.cpu()
        
        # Clear GPU cache periodically
        if i % (batch_size * 5) == 0:
            torch.cuda.empty_cache()
    
    return top_idx.numpy()

def brute_force_cpu(db, q, k):
    """CPU fallback brute force search"""
    # db, q: numpy arrays normalized
    # Use batched processing to avoid memory issues
    batch_size = 1000
    n_queries = q.shape[0]
    top_idx = np.zeros((n_queries, k), dtype=np.int32)
    
    for i in range(0, n_queries, batch_size):
        end_i = min(i + batch_size, n_queries)
        batch_q = q[i:end_i]
        
        sims = batch_q @ db.T
        idx = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]
        part = np.take_along_axis(sims, idx, axis=1)
        order = np.argsort(-part, axis=1)
        top_idx[i:end_i] = np.take_along_axis(idx, order, axis=1)
    
    return top_idx

def brute_force(db, q, k):
    sims = q @ db.T
    idx = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]
    part = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-part, axis=1)
    top_idx = np.take_along_axis(idx, order, axis=1)
    return top_idx

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--index_embeddings", type=str, required=False,
                    help="If provided rebuild index on-the-fly from embeddings (npy).")
    ap.add_argument("--index_state", type=str, required=False,
                    help="If not rebuilding, provide saved index state.")
    ap.add_argument("--queries", type=str, required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sample_queries", type=int, default=1000,
                    help="For brute force recall eval; speeds up large Q sets.")
    ap.add_argument("--max_radius", type=int, default=2)
    ap.add_argument("--target_mult", type=int, default=8)
    ap.add_argument("--no_fallback", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    return ap.parse_args()

def load_model(model_path, device):
    ck = torch.load(model_path, map_location=device)
    cfg = ck["config"]
    bor = ButterflyRotation(cfg["dim"])
    bor.load_state_dict(ck["bor"])
    ecvh = ECVH(cfg["dim"],
                cfg["m_vantages"],
                cfg["raw_bits"],
                cfg["code_bits"])
    ecvh.load_state_dict(ck["ecvh"])
    lrsq = LRSQ(cfg["dim"], cfg["rank"], cfg["blocks"])
    lrsq.load_state_dict(ck["lrsq"])
    return bor, ecvh, lrsq, cfg

def main():
    args = parse_args()
    device = args.device
    bor, ecvh, lrsq, cfg = load_model(args.model, device)

    bundle = {"bor": bor, "ecvh": ecvh, "lrsq": lrsq}

    if args.index_embeddings:
        emb = np.load(args.index_embeddings)
        db_t = torch.tensor(emb, dtype=torch.float32)
        index = AsteriaIndexCPU(bundle, device=device)
        index.add(db_t)
    elif args.index_state:
        # Loading saved raw vectors is simpler than reconstructing buckets; re-add.
        st = torch.load(args.index_state, map_location="cpu")
        db = np.stack(st["index_vectors_full"], axis=0)
        db_t = torch.tensor(db, dtype=torch.float32)
        index = AsteriaIndexCPU(bundle, device=device)
        index.add(db_t, ids=st["ids"])
    else:
        raise ValueError("Provide either --index_embeddings or --index_state")

    queries = np.load(args.queries)
    q_t = torch.tensor(queries, dtype=torch.float32)

    # Run search
    t0 = time.time()
    D, I = index.search(q_t,
                        k=args.k,
                        target_mult=args.target_mult,
                        max_radius=args.max_radius,
                        brute_force_fallback=not args.no_fallback)
    t1 = time.time()
    print(f"Search time: {t1 - t0:.3f}s  QPS: {queries.shape[0] / (t1 - t0):.2f}")

    # Recall evaluation (sample subset)
    sample = min(args.sample_queries, queries.shape[0])
    sel = np.random.choice(queries.shape[0], sample, replace=False)
    db = np.stack([v.numpy() for v in index.db_vectors_full], axis=0)
    db = db / np.linalg.norm(db, axis=1, keepdims=True)
    q_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    bf_idx = brute_force(db, q_norm[sel], args.k)
    hits = 0
    for i, qi in enumerate(sel):
        truth = set(bf_idx[i].tolist())
        got = set(I[qi][I[qi] >= 0].tolist())
        hits += len(truth & got) / len(truth)
    recall = hits / sample
    print(f"Sampled Recall@{args.k} (n={sample}): {recall:.4f}")

if __name__ == "__main__":
    main()