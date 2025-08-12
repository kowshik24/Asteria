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
    
    # Compute similarities in batches to manage GPU memory
    batch_size = min(1000, q.shape[0])  # Adjust based on GPU memory
    n_queries = q.shape[0]
    top_idx = torch.zeros((n_queries, k), dtype=torch.int64, device='cpu')
    
    print(f"Running GPU brute force with batch size {batch_size}...")
    
    for i in range(0, n_queries, batch_size):
        end_i = min(i + batch_size, n_queries)
        batch_q = q_gpu[i:end_i]
        
        # Compute similarities: batch_q @ db_gpu.T
        sims = torch.mm(batch_q, db_gpu.T)
        
        # Get top-k indices
        _, idx = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)
        top_idx[i:end_i] = idx.cpu()
    
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

def main():
    args = parse_args()
    print(f"Loading model from {args.model}...")
    ckpt = torch.load(args.model, map_location=args.device)
    
    cfg = ckpt["config"]
    bor = ButterflyRotation(cfg["dim"])
    bor.load_state_dict(ckpt["bor"])
    ecvh = ECVH(cfg["dim"], cfg["m_vantages"], cfg["k_raw"], cfg["m_code"])
    ecvh.load_state_dict(ckpt["ecvh"])
    lrsq = LRSQ(cfg["dim"], cfg["rank"], cfg["blocks"])
    lrsq.load_state_dict(ckpt["lrsq"])

    print(f"Loading index from {args.index}...")
    idx_state = torch.load(args.index, map_location="cpu", weights_only=False)
    bundle = {"bor": bor, "ecvh": ecvh, "lrsq": lrsq}
    index = AsteriaIndexCPU(bundle, device=args.device)
    
    # restore data
    index.db_vectors_full = [torch.tensor(v) for v in idx_state["index_vectors_full"]]
    index.db_proj_codes = [torch.tensor(c) for c in idx_state["index_proj_codes"]]
    index.ids = idx_state["ids"]
    index.bucket_map = idx_state["bucket_map"]
    
    print(f"Database size: {len(index.db_vectors_full)} vectors")

    print(f"Loading queries from {args.queries}...")
    queries = np.load(args.queries)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True).clip(1e-9)
    q_t = torch.tensor(queries, dtype=torch.float32)
    print(f"Query set size: {queries.shape[0]} vectors")

    print("Running Asteria search...")
    t0 = time.time()
    D, I = index.search(q_t, k=args.k, hamming_radius=args.hamming_radius)
    t1 = time.time()
    qps = queries.shape[0] / (t1 - t0)
    print(f"Search time: {t1 - t0:.3f}s  QPS: {qps:.2f}")

    if args.report == "recall":
        print("Computing brute force baseline (this may take a while)...")
        
        # Determine how many queries to evaluate
        if args.max_eval_queries == -1:
            eval_queries = queries
            print(f"Evaluating recall on all {len(queries)} queries")
        else:
            eval_queries = queries[:args.max_eval_queries]
            print(f"Evaluating recall on {len(eval_queries)} queries")
            
        db = np.stack([v.numpy() for v in index.db_vectors_full], axis=0)
        db = db / np.linalg.norm(db, axis=1, keepdims=True).clip(1e-9)
        
        # Use GPU if available for much faster brute force computation
        use_gpu = torch.cuda.is_available()
        device = 'cuda' if use_gpu else 'cpu'
        print(f"Using {'GPU' if use_gpu else 'CPU'} for brute force computation")
        
        t2 = time.time()
        if use_gpu:
            bf = brute_force_gpu(db, eval_queries, args.k, device)
        else:
            bf = brute_force_cpu(db, eval_queries, args.k)
        t3 = time.time()
        print(f"Brute force time for {eval_queries.shape[0]} queries: {t3-t2:.3f}s")
        
        # Recall@k on subset
        hits = 0
        for i in range(eval_queries.shape[0]):
            truth = set(bf[i].tolist())
            got = set(I[i][I[i] >= 0].tolist())
            hits += len(truth & got) / len(truth)
        recall = hits / eval_queries.shape[0]
        print(f"Recall@{args.k} (on {eval_queries.shape[0]} queries): {recall:.4f}")

if __name__ == "__main__":
    main()