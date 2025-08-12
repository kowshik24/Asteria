# Skeleton benchmark script (to be implemented)
import time
import numpy as np
import asteria as ast

def main():
    dim = 768
    n_train = 200_000
    n_db = 5_000_000
    n_q = 10_000

    train = np.random.randn(n_train, dim).astype(np.float32)
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    db = np.random.randn(n_db, dim).astype(np.float32)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    q = np.random.randn(n_q, dim).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    params = ast.IndexParams(dim=dim, m_bits=128, ecc_k=96, bor_layers=log2(dim), lrsq_rank=96, lrsq_blocks=24)
    index = ast.Index(params)
    index.fit(train)
    ids = np.arange(n_db, dtype=np.uint64)
    index.add(db, ids)

    t0 = time.time()
    D, I = index.search(q, k=10, max_probes=8)
    t1 = time.time()
    print("QPS:", n_q / (t1 - t0))

if __name__ == "__main__":
    main()