"""
Quick synthetic speed smoke test
"""
"""
Quick synthetic benchmark (no real semantics).
"""
import torch
import numpy as np
import time
from asteria.bor import ButterflyRotation
from asteria.ecvh import ECVH
from asteria.lrsq import LRSQ
from asteria.index_cpu import AsteriaIndexCPU

def main():
    dim = 768
    train = torch.randn(20000, dim)
    train = train / train.norm(dim=1, keepdim=True)
    bor = ButterflyRotation(dim)
    ecvh = ECVH(dim, 160, 96, 128)
    lrsq = LRSQ(dim, 96, 24)
    bundle = {"bor": bor, "ecvh": ecvh, "lrsq": lrsq}
    index = AsteriaIndexCPU(bundle)

    print("Adding database...")
    db = torch.randn(100000, dim)
    db = db / db.norm(dim=1, keepdim=True)
    index.add(db)

    q = torch.randn(1000, dim)
    q = q / q.norm(dim=1, keepdim=True)
    t0 = time.time()
    D, I = index.search(q, k=10, hamming_radius=1)
    t1 = time.time()
    print("QPS:", q.shape[0] / (t1 - t0))

if __name__ == "__main__":
    main()