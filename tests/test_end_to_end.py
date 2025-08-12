"""
Simple pytest verifying training + index + search
"""
import numpy as np
import torch
from asteria.bor import ButterflyRotation
from asteria.ecvh import ECVH
from asteria.lrsq import LRSQ
from asteria.index_cpu import AsteriaIndexCPU

def test_pipeline():
    dim = 64
    bor = ButterflyRotation(dim)
    ecvh = ECVH(dim, 32, 24, 32)
    lrsq = LRSQ(dim, rank=16, blocks=4)
    bundle = {"bor": bor, "ecvh": ecvh, "lrsq": lrsq}
    index = AsteriaIndexCPU(bundle)

    db = torch.randn(2000, dim)
    db = db / db.norm(dim=1, keepdim=True)
    index.add(db)

    q = torch.randn(10, dim)
    q = q / q.norm(dim=1, keepdim=True)
    D, I = index.search(q, k=5, hamming_radius=1)
    assert I.shape == (10,5)