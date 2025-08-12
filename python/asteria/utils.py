"""
Utility helpers (pair mining, hamming neighbors)
"""
import torch
import itertools

def mine_pairs(batch, top_k=5):
    """
    Simple nearest-neighbor mining within a batch to create positive/negative masks.
    Returns pos_mask, neg_mask (B,B) boolean.
    """
    with torch.no_grad():
        sim = batch @ batch.t()
        B = batch.size(0)
        # exclude self
        sim.fill_diagonal_(-1.0)
        topk = torch.topk(sim, k=min(top_k, B-1), dim=1).indices
        pos_mask = torch.zeros(B, B, dtype=torch.bool, device=batch.device)
        for i in range(B):
            pos_mask[i, topk[i]] = True
        neg_mask = ~pos_mask
        neg_mask.fill_diagonal_(False)
    return pos_mask, neg_mask

def hamming_neighbors(code_bits, radius, max_bits=None):
    """
    Generate neighbor codes (integer arrays) within given Hamming radius.
    code_bits: 1D tensor/list of {0,1} (length m)
    Returns list of tensors (neighbors) including the original at distance 0.
    WARNING: Exponential if radius large. Only for prototype.
    """
    if max_bits is None:
        max_bits = len(code_bits)
    base = torch.tensor(code_bits, dtype=torch.uint8)
    m = base.shape[0]
    results = [base.clone()]
    for r in range(1, radius+1):
        for idxs in itertools.combinations(range(m), r):
            nb = base.clone()
            nb[list(idxs)] = 1 - nb[list(idxs)]
            results.append(nb)
    return results

def pack_bits(bits):
    """
    Pack 0/1 bits into integer (Python int) for dict keys.
    """
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val