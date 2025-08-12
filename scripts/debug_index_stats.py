"""
Print bucket occupancy statistics and bit balance for a trained model + embeddings.
Usage:
  python scripts/debug_index_stats.py --model asteria_model.pt --embeddings db.npy
"""
import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from collections import Counter
from asteria.bor import ButterflyRotation
from asteria.ecvh import ECVH
from asteria.lrsq import LRSQ

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--device", default="cpu")
    return ap.parse_args()

def pack(bits):
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val

def main():
    args = parse()
    ck = torch.load(args.model, map_location=args.device)
    cfg = ck["config"]
    bor = ButterflyRotation(cfg["dim"])
    bor.load_state_dict(ck["bor"])
    
    # Use the actual config keys saved by training script
    ecvh = ECVH(cfg["dim"],
                cfg["m_vantages"],
                cfg["raw_bits"],  # Use "raw_bits" as saved by training
                cfg["code_bits"]) # Use "code_bits" as saved by training
    ecvh.load_state_dict(ck["ecvh"])
    
    emb = np.load(args.embeddings)
    X = torch.tensor(emb, dtype=torch.float32, device=args.device)
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-9)

    with torch.no_grad():
        rot = bor(X)
        _, raw_logits = ecvh(rot, return_logits=True)
        codes = ecvh(rot)
    
    # Bit balance
    raw_means = raw_logits.mean(0).cpu().numpy()
    print("Raw logit mean (first 10):", raw_means[:10])
    print("Mean abs(logit mean):", np.abs(raw_means).mean())

    # Bucket occupancy
    buckets = Counter()
    for i in range(codes.size(0)):
        bits = codes[i].cpu().numpy().astype(int).tolist()
        buckets[pack(bits)] += 1
    occ = np.array(list(buckets.values()))
    print(f"Buckets used: {len(buckets)}")
    print(f"Mean bucket size: {occ.mean():.2f}")
    print(f"Median bucket size: {np.median(occ):.2f}")
    print(f"Max bucket size: {occ.max()}")
    print(f"Percent empty (relative to 2^code_bits) (approx if large): "
          f"{(1 - len(buckets)/(2**min(cfg['code_bits'],20)))*100:.2f}% (capped view)")

if __name__ == "__main__":
    main()