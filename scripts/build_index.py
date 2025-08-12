"""
Builds an index from trained model + database vectors
"""
"""
Build an Asteria prototype index from a trained model and database embeddings.

Usage:
  python scripts/build_index.py --model asteria_model.pt \
      --db_embeddings db.npy --out index_state.pt
"""
import argparse
import numpy as np
import torch
from asteria.bor import ButterflyRotation
from asteria.ecvh import ECVH
from asteria.lrsq import LRSQ
from asteria.index_cpu import AsteriaIndexCPU

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--db_embeddings", type=str, required=True)
    ap.add_argument("--out", type=str, default="index_state.pt")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch", type=int, default=4096)
    return ap.parse_args()

def main():
    args = parse_args()
    ckpt = torch.load(args.model, map_location=args.device)
    cfg = ckpt["config"]

    bor = ButterflyRotation(cfg["dim"])
    bor.load_state_dict(ckpt["bor"])
    ecvh = ECVH(cfg["dim"], cfg["m_vantages"], cfg["k_raw"], cfg["m_code"])
    ecvh.load_state_dict(ckpt["ecvh"])
    lrsq = LRSQ(cfg["dim"], cfg["rank"], cfg["blocks"])
    lrsq.load_state_dict(ckpt["lrsq"])

    emb = np.load(args.db_embeddings)
    data = torch.tensor(emb, dtype=torch.float32)

    model_bundle = {"bor": bor, "ecvh": ecvh, "lrsq": lrsq}
    index = AsteriaIndexCPU(model_bundle, device=args.device)
    index.add(data, ids=list(range(data.shape[0])), batch=args.batch)

    torch.save({
        "model": ckpt,
        "index_vectors_full": [v.numpy() for v in index.db_vectors_full],
        "index_proj_codes": [c.numpy() for c in index.db_proj_codes],
        "ids": index.ids,
        "bucket_map": {k: v for k, v in index.bucket_map.items()}
    }, args.out)
    print(f"Saved index state to {args.out}")

if __name__ == "__main__":
    main()