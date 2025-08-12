"""
Joint training loop (BOR + Vantages + LRSQ projection)
"""
"""
Joint training loop (prototype) for BOR + ECVH + LRSQ projection.

Usage:
  python scripts/train_asteria.py --train_embeddings train.npy --dim 768 \
      --epochs 10 --batch_size 1024 --save_model asteria_model.pt
"""
import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Improved joint training including stronger hashing loss.

Key changes:
- Uses sign_pair_loss instead of raw_margin_loss.
- Uses bit balance on raw logits.
- Adds --code_bits (m_code) and --raw_bits (k_raw) so you can shorten codes
  for small toy datasets (e.g., --raw_bits 32 --code_bits 32).
- Adds structured synthetic augmentation option (--synthetic_clusters) for better signal
  if your training data is random Gaussian.
"""

import argparse
import numpy as np
import torch
import torch.optim as optim
from asteria.bor import ButterflyRotation
from asteria.ecvh import ECVH
from asteria.lrsq import LRSQ
from asteria.utils import mine_pairs

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_embeddings", type=str, required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--m_vantages", type=int, default=128)
    ap.add_argument("--raw_bits", type=int, default=64, help="k_raw")
    ap.add_argument("--code_bits", type=int, default=64, help="m_code")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--save_model", type=str, default="asteria_model.pt")
    ap.add_argument("--steps_per_epoch", type=int, default=400)
    ap.add_argument("--orth_openalty_weight", type=float, default=0.01)
    ap.add_argument("--quant_weight", type=float, default=0.5)
    ap.add_argument("--balance_weight", type=float, default=0.2)
    ap.add_argument("--sign_pair_weight", type=float, default=1.0)
    ap.add_argument("--pos_margin", type=float, default=0.8)
    ap.add_argument("--neg_margin", type=float, default=0.2)
    ap.add_argument("--synthetic_clusters", action="store_true",
                    help="Augment with random cluster shifts if data is random.")
    ap.add_argument("--augment_factor", type=int, default=2)
    ap.add_argument("--anomaly", action="store_true")
    return ap.parse_args()

def maybe_augment_clusters(x, factor=2):
    """
    Creates mild clustered structure:
      - Sample C cluster centers
      - For each sample, add small pull toward one center
    """
    if factor <= 1:
        return x
    B, D = x.shape
    C = max(4, B // 256)
    centers = torch.randn(C, D, device=x.device)
    centers = centers / centers.norm(dim=1, keepdim=True)
    assign = torch.randint(0, C, (B,), device=x.device)
    alpha = 0.3
    x_aug = x + alpha * centers[assign]
    x_aug = x_aug / x_aug.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return x_aug

def main():
    args = parse_args()
    if args.anomaly:
        torch.autograd.set_detect_anomaly(True)

    device = args.device
    emb = np.load(args.train_embeddings)
    assert emb.shape[1] == args.dim
    data = torch.tensor(emb, dtype=torch.float32, device=device)

    bor = ButterflyRotation(args.dim, device=device)
    ecvh = ECVH(args.dim,
                m_vantages=args.m_vantages,
                k_raw=args.raw_bits,
                m_code=args.code_bits,
                device=device)
    lrsq = LRSQ(args.dim, rank=args.rank, blocks=args.blocks, device=device)

    params = list(bor.parameters()) + list(ecvh.parameters()) + list(lrsq.parameters())
    opt = optim.Adam(params, lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        bor.train(); ecvh.train(); lrsq.train()
        loss_tracker = []
        for step in range(1, args.steps_per_epoch + 1):
            batch_idx = torch.randint(0, data.size(0), (args.batch_size,), device=device)
            x = data[batch_idx]
            x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-9)
            if args.synthetic_clusters:
                x = maybe_augment_clusters(x, args.augment_factor)

            x_rot = bor(x)
            code_bits, raw_logits = ecvh(x_rot, return_logits=True)
            proj = lrsq(x_rot)

            pos_mask, neg_mask = mine_pairs(x.detach(), top_k=5)

            # Loss components
            sign_pair = ecvh.sign_pair_loss(raw_logits,
                                            pos_mask,
                                            neg_mask,
                                            pos_margin=args.pos_margin,
                                            neg_margin=args.neg_margin)
            balance = ecvh.bit_balance_loss(raw_logits)
            # Quantization (detach projection for codes; gradient flows into lrsq.P via proj)
            q_codes = lrsq.quantize(proj.detach())
            recon = lrsq.dequantize(q_codes)
            quant = (proj - recon).pow(2).mean()
            ortho = bor.orthogonality_penalty(num_samples=min(128, args.dim))

            total = (args.sign_pair_weight * sign_pair +
                     args.balance_weight * balance +
                     args.quant_weight * quant +
                     args.orth_openalty_weight * ortho)

            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

            if step % 50 == 0:
                lrsq.reorthonormalize()

            loss_tracker.append(float(total))
            if step % 50 == 0:
                print(f"Epoch {epoch} Step {step}/{args.steps_per_epoch} "
                      f"Loss {np.mean(loss_tracker):.4f} "
                      f"SignPair {sign_pair.item():.3f} Bal {balance.item():.3f} "
                      f"Q {quant.item():.3f} Ortho {ortho.item():.4f}")
                loss_tracker = []

        torch.save({
            "bor": bor.state_dict(),
            "ecvh": ecvh.state_dict(),
            "lrsq": lrsq.state_dict(),
            "config": vars(args)
        }, args.save_model)
        print(f"[Epoch {epoch}] Saved -> {args.save_model}")

    print("Training complete.")

if __name__ == "__main__":
    main()