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
Joint training loop (prototype) for BOR + ECVH + LRSQ projection.

Added:
  --anomaly flag to help debug autograd
  Optional gradient check utility (run once at start if --grad_check)
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
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
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--m_vantages", type=int, default=160)
    ap.add_argument("--k_raw", type=int, default=96)
    ap.add_argument("--m_code", type=int, default=128)
    ap.add_argument("--rank", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=24)
    ap.add_argument("--save_model", type=str, default="asteria_model.pt")
    ap.add_argument("--steps_per_epoch", type=int, default=500)
    ap.add_argument("--anomaly", action="store_true")
    ap.add_argument("--grad_check", action="store_true")
    return ap.parse_args()

def gradient_smoke_test(bor, dim, device):
    x = torch.randn(8, dim, device=device, requires_grad=True)
    y = bor(x)
    loss = y.pow(2).sum()
    loss.backward()
    print("[Gradient smoke test] Success: BOR backward completed.")

def main():
    args = parse_args()
    if args.anomaly:
        torch.autograd.set_detect_anomaly(True)
        print("Anomaly detection ENABLED (will slow training).")

    device = args.device
    emb = np.load(args.train_embeddings)
    assert emb.shape[1] == args.dim
    data = torch.tensor(emb, dtype=torch.float32, device=device)

    bor = ButterflyRotation(args.dim, device=device)
    ecvh = ECVH(args.dim, args.m_vantages, args.k_raw, args.m_code, device=device)
    lrsq = LRSQ(args.dim, rank=args.rank, blocks=args.blocks, device=device)

    if args.grad_check:
        gradient_smoke_test(bor, args.dim, device)

    params = list(bor.parameters()) + list(ecvh.parameters()) + list(lrsq.parameters())
    opt = optim.Adam(params, lr=args.lr)

    for epoch in range(args.epochs):
        bor.train(); ecvh.train(); lrsq.train()
        losses = []
        for step in range(args.steps_per_epoch):
            batch_idx = torch.randint(0, data.size(0), (args.batch_size,), device=device)
            x = data[batch_idx]
            x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-9)

            x_rot = bor(x)
            code_bits, raw_logits = ecvh(x_rot, return_logits=True)
            proj = lrsq(x_rot)

            pos_mask, neg_mask = mine_pairs(x.detach())

            balance_loss = ecvh.bit_balance_loss(code_bits)
            hash_margin = ecvh.raw_margin_loss(raw_logits, pos_mask, neg_mask, margin=0.6)

            # Quantization round-trip (stop grad through codes)
            q_codes = lrsq.quantize(proj.detach())
            recon = lrsq.dequantize(q_codes)
            quant_loss = (proj - recon).pow(2).mean()

            ortho_pen = bor.orthogonality_penalty(num_samples=min(128, args.dim))

            total = (1.0 * hash_margin +
                     0.2 * balance_loss +
                     0.5 * quant_loss +
                     0.01 * ortho_pen)

            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

            # Re-orthonormalize LRSQ rows periodically
            if step % 50 == 0:
                lrsq.reorthonormalize()

            losses.append(float(total))
            if (step + 1) % 50 == 0:
                print(f"Epoch {epoch+1} Step {step+1}/{args.steps_per_epoch} "
                      f"Loss {np.mean(losses):.4f} "
                      f"HMargin {hash_margin.item():.3f} Bal {balance_loss.item():.3f} "
                      f"Q {quant_loss.item():.3f} Ortho {ortho_pen.item():.4f}")
                losses = []

        torch.save({
            "bor": bor.state_dict(),
            "ecvh": ecvh.state_dict(),
            "lrsq": lrsq.state_dict(),
            "config": vars(args)
        }, args.save_model)
        print(f"[Epoch {epoch+1}] Saved model -> {args.save_model}")

    print("Training complete.")

if __name__ == "__main__":
    main()