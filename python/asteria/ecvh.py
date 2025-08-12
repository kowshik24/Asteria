"""
Vantage hashing + simple linear ECC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ECVH(nn.Module):
    """
    Error-Correcting Vantage Hashing (simplified):
    - Learn m_v vantage directions in rotated space.
    - Produce k raw bits from the first k vantages (or a projection).
    - Encode raw bits via a fixed random linear generator G (k x m_code) mod 2.
    - For training differentiability, we use sign surrogate (tanh) and straight-through.
    NOTE: This is a prototype; advanced ECC decoding & margin-based code balancing
          can be layered later.
    """
    def __init__(self, dim, m_vantages=160, k_raw=96, m_code=128, device=None, dtype=torch.float32):
        super().__init__()
        assert k_raw <= m_vantages
        self.dim = dim
        self.m_vantages = m_vantages
        self.k_raw = k_raw
        self.m_code = m_code

        self.vantages = nn.Parameter(torch.randn(m_vantages, dim, device=device, dtype=dtype))
        nn.init.orthogonal_(self.vantages)

        # Random full-rank binary generator matrix G (k_raw x m_code)
        # We'll store as float for convenience; mod 2 arithmetic done with bit ops if needed later.
        with torch.no_grad():
            G = torch.randint(0, 2, (k_raw, m_code), device=device, dtype=torch.int64)
            # ensure rank (not strict guarantee; simple retry)
            self.register_buffer("G", G)

    def forward(self, x_rot: torch.Tensor, return_logits=False):
        """
        x_rot: (B, dim) already rotated.
        Returns:
          code_bits_float: (B, m_code) in {0,1} (STE) (float)
          logits (optional): (B, k_raw) pre-sign raw bit logits
        """
        B = x_rot.shape[0]
        # Compute vantage projections
        # (B, m_vantages)
        proj = x_rot @ self.vantages.t()
        # Raw bit logits from first k_raw vantages
        raw_logits = proj[:, :self.k_raw]
        # STE sign -> bits
        raw_bits = self._ste_sign(raw_logits)  # in {-1,1}
        raw_bits01 = (raw_bits > 0).float()  # in {0,1}

        # Linear encode: (B, k_raw) * (k_raw, m_code) -> (B, m_code) mod 2
        G = self.G.float()
        coded = raw_bits01 @ G
        coded = torch.remainder(coded, 2.0)
        if return_logits:
            return coded, raw_logits
        return coded

    @staticmethod
    def _ste_sign(x):
        """
        Straight-through sign: forward sign, backward identity (approx).
        """
        y = (x >= 0).float() * 2 - 1
        return y + (x - x.detach())  # pass gradient

    def bit_balance_loss(self, code_bits):
        # Encourage each bit ~0.5 occupancy => mean near 0.5
        mean_bits = code_bits.mean(dim=0)
        return ((mean_bits - 0.5) ** 2).mean()

    def raw_margin_loss(self, raw_logits, pos_mask, neg_mask, margin=1.0):
        """
        Encourage raw logits for positives to be close (bitwise) and
        for negatives to disagree. We approximate by pairwise logistic margin.

        raw_logits: (B, k_raw)
        pos_mask / neg_mask: (B,B) boolean
        """
        with torch.no_grad():
            # Pairwise sign expectation proxy
            pass
        # We'll compute pairwise distances in bit probability space:
        probs = torch.sigmoid(raw_logits)  # (B,k)
        # Pairwise L1 difference between codes:
        diff = torch.cdist(probs, probs, p=1) / probs.shape[1]  # normalized [0,1]
        pos_loss = (diff * pos_mask).sum() / (pos_mask.sum().clamp_min(1))
        neg_margin = F.relu(margin - diff) * neg_mask
        neg_loss = (neg_margin).sum() / (neg_mask.sum().clamp_min(1))
        return pos_loss + neg_loss