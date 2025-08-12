"""
Vantage hashing + simple linear ECC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ECVH(nn.Module):
    """
    Enhanced Error-Correcting Vantage Hashing (prototype):

    Changes vs previous version:
    - Stronger pairing loss (sign pair loss) to explicitly align signs for positives
      and anti-align for negatives.
    - Balance loss operates on raw logits (encouraging zero mean) NOT only post-encoded bits.
    - Optional temperature scaling.
    - Optional code length decoupled (m_code can equal k_raw if ECC not yet implemented).
    """

    def __init__(self,
                 dim,
                 m_vantages=128,
                 k_raw=96,
                 m_code=96,
                 temperature=1.0,
                 device=None,
                 dtype=torch.float32):
        super().__init__()
        assert k_raw <= m_vantages
        assert m_code >= k_raw  # if no ECC, they can be equal
        self.dim = dim
        self.m_vantages = m_vantages
        self.k_raw = k_raw
        self.m_code = m_code
        self.temperature = temperature

        self.vantages = nn.Parameter(torch.randn(m_vantages, dim, device=device, dtype=dtype))
        nn.init.orthogonal_(self.vantages)

        # Placeholder (identity / copy) generator if m_code == k_raw
        if m_code == k_raw:
            G = torch.eye(k_raw, m_code, device=device, dtype=torch.int64)
        else:
            # Random binary generator
            G = torch.randint(0, 2, (k_raw, m_code), device=device, dtype=torch.int64)
        self.register_buffer("G", G)

    def forward(self, x_rot: torch.Tensor, return_logits=False):
        """
        x_rot: (B, dim)
        Returns:
          code_bits_float: (B, m_code) ~ {0,1} STE
          raw_logits: (B, k_raw)
        """
        proj = x_rot @ self.vantages.t()  # (B, m_vantages)
        raw_logits = proj[:, :self.k_raw] / self.temperature

        # Straight-through binarization
        raw_sign = self._ste_sign(raw_logits)  # {-1,1}
        raw_bits01 = (raw_sign > 0).float()

        coded = (raw_bits01 @ self.G.float()) % 2.0  # (B, m_code)

        if return_logits:
            return coded, raw_logits
        return coded

    @staticmethod
    def _ste_sign(x):
        y = (x >= 0).float() * 2 - 1
        return y + (x - x.detach())

    def bit_balance_loss(self, raw_logits):
        """
        Encourage raw logits to be zero-mean => sign balanced.
        """
        mean_logits = raw_logits.mean(dim=0)
        return (mean_logits ** 2).mean()

    def sign_pair_loss(self,
                       raw_logits,
                       pos_mask,
                       neg_mask,
                       pos_margin=1.0,
                       neg_margin=0.5,
                       sample_negatives=True,
                       max_pairs=20000):
        """
        Explicit sign agreement / disagreement:

        For x_i, x_j:
           s_i = sign(raw_logits_i), but we approximate with logits.
           Positive pair: encourage (raw_logits_i * raw_logits_j) / k_raw to be >= pos_margin
           Negative pair: encourage (raw_logits_i * raw_logits_j) / k_raw <= -neg_margin

        Use hinge-style losses.

        To keep computation manageable, optionally subsample pairs.

        raw_logits: (B, k_raw)
        pos_mask / neg_mask: (B,B) boolean
        """
        B, K = raw_logits.shape
        # Pairwise dot in logit space
        # (B,K) @ (K,B) -> (B,B)
        sim = raw_logits @ raw_logits.t() / K  # scale

        # Extract positive pairs
        pos_idx = pos_mask.nonzero(as_tuple=False)
        neg_idx = neg_mask.nonzero(as_tuple=False)

        if sample_negatives and neg_idx.size(0) > max_pairs:
            perm = torch.randperm(neg_idx.size(0), device=neg_idx.device)[:max_pairs]
            neg_idx = neg_idx[perm]

        if pos_idx.size(0) > max_pairs:
            perm = torch.randperm(pos_idx.size(0), device=pos_idx.device)[:max_pairs]
            pos_idx = pos_idx[perm]

        pos_vals = sim[pos_idx[:, 0], pos_idx[:, 1]] if pos_idx.numel() > 0 else sim.new_zeros(0)
        neg_vals = sim[neg_idx[:, 0], neg_idx[:, 1]] if neg_idx.numel() > 0 else sim.new_zeros(0)

        pos_loss = F.relu(pos_margin - pos_vals).mean() if pos_vals.numel() else sim.new_tensor(0.0)
        neg_loss = F.relu(neg_vals + neg_margin).mean() if neg_vals.numel() else sim.new_tensor(0.0)
        return pos_loss + neg_loss