"""
Low-Rank Spherical Quantization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LRSQ(nn.Module):
    """
    Low-Rank Spherical Quantization prototype:
      - Learn P (rank x dim), enforced orthonormal rows via re-orthonormalization.
      - Uniform / learned affine quantization per block of the projected vector.
      - Provides quantize() for database storage and project() for queries.
    """
    def __init__(self, dim, rank=96, blocks=24, device=None, dtype=torch.float32):
        super().__init__()
        assert rank % blocks == 0, "rank must be divisible by blocks"
        self.dim = dim
        self.rank = rank
        self.blocks = blocks
        self.block_size = rank // blocks
        P = torch.randn(rank, dim, device=device, dtype=dtype)
        nn.init.orthogonal_(P)
        self.P = nn.Parameter(P)

        # Scale/zero for each block (learned, start neutral)
        self.scale = nn.Parameter(torch.ones(blocks, device=device, dtype=dtype))
        self.zero = nn.Parameter(torch.zeros(blocks, device=device, dtype=dtype))

    def forward(self, x_rot: torch.Tensor):
        """
        Project to low-rank.
        x_rot: (B,d)
        returns (B, rank)
        """
        return x_rot @ self.P.t()

    def reorthonormalize(self):
        with torch.no_grad():
            # QR-based row-orthonormalization
            Q, _ = torch.linalg.qr(self.P.t())  # Q: (dim, rank)
            self.P.data = Q.t()

    def quantize(self, Y: torch.Tensor):
        """
        Y: (N, rank) projected vectors
        Returns int8 codes per dimension (N, rank) (prototype).
        """
        N = Y.shape[0]
        Yb = Y.view(N, self.blocks, self.block_size)
        codes = []
        for b in range(self.blocks):
            block = Yb[:, b, :]
            sc = self.scale[b]
            zp = self.zero[b]
            # simple affine
            q = (block / sc + zp)
            # clamp to int8
            q = torch.clamp(torch.round(q), -128, 127)
            codes.append(q.to(torch.int8))
        return torch.stack(codes, dim=1).view(N, self.rank)

    def dequantize(self, codes: torch.Tensor):
        """
        codes: (N, rank) int8
        returns float (N, rank)
        """
        N = codes.shape[0]
        codes_b = codes.view(N, self.blocks, self.block_size).float()
        outs = []
        for b in range(self.blocks):
            sc = self.scale[b]
            zp = self.zero[b]
            out_block = (codes_b[:, b, :] - zp) * sc
            outs.append(out_block)
        return torch.stack(outs, dim=1).view(N, self.rank)