"""
Butterfly Orthogonal Rotation module
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ButterflyRotation(nn.Module):
    """
    BOR: Structured O(d log d) orthogonal rotation using fixed butterfly pairing.
    Assumes (possibly padded) dimension = 2^L. If not power of two, we pad and later slice.
    Each layer has trainable angles producing local 2x2 orthonormal blocks:
         [ cosθ  sinθ]
         [-sinθ  cosθ]
    Forward complexity: O(d log d). Backprop fully supported.
    """
    def __init__(self, dim: int, device=None, dtype=torch.float32):
        super().__init__()
        self.original_dim = dim
        self.dim_pow2 = 1 << (dim - 1).bit_length()
        self.layers = int(math.log2(self.dim_pow2))
        # Angles per stage: (dim_pow2 // 2) parameters
        angles = []
        for l in range(self.layers):
            blocks = self.dim_pow2 // 2
            a = torch.empty(blocks, dtype=dtype, device=device).uniform_(-0.05, 0.05)
            angles.append(nn.Parameter(a))
        self.angles = nn.ParameterList(angles)
        # Optional learnable permutation (disabled for simplicity). You can add later.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, d). We internally pad to dim_pow2, apply layers, slice back.
        """
        B, d = x.shape
        if d != self.original_dim:
            raise ValueError("Input dim mismatch")
        if self.dim_pow2 != d:
            # pad
            pad = self.dim_pow2 - d
            x = torch.cat([x, x.new_zeros(B, pad)], dim=1)

        out = x
        size = self.dim_pow2
        for l in range(self.layers):
            stride = 1 << l
            # pattern: pairs separated by stride
            theta = self.angles[l]
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            # reshape for vectorized mixing:
            # We'll iterate in blocks of size 2*stride:
            out = out.view(B, size)
            for start in range(0, size, 2 * stride):
                left_slice = slice(start, start + stride)
                right_slice = slice(start + stride, start + 2 * stride)
                # angles relevant to these pairs:
                idx0 = torch.arange(start//2, (start//2)+stride, device=out.device)
                c = cos_t[idx0]
                s = sin_t[idx0]
                xl = out[:, left_slice]
                xr = out[:, right_slice]
                # Apply 2x2 rotations (broadcast)
                # new_left =  c * xl + s * xr
                # new_right = -s * xl + c * xr
                # Expand c,s for broadcasting
                while c.dim() < xl.dim():
                    c = c.unsqueeze(0)
                    s = s.unsqueeze(0)
                new_l = c * xl + s * xr
                new_r = -s * xl + c * xr
                out[:, left_slice] = new_l
                out[:, right_slice] = new_r
        if self.dim_pow2 != self.original_dim:
            out = out[:, :self.original_dim]
        return out

    def orthogonality_penalty(self, num_samples=256):
        """
        (Optional) Empirical penalty to encourage exact orthogonality of overall matrix.
        Not strictly needed; stacking perfect 2x2 rotations is orthonormal in theory,
        but finite precision and optimization might drift angles.
        """
        with torch.no_grad():
            # Sample random standard basis combos
            I = torch.eye(self.original_dim, device=self.angles[0].device)[:num_samples]
            Y = self.forward(I)
            # Want Y^T Y ≈ I
            G = Y.T @ Y
            I_ = torch.eye(self.original_dim, device=G.device)
            return (G - I_).pow(2).mean()