"""
Butterfly Orthogonal Rotation module
"""
import math
import torch
import torch.nn as nn

class ButterflyRotation(nn.Module):
    """
    Gradient-safe Butterfly Orthogonal Rotation (BOR).

    Builds log2(next_power_of_two(dim)) layers of disjoint 2x2 rotations.
    Each layer is defined by a list of (left_idx, right_idx) pairs stored as two
    index tensors. Per-pair we have a single angle θ -> 2x2 block:
        [ cosθ  sinθ]
        [-sinθ  cosθ]

    Forward (training):
      - Pads input to power-of-two (if needed)
      - Applies each layer out-of-place (allocates a new tensor per layer)
      - Slices back to original dimension
    Forward (inference fast path):
      - If torch.is_grad_enabled() is False, we reuse a preallocated buffer.

    This avoids unsafe in-place modifications on views that cause autograd
    version counter errors.
    """

    def __init__(self, dim: int, device=None, dtype=torch.float32):
        super().__init__()
        self.original_dim = dim
        self.dim_pow2 = 1 << (dim - 1).bit_length()
        self.layers = int(math.log2(self.dim_pow2))
        self.device = device
        self.dtype = dtype

        # For each layer l, build disjoint index pairs.
        left_indices = []
        right_indices = []
        angles = []

        for l in range(self.layers):
            stride = 1 << l
            pairs_left = []
            pairs_right = []
            # Partition dimension into blocks of size 2*stride
            for start in range(0, self.dim_pow2, 2 * stride):
                for offset in range(stride):
                    left = start + offset
                    right = start + stride + offset
                    pairs_left.append(left)
                    pairs_right.append(right)
            left_idx_t = torch.tensor(pairs_left, device=device, dtype=torch.long)
            right_idx_t = torch.tensor(pairs_right, device=device, dtype=torch.long)
            left_indices.append(left_idx_t)
            right_indices.append(right_idx_t)

            # Angle per pair
            a = torch.empty(len(pairs_left), device=device, dtype=dtype).uniform_(-0.05, 0.05)
            angles.append(nn.Parameter(a))

        self.left_indices = nn.ParameterList(
            [nn.Parameter(t, requires_grad=False) for t in left_indices]
        )
        self.right_indices = nn.ParameterList(
            [nn.Parameter(t, requires_grad=False) for t in right_indices]
        )
        self.angles = nn.ParameterList(angles)

        # Buffer for optional inference fast path (allocated lazily)
        self.register_buffer("_tmp_infer", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, original_dim)
        returns: (B, original_dim) rotated
        """
        if x.dim() != 2 or x.size(1) != self.original_dim:
            raise ValueError("Input shape mismatch")

        needs_pad = self.dim_pow2 != self.original_dim
        if needs_pad:
            pad_cols = self.dim_pow2 - self.original_dim
            x_pad = torch.cat([x, x.new_zeros(x.size(0), pad_cols)], dim=1)
        else:
            x_pad = x

        y = x_pad
        grad_enabled = torch.is_grad_enabled()

        # Fast path: reuse a temporary buffer only if no grad.
        if not grad_enabled:
            if self._tmp_infer is None or self._tmp_infer.shape != y.shape:
                self._tmp_infer = torch.empty_like(y)

        for l in range(self.layers):
            left_idx = self.left_indices[l]
            right_idx = self.right_indices[l]
            theta = self.angles[l]
            c = torch.cos(theta)
            s = torch.sin(theta)

            # Gather
            xl = y.index_select(1, left_idx)       # (B, num_pairs)
            xr = y.index_select(1, right_idx)      # (B, num_pairs)

            # Broadcast angles (1, num_pairs)
            while c.dim() < xl.dim():
                c = c.unsqueeze(0)
                s = s.unsqueeze(0)

            new_left = xl * c + xr * s
            new_right = -xl * s + xr * c

            if grad_enabled:
                # Allocate new tensor each layer for safety in autograd
                new_y = torch.empty_like(y)
            else:
                # Reuse buffer
                new_y = self._tmp_infer

            # Scatter results
            new_y.index_copy_(1, left_idx, new_left)
            new_y.index_copy_(1, right_idx, new_right)

            y = new_y

        if needs_pad:
            y = y[:, :self.original_dim]
        return y

    @torch.no_grad()
    def orthogonality_penalty(self, num_samples=256):
        """
        Empirical orthogonality check. Can be weighted very lightly.
        """
        num = min(num_samples, self.original_dim)
        I = torch.eye(self.original_dim, device=self.angles[0].device)[:num]
        Y = self.forward(I)
        G = Y.T @ Y
        I_ = torch.eye(self.original_dim, device=G.device)
        return (G - I_).pow(2).mean()

    def extra_repr(self):
        return f"orig_dim={self.original_dim}, dim_pow2={self.dim_pow2}, layers={self.layers}"