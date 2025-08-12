"""
Minimal CPU-only index using the three components
"""
import torch
from collections import defaultdict
from .bor import ButterflyRotation
from .ecvh import ECVH
from .lrsq import LRSQ
from .utils import pack_bits
import itertools

def enumerate_hamming(code_bits, radius):
    """
    Efficient generator of bit flips up to given radius.
    code_bits: 1D tensor/list of {0,1}
    """
    m = len(code_bits)
    yield code_bits  # distance 0
    if radius == 0: 
        return
    idxs = list(range(m))
    for r in range(1, radius + 1):
        for comb in itertools.combinations(idxs, r):
            nb = list(code_bits)
            for c in comb:
                nb[c] = 1 - nb[c]
            yield nb

class AsteriaIndexCPU:
    """
    Improved CPU prototype with:
    - Adaptive radius escalation
    - Fallback brute-force if candidate set too small
    """
    def __init__(self, model_bundle: dict, device='cpu'):
        self.device = device
        self.rotation: ButterflyRotation = model_bundle['bor'].to(device).eval()
        self.hash: ECVH = model_bundle['ecvh'].to(device).eval()
        self.lrsq: LRSQ = model_bundle['lrsq'].to(device).eval()
        self.dim = self.rotation.original_dim
        self.db_vectors_full = []   # list of tensors (normalized)
        self.ids = []
        self.bucket_map = defaultdict(list)  # packed_code -> list of global indices

    @torch.no_grad()
    def add(self, vectors: torch.Tensor, ids=None, batch=4096):
        vectors = vectors.to(self.device)
        N = vectors.shape[0]
        if ids is None:
            ids = list(range(len(self.ids), len(self.ids)+N))

        for start in range(0, N, batch):
            chunk = vectors[start:start+batch]
            chunk = chunk / chunk.norm(dim=1, keepdim=True).clamp_min(1e-9)
            rot = self.rotation(chunk)
            codes = self.hash(rot)  # (B, m_code)
            for i in range(chunk.size(0)):
                bits = codes[i].cpu().numpy().astype(int).tolist()
                packed = pack_bits(bits)
                g_idx = len(self.ids)
                self.bucket_map[packed].append(g_idx)
                self.db_vectors_full.append(chunk[i].cpu())
                self.ids.append(int(ids[start + i]))

    def _gather_candidates(self, bits_list, min_candidates, max_radius):
        """
        Escalate radius until candidate count >= min_candidates or radius > max_radius.
        """
        m_code = len(bits_list)
        for radius in range(0, max_radius + 1):
            cands = set()
            for nb in enumerate_hamming(bits_list, radius):
                packed = pack_bits(nb)
                if packed in self.bucket_map:
                    cands.update(self.bucket_map[packed])
            if len(cands) >= min_candidates or radius == max_radius:
                return list(cands), radius
        return [], max_radius

    @torch.no_grad()
    def search(self,
               queries: torch.Tensor,
               k=10,
               target_mult=8,
               max_radius=3,
               brute_force_fallback=True,
               batch=512):
        """
        target_mult: minimum candidate count = k * target_mult before we stop expanding radius.
        max_radius: cap on Hamming expansion.
        brute_force_fallback: if after max_radius expansion < k candidates, fallback to brute-force full DB.

        Returns (scores, ids)
        """
        queries = queries.to(self.device)
        Q = queries.shape[0]
        db_mat = torch.stack(self.db_vectors_full, dim=0).to(self.device)  # (N,d)

        out_scores = torch.full((Q, k), -1.0, device=self.device)
        out_ids = torch.full((Q, k), -1, dtype=torch.long, device=self.device)

        for start in range(0, Q, batch):
            chunk = queries[start:start+batch]
            chunk = chunk / chunk.norm(dim=1, keepdim=True).clamp_min(1e-9)
            rot = self.rotation(chunk)
            codes = self.hash(rot)

            for qi in range(chunk.size(0)):
                bits = codes[qi].cpu().numpy().astype(int).tolist()
                cand_idx, used_radius = self._gather_candidates(
                    bits,
                    min_candidates=k * target_mult,
                    max_radius=max_radius
                )
                if len(cand_idx) < k and brute_force_fallback:
                    # Full brute-force
                    sims = (chunk[qi:qi+1] @ db_mat.t()).squeeze(0)
                    top = torch.topk(sims, k=min(k, sims.numel()))
                    out_scores[start+qi, :top.indices.numel()] = top.values
                    chosen_ids = [self.ids[i] for i in top.indices.tolist()]
                    out_ids[start+qi, :len(chosen_ids)] = torch.tensor(chosen_ids, device=self.device)
                    continue

                if not cand_idx:
                    continue

                cand_t = db_mat[cand_idx]
                sims = (chunk[qi:qi+1] @ cand_t.t()).squeeze(0)
                top = torch.topk(sims, k=min(k, sims.numel()))
                out_scores[start+qi, :top.indices.numel()] = top.values
                chosen_ids = [self.ids[cand_idx[j]] for j in top.indices.tolist()]
                out_ids[start+qi, :len(chosen_ids)] = torch.tensor(chosen_ids, device=self.device)

        return out_scores.cpu().numpy(), out_ids.cpu().numpy()