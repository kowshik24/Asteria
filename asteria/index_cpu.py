"""
Minimal CPU-only index using the three components
"""
import torch
from collections import defaultdict
from .bor import ButterflyRotation
from .ecvh import ECVH
from .lrsq import LRSQ
from .utils import hamming_neighbors, pack_bits

class AsteriaIndexCPU:
    """
    Minimal CPU-only prototype:
      - Stores original full vectors (optional) + low-rank quantized.
      - Buckets by encoded ECVH code (packed integer).
      - Brute-force inside buckets + Hamming expansion.
    Optimizations (SIMD, CAPRG graph, bound pruning) are not yet implemented.
    """
    def __init__(self, model_bundle: dict, device='cpu'):
        self.device = device
        self.rotation: ButterflyRotation = model_bundle['bor'].to(device).eval()
        self.hash: ECVH = model_bundle['ecvh'].to(device).eval()
        self.lrsq: LRSQ = model_bundle['lrsq'].to(device).eval()
        self.dim = self.rotation.original_dim
        self.rank = self.lrsq.rank
        self.db_vectors_full = []   # (optional) list of full float32 vectors
        self.db_proj_codes = []     # quantized low-rank codes
        self.ids = []
        self.bucket_map = defaultdict(list)  # code -> list of index positions

    @torch.no_grad()
    def add(self, vectors: torch.Tensor, ids=None, batch=4096):
        vectors = vectors.to(self.device)
        N = vectors.shape[0]
        if ids is None:
            ids = list(range(len(self.ids), len(self.ids)+N))

        for start in range(0, N, batch):
            chunk = vectors[start:start+batch]
            # Normalize
            chunk = chunk / chunk.norm(dim=1, keepdim=True).clamp_min(1e-9)
            rot = self.rotation(chunk)
            codes = self.hash(rot)
            proj = self.lrsq(rot)
            qcodes = self.lrsq.quantize(proj)

            for i in range(chunk.size(0)):
                cbits = codes[i].cpu().numpy().astype(int).tolist()
                packed = pack_bits(cbits)
                global_idx = len(self.ids)
                self.bucket_map[packed].append(global_idx)
                self.db_vectors_full.append(chunk[i].cpu())
                self.db_proj_codes.append(qcodes[i].cpu())
                self.ids.append(int(ids[start+i]))

    @torch.no_grad()
    def search(self, queries: torch.Tensor, k=10, hamming_radius=1, batch=1024):
        queries = queries.to(self.device)
        Q = queries.shape[0]
        all_scores = torch.full((Q, k), -1.0, device=self.device)
        all_ids = torch.full((Q, k), -1, dtype=torch.long, device=self.device)

        for start in range(0, Q, batch):
            chunk = queries[start:start+batch]
            chunk = chunk / chunk.norm(dim=1, keepdim=True).clamp_min(1e-9)
            rot = self.rotation(chunk)
            codes = self.hash(rot)
            proj = self.lrsq(rot)
            # (For this minimal prototype we will re-score using full uncompressed vectors)
            for qi in range(chunk.size(0)):
                cbits = codes[qi].cpu().numpy().astype(int).tolist()
                neigh_codes = hamming_neighbors(cbits, hamming_radius, max_bits=len(cbits))
                candidates = set()
                for nb in neigh_codes:
                    packed = pack_bits(nb)
                    if packed in self.bucket_map:
                        candidates.update(self.bucket_map[packed])
                if not candidates:
                    continue
                cand_idx = list(candidates)
                cand_vecs = torch.stack([self.db_vectors_full[c] for c in cand_idx]).to(self.device)
                qv = chunk[qi:qi+1]
                scores = (qv @ cand_vecs.t()).squeeze(0)  # cosine since normalized
                topk = torch.topk(scores, k=min(k, scores.numel()))
                all_scores[start+qi, :topk.indices.numel()] = topk.values
                # map back to ids
                sel_ids = [self.ids[cand_idx[j]] for j in topk.indices.tolist()]
                all_ids[start+qi, :len(sel_ids)] = torch.tensor(sel_ids, device=self.device)
        return all_scores.cpu().numpy(), all_ids.cpu().numpy()