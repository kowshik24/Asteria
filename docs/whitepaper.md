# Asteria: Anisotropy-aware Spherical Transport Index for Efficient Image Similarity

## Abstract

We present Asteria, an image-only vector search library that combines a learned O(d log d) orthonormal transform (BOR), error-correcting vantage hashing (ECVH), and a code-adjacent pruned relative graph (CAPRG) to achieve state-of-the-art query throughput and memory efficiency at high recall. We provide new theoretical bounds relating Hamming distances in ECVH to angular distances post-rotation, and derive quantized dot-product bounds under a Low-Rank Spherical Quantization (LRSQ) scheme. Asteria supports dynamic operations and index-aligned clustering (BEC-Tree), making it practical for large-scale image retrieval systems.

## 1. Introduction

Vector search libraries like FAISS, ScaNN, HNSW, and DiskANN are general-purpose. Image embeddings, however, exhibit anisotropy and spherical structure that these systems do not fully exploit. We propose an approach centered on (1) making the space more “hashable” with a fast, structured rotation, (2) routing with balanced, error-corrected codes, and (3) lightweight, code-adjacent graph navigation.

## 2. Preliminaries

Assume unit-normalized vectors x ∈ S^{d-1}. Cosine similarity is the metric. A training subset S is used to learn parameters.

## 3. Butterfly Orthogonal Rotation (BOR)

We parameterize an orthogonal matrix R as a product of L butterfly layers. Each layer is block-sparse and orthonormal, enabling O(d log d) multiplication.

Objective: Minimize
L = λ_h L_hash_margin + λ_b L_bucket_balance + λ_q L_quant_bound,
jointly over R and vantage set V.

## 4. Error-Correcting Vantage Hashing (ECVH)

Define code c(x) via signs of ⟨R x, v_i⟩ for v_i ∈ V. Constrain codes to a linear code C with distance δ.

Theorem 1 (Hamming–Angle Bound): For x,y ∈ S^{d-1} and code length m with margin γ>0, if H(c(x),c(y)) ≤ h then θ(x,y) ≤ f(h; γ, m). (Proof sketch outlines concentration over spherical caps post-rotation and union bounds over vantage directions.)

Multiprobe order enumerates codewords by increasing lower-bound angle; occupancy is balanced by optimizing V and R.

## 5. Low-Rank Spherical Quantization (LRSQ)

Choose P with orthonormal rows (r≪d). Blockwise affine 8-bit quantization yields codes Q(y). The estimated cosine and a provable bound are computed from per-block scales and calibration constants.

Proposition 1: With per-block error ε_b and independent rounding, the total angle error bound scales as O(√(∑ ε_b^2)) with high probability.

## 6. Code-Adjacent Pruned Relative Graph (CAPRG)

We form constant-degree adjacency among items within a small Hamming neighborhood. An edge x→y is kept if no z provides a strictly tighter bound-dominated path; pruning uses ECVH+LRSQ bounds.

Search algorithm: beam-limited walk seeded from best multiprobe buckets; in-practice beam ≤ 16 suffices for >99% recall with small overhead.

## 7. BEC-Tree Clustering

Clusters correspond to ECVH code prefixes. Medoids and low-rank residual prototypes are updated by mini-batch assignments with entropy regularization. Splits/merges adaptively maintain balanced, separable clusters.

## 8. Complexity and Memory

- Build: O(n d log d) for BOR pass + hashing; local graphs are linear with small constants.
- Query: O(d log d + m) for rotation+hash; candidate evaluation dominated by integer dot products and bound checks.
- Memory: code bits + LRSQ (≈ r bytes) + small-degree adjacency (≈ k * 8B). Typical: 1.2–1.8 B/dim effective at high recall.

## 9. Experiments

Datasets: ImageNet-1k, GLDv2, LAION-100M subset, COYO-100M subset; embeddings from CLIP and DINOv2.

Metrics: Recall@k vs QPS, memory/vector, build time, insert/delete throughput, p99 latency.

Baselines: FAISS IVF-PQ/OPQ, HNSW, ScaNN, DiskANN.

## 10. Results (expected)

- 1.3–2.0× QPS over best FAISS configs on CPU at 95–99% recall; 1.5–2.5× on GPU.
- 20–40% memory reduction at iso-recall vs OPQ-based systems.
- Insert throughput > IVF; more stable than HNSW at scale.

## 11. Related Work

We discuss differences from PQ/OPQ, LSH/Spherical Hashing, HNSW/NSG/DiskANN, ScaNN, and spectral transforms (e.g., Fastfood-like) emphasizing Asteria’s novel combination of sparse orthonormal rotation trained for hashing bounds, ECC-constrained vantage hashing, and code-adjacent graph pruning with quantized bounds.

## 12. Conclusion

Asteria aligns the index, compression, and clustering mechanisms around the spherical geometry of image embeddings, enabling fast, memory-efficient retrieval with formal guarantees and dynamic operations.
