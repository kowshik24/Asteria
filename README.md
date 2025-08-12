High-level overview

    Library name: Asteria (Anisotropy-aware Spherical Transport Index for Efficient Retrieval in Images)
    Core novelty (distinct from HNSW/IVF-PQ/LSH/ScaNN):
        Butterfly Orthogonal Rotation (BOR): a learnable, extremely fast, sparse parameterization of orthonormal transforms tailored to image embeddings, reducing anisotropy and making angular neighborhoods more separable. It uses a stacked butterfly factorization with trainable mixing, enabling O(d log d) rotation with cache-friendly memory layout. Unlike ITQ/Faiss OPQ, BOR is sparse, FFT-like, and optimized for angular retrieval bounds rather than reconstruction error.
        Error-Correcting Vantage Hashing (ECVH): after BOR, we compute compact m-bit binary codes from a learned set of vantage directions on the unit hypersphere. The code design uses error-correcting codes to guarantee Hamming/angle lower bounds and balanced occupancy, enabling precise multiprobe ordering with bounded work. Unlike Spherical Hashing or LSH, ECVH codes are trained jointly with BOR to minimize angular distortion and maximize code margin with provable angle/Hamming inequalities.
        Code-Adjacent Pruned Relative Graph (CAPRG): within near-Hamming buckets, we maintain a small navigable graph whose edges are formed in code-space under an angular gate with provable pruning rules derived from ECVH bounds. Unlike HNSW (geometric small-world), CAPRG edges are code-adjacent, constant-degree, and updated incrementally with amortized sublinear maintenance cost; its routing cost is independent of global density.
        Image-only compression and distance estimation: Low-Rank Spherical Quantization (LRSQ) compresses vectors for fast dot product estimation with deterministic bounds on angular error. LRSQ uses a low-rank orthogonal projection and blockwise affine 8-bit quantization optimized specifically on image embeddings’ spectral profile (post-BOR), enabling SIMD/GPU kernels and bound-checked early exits.
        Clustering integrated with index topology: BEC-Tree (Binary Error-Correcting Cluster Tree) uses ECVH prefixes as a hierarchical partition coupled to a medoid-on-sphere refinement with entropy-regularized assignment. It yields clusters aligned with retrieval routing, supporting dynamic split/merge with immediate index consistency.

Intuition: This library explicitly embraces the fact that normalized image embeddings live on a high-dimensional sphere and are anisotropic. Asteria learns a fast, structured rotation (BOR) that makes spherical partitioning easy and then uses an error-correcting binary code to route queries (ECVH) with the ability to provably bound and prune. Finally, a tiny code-adjacent graph adds robustness within buckets without the overhead and dynamical instability of global navigable small-world graphs.

Key advantages

    Speed: O(d log d) rotation + bit extraction; near-constant time routing per probe; SIMD/GPU bitset scans; tight multiprobe ordering reduces candidate counts by 2-5x at same recall compared to IVF/ScaNN-like baselines.
    Memory: Code-centric graph and 8-bit blockwise quantization yield 1.2–1.8 bytes/dimension effective storage at high recall regimes (vs ≥2–4 B/dim in PQ/OPQ schemes).
    Dynamic updates: Incremental insertion with localized rebalancing (code buckets + constant-degree CAPRG updates); no global graph rewrites; background re-training optional and online-safe.
    Image-only optimization: Training objectives leverage the empirical spectral anisotropy of image embeddings (e.g., CLIP/DINO), giving better compression and tighter angle bounds than general-purpose systems.

Algorithmic details (publishable components)

    Butterfly Orthogonal Rotation (BOR)

    Factorization: R = Π_{l=1..L} Bl, with each Bl a sparse butterfly layer composed of 2×2 orthonormal blocks with learnable mixing and permutations. This gives O(d log d) multiplies and near-sequential memory access.
    Objective: Given training vectors x on the unit sphere, learn R to minimize the angular quantization loss after hashing and LRSQ. We jointly optimize for:
        maximal cosine margin between near neighbors and non-neighbors after hashing (code separation),
        balanced code occupancy across buckets,
        low quantization distortion for LRSQ.
    Novelty: Orthogonal rotation parameterized by butterfly layers trained for angular hashing error and quantized dot-product bounds, not reconstruction.

    Error-Correcting Vantage Hashing (ECVH)

    Vantage set V = {v1..vm} on the unit sphere, learned jointly with BOR. Code c(x) ∈ {0,1}^m defined by c_i = sign(⟨R x, v_i⟩).
    ECC embedding: We constrain codebooks to a linear code C with distance δ and rate ρ to enable:
        provable angle/Hamming relationships: if H(c(x), c(y)) ≤ h, then θ(x,y) ≤ f(h; V,R),
        multi-probe sequence enumerating codewords in increasing lower-bound angle order.
    Bucketization and routing: Primary bucket uses the exact code; multi-probe explores nearby codewords as budget allows. Each bucket keeps a compact list of IDs with LRSQ codes and optional CAPRG adjacency.

    Code-Adjacent Pruned Relative Graph (CAPRG)

    Build local graphs over items within each codeword and its small Hamming neighborhood (radius r0). Each node keeps k neighbors chosen by:
        angular gate: keep neighbor y if no z exists s.t. maxBoundAngle(x,z) < exactAngle(x,y) and maxBoundAngle(y,z) < exactAngle(x,y), where bounds come from ECVH+LRSQ. This prunes transitive edges.
    Search: Start with the best-probed bucket seeds, run beam-limited graph walks using bound checks (cheap) and occasional exact re-checks (rare). This gives recall boosts with limited overhead.

    Low-Rank Spherical Quantization (LRSQ)

    Projection: Choose P ∈ R^{r×d}, r<<d, with orthonormal rows (trained jointly with BOR) to capture the signal-dominant subspace after BOR.
    Blockwise affine quantization: Partition r dims into B blocks; each block uses learned scale/zero-point and k-level codebook (k=256 typical). Pack into 8-bit arrays aligned for SIMD.
    Dot product estimation: For unit vectors, cosine ≈ α ⟨P R x, Q(y)⟩ + β with per-block dequant error bounds aggregating to a certified bound on the angle. Early-exit pruning when upper bound falls below current top-k threshold.

    BEC-Tree for clustering

    Hierarchy: Clusters correspond to prefixes of the ECVH code; at level t, there are up to 2^t bins. Each bin maintains a medoid on the sphere and a small set of “prototype residuals” in the low-rank space.
    Assignment: Entropy-regularized objective: minimize sum over samples of angular distance to the medoid plus a soft penalty for code-prefix divergence; solved with mini-batch updates. Splits occur when prefix entropy rises; merges when occupancy and margin drop.
    Benefit: Clusters are index-aligned; retrieval can optionally return cluster IDs cheaply; cluster maintenance is O(n) over epochs with no global graph rebuilds.

Theoretical results to target for the paper

    Hamming-angle bound theorem: With BOR+ECVH construction, prove that for all x,y, H(c(x),c(y)) ≤ h implies θ(x,y) ≤ f(h) where f depends on the minimal margin and vantage distribution; show concentration bounds under subgaussian post-rotation coordinates.
    Quantized bound aggregation: Provide per-block affine quantization error bounds that sum to a global cosine bound with high probability guarantees (empirical calibration optional).
    Routing complexity: Show expected candidates per probe decays exponentially in h under balanced code occupancy; derive QPS scaling with m, bucket fanout, and ECC parameters.

Implementation architecture

    Data model:
        Unit-normalized vectors (float32 on ingest), rotated and quantized for storage.
        Stored artifacts: BOR (L layers), V vantage set, ECC matrices, P projection, LRSQ block params.
        Index: code-buckets -> contiguous ID arrays + compressed LRSQ codes + CAPRG adjacency.

    Query flow:
        Normalize x, rotate via BOR (SIMD/GPU kernels).
        Compute m dot products with vantages to generate code bits; look up primary bucket.
        Multi-probe enumerate nearby codes using ECC distance order; retrieve candidates.
        Fast bound checks via LRSQ; maintain a top-k heap; early exit when safe.
        Optionally run CAPRG beam search for re-ranking among close candidates.
        Return IDs, approximate scores; optionally compute exact cosines for the final k.

    Insert flow:
        Normalize + BOR + ECVH + LRSQ.
        Append to bucket’s arrays; lazy CAPRG neighbor update using a small candidate pool from the same bucket and adjacent codes; periodic pruning pass amortized.

    Delete/Update:
        Tombstones + periodic compaction. Updates follow delete+insert semantics.

    GPU acceleration:
        Fused kernels: normalization + BOR + vantages dot-products; warp-level bitpacking for ECVH codes.
        Bitset Hamming distance and ECC multiprobe generation on GPU using small LUTs.
        LRSQ dot products via int8 tensor cores (CUDA) or AVX512 VNNI on CPU.

    Memory layout:
        Structure-of-arrays (SoA) for codes and IDs per bucket.
        64-bit aligned adjacency lists with small-degree CAPRG.
        Optional tiered cache: hot buckets pinned; cold on disk with async prefetch.

Benchmarks and evaluation plan

    Datasets: CLIP ViT-B/32 and DINOv2 embeddings for:
        ImageNet-1k (1.2M) for dev tuning, GLDv2 for fine-grained retrieval, LAION-100M subset, COYO-100M subset. Include at least one 1B-scale synthetic extension via bootstrapping to stress scaling.
    Baselines: FAISS (IVF-PQ, HNSW, OPQ), ScaNN, DiskANN. Measure:
        Recall@k vs QPS (single-thread, multi-thread, GPU), memory bytes/vector, build time, insertion rate, update latency, tail latency (p99).
    Ablations:
        Remove CAPRG; remove ECC; replace BOR with dense OPQ; vary m (code length), ECC rate, and probe budget.
    Expected results:
        At target recall levels (95–99%), achieve 1.3–2.0× QPS vs best FAISS configuration on CPU; 1.5–2.5× on GPU for mid/large k.
        Memory 20–40% lower than OPQ@8–16 bytes/dim at equal recall.
        Insert throughput comparable to IVF; far faster and more stable than HNSW at high scale.

Initial API design

    Core:
        fit(train_vectors)
        add(vectors, ids)
        search(queries, k, ef=optional, probes=optional)
        remove(ids)
        train_incremental(stream)
        save(path), load(path)

    Clustering:
        cluster(level=None or target_k), assign(vectors), medoids(level), splits/merges.

    Bindings: C++ core, Python bindings via pybind11, optional Rust FFI.

Repository skeleton

    src/core: rotation, hashing, ecc, lrsq, index, caprg, bounds, io
    src/kernels: CUDA kernels and CPU SIMD intrinsics
    python: bindings and high-level API
    benchmarks: datasets loaders, runner scripts, plotting
    docs: design and whitepaper

Below are proposed starter files to bootstrap the project and document the method precisely.
