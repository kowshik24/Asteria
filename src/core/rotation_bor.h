#pragma once
#include <vector>
#include <cstdint>

// Butterfly Orthogonal Rotation (BOR)
// R = product of L butterfly layers, each layer is a set of 2x2 orthonormal blocks and permutations.
// Forward applies R*x in O(d log d) with cache-friendly memory access.

namespace asteria {

struct ButterflyLayer {
  // Each layer defines pairs of indices and 2x2 orthonormal blocks:
  // [ a  b ]
  // [-b  a ]  with a^2 + b^2 = 1 (Givens-like), or a general 2x2 orthonormal block.
  std::vector<int> idx_left;
  std::vector<int> idx_right;
  std::vector<float> a;  // cos(theta)
  std::vector<float> b;  // sin(theta)
  // Optional permutations encoded separately for cache-friendly traversal.
  std::vector<int> perm; // applied before the 2x2 blocks
};

struct BOR {
  int dim;
  std::vector<ButterflyLayer> layers;

  explicit BOR(int d = 0) : dim(d) {}

  // Apply y = R * x (in-place allowed if y points to x)
  void apply(const float* x, float* y) const;

  // Initialize random sparse orthonormal butterfly with L layers.
  static BOR random_init(int dim, int L, uint64_t seed);

  // Train BOR parameters against a given loss (to be implemented elsewhere).
};

} // namespace asteria