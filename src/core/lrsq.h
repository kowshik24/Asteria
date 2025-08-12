#pragma once
#include <vector>
#include <cstdint>

namespace asteria {

// Low-Rank Spherical Quantization (LRSQ)
struct LRSQ {
  int dim;
  int rank;
  int blocks;
  // P: rank x dim orthonormal rows
  std::vector<float> P;

  // Per-block affine quantization params
  struct BlockParam {
    float scale;
    float zero;
  };
  std::vector<BlockParam> params;

  // Quantize y = P * (R*x) to int8 codes
  void quantize(const float* x_rot, std::vector<int8_t>& out_codes) const;

  // Estimate cosine and bounds between query (float low-rank) and db codes
  struct Estimator {
    const LRSQ* self;
    std::vector<float> q_proj; // precomputed P * (R*q)
    float estimate(const int8_t* db_codes) const;
    void bounds(const int8_t* db_codes, float& lo, float& hi) const;
  };
};

} // namespace asteria