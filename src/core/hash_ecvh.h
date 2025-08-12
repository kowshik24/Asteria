#pragma once
#include <vector>
#include <cstdint>

namespace asteria {

// Error-Correcting Vantage Hashing (ECVH)
struct ECVH {
  int dim;
  int m_bits;        // code length
  int ecc_k;         // ECC parameters (e.g., BCH (n,k))
  // Vantage directions stored row-major (m_bits x dim)
  std::vector<float> vantages;

  // ECC encoding/decoding small matrices or LUTs
  std::vector<uint8_t> ecc_gen;  // generator matrix packed
  std::vector<uint8_t> ecc_par;  // parity-check matrix packed

  // Compute raw sign bits s in {0,1}^m from rotated x (R*x passed in)
  void sign_bits(const float* x_rot, uint8_t* out_bits) const;

  // Encode to nearest ECC-valid codeword (optional during indexing).
  void ecc_encode_inplace(uint8_t* bits) const;

  // Multiprobe iterator: yields nearby codewords in increasing Hamming order
  // with ECC awareness.
  struct ProbeIter {
    const ECVH* self;
    const uint8_t* base_code;
    int max_radius;
    // Implementation detail omitted
    bool next(uint8_t* out_code);
  };
};

} // namespace asteria