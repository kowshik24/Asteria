#pragma once
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <mutex>

#include "rotation_bor.h"
#include "hash_ecvh.h"
#include "lrsq.h"

namespace asteria {

struct IndexParams {
  int dim;
  int m_bits;
  int ecc_k;
  int bor_layers;
  int lrsq_rank;
  int lrsq_blocks;
};

struct Bucket {
  std::vector<uint64_t> ids;
  std::vector<int8_t>  codes; // LRSQ packed codes (row-major by item)
  // CAPRG adjacency (fixed small degree)
  std::vector<uint32_t> adj_index; // CSR start indices
  std::vector<uint32_t> adj_list;  // neighbor indices within this bucket scope
  std::mutex mtx; // for concurrent inserts
};

struct Index {
  IndexParams params;
  BOR bor;
  ECVH ecvh;
  LRSQ lrsq;

  // Map code -> bucket storage (code packed to uint64 key)
  std::unordered_map<uint64_t, Bucket> buckets;

  void fit(const std::vector<float>& train, size_t n); // learns bor/ecvh/lrsq
  void add(const std::vector<float>& vecs, const std::vector<uint64_t>& ids, size_t n);
  void search(const std::vector<float>& queries, size_t nq, int k, int max_probes,
              std::vector<float>& out_D, std::vector<uint64_t>& out_I) const;

  void remove(const std::vector<uint64_t>& ids);
  void save(const char* path) const;
  void load(const char* path);
};

} // namespace asteria