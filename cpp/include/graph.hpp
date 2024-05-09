#pragma once
#include "memory_pool.hpp"
#include "util.hpp"
#include <cstdint>
#include <vector>
#include <span>

namespace ann {
template <typename id_t> struct FixedDegreeGraph {
  FixedDegreeGraph() = default;
  FixedDegreeGraph(int64_t d_max, int64_t v_num) {
    this->d_max = d_max;
    this->v_num = v_num;
    indices.resize(d_max * v_num);
  };
  int64_t v_num{0};
  int64_t d_max{0};
  std::vector<id_t> indices;

  // adj starts at offset 1
  __force_inline__ std::span<id_t> get_adj(id_t vid) {
    return std::span(indices.begin() + vid * d_max, d_max);
  };
};
/**
 * @brief
 * NSWGraph stores adjacency lists contiguously in an array
 * The first element in each adjacency list defines the degree of the adjacency
 * list The max size of the adj is defined by d_max
 *
 * @tparam id_t Data type for ids
 * @tparam id_t Data types for the indices
 */
template <typename id_t = uint32_t> class NSWGraph {
private:
  id_t d_max{0}; // max degree
  id_t v_num{0}; // number of vertices
  id_t *indices{nullptr};

public:
  NSWGraph() = default;
  NSWGraph(id_t v_num, id_t d_max) {
    this->v_num = v_num;
    this->d_max = d_max;
    size_t num_bytes = 1llu * sizeof(id_t) * v_num * (d_max + 1);
    indices = static_cast<id_t *>(MemoryPool::Global().malloc(num_bytes));
  };

  ~NSWGraph() {
    v_num = 0;
    d_max = 0;
    if (indices)
      MemoryPool::Global().free(indices);
  };

  __force_inline__ id_t get_d_max() { return d_max; };
  __force_inline__ id_t get_v_num() { return v_num; };
  __force_inline__ id_t *get_indices() { return indices; };

  // 0-th index is reserved for degree
  __force_inline__ id_t get_deg(id_t vid) {
    return *(indices + 1llu * vid * (d_max + 1));
  };

  // adj starts at offset 1
  __force_inline__ id_t *get_adj(id_t vid) {
    return indices + 1llu * vid * (d_max + 1) + 1;
  };

  // Swap function
  friend void swap(NSWGraph &first, NSWGraph &second) noexcept {
    std::swap(first.v_num, second.v_num);
    std::swap(first.d_max, second.d_max);
    std::swap(first.indices, second.indices);
  }

  NSWGraph(NSWGraph &&other) noexcept { swap(*this, other); }

  NSWGraph &operator=(NSWGraph &&other) noexcept {
    // Check for self-assignment
    if (this != &other) {
      // Release resources from the current object
      this->~NSWGraph();
      // Steal the resources from the source object
      swap(*this, other);
    }
    return *this;
  }

  NSWGraph(const NSWGraph &other) = delete;
  NSWGraph &operator=(const NSWGraph &other) = delete;
};

/**
 * @brief
 * NNDGraph stores adjacency lists contiguously in an array
 * Each adjacency list has the same length d_max
 * @tparam id_t
 * @tparam id_t
 */
template <typename id_t = uint32_t> class NNDGraph {
private:
  id_t d_max{0}; // max degree
  id_t v_num{0}; // number of vertices
  id_t *indices{nullptr};

public:
  NNDGraph() = default;
  NNDGraph(id_t v_num, id_t d_max) {
    this->v_num = v_num;
    this->d_max = d_max;
    size_t num_bytes = 1llu * v_num * d_max * sizeof(id_t);
    indices = static_cast<id_t *>(MemoryPool::Global().malloc(num_bytes));
  };

  ~NNDGraph() {
    v_num = 0;
    d_max = 0;
    if (indices)
      MemoryPool::Global().free(indices);
  };

  __force_inline__ id_t get_d_max() { return d_max; };
  __force_inline__ id_t get_v_num() { return v_num; };
  __force_inline__ id_t *get_indices() { return indices; };

  // 0-th index is reserved for degree
  __force_inline__ id_t get_deg(id_t vid) { return d_max; };

  // adj starts at offset 0
  __force_inline__ id_t *get_adj(id_t vid) { return indices + 1llu * vid * d_max; };

  friend void swap(NNDGraph &first, NNDGraph &second) noexcept {
    std::swap(first.v_num, second.v_num);
    std::swap(first.d_max, second.d_max);
    std::swap(first.indices, second.indices);
  }

  NNDGraph(NNDGraph &&other) noexcept { swap(*this, other); }

  NNDGraph &operator=(NNDGraph &&other) noexcept {
    // Check for self-assignment
    if (this != &other) {
      // Release resources from the current object
      this->~NNDGraph();
      // Steal the resources from the source object
      swap(*this, other);
    }
    return *this;
  }

  NNDGraph(const NNDGraph &other) = delete;
  NNDGraph &operator=(const NNDGraph &other) = delete;
};
} // namespace ann
