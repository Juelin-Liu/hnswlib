#pragma once
#include "memory_pool.hpp"
#include "util.hpp"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

namespace ann {
struct FixedDegreeGraph {
  FixedDegreeGraph() = default;
  FixedDegreeGraph(int64_t _d_max, int64_t _v_num) {
    d_max = _d_max;
    v_num = _v_num;
    indices.resize(d_max * v_num);
  };
  int64_t v_num{0};
  int64_t d_max{0};
  std::vector<id_t> indices;

  // adj starts at offset 1
  std::span<id_t> get_adj(id_t vid) {
    return std::span(indices.begin() + vid * d_max, d_max);
  };
  // adj starts at offset 1
  std::span<const id_t> get_adj(id_t vid) const {
    return std::span(indices.begin() + vid * d_max, d_max);
  };
  bool operator==(const FixedDegreeGraph &other) const {
    if (v_num != other.v_num)
      return false;
    if (d_max != other.d_max)
      return false;
    for (size_t i = 0; i < indices.size(); i++) {
      if (indices.at(i) != other.indices.at(i))
        return false;
    }
    return true;
  };

  float recall( FixedDegreeGraph& other)   {
    if (v_num != other.v_num)
      return 0;
    size_t deg = std::min(other.d_max, d_max);
    size_t total = deg * v_num;
    size_t matched = 0;
    for (size_t i = 0; i < v_num; i++) {
      auto adj = get_adj(i);
      auto oadj = other.get_adj(i);
      for (size_t j = 0; j != deg; j++) {
        matched += adj[j] == oadj[j];
      }
    }
    return 1.0 * matched / total;
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  FixedDegreeGraph &graph) {
    os << "v_num = " << graph.v_num << " d_max = " << graph.d_max << "\n";
    os << "indices:\n";
    for (size_t i = 0; i < graph.v_num; i++) {
      auto adj = graph.get_adj(i);
      os << i << ":";
      for (auto n : adj) {
        os << " " << n;
      }
      os << "\n";
    }
    return os;
  };
};

    /**
 * @brief
 * StaticNSWGraph stores adjacency lists contiguously in an array
 * The first element in each adjacency list defines the degree of the adjacency
 * list The max size of the adj is defined by d_max
 *
 * @tparam id_t Data type for ids
 * @tparam id_t Data types for the indices
 */
class NSWGraph {
private:
  id_t d_max{0}; // max degree
  id_t v_num{0}; // number of vertices
  id_t *indices{nullptr};

public:
  NSWGraph() = default;
  NSWGraph(id_t v_num, id_t d_max) {
    Init(v_num, d_max);
  };

  void Init(id_t v_num, id_t d_max) {
    this->v_num = v_num;
    this->d_max = d_max;
    size_t num_bytes = 1llu * sizeof(id_t) * v_num * (d_max + 1);
    indices = static_cast<id_t *>(MemoryPool::Global().malloc(num_bytes));
  }

  void Clear() {
    v_num = 0;
    d_max = 0;
    if (indices)
      MemoryPool::Global().free(indices);
  }
  ~NSWGraph() {
    Clear();
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
class NNDGraph {
private:
  id_t d_max{0}; // max degree
  id_t v_num{0}; // number of vertices
  id_t *indices{nullptr};

public:
  NNDGraph() = default;
  NNDGraph(id_t v_num, id_t d_max) {
    Init(v_num, d_max);
  };

  ~NNDGraph() {
    Clear();
  };

  void Init(id_t v_num, id_t d_max) {
    this->v_num = v_num;
    this->d_max = d_max;
    size_t num_bytes = 1llu * v_num * d_max * sizeof(id_t);
    indices = static_cast<id_t *>(MemoryPool::Global().malloc(num_bytes));
  }

  void Clear() {
    v_num = 0;
    d_max = 0;
    if (indices)
      MemoryPool::Global().free(indices);
  }

  __force_inline__ id_t get_d_max() { return d_max; };
  __force_inline__ id_t get_v_num() { return v_num; };
  __force_inline__ id_t *get_indices() { return indices; };

  // 0-th index is reserved for degree
  __force_inline__ id_t get_deg(id_t vid) { return d_max; };

  // adj starts at offset 0
  __force_inline__ id_t *get_adj(id_t vid) {
    return indices + 1llu * vid * d_max;
  };

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
