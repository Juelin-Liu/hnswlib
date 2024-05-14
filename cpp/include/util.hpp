#pragma once
#include <cstdint>
#include <limits.h>

namespace ann {
#define __force_inline__ inline __attribute__((always_inline))

#ifdef __clang__
#define float16 __fp16
#elifdef __GNUC__
#define float16 _Float16
#endif

#define Sum8(arr)                                                              \
  arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7]

#define Sum16(arr)                                                             \
  arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7] +      \
      arr[8] + arr[9] + arr[10] + arr[11] + arr[12] + arr[13] + arr[14] +      \
      arr[15]

// class E4M3
// {
//     private:
//     uint8_t data;
// };

// class E5M2 {
//     private:
//     uint8_t data;
// };

template <typename data_t> struct Matrix2D {
  int64_t num_elem{0};
  int64_t dim{0};
  data_t *data{nullptr};
  Matrix2D(int64_t _num_elem, int64_t _dim, data_t *_data)
      : data{_data}, dim{_dim}, num_elem(_num_elem){};

  data_t *get_feat(int64_t vid) const { return data + vid * dim; }
};

template <int scale, int dimension> constexpr int get_main(int dim) {
  if constexpr (dimension == INT_MAX) {
    return dim - dim % scale;
  } else {
    return dimension - dimension % scale;
  }
};

template <int scale, int dimension> constexpr int get_residual(int dim) {
  if constexpr (dimension == INT_MAX) {
    return dim % scale;
  } else {
    return dimension % scale;
  }
};

template <int scale, int dimension> constexpr int get_all(int dim) {
  if constexpr (dimension == INT_MAX) {
    return dim;
  } else {
    return dimension;
  }
};

enum class DistanceType {
  L2 = 0,     // (x - y)^2
  L1 = 1,     // |x - y|
  Cosine = 2, // (x . y) / (|x| * |y|)
  Ip = 3,     // (x . y)
  Hamming = 4 // popcnt(x ^ y)
};

/**
 * Whether minimal distance corresponds to similar elements (using the given
 * metric).
 */
inline constexpr bool is_min_close(DistanceType metric) {
  bool select_min;
  switch (metric) {
  case DistanceType::Ip:
    // Similarity metrics have the opposite meaning, i.e. nearest neighbors are
    // those with larger
    select_min = false;
    break;
  default:
    select_min = true;
  }
  return select_min;
}
} // namespace ann