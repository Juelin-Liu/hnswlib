#pragma once
#include <algorithm>
#include <cstdint>
#include <limits.h>
#include <random>
#include <vector>

namespace ann {

typedef int32_t id_t;

#define ALWAYS_ASSERT(expr)                                       \
    do {                                                          \
        if (!(expr)) {                                            \
            std::cerr << "Assertion failed: " << #expr            \
                      << ", file " << __FILE__                    \
                      << ", line " << __LINE__ << std::endl;      \
            std::abort();                                         \
        }                                                         \
    } while (0)

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

inline int get_random_level(const std::vector<double>& cum_probs) {
  std::default_random_engine gen(100);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  double prob = distribution(gen);
  for (size_t i = 0; i < cum_probs.size(); i++){
    if (prob < cum_probs.at(i)) return i;
  }
  return cum_probs.size() - 1;
}

inline std::vector<id_t> get_random_indices(size_t num_elem){
  auto seed = 42;
  std::vector<id_t> indices(num_elem);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));
  return indices;
}
} // namespace ann