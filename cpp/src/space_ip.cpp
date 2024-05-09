#include "space_ip.hpp"
#include <cpuid.h>
#include <immintrin.h>
#include <numeric>

namespace ann {
#define Sum8(arr)                                                              \
  arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7]

#define Sum16(arr)                                                             \
  arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7] +      \
      arr[8] + arr[9] + arr[10] + arr[11] + arr[12] + arr[13] + arr[14] +      \
      arr[15]

namespace impl {

template <typename dist_t, typename data_t, int K>
__force_inline__ dist_t InnerProductResidual(const data_t *pVect1, const data_t *pVect2,
                                   int dim) {
  dist_t dist{0};
#pragma unroll(K)
  for (int i = 0; i < dim; i += 1) {
    dist += pVect1[i] * pVect2[i];
  }

  return dist;
}

// K is the unroll factor
template <typename dist_t, typename data_t, int K>
__force_inline__ float InnerProductMain(const float *pVect1, const float *pVect2,
                              int dim) {

  static_assert(std::is_same<dist_t, float>::value,
                "Argument dist_t must be of type float");
  static_assert(std::is_same<data_t, float>::value,
                "Argument data_t must be of type float");

#ifdef __AVX512F__
  {
    __m512 temp = _mm512_set1_ps(0);
    constexpr int step = sizeof(temp) / sizeof(float);
    int bound = dim - dim % step;
    float __attribute__((aligned(sizeof(temp)))) TmpRes[step];

#pragma unroll(K)
    for (int i = 0; i < bound; i += step) {
      const auto diff = _mm512_mul_ps(_mm512_loadu_ps(pVect1 + i),
                                      _mm512_loadu_ps(pVect2 + i));
      temp = _mm512_add_ps(temp, diff);
    }
    _mm512_store_ps(TmpRes, temp);
    return Sum16(TmpRes);
  }

#elif __AVX__
  {
    __m256 temp = _mm256_set1_ps(0);
    constexpr int step = sizeof(temp) / sizeof(float);
    int bound = dim - dim % step;
    float __attribute__((aligned(sizeof(temp)))) TmpRes[step];

#pragma unroll(K)
    for (int i = 0; i < bound; i += step) {
      const auto diff = _mm256_mul_ps(_mm256_loadu_ps(pVect1 + i),
                                      _mm256_loadu_ps(pVect2 + i));
      temp = _mm256_add_ps(temp, diff);
    }
    _mm256_store_ps(TmpRes, temp);
    return Sum8(TmpRes);
  }
#else
  {
    dist_t dist{0};
#pragma unroll(K)
    for (int i = 0; i < dim; i += 1) {
      dist += pVect1[i] * pVect2[i];
    }

    return dist;
  }
#endif
}

// K is the unroll factor
template <typename dist_t, typename data_t, int K>
__force_inline__ float InnerProductMain(const float16 *pVect1, const float16 *pVect2,
                              int dim) {
  static_assert(std::is_same<dist_t, float>::value,
                "Argument dist_t must be of type float");
  static_assert(std::is_same<data_t, float16>::value,
                "Argument data_t must be of type float16");
#ifdef __AVX512F__
  {
    __m512 temp = _mm512_set1_ps(0);
    constexpr int step = sizeof(temp) / sizeof(float);
    int bound = dim - dim % step;
    float __attribute__((aligned(sizeof(temp)))) TmpRes[step];

#pragma unroll(K)
    for (int i = 0; i < bound; i += step) {
      const auto diff =
          _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256(pVect1 + i)),
                        _mm512_cvtph_ps(_mm256_loadu_si256(pVect2 + i)));
      temp = _mm512_add_ps(temp, diff);
    }
    _mm512_store_ps(TmpRes, temp);
    return Sum16(TmpRes);
  }
#elif __AVX2__ && __F16C__
  {
    __m256 temp = _mm256_set1_ps(0);
    constexpr int step = sizeof(temp) / sizeof(float);
    int bound = dim - dim % step;
    float __attribute__((aligned(sizeof(temp)))) TmpRes[step];

#pragma unroll(K)
    for (int i = 0; i < bound; i += step) {
      const __m256 diff = _mm256_mul_ps(
          _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u *)(pVect1 + i))),
          _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u *)(pVect2 + i))));
      temp = _mm256_add_ps(temp, diff);
    }
    _mm256_store_ps(TmpRes, temp);
    return Sum8(TmpRes);
  }
#else
  {
    float res{0};
#pragma unroll(K)
    for (int i = 0; i < dim; i += 1) {
      const dist_t diff = pVect1[i] - pVect2[i];
      res += diff * diff;
    }
    return res;
  }
#endif
}

// K is the unroll factor
template <typename dist_t, typename data_t, int K>
__force_inline__ int InnerProductMain(const uint8_t *pVect1, const uint8_t *pVect2,
                            int dim) {
  static_assert(std::is_same<dist_t, int>::value,
                "Argument dist_t must be of type int");
  static_assert(std::is_same<data_t, uint8_t>::value,
                "Argument data_t must be of type uint8_t");
#ifdef __AVX512F__
  {
    __m512i temp = _mm512_set1_epi32(0);
    constexpr int step = sizeof(temp) / sizeof(int16_t);
    int bound = dim - dim % step;
    int __attribute__((aligned(sizeof(temp)))) TmpRes[step];

#pragma unroll(K)
    for (int i = 0; i < bound; i += step) {
      const auto a_casted =
          _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i_u *)(pVect1 + i)));
      const auto b_casted =
          _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i_u *)(pVect2 + i)));
      const auto diff_sq = _mm512_mullo_epi16(a_casted, b_casted);
      temp = _mm512_add_epi32(
          temp, _mm512_cvtepu16_epi32(_mm512_castsi512_si256(diff_sq)));
      temp = _mm512_add_epi32(
          temp, _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(diff_sq, 1)));
    }
    _mm512_store_si512((__m512i *)TmpRes, temp);
    return Sum16(TmpRes);
  }
#elif __AVX2__ && __F16C__
  {
    __m256i temp = _mm256_set1_epi32(0);
    constexpr int step = sizeof(temp) / sizeof(int16_t);
    int bound = dim - dim % step;
    int __attribute__((aligned(sizeof(temp)))) TmpRes[step];

#pragma unroll(K)
    for (int i = 0; i < bound; i += step) {
      const auto a_casted =
          _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i_u *)(pVect1 + i)));
      const auto b_casted =
          _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i_u *)(pVect2 + i)));
      const auto diff_sq = _mm256_mullo_epi16(a_casted, b_casted);
      temp = _mm256_add_epi32(
          temp, _mm256_cvtepu16_epi32(_mm256_castsi256_si128(diff_sq)));
      temp = _mm256_add_epi32(
          temp, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(diff_sq, 1)));
    }
    _mm256_store_si256((__m256i *)TmpRes, temp);
    return Sum8(TmpRes);
  }
#else
  {
    dist_t dist{0};
#pragma unroll(K)
    for (int i = 0; i < dim; i += 1) {
      dist += pVect1[i] * pVect2[i];
    }

    return dist;
  }
#endif
}
template <typename dist_t, typename data_t, int scale,
          int dimension = INT32_MAX>
__force_inline__ dist_t InnerProductUnroll(const data_t *pVect1, const data_t *pVect2,
                                 int dim) {
  if (dimension == INT32_MAX) {
    int residual = dim % scale;
    if (residual == 0) {
      return InnerProductMain<dist_t, data_t, 1>(pVect1, pVect2, dim);
    } else {
      return InnerProductMain<dist_t, data_t, 1>(pVect1, pVect2,
                                                 dim - residual) +
             InnerProductResidual<dist_t, data_t, 1>(
                 pVect1 + dim - residual, pVect2 + dim - residual, residual);
    }
  } else {
    constexpr int unroll = (dimension - dimension % scale) / scale;
    constexpr int residual = dimension % scale;
    if constexpr(residual == 0) {
      return InnerProductMain<dist_t, data_t, unroll>(pVect1, pVect2, dim);
    } else {
      return InnerProductMain<dist_t, data_t, unroll>(pVect1, pVect2, dim) +
             InnerProductResidual<dist_t, data_t, residual>(
                 pVect1 + unroll, pVect2 + unroll, residual);
    }
  }
};
} // namespace impl

// data_t will be cast to cast_t in register to conduct the computation by
// default unless the hardware supports the computation in cast_t natively
/**
 * @brief Wrapper for inner product with unroll optimization
 *
 * @tparam dist_t
 * @tparam data_t
 * @tparam cast_t
 * @param pVect1
 * @param pVect2
 * @param dim
 * @return dist_t
 */
template <typename dist_t, typename data_t, typename cast_t>
__force_inline__ dist_t InnerProduct(const data_t *pVect1, const data_t *pVect2,
                           int dim) {
#ifdef __AVX512F__
  constexpr int scale = 512 / sizeof(cast_t) / 8;
#elif __AVX__
  constexpr int scale = 256 / sizeof(cast_t) / 8;
#else
  constexpr int scale = 1;
#endif

  // optimize for specific dimension
  switch (dim) {
  case 200:
    return impl::InnerProductUnroll<dist_t, data_t, scale, 200>(pVect1, pVect2,
                                                                dim);
  case 128:
    return impl::InnerProductUnroll<dist_t, data_t, scale, 128>(pVect1, pVect2,
                                                                dim);
  case 100:
    return impl::InnerProductUnroll<dist_t, data_t, scale, 100>(pVect1, pVect2,
                                                                dim);
  case 96:
    return impl::InnerProductUnroll<dist_t, data_t, scale, 96>(pVect1, pVect2,
                                                               dim);
  case 64:
    return impl::InnerProductUnroll<dist_t, data_t, scale, 96>(pVect1, pVect2,
                                                               dim);
  case 32:
    return impl::InnerProductUnroll<dist_t, data_t, scale, 96>(pVect1, pVect2,
                                                               dim);
  case 16:
    return impl::InnerProductUnroll<dist_t, data_t, scale, 16>(pVect1, pVect2,
                                                               dim);
  default:
    return impl::InnerProductUnroll<dist_t, data_t, scale>(pVect1, pVect2, dim);
    ;
  }
};

float InnerProduct(const float *pVect1, const float *pVect2, int dim) {
  return InnerProduct<float, float, float>(pVect1, pVect2, dim);
};

float InnerProduct(const float16 *pVect1, const float16 *pVect2, int dim) {
  return InnerProduct<float, float16, float>(pVect1, pVect2, dim);
};

int InnerProduct(const uint8_t *pVect1, const uint8_t *pVect2, int dim) {
  return InnerProduct<int, uint8_t, int16_t>(pVect1, pVect2, dim);
};

// float InnerProduct(const E5M2 *pVect1,
//                    const E5M2 *pVect2,
//                    int dim);

// float InnerProduct(const E4M3 *pVect1,
//                    const E4M3 *pVect2,
//                    int dim);
} // namespace ann
