#pragma once
#include <cstdint>
#include <float8.hpp>

namespace ann {

float L2Distance(const float *pVect1, const float *pVect2, int dim);

float L2Distance(const _Float16 *pVect1, const _Float16 *pVect2, int dim);

int L2Distance(const uint8_t *pVect1, const uint8_t *pVect2, int dim);

// float L2Distance(const E5M2 *pVect1,
//                    const E5M2 *pVect2,
//                    int dim);

// float L2Distance(const E4M3 *pVect1,
//                    const E4M3 *pVect2,
//                    int dim);
} // namespace ann
