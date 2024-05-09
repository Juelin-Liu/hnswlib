#pragma once
#include "util.hpp"

namespace ann {

float L1Distance(const float *pVect1, const float *pVect2, int dim);

float L1Distance(const float16 *pVect1, const float16 *pVect2, int dim);

int L1Distance(const uint8_t *pVect1, const uint8_t *pVect2, int dim);

// float L1Distance(const E5M2 *pVect1,
//                    const E5M2 *pVect2,
//                    int dim);

// float L1Distance(const E4M3 *pVect1,
//                    const E4M3 *pVect2,
//                    int dim);
} // namespace ann
