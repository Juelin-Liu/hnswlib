#pragma once
#include "util.hpp"
#include <graph.hpp>

namespace ann {

template <DistanceType dist_type, typename id_t = uint32_t,
          typename data_t = float>
FixedDegreeGraph<id_t> build_nnd(int num_iterations, int max_candidates,
                                 const Matrix2D<data_t> &data);

template <typename id_t = uint32_t>
FixedDegreeGraph<id_t> refine_nnd(const FixedDegreeGraph<id_t> &graph);
} // namespace ann
