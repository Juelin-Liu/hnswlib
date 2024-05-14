#pragma once

#include "graph.hpp"
#include "space.hpp"
#include "util.hpp"
#include <cstdint>
#include <functional>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <priority_queue.hpp>
#include <tbb/parallel_for.h>
#include <vector>
#include <algorithm>

namespace ann {

template <typename id_t, typename data_t, DistanceType dist_type>
FixedDegreeGraph<id_t> knn_brute_force(int num_iterations, int d_max,
                                       const Matrix2D<data_t> &data) {

  using namespace tbb;

  int64_t v_num = data.num_elem;
  int64_t e_num = data.num_elem * d_max;

  auto DistFunc = get_distance<dist_type, data_t>;

  FixedDegreeGraph<id_t> ret(d_max, v_num);
  for (int i = 0; i < num_iterations; i++) {
    parallel_for(
        blocked_range<int64_t>(0, v_num), [&](const blocked_range<int64_t> &r) {
          std::vector<std::pair<float, id_t>> distances(data.num_elem);
          for (int64_t vid = r.begin(); vid < r.end(); vid++) {
            auto n_list = ret.get_adj(vid);
            for (int64_t n = 0; n < v_num; n++) {
              float d = DistFunc(data.get_feat(vid), data.get_feat(n), data.dim);
              distances.at(n) = std::make_pair(d, n);
            }
            if (is_min_close(dist_type)) {
              std::sort(distances.begin(), distances.end());
            } else {
              std::sort(distances.begin(), distances.end(), std::greater<std::pair<float, int>>());
            }
            auto new_n_list = ret.get_adj(vid);
            for (int i = 0; i < d_max; i++) {
              new_n_list[i] = distances.at(i).second;
            }
          }
        });
  }

  return ret;
};
}