#pragma once
#include "graph.hpp"
#include "space.hpp"
#include "util.hpp"
#include <cstdint>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <priority_queue.hpp>
#include <queue>
#include <random>
#include <tbb/parallel_for.h>

namespace ann {

template<typename id_t, typename data_t, DistanceType dist_type>
FixedDegreeGraph<id_t> build_nnd(int num_iterations, int d_max,
                                 const Matrix2D<data_t> &data) {
  using namespace tbb;
  int64_t v_num = data.num_elem;
  int64_t e_num = data.num_elem * d_max;

  auto DistFunc = get_distance<dist_type, data_t>;

  FixedDegreeGraph<id_t> ret(d_max, v_num);
  parallel_for(blocked_range<int64_t>(0, v_num),
               [&](const blocked_range<int64_t> &r) {
                 for (int64_t i = r.begin(); i < r.end(); i++) {
                   auto adj = ret.get_adj(i);
                   std::mt19937 rng(std::random_device{}());
                   std::vector<int> vec(v_num);
                   std::iota(begin(vec), end(vec), 0);
                   std::shuffle(begin(vec), end(vec), rng);

                   for (int i = 0; i < d_max; i++) {
                     adj[i] = vec[i];
                   }
                 }
               });

  for (int i = 0; i < num_iterations; i++) {
    FixedDegreeGraph<id_t> tmp(d_max, v_num);
    parallel_for(
        blocked_range<int64_t>(0, v_num), [&](const blocked_range<int64_t> &r) {
          for (int64_t vid = r.begin(); vid < r.end(); vid++) {
            // find all the neighbors of v
            // and the neighbors of those neighbors

            std::vector<std::pair<float, id_t>> distances;
            auto n_list = ret.get_adj(vid);
            for (auto n : n_list) {
              float d =
                  DistFunc(data.get_feat(vid), data.get_feat(n), data.dim);
              distances.push_back(std::make_pair(d, n));

              auto nn_list = ret.get_adj(n);
              for (auto nn : nn_list) {
                float dd =
                    DistFunc(data.get_feat(vid), data.get_feat(nn), data.dim);
                distances.push_back(std::make_pair(dd, nn));
              }

              if (is_min_close(dist_type)) {
                std::sort(distances.begin(), distances.end());
              } else {
                std::sort(distances.begin(), distances.end(),
                          std::greater<std::pair<float, int>>());
              }
              distances.erase( unique( distances.begin(), distances.end() ), distances.end() );

              auto new_n_list = tmp.get_adj(vid);
              for (int i = 0; i < d_max; i++) {
                new_n_list[i] = distances.at(i).second;
              }
            }
          }
        });
    ret = tmp;
  }

  return ret;
};
} // namespace ann
