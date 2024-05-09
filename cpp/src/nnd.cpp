#include "nnd.hpp"
#include "graph.hpp"
#include "space.hpp"
#include "util.hpp"
#include <cstdint>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <queue>
#include <random>
#include <tbb/parallel_for.h>

namespace ann {

template <DistanceType dist_type, typename id_t, typename data_t>
FixedDegreeGraph<id_t> build_nnd(int num_iterations, int d_max,
                         const Matrix2D<data_t> &data) {

  using namespace tbb;

  int64_t v_num = data.num_elem;
  int64_t e_num = data.num_elem * d_max;

  auto Func = get_distance<dist_type, data_t>;

  std::random_device rd;
  std::mt19937 gen(rd());
  // Define range for random values (modify as needed)
  std::uniform_int_distribution<id_t> dis(0, v_num - 1);

  FixedDegreeGraph<id_t> ret(d_max, v_num);
  parallel_for(blocked_range<int64_t>(0, e_num),
               [&](const blocked_range<int64_t> &r) {
                 for (int64_t i = r.begin(); i < r.end(); i++) {
                   ret.indices.at(i) = dis(gen);
                 }
               });

  for (int i = 0; i < num_iterations; i++) {
    FixedDegreeGraph<id_t> tmp(d_max, v_num);
    parallel_for(blocked_range<int64_t>(0, v_num), [&](const blocked_range<int64_t>& r) {
        for (int64_t vid = r.begin(); vid < r.end(); vid++){
            // find all the neighbors of v
            // and the neighbors of those neighbors
            std::priority_queue<std::pair<float, id_t>> candidates;
            auto n_list = ret.get_adj(vid);
            for (auto n: n_list) {
                float d = Func(data.get_feat(vid), data.get_feat(n), data.dim);
                candidates.push(std::make_pair(d, n));
                auto nn_list = ret.get_adj(n);

                for (auto nn: nn_list) {
                    float dd = Func(data.get_feat(vid), data.get_feat(n), data.dim);
                    candidates.push(std::make_pair(dd, nn));
                }
            }

            auto new_n_list = tmp.get_adj(vid);
            for (int i = 0; i < d_max; i++){
                const auto [d, n] = candidates.top();
                candidates.pop();
                new_n_list[i] = n;
            }
        }
    });
    ret = std::move(tmp);
  }

  return std::move(ret);
};

template <>
FixedDegreeGraph<uint32_t>
build_nnd<DistanceType::L1, uint32_t, float>(int, int, const Matrix2D<float> &);
template <>
FixedDegreeGraph<uint32_t>
build_nnd<DistanceType::L2, uint32_t, float>(int, int, const Matrix2D<float> &);
template <>
FixedDegreeGraph<uint32_t>
build_nnd<DistanceType::Ip, uint32_t, float>(int, int, const Matrix2D<float> &);
// template <typename id_t, typename data_t>
// CSRGraph<id_t> build_undirected(id_t v_num, id_t d_max, id_t dim, data_t
// *data);

// template <typename id_t>
// CSRGraph<id_t> refine_undirected(const std::vector<id_t> &indices, id_t
// v_num);
} // namespace ann
