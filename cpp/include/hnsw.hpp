#pragma once
#include "graph.hpp"
#include "matrix2d.hpp"
#include "util.hpp"
#include "memory_pool.hpp"

#include <atomic>
#include <mutex>
#include <oneapi/tbb/concurrent_map.h>
#include <queue>
#include <string>
#include <vector>

namespace ann {

struct HNSWBuildConfig {
  id_t ef{0};
  id_t M{0};
  id_t M_base{0};
};

struct HNSWSearchConfig {
  id_t ef{0};
  id_t k{0};
};

struct HNSWLoadconfig {
  std::string index_path;
  std::string data_path;
  bool use_mmap_index{false};
  bool use_mmap_data{false};
  bool merge_index_data{false};
};

template <typename data_t, DistanceType distance_type, bool profiling>
class HNSW {
public:
  void SetBuildConfig(HNSWBuildConfig config) {
    build_config = config;
    cum_probs.clear();
    cum_edges.clear();
    dynamic_graphs.clear();
    int num_edges = 0;
    int cur_level = 0;
    double rlogm = 1.0 / log(build_config.M);
    double prob = 0.0;
    while (true) {
      double delta = exp(-1.0 * cur_level / rlogm) * (1 - exp(-1.0 / rlogm));
      if (delta < 1e-9)
        break;
      prob += delta;
      cur_level += 1;
      num_edges += (cur_level == 0) ? build_config.M_base : build_config.M;
      cum_probs.push_back(prob);
      cum_edges.push_back(num_edges);
    }
    n_layers = cum_probs.size();
    dynamic_graphs.resize(n_layers);
  }

  void SetSearchConfig(HNSWSearchConfig config) { search_config = config; }

  void BuildIndex(Matrix2D<data_t> matrix) {
    auto idx2vid = get_random_indices(matrix.num_elem);
    // build indices
    for (id_t i = 0; i < data.num_elem; i++) {
      auto vid = idx2vid.at(i);
      AddPoint(vid, matrix.get_feat_span(vid));
    }
  }

  void AddPoint(id_t vid, std::span<data_t> vdata) {
    int insert_level = get_random_level(cum_probs);
    id_t entry_point = -1;
    int search_level = -1;
    for (int visit_level = n_layers - 1; visit_level >= insert_level; visit_level--) {
      DGraph &g = dynamic_graphs.at(visit_level);
      if (g.empty()) continue;
      else {
        entry_point = g.begin()->first;
        search_level = visit_level;
        break;
      }
    }

    if (entry_point == -1) {
        // didn't find entry point insert directly at this level
        DGraph &g = dynamic_graphs.at(search_level);
        g[vid] = std::span<id_t>();
    }
  };

  std::priority_queue<std::pair<float, id_t>>
  SearchKnn(HNSWSearchConfig config, std::span<data_t> query);
  std::priority_queue<std::pair<float, id_t>>
  SearchKnnBatch(HNSWSearchConfig config, const Matrix2D<data_t> &queries);

private:
  std::vector<NSWGraph> graph;
  std::vector<std::mutex> locks;
  int n_layers{0};
  std::vector<double>
      cum_probs; // cummulative probability of insertion at each level
  std::vector<id_t> cum_edges; // cummulative number of edges for a vertex
                               // stored in the corresponding layer
  HNSWBuildConfig build_config;
  HNSWSearchConfig search_config;
  // profiling metric
  std::vector<std::atomic<int64_t>> num_hops;
  std::vector<std::atomic<int64_t>> num_dists;

  using DGraph = tbb::concurrent_map<id_t, std::span<id_t>>;
  std::vector<DGraph> dynamic_graphs;
  std::vector<std::span<data_t>> data;
};

} // namespace ann