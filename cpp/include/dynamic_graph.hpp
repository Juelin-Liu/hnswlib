#pragma once
#include "util.hpp"
#include <span>

namespace ann 
{

struct DynamicGraph {
  tbb::concurrent_map<id_t, std::span<id_t>> vid_to_adj;

  bool empty() const {
    return vid_to_adj.empty();
  }

  bool Contains(id_t vid) {return vid_to_adj.contains(vid);};

  void AddAdj(id_t vid, std::span<id_t> adj) {
    vid_to_adj[vid] = adj;
  };

  std::span<id_t> GetAdj(id_t vid) const {
    auto itr = vid_to_adj.find(vid);
    if (itr != vid_to_adj.end()) return itr->second;
    return std::span<id_t>{};
  };
};
}
