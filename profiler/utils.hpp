#pragma once

#include <string>
#include <stdint.h>
#include "timer.hpp"

namespace profiler {
    #define ATEN_DIST_TYPE_SWITCH(dist, DistType, ...)          \
    do {                                                     \
        if ((dist) == "l2") {                                \
            typedef  hnswlib::L2Space DistType;                 \
            { __VA_ARGS__ }                                  \
        } else if ((dist) == "ip") {                         \
            typedef  hnswlib::InnerProductSpace DistType;       \
            { __VA_ARGS__ }                                  \
        } else if ((dist) == "cosine") {                     \
            typedef  hnswlib::InnerProductSpace DistType;       \
            { __VA_ARGS__ }                                  \
        } else {                                             \
            spdlog::error("Unsupported distance type {}", dist); \
        }                                                    \
    } while (0)

    struct HNSWConfig {
        // graph construction parameters:
        std::string space; // name of the space (can be one of "l2", "ip", or "cosine").
        int64_t dim; // dimensionality of the space.
        int64_t M; // parameter that defines the maximum number of outgoing connections in the graph.
        int64_t ef_construction; // parameter that controls speed/accuracy trade-off during the index construction.
        int64_t max_elements; //  capacity of the index

        // query time parameters:
        int64_t ef;
        int64_t num_threads;
        int64_t k;

        std::string feat_path;
        std::string index_path;
        std::string query_path;
        std::string truth_path;

        std::string index_out;
        std::string profile_out;
    };

    struct HNSWProfile {
        HNSWConfig config;
    };

    HNSWConfig get_hnsw_config(int argc, char *argv[]);

} // namespace profiler
