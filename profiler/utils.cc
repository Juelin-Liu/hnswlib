#include "argparse.hpp"
#include "utils.hpp"

namespace profiler {
    HNSWConfig get_hnsw_config(int argc, char *argv[]) {
        HNSWConfig config;
        argparse::ArgumentParser program("HNSW profiler");
        program.add_argument("--space").help("one of l2, ip, or cosine").required();
        program.add_argument("--M").help(" maximum number of outgoing connections in the graph").scan<'i', int>().required();
        program.add_argument("--ef_construction").help("priority queue capacity during the index construction").scan<'i', int>().required();

        program.add_argument("--ef").help("priority queue capacity during the index construction").scan<'i', int>().required();
        program.add_argument("--num_threads").help("capacity of the index").scan<'i', int>().required();
        program.add_argument("--k").help("top k search index").scan<'i', int>().required();

        program.add_argument("--feat_path").help("path to the feature file").required();
        program.add_argument("--index_path").help("path to the graph index file").default_value("");
        program.add_argument("--query_path").help("path to the query feature file").default_value("");
        program.add_argument("--truth_path").help("path to the ground truth file").default_value("");

        program.add_argument("--index_out").help("index output directory").default_value("");
        program.add_argument("--profile_out").help("profile output directory").default_value("/tmp/out.log");
        program.parse_args(argc, argv);

        config.space = program.get<std::string>("--space");
        config.M = program.get<int>("--M");
        config.ef_construction = program.get<int>("--ef_construction");

        config.ef = program.get<int>("--ef");
        config.num_threads = program.get<int>("--num_threads");
        config.k = program.get<int>("--k");

        config.feat_path = program.get<std::string>("--feat_path");
        config.index_path = program.get<std::string>("--index_path");
        config.query_path = program.get<std::string>("--query_path");
        config.truth_path = program.get<std::string>("--truth_path");

        config.index_out = program.get<std::string>("--index_out");
        config.profile_out = program.get<std::string>("--profile_out");

        return config;
    };
} // namespace profiler
