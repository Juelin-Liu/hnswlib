#include "graph.hpp"

namespace ann
{

    // template instantiation for NSWGraph
    template class NSWGraph<int32_t>; 
    template class NSWGraph<int64_t>; 
    template class NSWGraph<uint32_t>; 
    template class NSWGraph<uint64_t>; 

    // template instantiation for NNDGraph
    template class NNDGraph<int32_t>; 
    template class NNDGraph<int64_t>; 
    template class NNDGraph<uint32_t>; 
    template class NNDGraph<uint64_t>; 
} // namespace ann
