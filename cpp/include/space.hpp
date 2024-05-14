#pragma once

#include "space_l1.hpp"
#include "space_l2.hpp"
#include "space_ip.hpp"
#include "util.hpp"
#include <iostream>
#include <stdlib.h>

namespace ann {
    template<DistanceType distance_type, typename data_t>
    __force_inline__ auto get_distance(const data_t * pVect1, const data_t * pVect2, int dim){
        if constexpr(distance_type == DistanceType::L2) {
            return ann::L2Distance(pVect1, pVect2, dim);
        } else if constexpr(distance_type == DistanceType::L1) {
            return ann::L1Distance(pVect1, pVect2, dim);
        } else if constexpr(distance_type == DistanceType::Ip) {
            return ann::InnerProduct(pVect1, pVect2, dim);
        } else {
            std::cerr << "unsupported distance type";
            exit(-1);
        }
    }
    
    template<typename data_t>
    __force_inline__ auto get_distance(DistanceType distance_type, const data_t * pVect1, const data_t * pVect2, int dim){
        if (distance_type == DistanceType::L2) {
            return ann::L2Distance(pVect1, pVect2, dim);
        } else if (distance_type == DistanceType::L1) {
            return ann::L1Distance(pVect1, pVect2, dim);
        } else if (distance_type == DistanceType::Ip) {
            return ann::InnerProduct(pVect1, pVect2, dim);
        } else {
            std::cerr << "unsupported distance type";
            exit(-1);
        }
    }

} // ann end