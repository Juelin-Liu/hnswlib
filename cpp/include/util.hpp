#pragma once
#include <cstdint>
#include <cstddef>

namespace ann
{
    #define __force_inline__ inline __attribute__((always_inline))

    #ifdef __clang__
        #define float16 __fp16
    #elifdef __GNUC__
        #define float16 _Float16
    #endif

    // class E4M3
    // {
    //     private:
    //     uint8_t data;
    // };

    // class E5M2 {
    //     private:
    //     uint8_t data;
    // };

    template<typename data_t>
    struct Matrix2D
    {
        int64_t num_elem{0};
        int64_t dim{0};
        data_t* data{nullptr};
        data_t* get_feat(int64_t vid) {
            return data + vid * dim;
        }
    };

    enum class DistanceType
    {
        L2 = 0,           // (x - y)^2
        L1 = 1,           // |x - y|
        Cosine = 2,       // (x . y) / (|x| * |y|)
        Ip = 3, // (x . y)
        Hamming = 4       // popcnt(x ^ y)
    };

    /**
     * Whether minimal distance corresponds to similar elements (using the given metric).
     */
    inline bool is_min_close(DistanceType metric)
    {
        bool select_min;
        switch (metric)
        {
        case DistanceType::Ip:
            // Similarity metrics have the opposite meaning, i.e. nearest neighbors are those with larger
            select_min = false;
            break;
        default:
            select_min = true;
        }
        return select_min;
    }
}