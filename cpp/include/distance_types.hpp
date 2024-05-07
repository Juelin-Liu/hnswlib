#pragma once

namespace ann
{
    typedef enum class DistanceType
    {
        L2 = 0,           // (x - y)^2
        L1 = 1,           // |x - y|
        Cosine = 2,       // (x . y) / (|x| * |y|)
        InnerProduct = 3, // (x . y)
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
        case DistanceType::InnerProduct:
            // Similarity metrics have the opposite meaning, i.e. nearest neighbors are those with larger
            select_min = false;
            break;
        default:
            select_min = true;
        }
        return select_min;
    }
}