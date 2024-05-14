#include "knn.hpp"
#include "nnd.hpp"
#include "test_util.hpp"
#include "util.hpp"
#include <gtest/gtest.h>

TEST(NND, L2Float32) {
  using namespace ann;
  using data_t = float;
  using dist_t = float;
  using id_t = uint32_t;
  int v_num = 100;
  int d_max = 20;
  int num_iteration = 15;
  constexpr DistanceType distance_type = DistanceType::L2;
  auto vec = generateRandomVector<data_t>(v_num * ArraySize, -1024, 1024);

  auto data = Matrix2D(v_num, ArraySize, vec.data());
  auto pred =
      build_nnd<id_t, data_t, distance_type>(num_iteration, d_max, data);
  auto gt =
      knn_brute_force<id_t, data_t, distance_type>(num_iteration, d_max, data);
  EXPECT_EQ(pred.indices, gt.indices);
  if (pred != gt) {
    std::cout << "recall:" << gt.recall(pred) << std::endl;
  } else {
    std::cout << "recall:" << 1 << std::endl;
  }
}