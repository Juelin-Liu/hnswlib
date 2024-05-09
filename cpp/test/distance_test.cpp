#include "space.hpp"
#include "test_util.hpp"
#include "util.hpp"
#include <cstdint>
TEST(L2Distance, Float32) {
  using data_t = float;
  using dist_t = float;
  auto func = ann::get_distance<ann::DistanceType::L2, data_t>;
  auto [vec1, vec2] = generateRandomVectors<data_t>(ArraySize, -1024, 1024);
  auto dist = L2DistanceGroundTruth<dist_t, data_t>(vec1.data(), vec2.data(),
                                                    vec1.size());
  auto pred = func(vec1.data(), vec2.data(), vec1.size());
  auto delta = std::abs(dist - pred);
  auto eps = std::abs(dist * 1e-5); // allow 0.01% diviation
  EXPECT_LT(delta, eps);
  if (delta > eps) {
    std::cout << "vec1: " << print(vec1) << std::endl;
    std::cout << "vec2: " << print(vec2) << std::endl;
    std::cout << "diff: " << print_l2(vec1, vec2) << std::endl;
    std::cout << "dist: " << (double)dist << std::endl;
    std::cout << "pred: " << (double)pred << std::endl;
    std::cout << "delta: " << (double)delta << std::endl;
    std::cout << "eps: " << (double)eps << std::endl;
  }
}

TEST(L2Distance, Float16) {
  using data_t = float16;
  using dist_t = float;
  auto func = ann::get_distance<ann::DistanceType::L2, data_t>;
  auto [vec1, vec2] = generateRandomVectors<data_t>(ArraySize, -1024, 1024);
  auto dist = L2DistanceGroundTruth<dist_t, data_t>(vec1.data(), vec2.data(),
                                                    vec1.size());
  auto pred = func(vec1.data(), vec2.data(), vec1.size());
  auto delta = std::abs(dist - pred);
  auto eps = std::abs(dist * 1e-5); // allow 0.01% diviation
  EXPECT_LT(delta, eps);
  if (delta > eps) {
    std::cout << "vec1: " << print(vec1) << std::endl;
    std::cout << "vec2: " << print(vec2) << std::endl;
    std::cout << "diff: " << print_l2(vec1, vec2) << std::endl;
    std::cout << "dist: " << (double)dist << std::endl;
    std::cout << "pred: " << (double)pred << std::endl;
    std::cout << "delta: " << (double)delta << std::endl;
    std::cout << "eps: " << (double)eps << std::endl;
  }
}

TEST(L2Distance, UInt8) {
  using data_t = uint8_t;
  using dist_t = int;
  auto func = ann::get_distance<ann::DistanceType::L2, data_t>;
  auto [vec1, vec2] = generateRandomVectors<data_t>(ArraySize, 0, 255);
  auto dist = L2DistanceGroundTruth<dist_t, data_t>(vec1.data(), vec2.data(),
                                                    vec1.size());
  auto pred = func(vec1.data(), vec2.data(), vec1.size());
  auto delta = std::abs(dist - pred);
  auto eps = std::abs(dist * 1e-5); // allow 0.01% diviation
  EXPECT_LT(delta, eps);
  if (delta > eps) {
    std::cout << "vec1: " << print(vec1) << std::endl;
    std::cout << "vec2: " << print(vec2) << std::endl;
    std::cout << "diff: " << print_l2(vec1, vec2) << std::endl;
    std::cout << "dist: " << (double)dist << std::endl;
    std::cout << "pred: " << (double)pred << std::endl;
    std::cout << "delta: " << (double)delta << std::endl;
    std::cout << "eps: " << (double)eps << std::endl;
  }
}

TEST(L1Distance, Float32) {
  using data_t = float;
  using dist_t = float;
  auto func = ann::get_distance<ann::DistanceType::L1, data_t>;
  auto [vec1, vec2] = generateRandomVectors<data_t>(ArraySize, -1024, 1024);
  auto dist = L1DistanceGroundTruth<dist_t, data_t>(vec1.data(), vec2.data(),
                                                    vec1.size());
  auto pred = func(vec1.data(), vec2.data(), vec1.size());
  auto delta = std::abs(dist - pred);
  auto eps = std::abs(dist * 1e-5); // allow 0.01% diviation
  EXPECT_LT(delta, eps);
  if (delta > eps) {
    std::cout << "vec1: " << print(vec1) << std::endl;
    std::cout << "vec2: " << print(vec2) << std::endl;
    std::cout << "diff: " << print_l2(vec1, vec2) << std::endl;
    std::cout << "dist: " << (double)dist << std::endl;
    std::cout << "pred: " << (double)pred << std::endl;
    std::cout << "delta: " << (double)delta << std::endl;
    std::cout << "eps: " << (double)eps << std::endl;
  }
}

TEST(L1Distance, Float16) {
  using data_t = float16;
  using dist_t = float;
  auto func = ann::get_distance<ann::DistanceType::L1, data_t>;
  auto [vec1, vec2] = generateRandomVectors<data_t>(ArraySize, -1024, 1024);
  auto dist = L1DistanceGroundTruth<dist_t, data_t>(vec1.data(), vec2.data(),
                                                    vec1.size());
  auto pred = func(vec1.data(), vec2.data(), vec1.size());
  auto delta = std::abs(dist - pred);
  auto eps = std::abs(dist * 1e-5); // allow 0.01% diviation
  EXPECT_LT(delta, eps);
  if (delta > eps) {
    std::cout << "vec1: " << print(vec1) << std::endl;
    std::cout << "vec2: " << print(vec2) << std::endl;
    std::cout << "diff: " << print_l2(vec1, vec2) << std::endl;
    std::cout << "dist: " << (double)dist << std::endl;
    std::cout << "pred: " << (double)pred << std::endl;
    std::cout << "delta: " << (double)delta << std::endl;
    std::cout << "eps: " << (double)eps << std::endl;
  }
}

TEST(L1Distance, UInt8) {
  using data_t = uint8_t;
  using dist_t = int;
  auto func = ann::get_distance<ann::DistanceType::L1, data_t>;
  auto [vec1, vec2] = generateRandomVectors<data_t>(ArraySize, 0, 255);
  auto dist = L1DistanceGroundTruth<dist_t, data_t>(vec1.data(), vec2.data(),
                                                    vec1.size());
  auto pred = func(vec1.data(), vec2.data(), vec1.size());
  auto delta = std::abs(dist - pred);
  auto eps = std::abs(dist * 1e-5); // allow 0.01% diviation
  EXPECT_LT(delta, eps);
  if (delta > eps) {
    std::cout << "vec1: " << print(vec1) << std::endl;
    std::cout << "vec2: " << print(vec2) << std::endl;
    std::cout << "diff: " << print_l2(vec1, vec2) << std::endl;
    std::cout << "dist: " << (double)dist << std::endl;
    std::cout << "pred: " << (double)pred << std::endl;
    std::cout << "delta: " << (double)delta << std::endl;
    std::cout << "eps: " << (double)eps << std::endl;
  }
}

TEST(InnerProduct, Float32) {
  using data_t = float;
  using dist_t = float;
  auto func = ann::get_distance<ann::DistanceType::Ip, data_t>;
  auto [vec1, vec2] = generateRandomVectors<data_t>(ArraySize, -1024, 1024);
  auto dist = InnerProductGroundTruth<dist_t, data_t>(vec1.data(), vec2.data(),
                                                    vec1.size());
  auto pred = func(vec1.data(), vec2.data(), vec1.size());
  auto delta = std::abs(dist - pred);
  auto eps = std::abs(dist * 1e-5); // allow 0.01% diviation
  EXPECT_LT(delta, eps);
  if (delta > eps) {
    std::cout << "vec1: " << print(vec1) << std::endl;
    std::cout << "vec2: " << print(vec2) << std::endl;
    std::cout << "diff: " << print_l2(vec1, vec2) << std::endl;
    std::cout << "dist: " << (double)dist << std::endl;
    std::cout << "pred: " << (double)pred << std::endl;
    std::cout << "delta: " << (double)delta << std::endl;
    std::cout << "eps: " << (double)eps << std::endl;
  }
}

TEST(InnerProduct, Float16) {
  using data_t = float16;
  using dist_t = float;
  auto func = ann::get_distance<ann::DistanceType::Ip, data_t>;
  auto [vec1, vec2] = generateRandomVectors<data_t>(ArraySize, -1024, 1024);
  auto dist = InnerProductGroundTruth<dist_t, data_t>(vec1.data(), vec2.data(),
                                                    vec1.size());
  auto pred = func(vec1.data(), vec2.data(), vec1.size());
  auto delta = std::abs(dist - pred);
  auto eps = std::abs(dist * 1e-5); // allow 0.01% diviation
  EXPECT_LT(delta, eps);
  if (delta > eps) {
    std::cout << "vec1: " << print(vec1) << std::endl;
    std::cout << "vec2: " << print(vec2) << std::endl;
    std::cout << "diff: " << print_l2(vec1, vec2) << std::endl;
    std::cout << "dist: " << (double)dist << std::endl;
    std::cout << "pred: " << (double)pred << std::endl;
    std::cout << "delta: " << (double)delta << std::endl;
    std::cout << "eps: " << (double)eps << std::endl;
  }
}

TEST(InnerProduct, UInt8) {
  using data_t = uint8_t;
  using dist_t = int;
  auto func = ann::get_distance<ann::DistanceType::Ip, data_t>;
  auto [vec1, vec2] = generateRandomVectors<data_t>(ArraySize, 0, 255);
  auto dist = InnerProductGroundTruth<dist_t, data_t>(vec1.data(), vec2.data(),
                                                    vec1.size());
  auto pred = func(vec1.data(), vec2.data(), vec1.size());
  auto delta = std::abs(dist - pred);
  auto eps = std::abs(dist * 1e-5); // allow 0.01% diviation
  EXPECT_LT(delta, eps);
  if (delta > eps) {
    std::cout << "vec1: " << print(vec1) << std::endl;
    std::cout << "vec2: " << print(vec2) << std::endl;
    std::cout << "diff: " << print_l2(vec1, vec2) << std::endl;
    std::cout << "dist: " << (double)dist << std::endl;
    std::cout << "pred: " << (double)pred << std::endl;
    std::cout << "delta: " << (double)delta << std::endl;
    std::cout << "eps: " << (double)eps << std::endl;
  }
}