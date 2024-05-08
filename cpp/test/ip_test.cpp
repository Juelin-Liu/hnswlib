#include "test_util.hpp"
#include "space_ip.hpp"

TEST(InnerProduct, Float32) {
  int arr_len = 128;

  auto [vec1, vec2] = generateRandomVectors<float>(arr_len, -1024, 1024);
  auto dist = InnerProductGroundTruth<float, float>(vec1.data(), vec2.data(),
                                                  vec1.size());
  auto pred = ann::InnerProduct(vec1.data(), vec2.data(), vec1.size());
  auto delta = std::abs(dist - pred);
  auto eps = std::abs(pred * 1e-5); // allow 0.01% diviation
  EXPECT_LT(delta, eps);
  if (delta > eps) {
    std::cout << "vec1: " << print(vec1) << std::endl;
    std::cout << "vec2: " << print(vec2) << std::endl;
    std::cout << "diff: " << print_l1(vec1, vec2) << std::endl;
    std::cout << "dist: " << (double)dist << std::endl;
    std::cout << "pred: " << (double)pred << std::endl;
    std::cout << "delta: " << (double)delta << std::endl;
    std::cout << "eps: " << (double)eps << std::endl;
  }
}

TEST(InnerProduct, Float16) {
  int arr_len = 128;

  auto [vec1, vec2] = generateRandomVectors<_Float16>(arr_len, -1024, 1024);
  auto dist = InnerProductGroundTruth<float, _Float16>(vec1.data(), vec2.data(),
                                                     vec1.size());
  auto pred = ann::InnerProduct(vec1.data(), vec2.data(), vec1.size());
  auto delta = std::abs(dist - pred);
  auto eps = std::abs(pred * 1e-5); // allow 0.01% diviation
  EXPECT_LT(delta, eps);
  if (delta > eps) {
    std::cout << "vec1: " << print(vec1) << std::endl;
    std::cout << "vec2: " << print(vec2) << std::endl;
    std::cout << "diff: " << print_l1(vec1, vec2) << std::endl;
    std::cout << "dist: " << (double)dist << std::endl;
    std::cout << "pred: " << (double)pred << std::endl;
    std::cout << "delta: " << (double)delta << std::endl;
    std::cout << "eps: " << (double)eps << std::endl;
  }
}

TEST(InnerProduct, UInt8) {
  int arr_len = 128;

  auto [vec1, vec2] = generateRandomVectors<uint8_t>(arr_len, -1024, 1024);
  auto dist = InnerProductGroundTruth<int, uint8_t>(vec1.data(), vec2.data(),
                                                  vec1.size());
  auto pred = ann::InnerProduct(vec1.data(), vec2.data(), vec1.size());
  auto delta = std::abs(dist - pred);
  auto eps = std::abs(pred * 1e-5); // allow 0.01% diviation
  EXPECT_LT(delta, eps);
  if (delta > eps) {
    std::cout << "vec1: " << print(vec1) << std::endl;
    std::cout << "vec2: " << print(vec2) << std::endl;
    std::cout << "diff: " << print_l1(vec1, vec2) << std::endl;
    std::cout << "dist: " << (double)dist << std::endl;
    std::cout << "pred: " << (double)pred << std::endl;
    std::cout << "delta: " << (double)delta << std::endl;
    std::cout << "eps: " << (double)eps << std::endl;
  }
}