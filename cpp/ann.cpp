#include "space_l2.hpp"
#include <iostream>
#include <random>
#include <span>
#include <sstream>
#include <utility>
#include <vector>

template <typename dist_t, typename data_t, int K = 1>
inline dist_t InnerProductGroundTruth(const data_t *pVect1,
                                      const data_t *pVect2,
                                      int dim)
{
  dist_t dist{0};
#pragma unroll(K)
  for (int i = 0; i < dim; i++)
  {
    const dist_t diff = pVect1[i] - pVect2[i];
    dist += diff * diff;
  }

  return dist;
};

template <typename DataType>
std::pair<std::span<DataType>, std::span<DataType>> generateRandomVectors(size_t arr_len)
{
  // Seed the random number generator (optional for better randomness)
  std::random_device rd;
  std::mt19937 gen(rd());

  // Define range for random values (modify as needed)
  std::uniform_real_distribution<float> dis(0, 256); 
  // std::uniform_real_distribution<DataType> dis(0.0, 1.0); //
  // Allocate aligned memory for vectors
  DataType *vec1 = (DataType *)std::aligned_alloc(64, arr_len * sizeof(DataType));
  DataType *vec2 = (DataType *)std::aligned_alloc(64, arr_len * sizeof(DataType));

  // Check if allocation was successful
  if (vec1 == nullptr || vec2 == nullptr)
  {
    throw std::bad_alloc(); // Handle allocation failure (optional)
  }
  //   // Generate random vectors
  for (int i = 0; i < arr_len; ++i)
  {
    vec1[i] = dis(gen);
    vec2[i] = dis(gen);
  }
  std::span<DataType> r1 = std::span(vec1, arr_len);
  std::span<DataType> r2 = std::span(vec2, arr_len);
  return std::make_pair(r1, r2);
};

template <typename T>
std::string print(std::span<T> vec)
{
  std::stringstream ss;
  for (auto x : vec)
  {
    ss << " " << (double)x;
  }
  return ss.str();
};

template <typename T>
std::string print_diff(std::span<T> vec1, std::span<T> vec2)
{
  std::stringstream ss;
  for (size_t i = 0; i < vec1.size(); i++)
  {
    auto dif = vec1[i] - vec2[i];
    ss << " " << (double)dif;
  }
  return ss.str();
};

template <typename T>
std::string print_l2(std::span<T> vec1, std::span<T> vec2)
{
  std::stringstream ss;
  for (size_t i = 0; i < vec1.size(); i++)
  {
    const auto dif = vec1[i] - vec2[i];
    ss << " " << (double)dif * dif;
  }
  return ss.str();
};

int main(int argc, char const *argv[])
{
  int arr_len = 16;
  auto [vec1, vec2] = generateRandomVectors<uint8_t>(arr_len);
  std::cout << "vec1: " << print(vec1) << std::endl;
  std::cout << "vec2: " << print(vec2) << std::endl;
  std::cout << "diff: " << print_l2(vec1, vec2) << std::endl;

  auto dist = InnerProductGroundTruth<int, uint8_t, 1>(vec1.data(), vec2.data(), vec1.size());
  auto pred = ann::InnerProduct(vec1.data(), vec2.data(), vec1.size());
  auto delta = dist - pred;
  std::cout << "dist: " << (double)dist << std::endl;
  std::cout << "pred: " << (double)pred << std::endl;
  std::cout << "delta: " << (double)delta << std::endl;


  return 0;
}
