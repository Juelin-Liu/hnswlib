#pragma once

#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <span>
#include <sstream>
#include <type_traits>
#include <utility>
#define ArraySize 128

template <typename dist_t, typename data_t>
inline dist_t L2DistanceGroundTruth(const data_t *pVect1, const data_t *pVect2,
                                    int dim) {
  dist_t dist{0};
  for (int i = 0; i < dim; i++) {
    const dist_t diff = pVect1[i] - pVect2[i];
    dist += diff * diff;
  }

  return dist;
};

template <typename dist_t, typename data_t>
inline dist_t L1DistanceGroundTruth(const data_t *pVect1, const data_t *pVect2,
                                    int dim) {
  dist_t dist{0};
  for (int i = 0; i < dim; i++) {
    const dist_t diff = pVect1[i] - pVect2[i];
    dist += std::abs(diff);
  }

  return dist;
};

template <typename dist_t, typename data_t>
inline dist_t InnerProductGroundTruth(const data_t *pVect1,
                                      const data_t *pVect2, int dim) {
  dist_t dist{0};
  for (int i = 0; i < dim; i++) {
    const dist_t diff = pVect1[i] * pVect2[i];
    dist += diff;
  }

  return dist;
};

template <typename data_t>
std::pair<std::span<data_t>, std::span<data_t>>
generateRandomVectors(size_t arr_len, float min, float max) {
  // Seed the random number generator (optional for better randomness)
  std::random_device rd;
  std::mt19937 gen(rd());
  // Allocate aligned memory for vectors
  data_t *vec1 = (data_t *)std::aligned_alloc(64, arr_len * sizeof(data_t));
  data_t *vec2 = (data_t *)std::aligned_alloc(64, arr_len * sizeof(data_t));

  // Check if allocation was successful
  if (vec1 == nullptr || vec2 == nullptr) {
    throw std::bad_alloc(); // Handle allocation failure (optional)
  }
  //   // Generate random vectors

  if (std::is_floating_point_v<data_t>) {
    auto dis = std::uniform_real_distribution<float>{min, max};
    for (int i = 0; i < arr_len; ++i) {
      vec1[i] = (data_t)dis(gen);
      vec2[i] = (data_t)dis(gen);
    }
  } else {
        auto dis = std::uniform_int_distribution<int>{(int)min, (int)max};

    for (int i = 0; i < arr_len; ++i) {
      vec1[i] = (data_t)dis(gen);
      vec2[i] = (data_t)dis(gen);
    }
  }

  std::span<data_t> r1 = std::span(vec1, arr_len);
  std::span<data_t> r2 = std::span(vec2, arr_len);
  return std::make_pair(r1, r2);
};

template <typename T> std::string print(std::span<T> vec) {
  std::stringstream ss;
  for (auto x : vec) {
    ss << " " << (double)x;
  }
  return ss.str();
};

template <typename T>
std::string print_diff(std::span<T> vec1, std::span<T> vec2) {
  std::stringstream ss;
  for (size_t i = 0; i < vec1.size(); i++) {
    auto dif = vec1[i] - vec2[i];
    ss << " " << (double)dif;
  }
  return ss.str();
};

template <typename T>
std::string print_l2(std::span<T> vec1, std::span<T> vec2) {
  std::stringstream ss;
  for (size_t i = 0; i < vec1.size(); i++) {
    const auto dif = vec1[i] - vec2[i];
    ss << " " << (double)dif * dif;
  }
  return ss.str();
};

template <typename T>
std::string print_l1(std::span<T> vec1, std::span<T> vec2) {
  std::stringstream ss;
  for (size_t i = 0; i < vec1.size(); i++) {
    const auto dif = vec1[i] - vec2[i];
    ss << " " << std::abs((double)dif);
  }
  return ss.str();
};

template <typename T>
std::string print_ip(std::span<T> vec1, std::span<T> vec2) {
  std::stringstream ss;
  for (size_t i = 0; i < vec1.size(); i++) {
    const auto dif = vec1[i] * vec2[i];
    ss << " " << (double)dif;
  }
  return ss.str();
};
