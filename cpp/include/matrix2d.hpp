#pragma once
#include "util.hpp"
#include <span>
namespace ann {

template <typename data_t> struct Matrix2D {
  id_t num_elem{0};
  id_t dim{0};
  data_t *data{nullptr};
  Matrix2D(id_t _num_elem, id_t _dim, data_t *_data)
      : data{_data}, dim{_dim}, num_elem(_num_elem){};
  ~Matrix2D() = default;
  const data_t *get_feat(id_t vid) const { return data + vid * dim; };
  const std::span<data_t> get_feat_span(id_t vid) const {
    return std::span{data + vid * dim, dim};
  };
  data_t *get_feat(id_t vid) { return data + vid * dim; };
  std::span<data_t> get_feat_span(id_t vid) {
    return std::span{data + vid * dim, dim};
  };
};

} // namespace ann