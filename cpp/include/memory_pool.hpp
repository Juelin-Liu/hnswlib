#pragma once
#include <stdint.h>
#include <vector>

namespace ann {
class MemoryPool {
public:
  MemoryPool() = default;
  ~MemoryPool();
  void *malloc(uint64_t num_bytes, uint64_t alignment = 64);
  void *cached_malloc(uint64_t num_bytes, uint64_t alignment = 64);
  void free(void *ptr);
  void release(void *ptr);
  static MemoryPool &Global(); // get thread local memory pool

  int64_t total_bytes() { return num_bytes; };
  int64_t total_calls() { return num_calls; };

private:
  struct Container {
    void *ptr{nullptr};
    size_t num_bytes{0};
    Container() = default;
    Container(size_t _num_bytes, void *_ptr)
        : num_bytes{_num_bytes}, ptr{_ptr} {};
  };

  int64_t num_calls{0};
  int64_t num_bytes{0};
  std::vector<Container> allocated_ptrs;
  std::vector<Container> free_ptrs;
};

} // namespace ann