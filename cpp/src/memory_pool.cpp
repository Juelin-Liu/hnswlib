#include "memory_pool.hpp"
#include <oneapi/tbb/scalable_allocator.h>

namespace ann {

MemoryPool::~MemoryPool() {
  for (auto container : allocated_ptrs) {
    free(container.ptr);
  }
  for (auto container : free_ptrs) {
    free(container.ptr);
  }
  allocated_ptrs.clear();
};

void *MemoryPool::malloc(uint64_t num_bytes, uint64_t alignment) {
  void *container = scalable_aligned_malloc(num_bytes, alignment);
  num_bytes += num_bytes;
  num_calls += 1;
  allocated_ptrs.push_back({num_bytes, container});
  return container;
};

void MemoryPool::free(void *ptr) {
  auto ptr_eq = [ptr](const Container &itr) {
    return itr.ptr == ptr;
  };
  auto alloc_itr =
      std::find_if(allocated_ptrs.begin(), allocated_ptrs.end(), ptr_eq);

  allocated_ptrs.erase(alloc_itr);

  scalable_aligned_free(ptr);
};

void *MemoryPool::cached_malloc(uint64_t num_bytes, uint64_t alignment) {
  auto size_ge = [num_bytes](const Container &itr) {
    return itr.num_bytes >= num_bytes;
  };
  auto free_itr = std::find_if(free_ptrs.begin(), free_ptrs.end(), size_ge);

  if (free_itr != free_ptrs.end()) {
    free_ptrs.erase(free_itr);
    allocated_ptrs.push_back(*free_itr);
    return (*free_itr).ptr;
  } else {
    return malloc(num_bytes, alignment);
  }
};

void MemoryPool::release(void *container) {
  auto ptr_eq = [container](const Container &itr) {
    return itr.ptr == container;
  };
  auto alloc_itr =
      std::find_if(allocated_ptrs.begin(), allocated_ptrs.end(), ptr_eq);

  size_t num_bytes = (*alloc_itr).num_bytes;

  auto size_ge = [num_bytes](const Container &itr) {
    return itr.num_bytes >= num_bytes;
  };

  auto free_itr = std::find_if(free_ptrs.begin(), free_ptrs.end(), size_ge);
  free_ptrs.insert(free_itr, *alloc_itr);
  allocated_ptrs.erase(alloc_itr);
};

} // namespace ann