// #pragma once
// #include "memory_pool.hpp"
// #include "util.hpp"
// #include <algorithm>
// #include <cstring>
// #include <limits>

// namespace ann {

// template <typename dist_t,  DistanceType distance_type>
// class PriorityQueue {
// private:
//   struct Entry {
//     id_t id{std::numeric_limits<id_t>::max()};
//     dist_t distance{is_min_close(distance_type) ? std::numeric_limits<dist_t>::max() : std::numeric_limits<dist_t>::min() };
//     Entry(id_t _id, dist_t _distance) : id{_id}, distance{_distance} {};

//     bool operator<(const Entry &e) const {
//       if constexpr (is_min_close(distance_type)) {
//         return this->distance < e.distance;
//       } else {
//         return this->distance > e.distance;
//       };
//     };
//   };

//   Entry *data{nullptr};
//   int capacity{0};
//   int num_item{0};

// public:
//   PriorityQueue() = default;

//   ~PriorityQueue() {
//     if (data) {
//       MemoryPool::Global().cached_free(data);
//       capacity = 0;
//       num_item = 0;
//     }
//   }
//   PriorityQueue(int _capacity) {
//     capacity = _capacity;
//     num_item = 0;
//     data = static_cast<Entry *>(MemoryPool::Global().cached_malloc(sizeof(Entry) * capacity));
//   }

//   id_t get_id(int idx) {
//     return data[idx].id;
//   }
  
//   id_t get_distance(int idx) {
//     return data[idx].distance;
//   }

//   void enqueue(id_t id, dist_t distance) {
//     // check if the insert at all

//     const Entry e{id, distance};
//     if (num_item == 0) {
//       data[num_item] = e;
//       return;
//     }

//     if constexpr(is_min_close(distance_type)) {
//         if (num_item == capacity && data[num_item - 1].distance < distance) return;
//     } else {
//         if (num_item == capacity && data[num_item - 1].distance > distance) return;
//     }

//     int idx{0};
//     for (int i = 0; i < num_item; i++){
//       if (e < data[i]) {
//         idx = i;
//         break;
//       };
//     }
//     for (int i = num_item - 1; i > idx; i--){
//       data[i+1] = data[i];
//     }
//     data[idx] = e;
//     num_item = std::min(capacity, num_item + 1);
//   }
  
//   // Swap function
//   friend void swap(PriorityQueue &first, PriorityQueue &second) noexcept {
//     std::swap(first.capacity, second.capacity);
//     std::swap(first.num_item, second.num_item);
//     std::swap(first.data, second.data);
//   }

//   PriorityQueue(PriorityQueue &&other) noexcept { swap(*this, other); }

//   PriorityQueue &operator=(PriorityQueue &&other) noexcept {
//     // Check for self-assignment
//     if (this != &other) {
//       // Release resources from the current object
//       this->~PriorityQueue();
//       // Steal the resources from the source object
//       swap(*this, other);
//     }
//     return *this;
//   }

//   PriorityQueue(const PriorityQueue &other) = delete;
//   PriorityQueue &operator=(const PriorityQueue &other) = delete;
// };
// } // namespace ann