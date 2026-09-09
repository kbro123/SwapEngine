#pragma once
// Process-wide allocation counter for the T4 hot-path invariants (PRINCIPLES.md P5): counts EVERY malloc /
// realloc / free in the process through libmalloc's logger hook (macOS), not Eigen's guard alone. On other
// platforms available() is false and the T4 assertions are skipped with a printed reason (never silently green).
#include <cstdint>
#if defined(__APPLE__)
extern "C" {
typedef void(swaps_malloc_logger_t)(uint32_t type, uintptr_t arg1, uintptr_t arg2, uintptr_t arg3, uintptr_t result,
                                    uint32_t num_hot_frames_to_skip);
extern swaps_malloc_logger_t* malloc_logger;
}
#endif
namespace swaps::testing {
struct AllocCounters { unsigned long allocs = 0, bytes = 0, frees = 0; };
inline AllocCounters& alloc_counters() { static AllocCounters c; return c; }
#if defined(__APPLE__)
inline void swaps_alloc_logger(uint32_t type, uintptr_t, uintptr_t a2, uintptr_t a3, uintptr_t, uint32_t) {
  auto& c = alloc_counters();
  if (type & 2) { ++c.allocs; c.bytes += (type & 4) ? a3 : a2; }
  if (type & 4) ++c.frees;
}
inline bool alloc_counting_available() { return true; }
// RAII: arm the hook for the scope; read allocs()/bytes() after (or during) the scope.
struct AllocScope {
  AllocScope() { alloc_counters() = AllocCounters{}; malloc_logger = swaps_alloc_logger; }
  ~AllocScope() { malloc_logger = nullptr; }
  unsigned long allocs() const { return alloc_counters().allocs; }
  unsigned long bytes() const { return alloc_counters().bytes; }
};
#else
inline bool alloc_counting_available() { return false; }
struct AllocScope {
  unsigned long allocs() const { return 0; }
  unsigned long bytes() const { return 0; }
};
#endif
}  // namespace swaps::testing
