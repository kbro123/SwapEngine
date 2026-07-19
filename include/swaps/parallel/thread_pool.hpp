#pragma once
// A small persistent thread pool -- created ONCE, reused for every fan-out, so the calibration and
// portfolio parallel paths pay thread-creation ONCE instead of per call (std::async spawns a fresh thread
// each task). Header-only, no dependencies beyond the standard library.
//
// The ONE primitive is `parallel_for(count, fn)`: run fn(i) for i in [0,count) across the pool's workers
// and BLOCK until all complete. fn is called concurrently for distinct i, so a caller must write DISJOINT
// state per i (which is exactly how calibrate_staged_parallel and ParallelPortfolio use it -- disjoint
// x-blocks / output segments). Determinism is unaffected: the pool changes WHEN each task runs, never
// WHAT it computes.
//
// Not for NESTED fan-out: a task submitted to a pool must not itself call parallel_for on the SAME pool
// (all workers could block waiting on sub-tasks -> deadlock). Our uses are flat (a block/slice solve is
// sequential internally), so this never arises.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace swaps::parallel {

class ThreadPool {
 public:
  // n <= 0 => hardware_concurrency (min 1). The pool owns n worker threads for its whole lifetime.
  explicit ThreadPool(int n = 0) {
    if (n <= 0) n = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    workers_.reserve(n);
    for (int i = 0; i < n; ++i) workers_.emplace_back([this] { worker_loop(); });
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return static_cast<int>(workers_.size()); }

  // Run fn(i) for i in [0,count) across the workers; blocks until all complete. With an empty pool (or
  // count<=1) it just runs inline on the calling thread. `fn` is copied once and shared by reference to
  // the tasks (it outlives them -- this call blocks), so `fn` must be safe to invoke concurrently.
  template <class Fn>
  void parallel_for(int count, Fn fn) {
    if (count <= 0) return;
    if (workers_.empty() || count == 1) {
      for (int i = 0; i < count; ++i) fn(i);
      return;
    }
    std::atomic<int> remaining{count};
    std::mutex done_m;
    std::condition_variable done_cv;
    {
      std::lock_guard<std::mutex> lk(m_);
      for (int i = 0; i < count; ++i)
        tasks_.push([i, &fn, &remaining, &done_m, &done_cv] {
          fn(i);
          if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> dl(done_m);  // pair with the waiter's lock so the notify is not lost
            done_cv.notify_one();
          }
        });
    }
    cv_.notify_all();
    std::unique_lock<std::mutex> lk(done_m);
    done_cv.wait(lk, [&] { return remaining.load(std::memory_order_acquire) == 0; });
  }

 private:
  void worker_loop() {
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
        if (stop_ && tasks_.empty()) return;
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      task();
    }
  }

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex m_;
  std::condition_variable cv_;
  bool stop_ = false;
};

}  // namespace swaps::parallel
