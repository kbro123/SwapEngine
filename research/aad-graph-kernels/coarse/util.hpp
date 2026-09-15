#pragma once
// Timing etiquette for the shared machine: never time while the engine's quiet-machine pipeline is active.
// Hold list (coordinator, 2026-09-15): shape_ladder_bench, check_perf.py, verify.sh, ctest, mutate.py, probe_dm_stall, ninja,
// clang++ (and its clang -cc1 children); plus this study's own other probes (guardtrace, branchprobe) so two of ours never time at
// once, and any extra names in $AADJIT_EXTRA_HOLD (comma-separated). The calling process itself is excluded. Resume only after
// QUIET_S seconds with none of them (the pipeline has idle gaps between steps). Each timing block runs under a watcher; if any
// listed process appears during the block, its numbers are discarded and the block is re-run after the next quiet window.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

namespace util {
inline constexpr int QUIET_S = 120;
inline double load1() { double l[3] = {0, 0, 0}; getloadavg(l, 3); return l[0]; }
inline const std::vector<std::string>& hold_names() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> n = {"shape_ladder_bench", "check_perf.py", "verify.sh", "ctest", "mutate.py", "probe_dm_stall",
                                  "ninja", "clang++", "clang -cc1", "guardtrace", "branchprobe"};
    if (const char* e = std::getenv("AADJIT_EXTRA_HOLD")) {
      std::stringstream ss(e);
      std::string item;
      while (std::getline(ss, item, ',')) if (!item.empty()) n.push_back(item);
    }
    return n;
  }();
  return names;
}
inline bool gate_busy(std::string* which = nullptr) {
  FILE* f = popen("ps -Ao pid=,args=", "r");
  if (!f) return true;  // cannot tell: treat as busy
  const long self = static_cast<long>(getpid());
  bool busy = false;
  char line[8192];
  while (std::fgets(line, sizeof line, f)) {
    char* end = nullptr;
    const long pid = std::strtol(line, &end, 10);
    if (pid == self) continue;
    const std::string args(end ? end : line);
    if (args.find("ps -Ao pid=,args=") != std::string::npos) continue;
    for (const auto& n : hold_names())
      if (args.find(n) != std::string::npos) { busy = true; if (which && which->empty()) *which = n; }
  }
  pclose(f);
  return busy;
}
inline void wait_quiet() {
  auto quiet_since = std::chrono::steady_clock::now();
  bool announced = false;
  for (;;) {
    std::string which;
    if (gate_busy(&which)) {
      if (!announced) { std::fprintf(stderr, "[hold: '%s' running, load %.2f]\n", which.c_str(), load1()); announced = true; }
      quiet_since = std::chrono::steady_clock::now();
    } else if (std::chrono::steady_clock::now() - quiet_since >= std::chrono::seconds(QUIET_S)) {
      if (announced) std::fprintf(stderr, "[resume: %d s quiet, load %.2f]\n", QUIET_S, load1());
      return;
    }
    sleep(10);
  }
}
// Run `block` (which assigns its timings) only in a quiet window, re-running it if the pipeline became active meanwhile.
template <class F> void guarded(F&& block) {
  for (;;) {
    wait_quiet();
    std::atomic<bool> seen{false}, stop{false};
    std::thread w([&] {
      while (!stop) {
        if (gate_busy()) seen = true;
        for (int i = 0; i < 20 && !stop; ++i) usleep(100000);
      }
    });
    block();
    stop = true;
    w.join();
    if (!seen) return;
    std::fprintf(stderr, "[gate activity during a timing block: discarded, re-running after the next quiet window]\n");
  }
}
template <class F> double time_us(F&& f, int reps) {
  std::vector<double> t;
  for (int b = 0; b < 7; ++b) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) f();
    t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps);
  }
  std::sort(t.begin(), t.end());
  return t[3];
}
}  // namespace util
