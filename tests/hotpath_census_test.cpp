// E5 taxonomy: T4 hot-path invariant (every streaming stage is reached by a committed scenario, within that scenario's allocation and factorisation pins)
// HOT-PATH CENSUS (2026-09-15). A streaming tick stamps each stage it passes through into StreamTick::stages (StreamStage, streaming.hpp).
// This test drives one committed scenario per stage and checks it against tests/hotpath_census.lock:
//   1. every StreamStage has a lock row (a new stage cannot land without naming the scenario that reaches it);
//   2. every row's scenario still reaches its stage (a scenario that silently stops exercising a stage is a lost premise);
//   3. every scenario stays within its allocation and factorisation pins (they may only DECREASE).
// tools/check_hotpath_census.py locks the streamer's refresh / factorisation CALL SITES, so a new site fails verify.sh until it gets a stage,
// a scenario and a pin. Set SWAPS_CENSUS_PRINT=1 to print the measured table in the lock's format instead of checking.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "malloc_count.hpp"  // bench/fixtures (an include dir of swaps_tests)
#include "shape_ladder.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/regularize.hpp"
#include "swaps/calibration/streaming.hpp"
#include "swaps/curve/curve_module.hpp"

namespace cal = swaps::calibration;
namespace cv = swaps::curve;
namespace px = swaps::pricing;
using swaps::shapes::Shape;
using SC = cal::StreamingCalibrator<cal::BundleProblem>;
using swaps::testing::AllocScope;
using Stage = cal::StreamStage;

namespace {

struct Outcome {
  unsigned stages = 0;
  unsigned long allocs = 0;
  int factors = 0;
  bool counted = true;  // false: threads involved, the calling-thread allocation count is not the scenario's cost
};

const Shape& rung(const char* name) {
  static const std::vector<Shape> L = swaps::shapes::ladder();
  for (const Shape& s : L)
    if (s.name == name) return s;
  throw std::invalid_argument(std::string("no rung ") + name);
}

const Eigen::VectorXd& calibrated(const Shape& s) {
  static std::map<std::string, Eigen::VectorXd> cache;
  auto it = cache.find(s.name);
  if (it == cache.end()) it = cache.emplace(s.name, cal::calibrate(s.prob, s.x0).x).first;
  return it->second;
}

SC::Options pinned() {
  SC::Options o;
  o.breakeven_steps = 64;
  return o;
}

std::vector<Eigen::VectorXd> walk_sequence(const Shape& s, double amp, int n) {
  std::vector<Eigen::VectorXd> seq;
  for (int k = 0; k < n; ++k) {
    Eigen::VectorXd q = s.q0;
    for (int i = 0; i < q.size(); ++i) {
      const auto& in = s.prob.instruments[static_cast<std::size_t>(i)];
      if (in.band_upper > in.band_lower || in.quote == cal::QuoteKind::FxForward || in.quote == cal::QuoteKind::TurnJump) continue;
      q[i] += amp * std::sin(0.8 * k + 0.9 * i);
    }
    seq.push_back(q);
  }
  return seq;
}

// Runs `ticks` (after `warm` uncounted ones) and ORs the stages of the counted ones.
template <class Tick>
Outcome run(SC& st, int warm, int ticks, Tick tick) {
  for (int k = 0; k < warm; ++k) tick(k);
  Outcome o;
  const int f0 = st.factor_count();
  AllocScope a;
  for (int k = warm; k < warm + ticks; ++k) o.stages |= tick(k).stages;
  o.allocs = a.allocs();
  o.factors = st.factor_count() - f0;
  return o;
}

Outcome walk_desk_3bp() {
  const Shape& s = rung("desk");
  cal::HybridBundleResidual eng(s.prob);
  SC st(eng, s.prob, calibrated(s), s.q0, pinned());
  const auto seq = walk_sequence(s, 3e-4, 26);
  return run(st, 6, 20, [&](int k) { return st.update(seq[static_cast<std::size_t>(k)]); });
}

Outcome square_big_ois() {
  const Shape& s = rung("ois_nolag");
  cal::HybridBundleResidual eng(s.prob);
  SC st(eng, s.prob, calibrated(s), s.q0, pinned());
  return run(st, 2, 10, [&](int k) { return st.update(k % 2 == 0 ? s.q_big : s.q0); });
}

Outcome intick_portfolio_rd0() {
  const Shape& s = rung("portfolio");
  cal::HybridBundleResidual eng(s.prob);
  SC::Options o = pinned();
  o.refresh_drift = 0.0;
  SC st(eng, s.prob, calibrated(s), s.q0, o);
  return run(st, 4, 10, [&](int k) { return st.update(k % 2 == 0 ? s.q_small : s.q0); });
}

Outcome reg_desk_small() {
  const Shape& s = rung("desk");
  std::vector<int> curves(s.prob.curves.size());
  for (std::size_t c = 0; c < curves.size(); ++c) curves[c] = static_cast<int>(c);
  SC::Options o = pinned();
  o.regularizer = cal::second_difference_operator(s.prob, 0.02, curves);
  cal::HybridBundleResidual eng(s.prob);
  SC st(eng, s.prob, calibrated(s), s.q0, o);
  return run(st, 6, 20, [&](int k) { return st.update(k % 2 == 0 ? s.q_small : s.q0); });
}

Outcome failed_portfolio_cap2() {
  const Shape& s = rung("portfolio");
  cal::HybridBundleResidual eng(s.prob);
  SC::Options o = pinned();
  o.max_steps = 2;
  SC st(eng, s.prob, calibrated(s), s.q0, o);
  return run(st, 0, 2, [&](int k) { return st.update(k == 0 ? s.q_big : s.q0); });
}

Outcome frozen_cap_ois() {
  const Shape& s = rung("ois_nolag");
  cal::HybridBundleResidual eng(s.prob);
  SC::Options o = pinned();
  o.adaptive_stall = false;
  o.max_frozen = 1;
  SC st(eng, s.prob, calibrated(s), s.q0, o);
  return run(st, 2, 10, [&](int k) { return st.update(k % 2 == 0 ? s.q_big : s.q0); });
}

Outcome desk_mixed_small() {
  const Shape& s = rung("desk_mixed");
  cal::HybridBundleResidual eng(s.prob);
  SC st(eng, s.prob, calibrated(s), s.q0, pinned());
  return run(st, 6, 20, [&](int k) { return st.update(k % 2 == 0 ? s.q_small : s.q0); });
}

Outcome desk_mixed_big() {  // the 25 bp four-number requote each way (bands move with their targets)
  const Shape& s = rung("desk_mixed");
  cal::HybridBundleResidual eng(s.prob);
  SC st(eng, s.prob, calibrated(s), s.q0, pinned());
  const Shape::Requote big = s.requote(s.q_big), base = s.requote(s.q0);
  return run(st, 2, 4, [&](int k) {
    const Shape::Requote& r = k % 2 == 0 ? big : base;
    for (int j = 0; j < s.prob.n_residuals(); ++j) eng.set_quote(j, r.target[j], r.lower[j], r.upper[j], r.decay[j]);
    if (!st.set_bands(r.lower, r.upper, r.decay)) throw std::logic_error("desk_mixed_big: a band move re-anchored");
    return st.update(r.target);
  });
}

// The C1 commit rule's fixture (streaming_near_singular_repro_test): a square 1y/2y/3y strip whose 2y row is banded +-5 bp with decay 1e-5.
cal::Instrument banded_par_swap(double T) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  double prev = 0.0;
  for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
    px::FloatCoupon c;
    c.obs.sub_start = {prev};
    c.obs.sub_end = {u};
    c.obs.tau_index = u - prev;
    c.pay = u;
    c.tau_pay = u - prev;
    ins.fwd.coupons.push_back(c);
    ins.fixed.coupons.push_back({u, u - prev, 1.0});
    prev = u;
  }
  return ins;
}

Eigen::VectorXd model_quotes(const cal::BundleProblem& p, const Eigen::VectorXd& x) {
  const auto C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[p.offset(c) + i]; });
  const auto cof = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  Eigen::VectorXd q(p.n_residuals());
  for (int i = 0; i < p.n_residuals(); ++i) q[i] = cal::instrument_model_quote<double>(p.instruments[i], cof);
  return q;
}

// Anchored OUTSIDE the tiny-decay band (knot 2 lifted 30 bp), solved inside it: the walk meets a weak direction the anchor did not have,
// truncates it, and the commit rule re-anchors at the shared threshold before committing.
Outcome weak_direction_strip() {
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .regions = cv::flat_hermite({}, {1, 2, 3})});
  for (int T = 1; T <= 3; ++T) p.instruments.push_back(banded_par_swap(T));
  Eigen::VectorXd xs(3);
  xs << 0.040, 0.042, 0.044;
  const Eigen::VectorXd q = model_quotes(p, xs);
  for (int i = 0; i < 3; ++i) p.instruments[static_cast<std::size_t>(i)].market = q[i];
  p.instruments[1].band_lower = q[1] - 5e-4;
  p.instruments[1].band_upper = q[1] + 5e-4;
  p.instruments[1].band_decay = 1e-5;
  Eigen::VectorXd x0 = xs;
  x0[1] += 30e-4;
  Eigen::VectorXd q1 = q;
  q1[0] += 0.5e-4;
  q1[2] -= 0.5e-4;
  SC st(p, x0, q, pinned());
  return run(st, 0, 1, [&](int) { return st.update(q1); });
}

Outcome prefetch_ois() {  // the background worker hands over M: a thread, so the allocation count is not the scenario's cost
  const Shape& s = rung("ois_nolag");
  cal::HybridBundleResidual eng(s.prob);
  SC::Options o = pinned();
  o.prefetch = true;
  o.adaptive_stall = false;
  o.max_frozen = 1;  // every tick refreshes, so a ready background M is taken
  SC st(eng, s.prob, calibrated(s), s.q0, o);
  Outcome out;
  out.counted = false;
  for (int k = 0; k < 400 && !(out.stages & (1u << static_cast<unsigned>(Stage::PrefetchTake))); ++k) {
    out.stages |= st.update(k % 2 == 0 ? s.q_big : s.q0).stages;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return out;
}

using Scenario = Outcome (*)();
const std::vector<std::pair<std::string, Scenario>>& scenarios() {  // cheapest first: the print mode names the first that reaches a stage
  static const std::vector<std::pair<std::string, Scenario>> v = {
      {"square_big_ois", square_big_ois},       {"frozen_cap_ois", frozen_cap_ois}, {"intick_portfolio_rd0", intick_portfolio_rd0},
      {"failed_portfolio_cap2", failed_portfolio_cap2}, {"weak_direction_strip", weak_direction_strip}, {"reg_desk_small", reg_desk_small}, {"walk_desk_3bp", walk_desk_3bp},
      {"desk_mixed_small", desk_mixed_small},   {"desk_mixed_big", desk_mixed_big}, {"prefetch_ois", prefetch_ois}};
  return v;
}

std::string lock_path() {
  const std::string f = __FILE__;
  return f.substr(0, f.find_last_of('/')) + "/hotpath_census.lock";
}

struct Lock {
  std::map<std::string, std::string> stage;                      // stage -> scenario | "unreachable"
  std::map<std::string, std::pair<long, long>> pins;             // scenario -> (allocs | -1, factors | -1)
};

Lock read_lock() {
  std::ifstream in(lock_path());
  if (!in) throw std::runtime_error("cannot open " + lock_path());
  Lock L;
  std::string line, section;
  while (std::getline(in, line)) {
    const auto hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    std::istringstream ss(line);
    std::string a, b, c;
    if (!(ss >> a)) continue;
    if (a.front() == '[') { section = a; continue; }
    if (section.empty()) {
      ss >> b;
      L.stage[a] = b;
    } else if (section == "[pins]") {
      ss >> b >> c;
      L.pins[a] = {b == "-" ? -1 : std::stol(b), c == "-" ? -1 : std::stol(c)};
    }
  }
  return L;
}

}  // namespace

TEST(HotpathCensus, EveryStageIsReachedByItsScenarioWithinItsPins) {
  std::map<std::string, Outcome> out;
  for (const auto& [name, fn] : scenarios()) out[name] = fn();
  const unsigned n = static_cast<unsigned>(Stage::kCount);
  if (std::getenv("SWAPS_CENSUS_PRINT")) {
    for (unsigned s = 0; s < n; ++s) {
      std::string first = "unreachable";
      for (const auto& [name, fn] : scenarios())
        if (out[name].stages & (1u << s)) { first = name; break; }
      std::printf("CENSUS-STAGE %s %s\n", cal::to_string(static_cast<Stage>(s)), first.c_str());
    }
    for (const auto& [name, fn] : scenarios())
      std::printf("CENSUS-PIN %s %s %s\n", name.c_str(), out[name].counted ? std::to_string(out[name].allocs).c_str() : "-",
                  out[name].counted ? std::to_string(out[name].factors).c_str() : "-");
    GTEST_SKIP() << "SWAPS_CENSUS_PRINT: printed the measured census, checked nothing";
  }
  const Lock L = read_lock();
  for (unsigned s = 0; s < n; ++s) {
    const std::string stage = cal::to_string(static_cast<Stage>(s));
    const auto row = L.stage.find(stage);
    if (row == L.stage.end()) {
      ADD_FAILURE() << "stage " << stage << " has no census lock row: name the scenario that reaches it (and add one if none does)";
      continue;
    }
    if (row->second == "unreachable") continue;
    const auto sc = out.find(row->second);
    if (sc == out.end()) {
      ADD_FAILURE() << "stage " << stage << ": unknown scenario " << row->second;
      continue;
    }
    EXPECT_TRUE(sc->second.stages & (1u << s)) << "scenario " << row->second << " no longer reaches stage " << stage << " (a lost premise)";
  }
  for (const auto& [stage, scenario] : L.stage) {
    bool known = false;
    for (unsigned s = 0; s < n; ++s) known = known || stage == cal::to_string(static_cast<Stage>(s));
    EXPECT_TRUE(known) << "the census lock names an unknown stage " << stage;
  }
  for (const auto& [name, fn] : scenarios()) {
    const auto p = L.pins.find(name);
    if (p == L.pins.end()) {
      ADD_FAILURE() << "scenario " << name << " has no [pins] row";
      continue;
    }
    const Outcome& o = out[name];
    std::printf("  [census] %-22s stages %05x allocs %lu factorisations %d\n", name.c_str(), o.stages, o.allocs, o.factors);
    if (p->second.second >= 0) EXPECT_LE(o.factors, p->second.second) << name << ": more factorisations than its pin";
    if (p->second.first >= 0 && o.counted && swaps::testing::alloc_counting_available())
      EXPECT_LE(static_cast<long>(o.allocs), p->second.first) << name << ": more allocations than its pin";
  }
}
