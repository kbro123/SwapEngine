// Proves the real-time reprice/residual hot path is ALLOCATION-FREE: after a warmup call has sized
// every scratch buffer, no further Eigen heap allocation occurs per tick. Uses Eigen's own real-time
// facility -- EIGEN_RUNTIME_NO_MALLOC + set_is_malloc_allowed(false) makes Eigen assert on ANY malloc
// while disabled. We route that assert to a catchable exception so a violation is a clean test failure
// (and names the deliverable of the single-threaded Workspace: zero per-tick allocation => jitter-free).
//
// Built as a SEPARATE executable: this TU compiles Eigen with the malloc guard, which must not ODR-clash
// with the other test TUs that compile Eigen normally.

#define EIGEN_RUNTIME_NO_MALLOC
#include <stdexcept>
#include <string>
// Fired when Eigen attempts a disallowed malloc (is_malloc_allowed() == false) -- throw instead of abort.
#define eigen_assert(X)                                              \
  do {                                                              \
    if (!(X)) throw std::runtime_error("eigen assert/malloc: " #X); \
  } while (0)

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <vector>

#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/problem.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;

namespace {

// An annual OIS as a generic instrument (one telescoped sub-period per coupon).
cal::Instrument annual(double T, int fc, int dc, cal::QuoteKind kind, int bench = 0) {
  cal::Instrument ins;
  ins.quote = kind;
  std::vector<px::FloatCoupon> flt;
  std::vector<px::FixedCoupon> fix;
  double prev = 0.0;
  for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
    px::FloatCoupon c;
    c.obs.sub_start = {prev};
    c.obs.sub_end = {u};
    c.obs.tau_index = u - prev;
    c.pay = u;
    c.tau_pay = u - prev;
    flt.push_back(c);
    fix.push_back({u, u - prev});
    prev = u;
  }
  ins.fwd = {flt, fc, dc};
  ins.fixed = {fix, dc};
  if (kind == cal::QuoteKind::ParSpread) ins.bench = {flt, bench, dc};
  return ins;
}

// A small 2-curve spread bundle (curve 0 outright, curve 1 = spread over 0): par-rate + basis rows.
cal::BundleProblem build() {
  cal::BundleProblem p;
  const std::vector<double> meeting{0.5}, back{1, 2, 3, 5, 10};
  p.curves.resize(2);
  p.curves[0] = {.base = -1, .regions = swaps::curve::flat_hermite(meeting, back)};
  p.curves[1] = {.base = 0, .regions = swaps::curve::flat_hermite(meeting, back)};
  for (double T : {1.0, 2.0, 3.0, 5.0, 10.0}) p.instruments.push_back(annual(T, 0, 0, cal::QuoteKind::ParRate));
  for (double T : {1.0, 2.0, 3.0, 5.0, 10.0})
    p.instruments.push_back(annual(T, 1, 0, cal::QuoteKind::ParSpread, 0));
  return p;
}

}  // namespace

// residuals()/model_rates() -- the per-tick reprice a streaming re-cal drives -- must not allocate.
TEST(AllocFree, CompiledResidualHotPathDoesNotAllocate) {
  const cal::BundleProblem p = build();
  const cal::CompiledBundleResidual eng(p);

  const int nk = 6;  // 1 front + 5 back per curve
  Eigen::VectorXd x(2 * nk);
  for (int i = 0; i < x.size(); ++i) x[i] = (i < nk ? 0.040 + 0.001 * i : 0.0020 + 0.0001 * i);

  // WARMUP: size every scratch buffer (df, per-batch coupon/sub/pv/num/rate/ann, out, res).
  eng.residuals(x);
  eng.model_rates(x);

  bool threw = false;
  std::string err;
  Eigen::internal::set_is_malloc_allowed(false);
  try {
    for (int i = 0; i < 500; ++i) {
      x[0] += (i % 2 ? 1e-9 : -1e-9);  // vary x so DF genuinely recomputes each tick
      const Eigen::VectorXd& r = eng.residuals(x);
      volatile double s = r.sum();  // force the reprice to actually run
      (void)s;
    }
  } catch (const std::exception& e) {
    threw = true;
    err = e.what();
  }
  Eigen::internal::set_is_malloc_allowed(true);
  EXPECT_FALSE(threw) << "hot-path allocation detected: " << err;
}
