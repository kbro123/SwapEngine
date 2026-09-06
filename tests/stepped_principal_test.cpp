// Priority-3 gate: the OPTIONAL booked structure added to portfolio::MultiCurveBook::Position -- a per-coupon
// STEPPED fixed rate (fixed_rates) and PRINCIPAL-EXCHANGE cashflows (principal_flows) -- plus the guarantee
// that these ride the templated fallback so the compiled hot path (nf_ = notional⊙fixed_rate scalar) is
// untouched. Assertions: (1) a constant stepped schedule reproduces the scalar fixed_rate; (2) a genuinely
// stepped schedule moves the NPV; (3) principal exchange adds exactly Σ amount·DF; (4) CompiledMultiCurveBook
// routes stepped/principal positions to the fallback AND its npv still equals the templated value. Hand-built
// bundle mirroring compiled_multi_test.cpp. QuantLib-free (swaps_tests).
#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <utility>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/portfolio/compiled_multi.hpp"
#include "swaps/portfolio/portfolio.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace pf = swaps::portfolio;

namespace {
const std::vector<double> kMeeting{0.5};
const std::vector<double> kBack{1.0, 2.0, 3.0, 5.0, 10.0};
constexpr int kNk = 6;

px::FloatCoupon ois_coupon(double a, double b) {
  px::FloatCoupon c;
  c.obs.sub_start = {a};
  c.obs.sub_end = {b};
  c.obs.tau_index = b - a;
  c.pay = b;
  c.tau_pay = b - a;
  return c;
}
px::FixedCoupon fixed_coupon(double a, double b) {
  px::FixedCoupon x;
  x.pay = b;
  x.tau = b - a;
  return x;
}
std::vector<px::FloatCoupon> annual_float(double T) {
  std::vector<px::FloatCoupon> v;
  double prev = 0.0;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) { v.push_back(ois_coupon(prev, t)); prev = t; }
  return v;
}
std::vector<px::FixedCoupon> annual_fixed(double T) {
  std::vector<px::FixedCoupon> v;
  double prev = 0.0;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) { v.push_back(fixed_coupon(prev, t)); prev = t; }
  return v;
}

std::vector<px::CurveStructure> make_curves() {
  std::vector<px::CurveStructure> v;
  v.push_back({.base = -1, .regions = swaps::curve::flat_hermite(kMeeting, kBack)});
  v.push_back({.base = 0, .regions = swaps::curve::flat_hermite(kMeeting, kBack)});
  return v;
}
constexpr int kNx = 2 * kNk;
Eigen::VectorXd base_state() {
  Eigen::VectorXd x(kNx);
  for (int i = 0; i < kNk; ++i) {
    x[i] = 0.030 + 0.0010 * i;
    x[kNk + i] = 0.0030 + 0.0002 * i;
  }
  return x;
}

pf::MultiCurveBook::Position swap_position(double T, int fc, int dc, double rate, double notional) {
  pf::MultiCurveBook::Position p;
  p.kind = pf::MultiCurveBook::Kind::Swap;
  p.notional = notional;
  p.float_coupons = annual_float(T);
  p.fwd_curve = fc;
  p.disc_curve = dc;
  p.fixed_coupons = annual_fixed(T);
  p.fixed_curve = dc;
  p.fixed_rate = rate;
  return p;
}

// Templated single-position value at x (the reference kernel BundleSession prices).
double value_of(const std::vector<px::CurveStructure>& specs, const pf::MultiCurveBook::Position& p,
                const Eigen::VectorXd& x) {
  pf::MultiCurveBook book;
  book.positions.push_back(p);
  std::vector<int> off(specs.size(), 0);
  for (std::size_t c = 1; c < specs.size(); ++c) off[c] = off[c - 1] + specs[c - 1].n_knots();
  const auto C = cal::build_bundle_curves<double>(specs, [&](int c, int i) { return x[off[c] + i]; });
  const auto cof = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  return book.value<double>(cof);
}
double disc(const std::vector<px::CurveStructure>& specs, int curve, double t, const Eigen::VectorXd& x) {
  std::vector<int> off(specs.size(), 0);
  for (std::size_t c = 1; c < specs.size(); ++c) off[c] = off[c - 1] + specs[c - 1].n_knots();
  const auto C = cal::build_bundle_curves<double>(specs, [&](int c, int i) { return x[off[c] + i]; });
  return C[curve]->discount(t);
}
}  // namespace

// (1) A CONSTANT stepped schedule reproduces the scalar fixed_rate (FP summation order => ~1e-13 relative).
TEST(SteppedPrincipal, ConstantScheduleEqualsScalarRate) {
  const auto specs = make_curves();
  const Eigen::VectorXd x = base_state();
  const double r = 0.027;

  pf::MultiCurveBook::Position scalar = swap_position(5.0, 1, 0, r, 1e7);
  pf::MultiCurveBook::Position stepped = scalar;
  stepped.fixed_rates.assign(stepped.fixed_coupons.size(), r);  // all equal to the scalar

  const double a = value_of(specs, scalar, x);
  const double b = value_of(specs, stepped, x);
  EXPECT_LE(std::abs(a - b), 1e-12 * (std::abs(a) + 1.0)) << "constant stepped must match scalar rate";
}

// (2) A genuinely stepped schedule moves the NPV vs the scalar-rate swap.
TEST(SteppedPrincipal, VaryingScheduleMovesNpv) {
  const auto specs = make_curves();
  const Eigen::VectorXd x = base_state();

  pf::MultiCurveBook::Position scalar = swap_position(5.0, 1, 0, 0.027, 1e7);
  pf::MultiCurveBook::Position stepped = scalar;
  stepped.fixed_rates.clear();
  for (std::size_t i = 0; i < stepped.fixed_coupons.size(); ++i)
    stepped.fixed_rates.push_back(0.020 + 0.003 * double(i));  // a rising step-up

  EXPECT_GT(std::abs(value_of(specs, stepped, x) - value_of(specs, scalar, x)), 1.0);
}

// (3) Principal exchange adds exactly Σ amount·DF(time) on the discount curve.
TEST(SteppedPrincipal, PrincipalExchangeAddsNotionalFlows) {
  const auto specs = make_curves();
  const Eigen::VectorXd x = base_state();
  const double N = 1e7;

  pf::MultiCurveBook::Position base = swap_position(5.0, 1, 0, 0.026, N);
  pf::MultiCurveBook::Position withpx = base;
  withpx.principal_flows = {{0.0, 1.0}, {5.0, -1.0}};  // +N initial, -N final, per unit notional

  const double expect_delta = N * (1.0 * disc(specs, 0, 0.0, x) + (-1.0) * disc(specs, 0, 5.0, x));
  const double got_delta = value_of(specs, withpx, x) - value_of(specs, base, x);
  EXPECT_LE(std::abs(got_delta - expect_delta), 1e-8 * (std::abs(expect_delta) + 1.0));
}

// (4) The compiled book routes stepped/principal positions to the FALLBACK, and npv still matches the
// templated value -- proving the hot path stays a scalar-nf_ book while the new structure prices correctly.
TEST(SteppedPrincipal, CompiledRoutesToFallbackAndMatches) {
  const auto specs = make_curves();
  const Eigen::VectorXd x = base_state();

  pf::MultiCurveBook book;
  book.positions.push_back(swap_position(5.0, 1, 0, 0.027, 1e7));  // plain -> compiled
  pf::MultiCurveBook::Position stepped = swap_position(5.0, 1, 0, 0.027, 1e7);
  for (std::size_t i = 0; i < stepped.fixed_coupons.size(); ++i)
    stepped.fixed_rates.push_back(0.020 + 0.003 * double(i));
  book.positions.push_back(stepped);  // stepped -> fallback
  pf::MultiCurveBook::Position withpx = swap_position(3.0, 1, 0, 0.025, 5e6);
  withpx.principal_flows = {{0.0, 1.0}, {3.0, -1.0}};
  book.positions.push_back(withpx);  // principal exchange -> fallback

  EXPECT_FALSE(pf::CompiledMultiCurveBook::swap_is_compilable(stepped));
  EXPECT_FALSE(pf::CompiledMultiCurveBook::swap_is_compilable(withpx));
  EXPECT_TRUE(pf::CompiledMultiCurveBook::swap_is_compilable(book.positions[0]));

  const pf::CompiledMultiCurveBook cmb(specs, book);
  EXPECT_EQ(cmb.n_fallback(), 2);
  EXPECT_EQ(cmb.n_compiled(), 1);

  // Compiled total (compiled + fallback halves) == templated book value.
  pf::MultiCurveBook whole = book;
  std::vector<int> off(specs.size(), 0);
  for (std::size_t c = 1; c < specs.size(); ++c) off[c] = off[c - 1] + specs[c - 1].n_knots();
  const auto C = cal::build_bundle_curves<double>(specs, [&](int c, int i) { return x[off[c] + i]; });
  const auto cof = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  const double want = whole.value<double>(cof);
  const double got = cmb.npv(x);
  EXPECT_LE(std::abs(got - want), 1e-10 * (std::abs(want) + 1.0));
}
