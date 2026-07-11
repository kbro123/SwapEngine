#pragma once
// MultiRegionCurve<Scalar, Regions...> -- a forward curve partitioned into an arbitrary compile-time
// sequence of region policies, each linear in its knot values (CLAUDE.md §2). The region TYPES are
// compile-time (fully inlined, no virtual dispatch on the hot path); the knot times/values are
// runtime. Regions are stitched left-to-right with a C0 (level) Boundary handoff by default.
//
// Because every region is a linear map of its values, integral(t) = w(t)*x stays linear in x, so the
// W-cache / analytic-Jacobian / linear warm-recal fast path applies unchanged. `is_linear_map` is the
// AND over the regions and gates that fast path.

#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "swaps/curve/regions.hpp"

namespace swaps::curve {

template <class Scalar, template <class> class... Regions>
class MultiRegionCurve {
 public:
  static constexpr bool is_linear_map = (Regions<Scalar>::is_linear_map && ...);
  static constexpr std::size_t n_regions = sizeof...(Regions);

  explicit MultiRegionCurve(Regions<Scalar>... regions) : regions_(std::move(regions)...) {}

  int n_knots() const {
    int n = 0;
    std::apply([&](const auto&... r) { ((n += r.n_values()), ...); }, regions_);
    return n;
  }
  double max_time() const { return std::get<n_regions - 1>(regions_).t_end(); }

  // x = all knot forwards, region by region. Stitch left-to-right with a C0 boundary handoff.
  template <class Vec>
  void set_forwards(const Vec& x) {
    if (static_cast<int>(x.size()) != n_knots())
      throw std::invalid_argument("set_forwards: wrong size");
    Boundary<Scalar> b{};  // origin: t=0, value/slope/integral = 0
    int off = 0;
    std::apply(
        [&](auto&... r) {
          auto step = [&](auto& region) {
            region.build(x, off, region.n_values(), b);
            off += region.n_values();
            b = region.out();
          };
          (step(r), ...);
        },
        regions_);
  }

  Scalar forward(double t) const {
    return locate(t, [](const auto& r, double u) { return r.forward(u); });
  }
  Scalar integral(double t) const {
    if (t <= 0.0) return Scalar(0.0);
    return locate(t, [](const auto& r, double u) { return r.integral(u); });
  }
  Scalar discount(double t) const {
    using std::exp;
    return exp(-integral(t));
  }
  Scalar zero(double t) const { return t <= 0.0 ? forward(0.0) : integral(t) / t; }

 private:
  // Apply `fn(region, t)` to the FIRST region whose end >= t; the last region handles t beyond it
  // (extrapolation). Compile-time-unrolled over the tuple with a runtime early-out.
  template <class Fn>
  Scalar locate(double t, Fn fn) const {
    Scalar result{};
    bool done = false;
    std::apply(
        [&](const auto&... r) {
          auto check = [&](const auto& region) {
            if (!done && t <= region.t_end()) {
              result = fn(region, t);
              done = true;
            }
          };
          (check(r), ...);
        },
        regions_);
    if (!done) result = fn(std::get<n_regions - 1>(regions_), t);  // beyond the last region
    return result;
  }

  std::tuple<Regions<Scalar>...> regions_;
};

}  // namespace swaps::curve
