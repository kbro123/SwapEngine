#pragma once
// Two-region instantaneous-forward curve (CLAUDE.md §2): piecewise-flat forwards on the CB
// meeting-date front, natural cubic spline on the forwards in the back, level-continuous at the join.
//
// As of the multi-region refactor this is a THIN WRAPPER over the general engine
// MultiRegionCurve<Scalar, Flat, NaturalCubic> -- verified bit-identical to the previous hand-written
// implementation (tests/multi_region_test.cpp). It keeps the specific constructor and the
// n_front()/n_back()/join_time() accessors so the rest of the engine is unchanged; new curves use
// MultiRegionCurve directly with whatever linear region policies they need.

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "swaps/curve/multi_region_curve.hpp"
#include "swaps/curve/regions.hpp"

namespace swaps::curve {

template <class Scalar>
class TwoRegionForwardCurve : public MultiRegionCurve<Scalar, Flat, NaturalCubic> {
  using Base = MultiRegionCurve<Scalar, Flat, NaturalCubic>;

 public:
  TwoRegionForwardCurve(std::vector<double> meeting_times, std::vector<double> back_times)
      : Base(Flat<Scalar>(meeting_times), NaturalCubic<Scalar>(back_times)),
        n_front_(static_cast<int>(meeting_times.size())),
        n_back_(static_cast<int>(back_times.size())),
        join_(meeting_times.empty() ? 0.0 : meeting_times.back()) {
    if (meeting_times.empty() || back_times.empty())
      throw std::invalid_argument("curve: need >=1 knot per region");
    if (!std::is_sorted(meeting_times.begin(), meeting_times.end()))
      throw std::invalid_argument("meeting_times unsorted");
    if (!std::is_sorted(back_times.begin(), back_times.end()))
      throw std::invalid_argument("back_times unsorted");
    if (meeting_times.front() <= 0.0) throw std::invalid_argument("meeting_times must be > 0");
    if (back_times.front() <= meeting_times.back())
      throw std::invalid_argument("back_times must start after the join");
  }

  int n_front() const { return n_front_; }
  int n_back() const { return n_back_; }
  double join_time() const { return join_; }
  // n_knots(), max_time(), set_forwards(), forward(), integral(), discount(), zero() are inherited.

 private:
  int n_front_, n_back_;
  double join_;
};

}  // namespace swaps::curve
