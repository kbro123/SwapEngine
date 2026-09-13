#pragma once
// calibration/consistent_risk.hpp — ONE book, N curve bundles, ALL risk implied off the FIRST bundle's curve (E7 stage
// 3.7): the `generate_risk` verb's whole computation as calibration-layer library code over any session type meeting
// RiskSession (the verb instantiates it with api::BundleSession; this header never includes api).
//
// The book is priced once on bundle[0]'s calibrated curve C0 (the anchor). Every other bundle is RE-LEVELED onto C0 --
// its quotes are set to the levels C0 implies, so it describes the SAME curve, not its own saved market -- and the
// book's curve risk is re-expressed in that bundle's instrument basis. So NPV and the parallel-curve DV01 are
// properties of C0 and the book, and only the per-instrument decomposition differs. A bundle with more knots than
// quotes is rank-completed by self-quoting its null directions as SYNTHETIC pillars (null_completed_ladder,
// calibration/risk.hpp) rather than smoothed, so the real pillars are not biased by a curvature penalty.
//
// The ladder is in QUOTE units: every real row carries the residual market scale D (FIXED 2026-09-13; banded rows
// were overstated by 1/decay and FX forwards by q·T -- calibration/risk.hpp).

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/diagnostics.hpp"  // CalibrationSession
#include "swaps/calibration/regularize.hpp"
#include "swaps/calibration/risk.hpp"  // null_completed_ladder
#include "swaps/portfolio/portfolio.hpp"

namespace swaps::calibration {

inline constexpr double kBasisPoint = 1e-4;  // the ladder's DV01 unit

// One bundle's risk, expressed in ITS OWN instruments, off the anchor curve.
struct ConsistentBundleRisk {
  Eigen::VectorXd ladder;           // dP/dq over this bundle's REAL calibration instruments (n_residuals)
  Eigen::VectorXd synthetic;        // dP/dq over the self-quoted null pillars (implied free knots), if any
  std::vector<int> synthetic_knot;  // the dominant knot index of each synthetic pillar (for labelling)
  double npv = 0.0;                 // book NPV on the anchor curve
  double pv01 = 0.0;                // parallel-curve DV01 on the anchor curve
  double ladder_dv01 = 0.0;         // 1bp · Σ(ladder + synthetic) — reconciles to pv01 up to the basis gap
  int n_residuals = 0, n_synthetic = 0;
};

// The full result: the shared invariants plus one ConsistentBundleRisk per input bundle (in input order).
struct ConsistentRisk {
  double npv = 0.0, pv01 = 0.0;  // the invariants (from the anchor bundle)
  int n = 0;                     // number of positions
  std::vector<ConsistentBundleRisk> bundles;
};

// The calibration that re-levels an under-determined bundle must be well-posed: the caller's regulariser when it is
// on, else the light preset over every curve. (The RISK operator itself is rank-completed, never smoothed.)
inline RegSpec relevel_calibration_reg(const RegSpec& reg, const BundleProblem& b) {
  if (reg.on()) return reg;
  return smoothing_preset(Smoothing::Light, static_cast<int>(b.curves.size()));
}

template <class S, class Book>
concept RiskSession = CalibrationSession<S> &&
    requires(const S cs, const Book& b, const BundleProblem& p, const Instrument& i) {
      { cs.price_portfolio(b).npv } -> std::convertible_to<double>;
      { cs.price_portfolio(b).pv01 } -> std::convertible_to<double>;
      { cs.price_portfolio_risk(b).curve_grad } -> std::convertible_to<Eigen::VectorXd>;
      { cs.model_quote(i) } -> std::convertible_to<double>;
      { cs.same_curve_set(p) } -> std::convertible_to<bool>;
      { b.positions.size() } -> std::convertible_to<std::size_t>;
    };

template <class Session, class Book>
  requires RiskSession<Session, Book>
ConsistentRisk consistent_risk(const Book& book, const std::vector<BundleProblem>& bundles, const RegSpec& reg) {
  if (bundles.empty()) throw std::invalid_argument("generate_risk: need at least one bundle");

  // Anchor: calibrate bundle[0] to ITS market -> the curve C0 everything is priced on.
  Session anchor(bundles[0]);
  anchor.calibrate(flat_x0(bundles[0]), relevel_calibration_reg(reg, bundles[0]));
  const auto p0 = anchor.price_portfolio(book);

  ConsistentRisk out;
  out.n = static_cast<int>(book.positions.size());
  out.npv = p0.npv;
  out.pv01 = p0.pv01;

  for (std::size_t k = 0; k < bundles.size(); ++k) {
    BundleProblem bk = bundles[k];  // copy: its markets are overwritten to re-level onto C0
    if (k > 0) {
      if (!anchor.same_curve_set(bk))
        throw std::invalid_argument(
            "generate_risk: every bundle must share bundle[0]'s curve set (same currencies / outright-or-"
            "spread, same order) so one book is valid on all and their forwards imply off the anchor.");
      // RE-LEVEL: each quote takes the value the anchor curve C0 implies -> bk now describes C0, not its own
      // saved market. (Skipped for the anchor itself.)
      for (auto& ins : bk.instruments) ins.market = anchor.model_quote(ins);
    }
    Session sk(bk);
    sk.calibrate(flat_x0(bk), relevel_calibration_reg(reg, bk));  // for k>0 the markets are C0-implied -> curve ~ C0

    const auto pk = sk.price_portfolio_risk(book);  // curve_grad = dP/dx_k on C0
    const auto pr = sk.price_portfolio(book);
    const Eigen::MatrixXd J = sk.jacobian(RegSpec{});

    const NullCompletedLadder lad =
        null_completed_ladder(J, pk.curve_grad, residual_market_scale(sk.problem().instruments));
    ConsistentBundleRisk item;
    item.n_residuals = lad.n_residuals;
    item.n_synthetic = static_cast<int>(lad.synthetic_knot.size());
    item.ladder = lad.full.head(lad.n_residuals);
    item.synthetic = lad.full.tail(lad.full.size() - lad.n_residuals);
    item.synthetic_knot = lad.synthetic_knot;
    item.npv = pr.npv;
    item.pv01 = pr.pv01;
    item.ladder_dv01 = kBasisPoint * lad.full.sum();
    out.bundles.push_back(std::move(item));
  }
  return out;
}

// The `generate_risk` request, decoded.
struct ConsistentRiskRequest {
  portfolio::MultiCurveBook book;
  std::vector<BundleProblem> bundles;
  RegSpec reg;  // only makes the RE-LEVELING calibration well-posed (relevel_calibration_reg)
};

template <class Session>
ConsistentRisk consistent_risk(const ConsistentRiskRequest& r) {
  return consistent_risk<Session, portfolio::MultiCurveBook>(r.book, r.bundles, r.reg);
}

}  // namespace swaps::calibration
