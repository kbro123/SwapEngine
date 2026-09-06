#pragma once
// VEGA LADDER — a swaption book's aggregate sensitivity to the vol surface's PARAMETERS. The vol analogue of
// the rates delta ladder (CLAUDE.md §1.4: analytic bucketed risk): the delta ladder buckets d(book)/d(quote)
// per calibration instrument; this buckets d(book)/d(vol-parameter) per surface CELL.
//
// A vol surface is a grid of CELLS, each a (expiry, tenor) point carrying a vol MODEL — either a single normal
// (Bachelier) vol, or a SABR triple {alpha, rho, nu}. A swaption book is a set of positions, each an
// (expiry, tenor, strike, payer, notional) that maps to ONE cell (its (expiry, tenor)) and prices off that
// cell's forward/annuity/expiry (the CURVE-dependent data, resolved once off the calibrated curve) at its own
// strike. The ladder is the book value's gradient wrt each cell's vol parameter(s):
//
//   NORMAL-VOL cell:  the implied vol is the cell's flat normal_vol at every strike, so
//                     d(V)/d(normal_vol) = Σ_swaptions-in-cell  notional · dV/dσ  (the BACHELIER VEGA).
//                     bachelier.hpp gives dV/dσ analytically, so this bucket is exact.
//   SABR cell:        the implied vol is σ_impl(K) = sabr_normal_vol(F,K,T;α,ρ,ν), so by the chain rule
//                     d(V)/d{α,ρ,ν} = Σ  notional · (dV/dσ)|_{σ=σ_impl} · dσ_impl/d{α,ρ,ν}.
//                     dV/dσ is again the analytic Bachelier vega AT the SABR vol; dσ_impl/d{param} comes from a
//                     TIGHT CENTRAL FINITE-DIFFERENCE of sabr_normal_vol (sabr.hpp exposes only the vol itself,
//                     not analytic param sensitivities — see sabr_normal_vol_param_grad below). SABR is smooth
//                     in its parameters, so a central FD at h≈1e-6 recovers dσ/dparam to ~1e-11.
//
// REUSES the existing pricing — bachelier_price / bachelier_vega / sabr_normal_vol — and reimplements nothing.
// QuantLib-free and header-only (double-only: this is a bucketed sum of analytic vegas, no calibration solve),
// so the api/ seam samples the calibrated curve to fill each cell's forward/annuity/expiry and calls in.
#include <cmath>
#include <vector>

#include "swaps/vol/bachelier.hpp"
#include "swaps/vol/sabr.hpp"

namespace swaps::vol {

// One surface cell: (expiry, tenor) resolved onto the calibrated curve (forward/annuity/expiry) + a vol model.
// `has_sabr` selects the model: SABR {alpha,rho,nu} when true, else the flat `normal_vol`.
struct VegaCell {
  double forward = 0.0;        // F — forward swap rate off the calibrated curve
  double annuity = 0.0;        // A = Σ τ_i·DF_i (the swaption numeraire)
  double expiry_years = 0.0;   // T — option expiry in curve time (years)
  bool has_sabr = false;
  double normal_vol = 0.0;                        // flat Bachelier vol (has_sabr == false)
  double sabr_alpha = 0.0, sabr_rho = 0.0, sabr_nu = 0.0;  // SABR triple (has_sabr == true)
};

// One swaption position, mapped to a cell. It inherits F/A/T from `cells[cell]` (same underlying swap) and
// carries its own strike/payer/notional. `notional` is the signed position size (the ladder scales linearly).
struct VegaSwaption {
  int cell = 0;
  double strike = 0.0;
  bool payer = true;
  double notional = 1.0;
};

// dσ_impl/d{alpha,rho,nu} for the beta=0 SABR normal vol, by a tight CENTRAL finite-difference of
// sabr_normal_vol (sabr.hpp does not expose analytic parameter sensitivities). Central FD is O(h²) and SABR is
// smooth in its parameters, so h = 1e-6 recovers each derivative to ~1e-11. Independent of payer/receiver (the
// SABR vol is a smile, not a payoff). `rho` bumps are clamped just inside (-1, 1) for safety near the boundary.
struct SabrVolParamGrad {
  double d_alpha = 0.0, d_rho = 0.0, d_nu = 0.0;
};

inline SabrVolParamGrad sabr_normal_vol_param_grad(double fwd, double strike, double expiry,
                                                   const SabrParams& p, double h = 1e-6) {
  const auto vol = [&](double a, double r, double n) {
    return sabr_normal_vol<double>(fwd, strike, expiry, a, r, n);
  };
  const double rho_hi = std::min(p.rho + h, 1.0 - 1e-12);
  const double rho_lo = std::max(p.rho - h, -1.0 + 1e-12);
  SabrVolParamGrad g;
  g.d_alpha = (vol(p.alpha + h, p.rho, p.nu) - vol(p.alpha - h, p.rho, p.nu)) / (2.0 * h);
  g.d_rho = (vol(p.alpha, rho_hi, p.nu) - vol(p.alpha, rho_lo, p.nu)) / (rho_hi - rho_lo);
  g.d_nu = (vol(p.alpha, p.rho, p.nu + h) - vol(p.alpha, p.rho, p.nu - h)) / (2.0 * h);
  return g;
}

// The implied normal vol a swaption in `c` sees at `strike` (flat cell vol, or the SABR vol at the strike).
inline double vega_cell_vol(const VegaCell& c, double strike) {
  if (!c.has_sabr) return c.normal_vol;
  return sabr_normal_vol(c.forward, strike, c.expiry_years, SabrParams{c.sabr_alpha, c.sabr_rho, c.sabr_nu});
}

// Total book value = Σ notional · Bachelier(F, K, σ_impl, T, A) — the quantity the ladder differentiates.
// Provided so the api/ layer and the FD cross-check reprice the book with the exact same pricing path.
inline double swaption_book_value(const std::vector<VegaCell>& cells, const std::vector<VegaSwaption>& book) {
  double v = 0.0;
  for (const VegaSwaption& sw : book) {
    const VegaCell& c = cells.at(static_cast<std::size_t>(sw.cell));
    const Payoff cp = sw.payer ? Payoff::Payer : Payoff::Receiver;
    const double sigma = vega_cell_vol(c, sw.strike);
    v += sw.notional * bachelier_price<double>(c.forward, sw.strike, sigma, c.expiry_years, c.annuity, cp);
  }
  return v;
}

// The vega ladder: per-cell buckets of d(book value)/d(vol parameter). For a NORMAL-vol cell only d_normal_vol
// is populated (d_alpha/d_rho/d_nu stay 0); for a SABR cell only d_alpha/d_rho/d_nu are (d_normal_vol stays 0).
// All four vectors have length n_cells. A cell with no swaption keeps every bucket at 0. `book_value` is the
// total book value at the current surface (the point the ladder is a gradient OF).
struct VegaLadder {
  std::vector<double> d_normal_vol;  // per cell (normal-vol cells)
  std::vector<double> d_alpha, d_rho, d_nu;  // per cell (SABR cells)
  double book_value = 0.0;
  int n_cells = 0;
};

inline VegaLadder vega_ladder(const std::vector<VegaCell>& cells, const std::vector<VegaSwaption>& book) {
  VegaLadder L;
  L.n_cells = static_cast<int>(cells.size());
  L.d_normal_vol.assign(cells.size(), 0.0);
  L.d_alpha.assign(cells.size(), 0.0);
  L.d_rho.assign(cells.size(), 0.0);
  L.d_nu.assign(cells.size(), 0.0);

  for (const VegaSwaption& sw : book) {
    const std::size_t ci = static_cast<std::size_t>(sw.cell);
    const VegaCell& c = cells.at(ci);
    const Payoff cp = sw.payer ? Payoff::Payer : Payoff::Receiver;
    const double sigma = vega_cell_vol(c, sw.strike);
    // dV/dσ AT the vol this swaption sees — analytic Bachelier vega (symmetric in payer/receiver).
    const double vega = bachelier_vega<double>(c.forward, sw.strike, sigma, c.expiry_years, c.annuity);
    L.book_value += sw.notional * bachelier_price<double>(c.forward, sw.strike, sigma, c.expiry_years,
                                                          c.annuity, cp);
    if (!c.has_sabr) {
      // σ_impl == normal_vol at every strike -> dσ/d(normal_vol) = 1.
      L.d_normal_vol[ci] += sw.notional * vega;
    } else {
      const SabrParams sp{c.sabr_alpha, c.sabr_rho, c.sabr_nu};
      const SabrVolParamGrad ds = sabr_normal_vol_param_grad(c.forward, sw.strike, c.expiry_years, sp);
      // Chain rule: d(V)/d{param} = (dV/dσ) · (dσ_impl/d{param}).
      L.d_alpha[ci] += sw.notional * vega * ds.d_alpha;
      L.d_rho[ci] += sw.notional * vega * ds.d_rho;
      L.d_nu[ci] += sw.notional * vega * ds.d_nu;
    }
  }
  return L;
}

}  // namespace swaps::vol
