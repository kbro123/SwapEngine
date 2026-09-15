// CASHFLOW-TABLE INSTRUMENT IR (research probe). Lowers today's cal::Instrument straight into flat tables -- DF atoms,
// growth sub-periods, coupon rows, MtM rows, fixed (annuity) rows, quote-transform rows, portfolio combinations -- and prices
// them with FIXED kernels (one loop per table), no recording. Parity: exact DF atoms (templated curves) => compare to the
// templated double kernel; W-cache DF atoms => compare to the engine's HybridBundleResidual::model_rates.
// Also: a coverage census of every shape-ladder row against the IR.
#include <Eigen/Core>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "shape_ladder.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/pricing/compiled_book.hpp"
#include "util.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;

namespace ir {
struct Unsupported : std::runtime_error { using std::runtime_error::runtime_error; };

enum class Mode : uint8_t { CmpEmpty, Cmp, Plain, GenEmpty, Gen };
struct Sub { int s, e; double w; };            // growth atom: DF[s]/DF[e] - 1  (w = weight, 1 if unweighted)
struct Coupon {                                 // one floating cashflow row
  Mode mode = Mode::Gen;
  int pay = -1;                                 // DF atom at pay (-1: observation only, e.g. a future)
  int sub0 = 0, sub1 = 0;
  bool weighted = false;
  double konst = 0.0, k = 1.0, rf = 1.0;
};
struct Mtm { int cpn, ex_s, ex_e, rn, rd, sn, sd; double fx_spot, reset_fx; };  // FX-reset notional row
struct Fixed { int pay; double w; };            // annuity row: DF[pay] * (tau*scale)
struct Range { int b = 0, e = 0; bool empty() const { return b == e; } };
enum class Tf : uint8_t { ParRate, ParSpread, ZeroCoupon, Xccy, Rate, Fx, StatePin, Portfolio };
struct Quote {
  Tf tf = Tf::ParRate;
  Range pos, neg, fix, mtm, comps;
  double tau = 0, conv = 0, fx_spot = 1, fx_spot_time = 0, realized = 0, tau_index = 1, rfac = 1;
  bool compounded = false;
  int obs = -1;                                 // Rate: its observation row (a Coupon with pay = -1)
  int fn = -1, fd = -1, fsn = -1, fsd = -1;     // FX atoms at T and at the spot date
  int state = -1;
};
struct Comp { double w; int q; };

struct Features { int compounded = 0, weighted = 0, mtm = 0, fx = 0, zc = 0, portfolio = 0, pin = 0, rate = 0, spread = 0, par = 0; };

struct Table {
  std::vector<std::pair<int, double>> atoms;
  std::map<std::pair<int, double>, int> atom_ix;
  std::vector<Sub> subs;
  std::vector<Coupon> cpns;
  std::vector<Mtm> mtms;
  std::vector<Fixed> fixed;
  std::vector<Quote> quotes;
  std::vector<Comp> comps;
  std::vector<int> row_quote;
  std::vector<double> DF, G, CP, MV, Q;
  Features feat;
  const cal::BundleProblem* prob = nullptr;

  int atom(int c, double t) {
    auto it = atom_ix.find({c, t});
    if (it != atom_ix.end()) return it->second;
    atoms.push_back({c, t});
    return atom_ix[{c, t}] = static_cast<int>(atoms.size()) - 1;
  }
  int coupon(const px::FloatCoupon& c, int fc, int dc, bool no_pay = false) {
    const auto& o = c.obs;
    if (!o.fixing_schedule.empty() && !o.resolved) throw Unsupported("unresolved fixings (input error)");
    if (o.fixing_step > 0.0) throw Unsupported("moment path (quadratic form in x, not DF-based)");
    Coupon cp;
    cp.pay = no_pay ? -1 : atom(dc, c.pay);
    cp.sub0 = static_cast<int>(subs.size());
    cp.weighted = !o.weight.empty();
    for (std::size_t k = 0; k < o.sub_start.size(); ++k)
      subs.push_back({atom(fc, o.sub_start[k]), atom(fc, o.sub_end[k]), cp.weighted ? o.weight[k] : 1.0});
    cp.sub1 = static_cast<int>(subs.size());
    if (cp.weighted) ++feat.weighted;
    if (o.compounded) {
      ++feat.compounded;
      cp.k = c.tau_pay / o.tau_index * c.scale;
      cp.konst = c.spread * o.tau_index - 1.0;
      cp.rf = o.realized_factor;
      cp.mode = o.sub_start.empty() ? Mode::CmpEmpty : Mode::Cmp;
    } else if (o.sub_start.size() == 1 && o.weight.empty() && o.fixing_step == 0.0 && o.realized == 0.0 && c.spread == 0.0 &&
               c.tau_pay == o.tau_index && c.scale == 1.0) {
      cp.mode = Mode::Plain;
    } else {
      cp.k = c.tau_pay / c.obs.tau_index * c.scale;
      cp.konst = c.obs.realized + c.spread * c.obs.tau_index;
      cp.mode = o.sub_start.empty() ? Mode::GenEmpty : Mode::Gen;
    }
    cpns.push_back(cp);
    return static_cast<int>(cpns.size()) - 1;
  }
  Range leg(const cal::FloatLeg& l) {
    Range r{static_cast<int>(cpns.size()), 0};
    for (const auto& c : l.coupons) coupon(c, l.forecast, l.discount);
    r.e = static_cast<int>(cpns.size());
    return r;
  }
  Range fixed_leg(const cal::FixedLeg& l) {
    Range r{static_cast<int>(fixed.size()), 0};
    for (const auto& c : l.coupons) fixed.push_back({atom(l.discount, c.pay), c.tau * c.scale});
    r.e = static_cast<int>(fixed.size());
    return r;
  }
  Range mtm_leg(const cal::FloatLeg& l) {
    Range r{static_cast<int>(mtms.size()), 0};
    for (const auto& c : l.coupons) {
      double s, e;
      if (c.accrual_set) { s = c.accrual_start; e = c.accrual_end; }
      else {
        if (c.obs.sub_start.empty() || c.obs.sub_end.empty()) throw Unsupported("fully-fixed MtM coupon without accrual (input error)");
        s = c.obs.sub_start.front(); e = c.obs.sub_end.back();
      }
      const double reset = c.reset_time >= 0.0 ? c.reset_time : s;
      Mtm mm{coupon(c, l.forecast, l.discount), s >= 0.0 ? atom(l.discount, s) : -1, e >= 0.0 ? atom(l.discount, e) : -1, -1, -1, -1, -1, l.fx_spot, c.reset_fx};
      if (c.reset_fx < 0.0) {
        if ((c.fx_fixing_set && c.fx_fixing_time < 0.0) || reset < 0.0) throw Unsupported("past MtM reset without reset_fx (input error)");
        mm.rn = atom(l.reset_num, reset); mm.rd = atom(l.reset_den, reset);
        if (l.fx_spot_time != 0.0) { mm.sd = atom(l.reset_den, l.fx_spot_time); mm.sn = atom(l.reset_num, l.fx_spot_time); }
      }
      mtms.push_back(mm);
    }
    r.e = static_cast<int>(mtms.size());
    return r;
  }
  int lower(const cal::Instrument& ins) {
    Quote q;
    switch (ins.quote) {
      case cal::QuoteKind::Portfolio: {
        ++feat.portfolio;
        std::vector<Comp> cs;
        for (const auto& c : ins.combination) cs.push_back({c.weight, lower(c.instrument)});
        q.tf = Tf::Portfolio;
        q.comps.b = static_cast<int>(comps.size());
        for (auto& c : cs) comps.push_back(c);
        q.comps.e = static_cast<int>(comps.size());
        break;
      }
      case cal::QuoteKind::Rate: {
        ++feat.rate;
        const auto& o = ins.obs;
        if (o.fixing_step > 0.0) throw Unsupported("moment path future");
        px::FloatCoupon fake;
        fake.obs = o; fake.tau_pay = o.tau_index;
        q.tf = Tf::Rate;
        q.obs = coupon(fake, ins.forecast, ins.forecast, true);
        q.compounded = o.compounded; q.realized = o.realized; q.tau_index = o.tau_index; q.rfac = o.realized_factor; q.conv = ins.convexity;
        break;
      }
      case cal::QuoteKind::TurnJump: {
        ++feat.pin;
        q.tf = Tf::StatePin;
        const auto& p = *prob;
        q.state = p.offset(ins.turn_curve) + p.curves[ins.turn_curve].n_interp_knots() + ins.turn_index;
        break;
      }
      case cal::QuoteKind::ParSpread:
        ++feat.spread;
        q.tf = Tf::ParSpread; q.pos = leg(ins.bench); q.neg = leg(ins.fwd); q.fix = fixed_leg(ins.fixed);
        break;
      case cal::QuoteKind::FxForward:
        ++feat.fx;
        q.tf = Tf::Fx; q.fx_spot = ins.fx_spot; q.fx_spot_time = ins.fx_spot_time;
        q.fn = atom(ins.fx_num, ins.fx_time); q.fd = atom(ins.fx_den, ins.fx_time);
        if (ins.fx_spot_time != 0.0) { q.fsd = atom(ins.fx_den, ins.fx_spot_time); q.fsn = atom(ins.fx_num, ins.fx_spot_time); }
        break;
      case cal::QuoteKind::XccyMtmBasis:
        ++feat.mtm;
        q.tf = Tf::Xccy; q.fix = fixed_leg(ins.fixed); q.pos = leg(ins.fwd); q.neg = leg(ins.bench); q.mtm = mtm_leg(ins.mtm);
        q.fx_spot = ins.mtm.fx_spot; q.fx_spot_time = ins.mtm.fx_spot_time;
        if (ins.mtm.fx_spot_time != 0.0) { q.fsd = atom(ins.mtm.reset_den, ins.mtm.fx_spot_time); q.fsn = atom(ins.mtm.reset_num, ins.mtm.fx_spot_time); }
        break;
      case cal::QuoteKind::ZeroCouponRate:
        ++feat.zc;
        q.tf = Tf::ZeroCoupon; q.pos = leg(ins.fwd); q.fix = fixed_leg(ins.fixed); q.tau = cal::zero_coupon_tau(ins);
        break;
      case cal::QuoteKind::ParRate:
      default:
        ++feat.par;
        q.tf = Tf::ParRate; q.pos = leg(ins.fwd); q.fix = fixed_leg(ins.fixed);
        break;
    }
    quotes.push_back(q);
    return static_cast<int>(quotes.size()) - 1;
  }
  void size_scratch() { DF.assign(atoms.size(), 0.0); G.assign(subs.size(), 0.0); CP.assign(cpns.size(), 0.0); MV.assign(mtms.size(), 0.0); Q.assign(quotes.size(), 0.0); }

  // ---- the fixed kernels (templated evaluation order reproduced term for term) ----
  double num(const Coupon& c) const {
    const double* g = G.data();
    if (c.mode == Mode::Cmp) {
      auto f = [&](int k) { return c.weighted ? 1.0 + subs[k].w * g[k] : 1.0 + g[k]; };
      double prod = f(c.sub0);
      for (int k = c.sub0 + 1; k < c.sub1; ++k) prod = prod * f(k);
      return prod;
    }
    double s = g[c.sub0];
    if (c.weighted) s = s * subs[c.sub0].w;
    for (int k = c.sub0 + 1; k < c.sub1; ++k) { if (c.weighted) s += g[k] * subs[k].w; else s += g[k]; }
    return s;
  }
  double leg_pv(Range r) const {
    if (r.empty()) return 0.0;
    double pv = CP[r.b];
    for (int i = r.b + 1; i < r.e; ++i) pv += CP[i];
    return pv;
  }
  double annuity(Range r) const {
    if (r.empty()) return 0.0;
    double a = DF[fixed[r.b].pay] * fixed[r.b].w;
    for (int i = r.b + 1; i < r.e; ++i) a += DF[fixed[i].pay] * fixed[i].w;
    return a;
  }
  double mtm_pv(Range r) const {
    if (r.empty()) return 0.0;
    double pv = MV[r.b];
    for (int i = r.b + 1; i < r.e; ++i) pv += MV[i];
    return pv;
  }
  void eval(const double* x) {
    const double* df = DF.data();
    // K2: growth over every sub-period, all instruments at once
    for (std::size_t k = 0; k < subs.size(); ++k) G[k] = df[subs[k].s] / df[subs[k].e] - 1.0;
    // K3: coupon rows
    for (std::size_t i = 0; i < cpns.size(); ++i) {
      const Coupon& c = cpns[i];
      if (c.pay < 0) continue;  // observation-only rows are read by their Rate quote
      switch (c.mode) {
        case Mode::CmpEmpty: CP[i] = df[c.pay] * ((c.rf + c.konst) * c.k); break;
        case Mode::Cmp: { const double a = c.rf * num(c) + c.konst; CP[i] = df[c.pay] * a * c.k; break; }
        case Mode::Plain: CP[i] = df[c.pay] * G[c.sub0]; break;
        case Mode::GenEmpty: CP[i] = df[c.pay] * (c.konst * c.k); break;
        case Mode::Gen: { const double a = num(c) + c.konst; CP[i] = df[c.pay] * a * c.k; break; }
      }
    }
    // K4: MtM reset rows
    for (std::size_t i = 0; i < mtms.size(); ++i) {
      const Mtm& mm = mtms[i];
      double v = CP[mm.cpn];
      if (mm.ex_e >= 0) v += df[mm.ex_e];
      if (mm.ex_s >= 0) v -= df[mm.ex_s];
      if (mm.reset_fx >= 0.0) { MV[i] = v * mm.reset_fx; continue; }
      double N = mm.fx_spot * (df[mm.rn] / df[mm.rd]);
      if (mm.sn >= 0) N = N * (df[mm.sd] / df[mm.sn]);
      MV[i] = N * v;
    }
    // K5: quote transforms (components precede their portfolio)
    for (std::size_t i = 0; i < quotes.size(); ++i) {
      const Quote& q = quotes[i];
      switch (q.tf) {
        case Tf::ParRate: Q[i] = leg_pv(q.pos) / annuity(q.fix); break;
        case Tf::ParSpread: Q[i] = (leg_pv(q.pos) - leg_pv(q.neg)) / annuity(q.fix); break;
        case Tf::ZeroCoupon: { const double r = leg_pv(q.pos) / annuity(q.fix); Q[i] = std::exp(std::log(1.0 + q.tau * r) / q.tau) - 1.0; break; }
        case Tf::Xccy: {
          const double ann = annuity(q.fix), ps = leg_pv(q.pos), pf = leg_pv(q.neg), mt = mtm_pv(q.mtm);
          if (q.fx_spot_time == 0.0) Q[i] = (ps - pf) / ann + mt / (q.fx_spot * ann);
          else { const double fx0 = q.fx_spot * (df[q.fsd] / df[q.fsn]); Q[i] = (ps - pf) / ann + mt / (fx0 * ann); }
          break;
        }
        case Tf::Rate: {
          const Coupon& c = cpns[q.obs];
          double r;
          if (q.compounded) r = (c.sub1 == c.sub0) ? (q.rfac - 1.0) / q.tau_index : (q.rfac * num(c) - 1.0) / q.tau_index;
          else r = (c.sub1 == c.sub0) ? q.realized / q.tau_index : (num(c) + q.realized) / q.tau_index;
          Q[i] = r + q.conv;
          break;
        }
        case Tf::Fx:
          if (q.fx_spot_time == 0.0) Q[i] = q.fx_spot * (df[q.fn] / df[q.fd]);
          else Q[i] = q.fx_spot * (df[q.fn] / df[q.fd]) * (df[q.fsd] / df[q.fsn]);
          break;
        case Tf::StatePin: Q[i] = x[q.state]; break;
        case Tf::Portfolio: {
          double acc = 0.0;
          for (int c = q.comps.b; c < q.comps.e; ++c) acc += comps[c].w * Q[comps[c].q];
          Q[i] = acc;
          break;
        }
      }
    }
  }
};
}  // namespace ir

int main() {
  std::printf("# cashflow-table IR probe\n");
  std::printf("%-20s %5s %5s %6s | %-60s | %6s %6s %6s %6s | %9s %9s | %8s %8s %8s\n", "shape", "rows", "IR ok", "cmpld", "exceptions / features",
              "atoms", "subs", "cpns", "mtm", "d_vs_tmpl", "d_vs_eng", "eng_mr", "ir_W", "load");
  int tot_rows = 0, tot_ok = 0, tot_cmp = 0;
  for (const auto& s : swaps::shapes::ladder()) {
    const auto& p = s.prob;
    const int m = p.n_residuals(), n = p.n_knots();
    ir::Table T;
    T.prob = &p;
    int ok = 0;
    std::map<std::string, int> exc;
    std::vector<char> row_ok(m, 0);
    for (int r = 0; r < m; ++r) {
      ir::Table probe;  // per-row trial so one exception does not poison the table
      probe.prob = &p;
      try { probe.lower(p.instruments[r]); row_ok[r] = 1; ++ok; }
      catch (const ir::Unsupported& e) { ++exc[e.what()]; }
    }
    const bool all_ok = ok == m;
    cal::HybridBundleResidual h(p);
    int compiled = 0;
    for (int r = 0; r < m; ++r) compiled += h.compiled_row(r) >= 0;
    tot_rows += m; tot_ok += ok; tot_cmp += compiled;
    std::string desc;
    for (auto& kv : exc) desc += kv.first + " x" + std::to_string(kv.second) + "; ";
    double d_t = -1, d_e = -1, t_eng = -1, t_ir = -1, load = util::load1();
    if (all_ok) {
      for (int r = 0; r < m; ++r) T.row_quote.push_back(T.lower(p.instruments[r]));
      const auto& F = T.feat;
      desc += "par " + std::to_string(F.par) + " spr " + std::to_string(F.spread) + " rate " + std::to_string(F.rate) + " cmpd-cpn " +
              std::to_string(F.compounded) + " wtd-cpn " + std::to_string(F.weighted) + " fx " + std::to_string(F.fx) + " mtm " + std::to_string(F.mtm) +
              " zc " + std::to_string(F.zc) + " pf " + std::to_string(F.portfolio) + " pin " + std::to_string(F.pin);
      T.size_scratch();
      Eigen::VectorXd x1 = s.x_true;
      for (int i = 0; i < n; ++i) x1[i] += 1e-5 * std::sin(1.3 * i + 0.2);
      // (i) exact atoms from the templated curves -> compare with the templated double kernel
      const auto Cd = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return x1[p.offset(c) + i]; });
      for (std::size_t a = 0; a < T.atoms.size(); ++a) T.DF[a] = Cd[T.atoms[a].first]->discount(T.atoms[a].second);
      T.eval(x1.data());
      const auto ofd = [&](int i) -> const px::CurveHandle<double>& { return *Cd[i]; };
      d_t = 0;
      for (int r = 0; r < m; ++r) d_t = std::max(d_t, std::abs(T.Q[T.row_quote[r]] - cal::instrument_model_quote<double>(p.instruments[r], ofd)));
      // (ii) W-cache atoms (linear curves only) -> compare with the engine's model rates
      const auto horizons = px::curve_linear_horizons(p.curves);
      bool linear = true;
      for (double hz : horizons) if (hz < std::numeric_limits<double>::infinity()) linear = false;
      if (linear) {
        px::CompiledCurveSet cs;
        cs.init(p.curves);
        for (const auto& a : T.atoms) cs.reg(a.first, a.second);
        cs.finalize();
        Eigen::VectorXd DFv;
        cs.df_into(x1, DFv);
        for (std::size_t a = 0; a < T.atoms.size(); ++a) T.DF[a] = DFv[static_cast<int>(a)];
        T.eval(x1.data());
        const Eigen::VectorXd mr = h.model_rates(x1);
        d_e = 0;
        for (int r = 0; r < m; ++r) d_e = std::max(d_e, std::abs(T.Q[T.row_quote[r]] - mr[r]));
        if (!std::getenv("AADJIT_NO_TIMING") && (s.name == "ois_nolag" || s.name == "fx_xccy" || s.name == "desk")) {
          util::guarded([&] {
          load = util::load1();
          Eigen::VectorXd xb = s.x_true;
          bool flip = false;
          t_eng = util::time_us([&] { flip = !flip; volatile double d = h.model_rates(flip ? x1 : xb)[0]; (void)d; }, 2000);
          t_ir = util::time_us([&] {
            flip = !flip;
            cs.df_into(flip ? x1 : xb, DFv);
            for (std::size_t a = 0; a < T.atoms.size(); ++a) T.DF[a] = DFv[static_cast<int>(a)];
            T.eval((flip ? x1 : xb).data());
            volatile double d = T.Q[0]; (void)d;
          }, 2000);
          });
        }
      }
    }
    std::printf("%-20s %5d %5d %6d | %-60s | %6zu %6zu %6zu %6zu | %9.1e %9.1e | %8.2f %8.2f %8.2f\n", s.name.c_str(), m, ok, compiled, desc.c_str(),
                T.atoms.size(), T.subs.size(), T.cpns.size(), T.mtms.size(), d_t, d_e, t_eng, t_ir, load);
    std::fflush(stdout);
  }
  std::printf("TOTAL rows %d, IR ok %d (%.1f %%), compiled-table rows today %d (%.1f %%)\n", tot_rows, tot_ok, 100.0 * tot_ok / tot_rows, tot_cmp,
              100.0 * tot_cmp / tot_rows);
}
