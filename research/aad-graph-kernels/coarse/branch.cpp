// BRANCH REPRESENTATIONS on mixed_scheme's MonotoneCubic region (SOFR 12Y..30Y + the C0 join node: N = 7 nodes).
// Inputs: the join value (= the Hermite front's last knot), the 6 region knots and the join integral (8 inputs). Outputs: the
// region's integral(t) at every time mixed_scheme's instruments query on SOFR at/after the join.
//   (E)  engine reference: curve::MonotoneCubic<double> (value) and <ad::DualPooled<8>> (value + J)
//   (1)  PREDICATE/SELECT (preferred default): every value branch computes BOTH arms and selects by mask; scalar, width-8 tangent,
//        and 4-lane (SIMD across scenarios) instantiations of ONE template. Also a BRANCHY instantiation (same code, `if`) to
//        price "both arms".
//   (2)  GUARD-AND-REPLAY: the engine's own MonotoneCubic<Rec> recorded (78 guards), affine-collapsed under its guards into W.
//   (3)  PER-PATTERN CACHE: (2) keyed by the guard bit pattern.
// Studies: (a) NaN/inf in the discarded arm (value, tangent, adjoint) and the fixes; (b) cost of both arms vs guard+cache;
// (c) SIMD lanes vs threads (vs GPU by reference); (d) the mask bits as a free kink/margin output.
#include "rec.hpp"
#include "shape_ladder.hpp"
#include "swaps/ad/dual.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "kernel.hpp"
#include "util.hpp"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <thread>

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace cv = swaps::curve;
using clk = std::chrono::steady_clock;

// ================= select-mode knobs (study (a)) =================
enum class SignMode { Naive, SafeArm, Copysign };  // Naive = the engine's m/|m| evaluated in BOTH arms
inline SignMode g_sign = SignMode::Copysign;
inline bool g_arith_blend = false;                 // false: bitwise/ternary select; true: c·a + (1−c)·b
// ================= mask log (study (d)) =================
inline bool g_log_masks = false;
inline int g_nmask = 0;
inline std::array<uint8_t, 512> g_mask{};
inline std::array<double, 512> g_margin{};

// ---- scalar ----
inline double val(double x) { return x; }
inline bool mgt0(double x) { if (g_log_masks && g_nmask < 512) { g_margin[g_nmask] = x; g_mask[g_nmask++] = x > 0.0; } return x > 0.0; }
inline bool mlt(double a, double b) { if (g_log_masks && g_nmask < 512) { g_margin[g_nmask] = b - a; g_mask[g_nmask++] = a < b; } return a < b; }
inline bool meq0(double x) { return x == 0.0; }
inline bool mand(bool a, bool b) { return a && b; }
inline double sel(bool c, double a, double b) { return g_arith_blend ? (double(c) * a + double(!c) * b) : (c ? a : b); }
inline double absx(double x) { return std::abs(x); }
inline double sgn1(double x) { return std::copysign(1.0, x); }

// ---- width-8 forward tangent ----
struct FD {
  double v = 0.0;
  std::array<double, 8> d{};
  FD() = default;
  FD(double x) : v(x) {}
};
inline FD operator+(const FD& a, const FD& b) { FD r(a.v + b.v); for (int k = 0; k < 8; ++k) r.d[k] = a.d[k] + b.d[k]; return r; }
inline FD operator-(const FD& a, const FD& b) { FD r(a.v - b.v); for (int k = 0; k < 8; ++k) r.d[k] = a.d[k] - b.d[k]; return r; }
inline FD operator-(const FD& a) { FD r(-a.v); for (int k = 0; k < 8; ++k) r.d[k] = -a.d[k]; return r; }
inline FD operator*(const FD& a, const FD& b) { FD r(a.v * b.v); for (int k = 0; k < 8; ++k) r.d[k] = a.d[k] * b.v + b.d[k] * a.v; return r; }
inline FD operator*(double c, const FD& b) { FD r(c * b.v); for (int k = 0; k < 8; ++k) r.d[k] = c * b.d[k]; return r; }
inline FD operator*(const FD& a, double c) { FD r(a.v * c); for (int k = 0; k < 8; ++k) r.d[k] = a.d[k] * c; return r; }
inline FD operator/(const FD& a, double c) { FD r(a.v / c); for (int k = 0; k < 8; ++k) r.d[k] = a.d[k] / c; return r; }
inline FD operator/(const FD& a, const FD& b) { FD r(a.v / b.v); const double ib = 1.0 / b.v; for (int k = 0; k < 8; ++k) r.d[k] = (a.d[k] - r.v * b.d[k]) * ib; return r; }
inline double val(const FD& x) { return x.v; }
inline bool mgt0(const FD& x) { return x.v > 0.0; }
inline bool mlt(const FD& a, const FD& b) { return a.v < b.v; }
inline bool meq0(const FD& x) { return x.v == 0.0; }
inline FD sel(bool c, const FD& a, const FD& b) {
  if (!g_arith_blend) return c ? a : b;
  const double ca = double(c), cb = double(!c);
  FD r(ca * a.v + cb * b.v);
  for (int k = 0; k < 8; ++k) r.d[k] = ca * a.d[k] + cb * b.d[k];
  return r;
}
inline FD absx(const FD& x) { return x.v < 0.0 ? -x : x; }  // Eigen AutoDiff abs: derivative sign(x), +1 at 0
inline FD sgn1(const FD& x) { return FD(std::copysign(1.0, x.v)); }

// ---- 4 scenario lanes (value only) ----
struct L4 {
  std::array<double, 4> v{};
  L4() = default;
  L4(double x) { v.fill(x); }
};
inline L4 operator+(const L4& a, const L4& b) { L4 r; for (int k = 0; k < 4; ++k) r.v[k] = a.v[k] + b.v[k]; return r; }
inline L4 operator-(const L4& a, const L4& b) { L4 r; for (int k = 0; k < 4; ++k) r.v[k] = a.v[k] - b.v[k]; return r; }
inline L4 operator-(const L4& a) { L4 r; for (int k = 0; k < 4; ++k) r.v[k] = -a.v[k]; return r; }
inline L4 operator*(const L4& a, const L4& b) { L4 r; for (int k = 0; k < 4; ++k) r.v[k] = a.v[k] * b.v[k]; return r; }
inline L4 operator*(double c, const L4& b) { L4 r; for (int k = 0; k < 4; ++k) r.v[k] = c * b.v[k]; return r; }
inline L4 operator*(const L4& a, double c) { L4 r; for (int k = 0; k < 4; ++k) r.v[k] = a.v[k] * c; return r; }
inline L4 operator/(const L4& a, double c) { L4 r; for (int k = 0; k < 4; ++k) r.v[k] = a.v[k] / c; return r; }
inline L4 operator/(const L4& a, const L4& b) { L4 r; for (int k = 0; k < 4; ++k) r.v[k] = a.v[k] / b.v[k]; return r; }
using M4 = std::array<double, 4>;  // all-ones / all-zeros lane masks, as a blendv would see them
inline M4 mgt0(const L4& x) { M4 m; for (int k = 0; k < 4; ++k) m[k] = x.v[k] > 0.0 ? 1.0 : 0.0; return m; }
inline M4 mlt(const L4& a, const L4& b) { M4 m; for (int k = 0; k < 4; ++k) m[k] = a.v[k] < b.v[k] ? 1.0 : 0.0; return m; }
inline M4 meq0(const L4& x) { M4 m; for (int k = 0; k < 4; ++k) m[k] = x.v[k] == 0.0 ? 1.0 : 0.0; return m; }
inline M4 mand(const M4& a, const M4& b) { M4 m; for (int k = 0; k < 4; ++k) m[k] = a[k] * b[k]; return m; }
inline L4 sel(const M4& c, const L4& a, const L4& b) { L4 r; for (int k = 0; k < 4; ++k) r.v[k] = c[k] != 0.0 ? a.v[k] : b.v[k]; return r; }
inline L4 absx(const L4& x) { L4 r; for (int k = 0; k < 4; ++k) r.v[k] = std::abs(x.v[k]); return r; }
inline L4 sgn1(const L4& x) { L4 r; for (int k = 0; k < 4; ++k) r.v[k] = std::copysign(1.0, x.v[k]); return r; }

template <class T> inline T smin(const T& a, const T& b) { return sel(mlt(b, a), b, a); }
template <class T> inline T smax(const T& a, const T& b) { return sel(mlt(a, b), b, a); }
template <class T> inline T sgn(const T& m) {
  switch (g_sign) {
    case SignMode::Naive: return m / absx(m);
    case SignMode::SafeArm: { const T ms = sel(meq0(m), T(1.0), m); return ms / absx(ms); }
    default: return sgn1(m);
  }
}

// (1) the MonotoneCubic region. BF = true: every value branch is predicate/select; BF = false: the same code with `if`.
template <class T, bool BF>
struct SelMC {
  std::vector<double> xs, h, lower, diag0, upper, di, qu;
  std::vector<int> qseg;
  int N = 0;
  std::vector<T> ys, S, rhs, m, a, b, c, d, Is;
  void init(const std::vector<double>& knots, double join_time, const std::vector<double>& qt) {
    N = static_cast<int>(knots.size()) + 1;
    xs.assign(N, 0.0); xs[0] = join_time;
    for (int i = 0; i + 1 < N; ++i) xs[i + 1] = knots[i];
    h.assign(N - 1, 0.0);
    for (int i = 0; i < N - 1; ++i) h[i] = xs[i + 1] - xs[i];
    lower.assign(N, 0.0); diag0.assign(N, 0.0); upper.assign(N, 0.0);
    diag0[0] = 2.0; upper[0] = 1.0;
    for (int i = 1; i < N - 1; ++i) { lower[i] = h[i]; diag0[i] = 2.0 * (h[i] + h[i - 1]); upper[i] = h[i - 1]; }
    lower[N - 1] = 1.0; diag0[N - 1] = 2.0;
    for (double t : qt) {  // structural: each query time's segment folded once
      if (t <= xs.front()) { qseg.push_back(-1); qu.push_back(xs.front() - t); }
      else if (t >= xs.back()) { qseg.push_back(-2); qu.push_back(t - xs.back()); }
      else { const int i = static_cast<int>(std::upper_bound(xs.begin(), xs.end(), t) - xs.begin()) - 1; qseg.push_back(i); qu.push_back(t - xs[i]); }
    }
    ys.resize(N); S.resize(N - 1); rhs.resize(N); m.resize(N); a.resize(N - 1); b.resize(N - 1); c.resize(N - 1); d.resize(N - 1); Is.resize(N);
    di.resize(N);
  }
  void eval(const T* in, T* out) {
    const int nseg = N - 1;
    for (int i = 0; i < N; ++i) ys[i] = in[i];
    for (int i = 0; i < nseg; ++i) S[i] = (ys[i + 1] - ys[i]) / h[i];
    rhs[0] = 3.0 * S[0];
    for (int i = 1; i < N - 1; ++i) rhs[i] = 3.0 * (h[i] * S[i - 1] + h[i - 1] * S[i]);
    rhs[N - 1] = 3.0 * S[nseg - 1];
    for (int i = 0; i < N; ++i) di[i] = diag0[i];
    for (int i = 1; i < N; ++i) { const double w = lower[i] / di[i - 1]; di[i] -= w * upper[i - 1]; rhs[i] = rhs[i] - w * rhs[i - 1]; }
    m[N - 1] = rhs[N - 1] / di[N - 1];
    for (int i = N - 1; i-- > 0;) m[i] = (rhs[i] - upper[i] * m[i + 1]) / di[i];
    for (int i = 0; i < N; ++i) {
      if (i == 0 || i == N - 1) {  // structural
        const T& Se = (i == 0) ? S[0] : S[N - 2];
        if constexpr (BF) {
          const T clamp = sgn(m[i]) * smin(absx(m[i]), absx(3.0 * Se));
          m[i] = sel(mgt0(m[i] * Se), clamp, m[i] * 0.0);
        } else {
          if (val(m[i] * Se) > 0.0) m[i] = sgn(m[i]) * smin(absx(m[i]), absx(3.0 * Se));
          else m[i] = m[i] * 0.0;
        }
      } else {
        const T pm = (S[i - 1] * h[i] + S[i] * h[i - 1]) / (h[i - 1] + h[i]);
        T M = 3.0 * smin(smin(absx(S[i - 1]), absx(S[i])), absx(pm));
        if (i > 1) {
          if constexpr (BF) {
            const T pd = (S[i - 1] * (2.0 * h[i - 1] + h[i - 2]) - S[i - 2] * h[i - 1]) / (h[i - 2] + h[i - 1]);
            const auto cd = mand(mand(mgt0((S[i - 1] - S[i - 2]) * (S[i] - S[i - 1])), mgt0(pm * pd)), mgt0(pm * (S[i - 1] - S[i - 2])));
            M = sel(cd, smax(M, 1.5 * smin(absx(pm), absx(pd))), M);
          } else if (val((S[i - 1] - S[i - 2]) * (S[i] - S[i - 1])) > 0.0) {
            const T pd = (S[i - 1] * (2.0 * h[i - 1] + h[i - 2]) - S[i - 2] * h[i - 1]) / (h[i - 2] + h[i - 1]);
            if (val(pm * pd) > 0.0 && val(pm * (S[i - 1] - S[i - 2])) > 0.0) M = smax(M, 1.5 * smin(absx(pm), absx(pd)));
          }
        }
        if (i < N - 2) {
          if constexpr (BF) {
            const T pu = (S[i] * (2.0 * h[i] + h[i + 1]) - S[i + 1] * h[i]) / (h[i] + h[i + 1]);
            const auto cu = mand(mand(mgt0((S[i] - S[i - 1]) * (S[i + 1] - S[i])), mgt0(pm * pu)), mgt0(-pm * (S[i] - S[i - 1])));
            M = sel(cu, smax(M, 1.5 * smin(absx(pm), absx(pu))), M);
          } else if (val((S[i] - S[i - 1]) * (S[i + 1] - S[i])) > 0.0) {
            const T pu = (S[i] * (2.0 * h[i] + h[i + 1]) - S[i + 1] * h[i]) / (h[i] + h[i + 1]);
            if (val(pm * pu) > 0.0 && val(-pm * (S[i] - S[i - 1])) > 0.0) M = smax(M, 1.5 * smin(absx(pm), absx(pu)));
          }
        }
        if constexpr (BF) {
          const T clamp = sgn(m[i]) * smin(absx(m[i]), M);
          m[i] = sel(mgt0(m[i] * pm), clamp, m[i] * 0.0);
        } else {
          if (val(m[i] * pm) > 0.0) m[i] = sgn(m[i]) * smin(absx(m[i]), M);
          else m[i] = m[i] * 0.0;
        }
      }
    }
    for (int i = 0; i < nseg; ++i) {
      const double hi = h[i];
      a[i] = ys[i];
      b[i] = m[i];
      c[i] = 3.0 * S[i] / hi - (2.0 * m[i] + m[i + 1]) / hi;
      d[i] = (m[i] + m[i + 1]) / (hi * hi) - 2.0 * S[i] / (hi * hi);
    }
    Is[0] = in[N];
    for (int i = 0; i < nseg; ++i) { const double u = h[i]; Is[i + 1] = Is[i] + u * (a[i] + u * (b[i] / 2.0 + u * (c[i] / 3.0 + u * d[i] / 4.0))); }
    for (std::size_t q = 0; q < qseg.size(); ++q) {
      const int i = qseg[q];
      const double u = qu[q];
      if (i == -1) out[q] = Is.front() - ys.front() * u;
      else if (i == -2) out[q] = Is.back() + ys.back() * u;
      else out[q] = Is[i] + u * (a[i] + u * (b[i] / 2.0 + u * (c[i] / 3.0 + u * d[i] / 4.0)));
    }
  }
};

struct TimeProbe : px::CurveHandle<double> {
  const px::CurveHandle<double>* real = nullptr;
  std::vector<double>* ts = nullptr;
  double forward(double t) const override { ts->push_back(t); return real->forward(t); }
  double integral(double t) const override { ts->push_back(t); return real->integral(t); }
  double discount(double t) const override { ts->push_back(t); return real->discount(t); }
  double turn_jump(int j) const override { return real->turn_jump(j); }
  void pieces_into(std::vector<double>& o) const override { real->pieces_into(o); }
  void set_forwards(const Eigen::VectorXd&) override {}
};

// a persistent worker: the per-dispatch latency a per-tick thread hand-off would pay
struct Worker {
  std::mutex mu;
  std::condition_variable cv_go, cv_done;
  std::function<void()> job;
  bool has = false, done = false, quit = false;
  std::thread th;
  Worker() : th([this] {
      std::unique_lock<std::mutex> lk(mu);
      for (;;) {
        cv_go.wait(lk, [&] { return has || quit; });
        if (quit) return;
        has = false;
        job();
        done = true;
        cv_done.notify_one();
      }
    }) {}
  void run(std::function<void()> j) {
    std::unique_lock<std::mutex> lk(mu);
    job = std::move(j); has = true; done = false;
    cv_go.notify_one();
    cv_done.wait(lk, [&] { return done; });
  }
  ~Worker() { { std::lock_guard<std::mutex> lk(mu); quit = true; } cv_go.notify_one(); th.join(); }
};

int main() {
  static const auto Lad = swaps::shapes::ladder();
  const swaps::shapes::Shape* ms = nullptr;
  for (const auto& s : Lad) if (s.name == "mixed_scheme") ms = &s;
  const auto& s = *ms;
  const auto& p = s.prob;
  const auto& regions = p.curves[0].regions;
  const std::vector<double>& front = regions[0].knots;
  const std::vector<double>& back = regions[1].knots;
  const int nf = static_cast<int>(front.size()), nb = static_cast<int>(back.size());
  std::vector<double> qt;
  {
    const auto C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return s.x_true[p.offset(c) + i]; });
    std::vector<TimeProbe> H(C.size());
    std::vector<std::vector<double>> logs(C.size());
    for (std::size_t c = 0; c < C.size(); ++c) { H[c].real = C[c].get(); H[c].ts = &logs[c]; }
    const auto of = [&](int i) -> const px::CurveHandle<double>& { return H[i]; };
    for (const auto& ins : p.instruments) (void)cal::instrument_model_quote<double>(ins, of);
    for (auto& lg : logs) for (double t : lg) if (t > front.back()) qt.push_back(t);
    std::sort(qt.begin(), qt.end());
    qt.erase(std::unique(qt.begin(), qt.end()), qt.end());
  }
  const int Q = static_cast<int>(qt.size());
  const double join_t = front.back();
  auto region_in = [&](const Eigen::VectorXd& x) {
    cv::Hermite<double> hf(front);
    cv::Boundary<double> b0{};
    hf.build(x.data(), 0, nf, b0);
    std::vector<double> in(8);
    in[0] = x[nf - 1];
    for (int i = 0; i < nb; ++i) in[1 + i] = x[nf + i];
    in[7] = hf.out().integral;
    return in;
  };
  auto engine_value = [&](const std::vector<double>& in, double* out) {
    cv::MonotoneCubic<double> mc(back);
    cv::Boundary<double> b{};
    b.time = join_t; b.value = in[0]; b.integral = in[7]; b.has_predecessor = true;
    mc.build(in.data() + 1, 0, nb, b);
    for (int q = 0; q < Q; ++q) out[q] = mc.integral(qt[q]);
  };
  using D8 = swaps::ad::DualPooled<8>;
  auto engine_vj = [&](const std::vector<double>& in, double* out, double* J) {
    std::vector<D8> xd(nb);
    for (int i = 0; i < nb; ++i) xd[i] = D8(in[1 + i], 8, 1 + i);
    cv::MonotoneCubic<D8> mc(back);
    cv::Boundary<D8> b{};
    b.time = join_t; b.value = D8(in[0], 8, 0); b.integral = D8(in[7], 8, 7); b.has_predecessor = true;
    mc.build(xd.data(), 0, nb, b);
    for (int q = 0; q < Q; ++q) {
      const D8 I = mc.integral(qt[q]);
      out[q] = I.value();
      for (int k = 0; k < 8; ++k) J[q + k * Q] = I.derivatives().size() == 8 ? I.derivatives()[k] : 0.0;
    }
  };
  SelMC<double, true> k1v; k1v.init(back, join_t, qt);
  SelMC<double, false> kbv; kbv.init(back, join_t, qt);
  SelMC<FD, true> k1j; k1j.init(back, join_t, qt);
  SelMC<FD, false> kbj; kbj.init(back, join_t, qt);
  SelMC<L4, true> k4; k4.init(back, join_t, qt);
  std::vector<FD> fin(8), fout(Q);
  auto vj_with = [&](auto& K, const std::vector<double>& in, double* out, double* J) {
    for (int k = 0; k < 8; ++k) { fin[k] = FD(in[k]); fin[k].d[k] = 1.0; }
    K.eval(fin.data(), fout.data());
    for (int q = 0; q < Q; ++q) { out[q] = fout[q].v; for (int k = 0; k < 8; ++k) J[q + k * Q] = fout[q].d[k]; }
  };
  // (2) guard tape of the engine's own region
  struct GuardKernel { aj::Kernel K; aj::Collapsed C; Eigen::MatrixXd W; double rec_us = 0; };
  auto record = [&](const std::vector<double>& in) {
    const auto t0 = clk::now();
    auto G = std::make_unique<GuardKernel>();
    aj::Recorder rec;
    aj::R = &rec;
    std::vector<aj::Rec> X(8);
    for (int k = 0; k < 8; ++k) {
      rec.nodes.push_back({aj::INPUT, -1, -1, in[k]});
      rec.inputs.push_back(static_cast<int>(rec.nodes.size()) - 1);
      X[k] = aj::Rec(in[k], rec.inputs.back());
    }
    cv::MonotoneCubic<aj::Rec> mc(back);
    cv::Boundary<aj::Rec> b{};
    b.time = join_t; b.value = X[0]; b.integral = X[7]; b.has_predecessor = true;
    mc.build(X.data() + 1, 0, nb, b);
    std::vector<int> outs;
    for (int q = 0; q < Q; ++q) outs.push_back(mc.integral(qt[q]).slot());
    aj::R = nullptr;
    G->K.build(rec, outs);
    G->C = aj::collapse(G->K);
    G->C.forward(in.data());
    G->W.resize(Q, 8);
    G->C.jacobian_reverse(G->W.data());
    G->rec_us = std::chrono::duration<double, std::micro>(clk::now() - t0).count();
    return G;
  };
  auto bits_of = [&](aj::Collapsed& C) {
    std::vector<char> bits(C.guards.size());
    for (std::size_t g = 0; g < C.guards.size(); ++g) {
      const auto& gd = C.guards[g];
      const double a = C.val[gd.a], b = C.val[gd.b];
      bool r;
      switch (gd.c) { case aj::LT: r = a < b; break; case aj::LE: r = a <= b; break; case aj::GT: r = a > b; break;
                      case aj::GE: r = a >= b; break; case aj::EQ: r = a == b; break; default: r = a != b; }
      bits[g] = r;
    }
    return bits;
  };

  const std::vector<double> in0 = region_in(s.x_true);
  auto G0 = record(in0);
  std::printf("# mixed_scheme MonotoneCubic region: N = %d nodes, %d query times at/after the join, 8 inputs\n", nb + 1, Q);
  std::printf("# guard tape: %zu guards, %d live ops -> collapsed %d affine rows / %zu nnz / %d nonlinear ops; record+collapse %.0f us\n",
              G0->K.guards.size(), G0->K.nops, G0->C.nrows, G0->C.w.size(), G0->C.nnl, G0->rec_us);

  std::vector<double> ve(Q), v1(Q), v2(Q), Je(static_cast<std::size_t>(Q) * 8), J1(Je.size());
  auto parity = [&](const char* tag, const std::vector<double>& in, GuardKernel& G) {
    engine_vj(in, ve.data(), Je.data());
    std::vector<double> vev(Q), vb(Q), vbj(Q), Jb(Je.size());
    engine_value(in, vev.data());
    k1v.eval(in.data(), v1.data());
    kbv.eval(in.data(), vb.data());
    std::vector<double> v1j(Q);
    vj_with(k1j, in, v1j.data(), J1.data());
    vj_with(kbj, in, vbj.data(), Jb.data());
    const int flips = G.C.forward(in.data());
    double d1 = 0, db = 0, dJ1 = 0, dJb = 0, d2 = 0, dJ2 = 0;
    for (int q = 0; q < Q; ++q) {
      d1 = std::max(d1, std::abs(v1[q] - vev[q]));
      db = std::max(db, std::abs(vb[q] - vev[q]));
      d2 = std::max(d2, std::abs(G.C.val[G.C.out[q]] - vev[q]));
      for (int k = 0; k < 8; ++k) {
        dJ1 = std::max(dJ1, std::abs(J1[q + k * Q] - Je[q + k * Q]));
        dJb = std::max(dJb, std::abs(Jb[q + k * Q] - Je[q + k * Q]));
        dJ2 = std::max(dJ2, std::abs(G.W(q, k) - Je[q + k * Q]));
      }
    }
    std::printf("  parity %-24s (1) select: value vs engine %.1e, J vs engine AAD %.1e | branchy: value %.1e, J %.1e | "
                "(2) guard replay: value %.1e, J=W %.1e, flips vs recording %d\n", tag, d1, dJ1, db, dJb, d2, dJ2, flips);
  };
  parity("x_true", in0, *G0);
  Eigen::VectorXd xs1 = s.x_true;
  for (int i = 0; i < xs1.size(); ++i) xs1[i] += 1e-5 * std::sin(1.3 * i + 0.2);
  const auto in1 = region_in(xs1);
  parity("0.1bp sine", in1, *G0);

  // ================= (a) the discarded arm: NaN/inf poisoning =================
  {
    // A FLAT region (every node value equal): every secant S = 0, the spline tangents m = 0 exactly, so every Hyman predicate
    // m·S > 0 is FALSE and the clamp arm's sign m/|m| is 0/0 = NaN -- computed and thrown away by a select.
    std::vector<double> flat(8, 0.031);
    flat[7] = in0[7];
    engine_vj(flat, ve.data(), Je.data());
    std::printf("\n  (a) flat region, engine AAD value[0] %.6f finite J %d\n", ve[0], static_cast<int>(std::all_of(Je.begin(), Je.end(), [](double v) { return std::isfinite(v); })));
    for (auto sm : {SignMode::Naive, SignMode::SafeArm, SignMode::Copysign})
      for (bool arith : {false, true}) {
        g_sign = sm; g_arith_blend = arith;
        k1v.eval(flat.data(), v1.data());
        std::vector<double> vj(Q);
        vj_with(k1j, flat, vj.data(), J1.data());
        bool vfin = true, jfin = true;
        double dv = 0, dJ = 0;
        for (int q = 0; q < Q; ++q) { vfin &= std::isfinite(v1[q]); if (vfin) dv = std::max(dv, std::abs(v1[q] - ve[q])); }
        for (std::size_t k = 0; k < J1.size(); ++k) { jfin &= std::isfinite(J1[k]); if (jfin) dJ = std::max(dJ, std::abs(J1[k] - Je[k])); }
        std::printf("      sign=%-8s select=%-10s value finite %d (|d| %.1e)  tangent J finite %d (|d| %.1e)\n",
                    sm == SignMode::Naive ? "m/|m|" : sm == SignMode::SafeArm ? "safe-arm" : "copysign", arith ? "arith-blend" : "bitwise", vfin, dv, jfin, dJ);
      }
    g_sign = SignMode::Copysign; g_arith_blend = false;
    // Reverse mode through a select: y = select(c, f(u), g(u)) with f(u) = 1/u at u = 0 (the discarded arm). The adjoint
    // reaching u from the dead arm is (mask = 0)·ybar·f'(u) = 0·(−inf) = NaN unless the dead arm's adjoint is masked or its input made safe.
    const double u = 0.0, ybar = 1.0, mask = 0.0;
    const double fprime = -1.0 / (u * u), gprime = 1.0;
    const double ubar_naive = mask * ybar * fprime + (1.0 - mask) * ybar * gprime;
    const double ubar_masked = (mask != 0.0 ? mask * ybar * fprime : 0.0) + (1.0 - mask) * ybar * gprime;
    const double us = (u == 0.0) ? 1.0 : u;  // safe arm input
    const double ubar_safe = mask * ybar * (-1.0 / (us * us)) + (1.0 - mask) * ybar * gprime;
    std::printf("      reverse through select(c, 1/u, u) at u=0, c=false: naive adjoint %g | masked propagation %g | safe-arm input %g\n",
                ubar_naive, ubar_masked, ubar_safe);
  }

  // ================= flip statistics and (d) mask bits vs guard flips =================
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> ph(0.0, 6.283185307179586);
  g_log_masks = true; g_nmask = 0;
  k1v.eval(in0.data(), v1.data());
  const int NM = g_nmask;
  const std::array<uint8_t, 512> mask0 = g_mask;
  g_log_masks = false;
  std::printf("\n  select masks per eval: %d predicates (vs %zu recorded guards)\n", NM, G0->K.guards.size());
  for (double amp : {1e-5, 1e-4, 25e-4}) {
    std::map<std::vector<char>, int> patterns;
    int flipped = 0, max_flips = 0, both = 0, mask_only = 0, guard_only = 0;
    long tot_flips = 0;
    double worst_stale = 0.0, min_margin_flip = 1e300;
    const int trials = 2000;
    for (int tr = 0; tr < trials; ++tr) {
      const double phase = ph(rng);
      Eigen::VectorXd x = s.x_true;
      for (int i = 0; i < x.size(); ++i) x[i] += amp * std::sin(0.7 * i + phase);
      const auto in = region_in(x);
      const int fl = G0->C.forward(in.data());
      ++patterns[bits_of(G0->C)];
      g_log_masks = true; g_nmask = 0;
      k1v.eval(in.data(), v1.data());
      g_log_masks = false;
      bool mchg = false;
      for (int k = 0; k < NM; ++k) if (g_mask[k] != mask0[k]) { mchg = true; min_margin_flip = std::min(min_margin_flip, std::abs(g_margin[k])); }
      both += (fl > 0) && mchg; mask_only += (fl == 0) && mchg; guard_only += (fl > 0) && !mchg;
      if (fl) {
        ++flipped; tot_flips += fl; max_flips = std::max(max_flips, fl);
        engine_value(in, ve.data());
        for (int q = 0; q < Q; ++q) worst_stale = std::max(worst_stale, std::abs(G0->C.val[G0->C.out[q]] - ve[q]));
      }
    }
    std::printf("  stress amp %.1e x %d: evals with a guard flip %d (%.1f %%), mean flips %.1f, max %d, distinct guard patterns %zu, "
                "worst |stale replay - engine| %.1e | mask-change vs guard-flip: both %d, mask-only %d, guard-only %d; smallest flipped margin %.1e\n",
                amp, trials, flipped, 100.0 * flipped / trials, flipped ? double(tot_flips) / flipped : 0.0, max_flips, patterns.size(),
                worst_stale, both, mask_only, guard_only, min_margin_flip == 1e300 ? 0.0 : min_margin_flip);
  }
  {
    Eigen::VectorXd xb = s.x_true;
    for (int i = 0; i < xb.size(); ++i) xb[i] += 25e-4 * std::sin(0.7 * i + 0.3);
    const auto inb = region_in(xb);
    const int fl = G0->C.forward(inb.data());
    auto Gb = record(inb);
    std::printf("  deterministic 25bp sine (phase 0.3): region guard flips %d; re-record+collapse %.0f us\n", fl, Gb->rec_us);
    parity("25bp sine (re-recorded)", inb, *Gb);
  }

  // ================= timings (held): (b) both arms vs branchy vs guard; (c) lanes vs threads =================
  const int B = 4096;
  std::vector<std::vector<double>> scen(B);
  for (int bi = 0; bi < B; ++bi) {
    Eigen::VectorXd x = s.x_true;
    const double phase = ph(rng);
    for (int i = 0; i < x.size(); ++i) x[i] += 25e-4 * std::sin(0.7 * i + phase);
    scen[bi] = region_in(x);
  }
  std::vector<L4> lin(8), lout(Q);
  auto lanes_batch = [&](int b0, int b1, std::vector<L4>& li, std::vector<L4>& lo, SelMC<L4, true>& K) {
    for (int bi = b0; bi + 3 < b1; bi += 4) {
      for (int k = 0; k < 8; ++k) for (int l = 0; l < 4; ++l) li[k].v[l] = scen[bi + l][k];
      K.eval(li.data(), lo.data());
    }
  };
  // lane parity vs scalar select
  {
    lanes_batch(0, 4, lin, lout, k4);
    double dl = 0;
    for (int l = 0; l < 4; ++l) { k1v.eval(scen[l].data(), v1.data()); for (int q = 0; q < Q; ++q) dl = std::max(dl, std::abs(lout[q].v[l] - v1[q])); }
    std::printf("\n  4-lane select vs scalar select, max |d| %.1e\n", dl);
  }
  double t_ev = 0, t_evj = 0, t_1v = 0, t_bv = 0, t_1vj = 0, t_bvj = 0, t_2v = 0, t_2vj = 0, t_rec = 0, t_lookup = 0, t_mask = 0;
  double t_batch_scalar = 0, t_batch_l4 = 0, t_batch_thr = 0, t_dispatch = 0, l0 = 0, l1 = 0;
  const unsigned nthr = 8;
  Eigen::MatrixXd Jout(Q, 8);
  std::map<std::vector<char>, int> cache;
  cache[bits_of(G0->C)] = 0;
  util::guarded([&] {
    l0 = util::load1();
    bool f = false;
    t_ev = util::time_us([&] { f = !f; engine_value(f ? in0 : in1, ve.data()); }, 20000);
    t_evj = util::time_us([&] { f = !f; engine_vj(f ? in0 : in1, ve.data(), Je.data()); }, 5000);
    t_1v = util::time_us([&] { f = !f; k1v.eval((f ? in0 : in1).data(), v1.data()); }, 50000);
    t_bv = util::time_us([&] { f = !f; kbv.eval((f ? in0 : in1).data(), v1.data()); }, 50000);
    t_1vj = util::time_us([&] { f = !f; vj_with(k1j, f ? in0 : in1, v1.data(), J1.data()); }, 20000);
    t_bvj = util::time_us([&] { f = !f; vj_with(kbj, f ? in0 : in1, v1.data(), J1.data()); }, 20000);
    t_2v = util::time_us([&] { f = !f; volatile int fl = G0->C.forward((f ? in0 : in1).data()); (void)fl; }, 50000);
    t_2vj = util::time_us([&] { f = !f; volatile int fl = G0->C.forward((f ? in0 : in1).data()); (void)fl; Jout = G0->W; }, 50000);
    t_rec = util::time_us([&] { f = !f; volatile auto g = record(f ? in0 : in1); (void)g; }, 50);
    t_lookup = util::time_us([&] { G0->C.forward(in1.data()); volatile auto it = cache.find(bits_of(G0->C)); (void)it; }, 20000);
    t_mask = util::time_us([&] { f = !f; g_log_masks = true; g_nmask = 0; k1v.eval((f ? in0 : in1).data(), v1.data()); g_log_masks = false; }, 50000);
    t_batch_scalar = util::time_us([&] { for (int bi = 0; bi < B; ++bi) k1v.eval(scen[bi].data(), v1.data()); }, 3);
    t_batch_l4 = util::time_us([&] { lanes_batch(0, B, lin, lout, k4); }, 3);
    t_batch_thr = util::time_us([&] {
      std::vector<std::thread> ths;
      for (unsigned t = 0; t < nthr; ++t)
        ths.emplace_back([&, t] {
          SelMC<L4, true> K; K.init(back, join_t, qt);
          std::vector<L4> li(8), lo(Q);
          lanes_batch(static_cast<int>(t * B / nthr), static_cast<int>((t + 1) * B / nthr), li, lo, K);
        });
      for (auto& th : ths) th.join();
    }, 3);
    Worker w;
    t_dispatch = util::time_us([&] { w.run([&] { k1v.eval(in0.data(), v2.data()); }); }, 2000);
    l1 = util::load1();
  });
  std::printf("  TIMINGS µs (load %.2f -> %.2f)\n", l0, l1);
  std::printf("    engine MonotoneCubic<double> value %.3f | engine <DualPooled<8>> value+J %.3f\n", t_ev, t_evj);
  std::printf("    (1) select value %.3f | branchy value %.3f | select value+J %.3f | branchy value+J %.3f | select value + mask/margin export %.3f\n",
              t_1v, t_bv, t_1vj, t_bvj, t_mask);
  std::printf("    (2) guard replay value %.3f | value+J (J = W copy) %.3f | on a flip: re-record+collapse %.1f | (3) pattern lookup %.3f\n",
              t_2v, t_2vj, t_rec, t_lookup);
  std::printf("    (c) batch of %d scenarios: scalar select %.1f us (%.3f/scen) | 4 lanes %.1f us (%.3f/scen) | %u threads x 4 lanes (incl. spawn) %.1f us (%.3f/scen) | "
              "one per-tick thread hand-off (condvar round trip + eval) %.3f\n",
              B, t_batch_scalar, t_batch_scalar / B, t_batch_l4, t_batch_l4 / B, nthr, t_batch_thr, t_batch_thr / B, t_dispatch);
  const double pstar_new = (t_1vj - t_2vj) / std::max(1e-9, t_rec);
  const double pstar_cached = (t_1vj - t_2vj) / std::max(1e-9, t_lookup);
  std::printf("    break-even flip rate per eval for guard+replay over select (value+J): re-record %.2e | cached pattern %.2e\n", pstar_new, pstar_cached);
}
