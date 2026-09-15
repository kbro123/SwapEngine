#pragma once
// AAD-JIT spike: record the ENGINE'S OWN templated pricing (instrument_model_quote<Rec>) once per structure into a flat op
// list (constant folding + CSE at record time, value-dependent comparisons recorded as GUARDS), then replay value + Jacobian
// allocation-free. Compared against the engine's hot path (HybridBundleResidual) on the same fixture, in the same process.
// Research probe only -- not engine code.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>
#include <dlfcn.h>

#include <Eigen/Core>

namespace aj {
enum Op : uint8_t { CONST, INPUT, ADD, SUB, MUL, DIV, NEG, EXP, LOG, SQRT };
enum Cmp : uint8_t { LT, LE, GT, GE, EQ, NE };
struct Node { Op op; int a, b; double v; };
struct Guard { Cmp c; int a, b; bool expect; };

struct Recorder {
  std::vector<Node> nodes;
  std::vector<Guard> guards;
  std::vector<int> inputs;  // node id of input i
  std::unordered_map<uint64_t, int> konst;
  struct K { uint64_t h; bool operator==(const K& o) const { return h == o.h; } };
  std::unordered_map<std::string, int> cse;  // (op,a,b) -> node  (string key: probe code, not hot)
  long cse_hits = 0, folded = 0, recorded = 0;
  int constant(double v) {
    uint64_t bits; std::memcpy(&bits, &v, 8);
    auto it = konst.find(bits);
    if (it != konst.end()) return it->second;
    nodes.push_back({CONST, -1, -1, v});
    return konst[bits] = static_cast<int>(nodes.size()) - 1;
  }
  int emit(Op op, int a, int b, double v) {
    ++recorded;
    char key[16]; std::memcpy(key, &op, 1); std::memcpy(key + 1, &a, 4); std::memcpy(key + 5, &b, 4);
    std::string k(key, 9);
    auto it = cse.find(k);
    if (it != cse.end()) { ++cse_hits; return it->second; }
    nodes.push_back({op, a, b, v});
    return cse[k] = static_cast<int>(nodes.size()) - 1;
  }
};
inline thread_local Recorder* R = nullptr;

struct Rec {
  double v = 0.0;
  int id = -1;  // -1: constant (folded at record time)
  Rec() = default;
  Rec(double d) : v(d) {}
  Rec(int i) : v(i) {}
  Rec(double d, int i) : v(d), id(i) {}
  int slot() const { return id >= 0 ? id : R->constant(v); }
  Rec& operator+=(const Rec& o);
  Rec& operator-=(const Rec& o);
  Rec& operator*=(const Rec& o);
  Rec& operator/=(const Rec& o);
};
inline Rec bin(Op op, const Rec& x, const Rec& y, double v) {
  if (x.id < 0 && y.id < 0) { ++R->folded; return Rec(v); }
  return Rec(v, R->emit(op, x.slot(), y.slot(), v));
}
inline Rec operator+(const Rec& x, const Rec& y) { return bin(ADD, x, y, x.v + y.v); }
inline Rec operator-(const Rec& x, const Rec& y) { return bin(SUB, x, y, x.v - y.v); }
inline Rec operator*(const Rec& x, const Rec& y) { return bin(MUL, x, y, x.v * y.v); }
inline Rec operator/(const Rec& x, const Rec& y) { return bin(DIV, x, y, x.v / y.v); }
inline Rec operator-(const Rec& x) { return x.id < 0 ? Rec(-x.v) : Rec(-x.v, R->emit(NEG, x.id, -1, -x.v)); }
inline Rec exp(const Rec& x) { double v = std::exp(x.v); return x.id < 0 ? Rec(v) : Rec(v, R->emit(EXP, x.id, -1, v)); }
inline Rec log(const Rec& x) { double v = std::log(x.v); return x.id < 0 ? Rec(v) : Rec(v, R->emit(LOG, x.id, -1, v)); }
inline Rec sqrt(const Rec& x) { double v = std::sqrt(x.v); return x.id < 0 ? Rec(v) : Rec(v, R->emit(SQRT, x.id, -1, v)); }
inline bool cmp(Cmp c, const Rec& x, const Rec& y, bool res) {
  if (x.id >= 0 || y.id >= 0) R->guards.push_back({c, x.slot(), y.slot(), res});  // value-dependent branch: GUARD
  return res;
}
inline bool operator<(const Rec& x, const Rec& y) { return cmp(LT, x, y, x.v < y.v); }
inline bool operator<=(const Rec& x, const Rec& y) { return cmp(LE, x, y, x.v <= y.v); }
inline bool operator>(const Rec& x, const Rec& y) { return cmp(GT, x, y, x.v > y.v); }
inline bool operator>=(const Rec& x, const Rec& y) { return cmp(GE, x, y, x.v >= y.v); }
inline bool operator==(const Rec& x, const Rec& y) { return cmp(EQ, x, y, x.v == y.v); }
inline bool operator!=(const Rec& x, const Rec& y) { return cmp(NE, x, y, x.v != y.v); }
// |x| = guard(x >= 0) then x or -x: bit-identical value, and the branch becomes a recorded guard (affine under it).
inline Rec abs(const Rec& x) { if (x.id < 0) return Rec(std::abs(x.v)); return cmp(GE, x, Rec(0.0), x.v >= 0.0) ? x : -x; }
inline Rec& Rec::operator+=(const Rec& o) { return *this = *this + o; }
inline Rec& Rec::operator-=(const Rec& o) { return *this = *this - o; }
inline Rec& Rec::operator*=(const Rec& o) { return *this = *this * o; }
inline Rec& Rec::operator/=(const Rec& o) { return *this = *this / o; }
}  // namespace aj

namespace Eigen {
template <> struct NumTraits<aj::Rec> : NumTraits<double> {
  using Real = aj::Rec; using NonInteger = aj::Rec; using Nested = aj::Rec; using Literal = aj::Rec;
  enum { IsComplex = 0, IsInteger = 0, IsSigned = 1, RequireInitialization = 1, ReadCost = 1, AddCost = 3, MulCost = 3 };
};
}  // namespace Eigen
