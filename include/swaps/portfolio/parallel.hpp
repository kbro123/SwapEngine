#pragma once
// COHERENT parallel portfolio reprice -- the data-parallel pricing side of the async split (Q2). A large
// book is partitioned into N contiguous slices, each a self-contained CompiledPortfolio with its OWN
// scratch; a reprice fans the slices across threads, ALL pricing the SAME pinned knot vector `x`, and
// writes each slice's per-swap NPVs into a disjoint output segment. Because a position's NPV depends only
// on `x` and its own cashflows (never on the other positions in the book), the sliced result is
// BIT-IDENTICAL to a single-thread CompiledPortfolio over the whole book -- so the parallel cut is
// DETERMINISTIC and internally COHERENT (one curve version across every thread).
//
// Usage with the live feed: snapshot ONCE (pin a version) then reprice -- that single snapshot is the
// coherent point-in-time curve every worker prices off:
//     feed.snapshot(x);                       // one version, shared read-only
//     const auto& npvs = book.reprice(x);     // N threads, all off `x`, per-swap NPVs in book order

#include <Eigen/Core>

#include <algorithm>
#include <future>
#include <memory>
#include <vector>

#include "swaps/parallel/thread_pool.hpp"
#include "swaps/portfolio/compiled.hpp"
#include "swaps/portfolio/portfolio.hpp"

namespace swaps::portfolio {

class ParallelPortfolio {
 public:
  // Partition `pf` into `n_slices` contiguous CompiledPortfolio blocks (capped at one position per slice).
  // Optional `pool`: a persistent thread pool to reprice on (reused across calls, no per-reprice thread
  // creation). nullptr => a std::async fan-out per reprice.
  ParallelPortfolio(const std::vector<curve::CurveModule>& modules, const Portfolio& pf, int n_slices,
                    swaps::parallel::ThreadPool* pool = nullptr)
      : n_swaps_(static_cast<int>(pf.positions.size())), pool_(pool) {
    const int P = n_swaps_;
    n_slices = std::max(1, std::min(n_slices, P));
    const int per = (P + n_slices - 1) / n_slices;  // ceil, so slices are as even as possible
    for (int lo = 0; lo < P; lo += per) {
      const int hi = std::min(P, lo + per);
      Portfolio slice;
      slice.positions.assign(pf.positions.begin() + lo, pf.positions.begin() + hi);
      slices_.push_back(std::make_unique<CompiledPortfolio>(modules, slice));
      offset_.push_back(lo);
    }
  }

  int n_swaps() const { return n_swaps_; }
  int n_slices() const { return static_cast<int>(slices_.size()); }

  // Coherent parallel reprice off ONE `x`: per-swap NPVs in original book order. Each slice reads `x`
  // (immutable) and writes a DISJOINT output segment, so there is no data race and the result is
  // order-independent -> bit-identical to a serial full-book reprice. Allocation-free after warm-up.
  const Eigen::VectorXd& reprice(const Eigen::VectorXd& x) const {
    out_.resize(n_swaps_);
    if (slices_.size() == 1) {  // no threading to be had
      out_ = slices_[0]->npv(x);
      return out_;
    }
    auto price_slice = [this, &x](int s) {
      const Eigen::VectorXd& npv = slices_[s]->npv(x);  // this slice's scratch (thread-private)
      out_.segment(offset_[s], npv.size()) = npv;       // disjoint segment -> race-free
    };
    if (pool_) {  // persistent pool: reuse workers across reprices
      pool_->parallel_for(static_cast<int>(slices_.size()), price_slice);
    } else {  // fallback: one std::async per slice
      std::vector<std::future<void>> futs;
      futs.reserve(slices_.size());
      for (std::size_t s = 0; s < slices_.size(); ++s)
        futs.push_back(std::async(std::launch::async, [&price_slice, s] { price_slice(static_cast<int>(s)); }));
      for (auto& f : futs) f.get();
    }
    return out_;
  }

  double total_npv(const Eigen::VectorXd& x) const { return reprice(x).sum(); }

 private:
  std::vector<std::unique_ptr<CompiledPortfolio>> slices_;
  std::vector<int> offset_;  // first book index of each slice
  int n_swaps_;
  swaps::parallel::ThreadPool* pool_;  // optional persistent pool (nullptr => std::async fan-out)
  mutable Eigen::VectorXd out_;  // reusable per-reprice result (disjoint segments written by workers)
};

}  // namespace swaps::portfolio
