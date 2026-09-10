#pragma once
// trade::NettingSet — the exposure-aggregation grouping of Trades (the `exposure` verb's "netting_sets").
// (trade::Book, the organisational desk -> strategy -> trades tree, lived here too until E6.1 (2026-09-10):
// it had no consumer and was deleted; a MultiCurveBook is built from trades directly.)
//
// NettingSet is the COUNTERPARTY/exposure view: a flat group of trades under one CSA — the exposure
// aggregation unit. This replaces the engine's hard-coded "netting set = the whole book" (xva/exposure.hpp)
// with a first-class object, and its discount curve is DETERMINED by the CSA (collateral currency's OIS),
// not a bare int role. A trade typically belongs to one Book (org) and one NettingSet (counterparty).

#include <string>
#include <utility>
#include <vector>

#include "swaps/build/conventions.hpp"     // SwapConv
#include "swaps/build/date.hpp"            // Date
#include "swaps/portfolio/portfolio.hpp"   // MultiCurveBook
#include "swaps/trade/csa.hpp"             // CSA
#include "swaps/trade/trade.hpp"           // Trade

namespace swaps::trade {

namespace portfolio = swaps::portfolio;
namespace build = swaps::build;

// A netting set: the exposure-aggregation unit — trades under one collateral agreement.
class NettingSet {
 public:
  NettingSet(std::string id, CSA csa) : id_(std::move(id)), csa_(std::move(csa)) {}

  const std::string& id() const { return id_; }
  const CSA& csa() const { return csa_; }
  // The discount curve for this set is the CSA's collateral-currency OIS — an object, not an int role.
  build::Index discount_index() const { return csa_.discount_index(); }
  std::string discount_index_id() const { return csa_.discount_index_id(); }

  NettingSet& add(Trade t) {
    trades_.push_back(std::move(t));
    return *this;
  }
  const std::vector<Trade>& trades() const { return trades_; }
  int count() const { return static_cast<int>(trades_.size()); }
  bool empty() const { return trades_.empty(); }

  // Materialize this netting set for valuation/exposure, with the CSA DECIDING the discount role: every
  // trade's discount_curve (and fixed-leg discounting) is overridden to `csa_discount_role` — the bundle
  // index of the collateral currency's OIS curve (the caller resolves discount_index_id() -> role via its
  // curve binding). Each trade still rolls under its own index's conventions. This is what makes the CSA
  // an OBJECT that actually reaches pricing, not an annotation.
  portfolio::MultiCurveBook to_book(const build::Date& value_date, int csa_discount_role) const {
    portfolio::MultiCurveBook mb;
    for (Trade t : trades_) {
      t.discount_curve = csa_discount_role;
      mb.positions.push_back(t.to_position(value_date));
    }
    return mb;
  }

 private:
  std::string id_;
  CSA csa_;
  std::vector<Trade> trades_;
};

}  // namespace swaps::trade
