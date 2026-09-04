#pragma once
// trade::Book and trade::NettingSet — the portfolio grouping objects that compose Trades.
//
// Book is the ORGANISATIONAL view: a hierarchy of trades and nested sub-books (desk -> strategy -> trades).
// It materializes into the engine's fast valuation form (portfolio::MultiCurveBook of Positions) via each
// trade's to_position, so the whole tree prices/risks through the existing hot path.
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

// A hierarchical portfolio of booked trades.
class Book {
 public:
  Book() = default;
  explicit Book(std::string name) : name_(std::move(name)) {}

  const std::string& name() const { return name_; }
  Book& add(Trade t) {
    trades_.push_back(std::move(t));
    return *this;
  }
  Book& add_subbook(Book b) {
    children_.push_back(std::move(b));
    return *this;
  }
  const std::vector<Trade>& trades() const { return trades_; }      // trades booked directly here
  const std::vector<Book>& children() const { return children_; }

  // Every trade in this book and all sub-books, flattened.
  std::vector<Trade> all_trades() const {
    std::vector<Trade> out = trades_;
    for (const auto& c : children_) {
      std::vector<Trade> ct = c.all_trades();
      out.insert(out.end(), ct.begin(), ct.end());
    }
    return out;
  }
  // Total trade count across the whole tree.
  int count() const {
    int n = static_cast<int>(trades_.size());
    for (const auto& c : children_) n += c.count();
    return n;
  }

  // Materialize the whole tree into the fast valuation book — one Position per trade, priced off the
  // calibrated bundle curves through the existing MultiCurveBook kernel. This overload applies ONE
  // caller-supplied convention to every trade — correct only for a single-index book; a mixed book
  // should use the per-trade-convention overload below.
  portfolio::MultiCurveBook to_book(const build::Date& value_date, const build::SwapConv& conv) const {
    portfolio::MultiCurveBook mb;
    for (const Trade& t : all_trades()) mb.positions.push_back(t.to_position(value_date, conv));
    return mb;
  }

  // Materialize with each trade rolling under ITS OWN index's conventions (Trade::to_position(vd) —
  // conventions DB via build::Index). The honest form for a mixed multi-index book.
  portfolio::MultiCurveBook to_book(const build::Date& value_date) const {
    portfolio::MultiCurveBook mb;
    for (const Trade& t : all_trades()) mb.positions.push_back(t.to_position(value_date));
    return mb;
  }

 private:
  std::string name_;
  std::vector<Trade> trades_;
  std::vector<Book> children_;
};

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
