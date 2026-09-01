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
  // calibrated bundle curves through the existing MultiCurveBook kernel.
  portfolio::MultiCurveBook to_book(const build::Date& value_date, const build::SwapConv& conv) const {
    portfolio::MultiCurveBook mb;
    for (const Trade& t : all_trades()) mb.positions.push_back(t.to_position(value_date, conv));
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

 private:
  std::string id_;
  CSA csa_;
  std::vector<Trade> trades_;
};

}  // namespace swaps::trade
