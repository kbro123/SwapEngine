#pragma once
// Minimal CSV reader for the committed golden files. Deliberately dependency-free.
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace swaps::golden {

/// Every non-comment, non-empty row, split on ','. Comments start with '#'.
inline std::vector<std::vector<std::string>> read_csv(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open golden file: " + path);
  std::vector<std::vector<std::string>> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::vector<std::string> cells;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) cells.push_back(cell);
    if (!cells.empty()) rows.push_back(std::move(cells));
  }
  if (rows.empty()) throw std::runtime_error("golden file is empty: " + path);
  return rows;
}

inline double as_double(const std::string& s) { return std::stod(s); }

}  // namespace swaps::golden
