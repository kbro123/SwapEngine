// Thin CLI over swaps::api::run_json: read a bundle request (JSON) from a file argument or stdin, run
// the one-shot calibrate/sample/price/risk dispatch, and print the JSON response to stdout.
//
// This is a convenience for testing and for a subprocess-style backend; a C++ web server should link
// swaps_api and hold a swaps::api::BundleSession directly (keeps streaming state across requests).
//
//   ./swaps_api_cli request.json          # from a file
//   cat request.json | ./swaps_api_cli    # from stdin

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "swaps/api/bundle_api.hpp"

int main(int argc, char** argv) {
  std::string input;
  if (argc > 1) {
    std::ifstream f(argv[1]);
    if (!f) {
      std::cerr << "cannot open " << argv[1] << "\n";
      return 1;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    input = ss.str();
  } else {
    std::stringstream ss;
    ss << std::cin.rdbuf();
    input = ss.str();
  }
  std::cout << swaps::api::run_json(input) << "\n";
  return 0;
}
