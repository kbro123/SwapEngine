# Test taxonomy (E5, 2026-09-10)

Every file in `tests/` carries a `// E5 taxonomy: …` label in its first three lines (after the
`@oracle-test` / `@consistency-test` banner where one exists); `tools/check_test_taxonomy.sh` (run by
`tools/verify.sh`) fails the gate on an unlabelled file. A file may carry several labels; the FIRST is what it
mainly is. The label says what the file's assertions compare an engine number TO — the question every
review of the suite kept having to answer by reading the code.

| Label | What the reference is | Examples |
|---|---|---|
| **T1 oracle** | an INDEPENDENT number: QuantLib (`swaps_oracle_tests`, `tests/ORACLE_TESTS.md`), Rateslib, a published table | `sabr_ql_oracle_test`, `scheme_value_oracle_test`, `bond_reference_test` |
| **T2 calibration** | the optimum itself: zero residual / stationarity / recovery of a generating curve, rank-deficient and banded cases | `calibration_test`, `streaming_band_test`, `rank_safety_test` |
| **T3 cross-path parity** | ANOTHER engine path on the same inputs: compiled vs templated vs AAD vs pooled, streamer vs cold LM, bound book vs one-shot | `compiled_generic_test`, `portfolio_compiled_test`, `hotpath_shapes_test` |
| **T4 hot-path invariant** | a property of the hot path: allocation counts (malloc logger), determinism, structure_equal completeness | `alloc_free_test`, `api_hotpath_test`, `structure_fingerprint_test` |
| **T5 properties + value pins** | a HAND / closed-form literal, an identity, a finite difference, a monotonicity or shape property | `kernel_pins_test`, `fx_black_grid_test`, `cds_closed_form_test` |
| **T6 regression** | a bug that was reproduced and fixed: the test fails on the reverted code | `bug_hunt_2026_09_test`, `kernel_edges_test`, `api_status_test` |

Rules the labels imply:

* A T5 literal must be derivable WITHOUT the engine (hand arithmetic, a textbook formula, an independent
  script). A value captured from the engine's own output is a REGRESSION FREEZE and says so in the comment
  (e.g. `calendar_data_test`'s G20 count table, the legacy SABR copies in `rates_fx_grid_test`).
* A T3 parity test proves two paths agree, not that either is right — it needs a T1/T5 sibling somewhere.
* A tolerance looser than the quantity's natural precision states its reason inline (a discretisation, a
  bump-noise floor, a measured regression pin with the date) — `tests/tolerances.hpp` holds the shared
  constants. The E5.1 sweep (commit d942702) lists the ones that were fixed.
* T6 is earned by a mutation: `tools/mutate.py` re-introduces a curated set of bugs into scratch copies of
  the headers, compiles the tests that claim to pin them, and reports the kill rate (gate 90 %). Add a
  mutation when you fix a bug a test should have caught.
