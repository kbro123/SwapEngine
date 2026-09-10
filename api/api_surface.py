"""SINGLE SOURCE OF TRUTH for the SwapEngine binding surface.

Every language interface — the pybind11 module (web), the typed `.pyi` stub, and the Excel/C UDFs over the
`extern "C" swaps_run_json` seam — is GENERATED from this one descriptor by `tools/gen_bindings.py`. Add or
change an engine operation here (and implement the C++ method in the engine), regenerate, and every binding
updates together; `tools/check_bindings.sh` fails the build if a generated file drifts from this descriptor.

Type kinds (arg + return), mapped by the generator to each target language:

    STR     std::string            <-> Python str / Excel text            / JSON string
    INT     int                    <-> Python int / Excel number          / JSON int
    SCALAR  double                 <-> Python float / Excel number        / JSON number
    BOOL    bool                   <-> Python bool                        / JSON bool
    VEC     list[float]            <-> Eigen::VectorXd (to_vec/to_list)   / JSON number[]
    IVEC    list[int]                                                     / JSON int[]
    MATRIX  list[list[float]]      <-> Eigen::MatrixXd (to_matrix)        / JSON number[][]
    REG     optional dict          <-> api::RegSpec (reg_from)            / JSON {lambda,curves,tension,sigma}
    VOID    None
    STRUCT:<Name>  a dict/JSON object whose fields are listed in STRUCTS[<Name>]

An op is either REGULAR (`cpp` = a BundleSession method; body is `MARSHAL(sess_.cpp(conv(args)))`) or CUSTOM
(`body` = a raw C++ expression/statement block, for the few irregular shapes). CUSTOM ops still declare their
args + ret so the `.pyi`/Excel bindings stay typed. `verb` names the `run_json` request key when the op is
reachable statelessly (Excel goes through that); ops with `verb=None` are pybind-only (e.g. streaming state).
"""

# ---- return structs: one field list generates both the pybind py::dict and the JSON object ----------
STRUCTS = {
    "PortfolioReprice": [("npv", "SCALAR"), ("pv01", "SCALAR"), ("price_us", "SCALAR"), ("n", "INT")],
    "PortfolioRisk": [("npv", "SCALAR"), ("curve_grad", "VEC"), ("ladder", "VEC"),
                      ("risk_us", "SCALAR"), ("n", "INT")],
    # Batched swaption vol cube (flat SoA): per-cell forward/annuity/expiry + per-point strike/vol/price/greeks.
    # All arrays are VEC(double) so one marshaller covers every field; point i belongs to cell point_cell[i].
    "VolCube": [("cell_forward", "VEC"), ("cell_annuity", "VEC"), ("cell_expiry_years", "VEC"),
                ("point_cell", "VEC"), ("strike", "VEC"), ("moneyness_bp", "VEC"), ("normal_vol", "VEC"),
                ("price", "VEC"), ("vega", "VEC"), ("delta", "VEC"), ("gamma", "VEC"), ("vanna", "VEC"),
                ("volga", "VEC"), ("payer", "VEC"),
                ("price_us", "SCALAR"), ("n_cells", "INT"), ("n_points", "INT")],
}

# Session constructor.
CTOR = {"args": [("bundle_json", "STR")],
        "doc": "Construct from a bundle object-graph JSON string ({curves:[...], instruments:[...]})."}

# ---- operations. Each: name, [regular: cpp | custom: body], args, ret, verb, prop?, doc ----------------
# args: list of (name, kind) or (name, kind, default_literal) — a default makes the arg optional.
# A REG arg is always optional (default None). prop=True registers a def_property_readonly (no args).
METHODS = [
    # regular struct/vector/matrix returns (generated wrapper = MARSHAL(sess_.<cpp>(conv args)))
    {"name": "anchor_market", "cpp": "problem().market", "args": [], "ret": "VEC", "verb": None,
     "doc": "Self-consistent target quotes in residual order — the streaming anchor q0."},
    {"name": "model_quote", "cpp": None, "args": [("instrument_json", "STR")], "ret": "SCALAR", "verb": None,
     "body": "return sess_.model_quote(api::instrument_from_json(json::parse(instrument_json)));",
     "doc": "Model quote (par rate/spread/future rate/FX fwd) of an arbitrary instrument off the curves."},
    {"name": "same_structure", "cpp": None, "args": [("bundle_json", "STR")], "ret": "INT", "verb": None,
     "body": "return sess_.same_structure(api::bundle_from_json(json::parse(bundle_json))) ? 1 : 0;",
     "doc": "1 if `bundle_json` differs from this session's problem ONLY in market levels (warm-tickable "
            "over the compiled W-cache); 0 if the topology changed (a recompile is needed). The OO/hot-path "
            "switch a stateful Model asks before choosing warm-vs-recompile."},
    {"name": "price_portfolio", "cpp": "price_portfolio_json", "args": [("book_json", "STR")],
     "ret": "STRUCT:PortfolioReprice", "verb": "portfolio",
     "doc": "Reprice a multi-curve+xccy book off the calibrated curves -> {npv,pv01,price_us,n}."},
    {"name": "bind_portfolio", "cpp": None, "args": [("book_json", "STR")], "ret": "VOID", "verb": None,
     "body": "sess_.bind_portfolio(api::book_from_json(json::parse(book_json)));",
     "doc": "Bind a book and build its compiled W-cache reprice twin ONCE, for the streaming regime "
            "(the book repriced every tick against the recalibrating curve). Amortizes the W build across "
            "ticks -> pair with reprice_bound(); a one-shot reprice keeps using price_portfolio."},
    {"name": "reprice_bound", "cpp": "reprice_bound", "args": [], "ret": "STRUCT:PortfolioReprice",
     "verb": None,
     "doc": "Reprice the bound book (bind_portfolio first) off the current calibrated x, reusing the cached "
            "compiled twin -> {npv,pv01,price_us,n}. ~300x the per-tick price_portfolio on the streaming path; "
            "npv==price_portfolio to 1e-9, pv01 to 1e-8. Rebuilds lazily after a structural bundle change."},
    {"name": "price_portfolio_risk", "cpp": "price_portfolio_risk_json",
     "args": [("book_json", "STR"), ("reg", "REG")], "ret": "STRUCT:PortfolioRisk", "verb": "portfolio_risk",
     "doc": "Book risk -> {npv, curve_grad=dP/dx, ladder=dP/dq, risk_us, n}. reg smooths the RISK operator."},
    {"name": "price_vol_cube", "cpp": "price_vol_cube_json", "args": [("cube_json", "STR")],
     "ret": "STRUCT:VolCube", "verb": "vol_cube",
     "doc": ("Batched swaption vol cube off the calibrated curve: {value_date, index?, cells:[{expiry, tenor, "
             "sabr?|normal_vol?, strikes?|moneyness_bp?|atm?, payer?}]} -> flat SoA {cell_forward, cell_annuity, "
             "cell_expiry_years, point_cell, strike, moneyness_bp, normal_vol, price, vega, delta, gamma, payer, "
             "price_us, n_cells, n_points}. Reuses the calibrated/streaming session (reprice a live surface with "
             "no recalibration).")},
    {"name": "jacobian", "cpp": "jacobian", "args": [], "ret": "MATRIX", "verb": None,
     "doc": "Calibration Jacobian J = dq/dx (n_res x n_knots)."},
    {"name": "risk_operator", "cpp": "risk_operator", "args": [("reg", "REG")], "ret": "MATRIX", "verb": "risk",
     "doc": "Analytic risk operator M = (JtJ+RtR)^-1 Jt = dx/dq (n_knots x n_res)."},
    {"name": "transform_matrix", "cpp": "transform_matrix_json",
     "args": [("source_bundle_json", "STR"), ("reg", "REG")], "ret": "MATRIX", "verb": "transform",
     "doc": "Cross-bundle risk transform T = cross_jacobian(source).risk_operator(reg) (adaptor Jacobian)."},
    {"name": "cross_jacobian", "cpp": "cross_jacobian_json", "args": [("source_bundle_json", "STR")],
     "ret": "MATRIX", "verb": None,
     "doc": "Raw cross Jacobian dq_source/dx_this (n_res_source x n_knots_this)."},
    {"name": "apply_transform", "cpp": None, "args": [("ladder", "VEC"), ("T", "MATRIX")], "ret": "VEC",
     "verb": None,
     "body": ("const std::size_t nr = T.size(), nc = nr ? T[0].size() : 0;\n"
              "    if (ladder.size() != nr) throw std::invalid_argument(\n"
              "        \"apply_transform: ladder length must equal the transform's row count\");\n"
              "    std::vector<double> out(nc, 0.0);\n"
              "    for (std::size_t i = 0; i < nr; ++i)\n"
              "      for (std::size_t j = 0; j < nc; ++j) out[j] += ladder[i] * T[i][j];\n"
              "    return out;"),
     "doc": "Apply a transform: delta_this = delta_source . T (row-vector . matrix)."},
    # side-effecting / stateful (custom bodies; pybind-only)
    {"name": "update", "cpp": "stream_update", "args": [("market", "VEC")], "ret": "VEC", "verb": None,
     "doc": "One frozen-Newton streaming tick to a new market -> the new knot vector x."},
    {"name": "recalibrate", "cpp": None, "args": [("market", "VEC"), ("reg", "REG")], "ret": "VEC",
     "verb": None,
     "body": "sess_.recalibrate(to_vec(market), reg_from(reg)); return to_list(sess_.x());",
     "doc": "Warm-recalibrate to a new market (custom-region/non-linear fallback) -> the new x."},
    {"name": "rebind", "cpp": None, "args": [("bundle_json", "STR"), ("reg", "REG")], "ret": "VEC", "verb": None,
     "body": "sess_.rebind(api::bundle_from_json(json::parse(bundle_json)), reg_from(reg)); return to_list(sess_.x());",
     "doc": "Warm re-solve to a structurally-identical bundle, updating market targets AND soft-quote bands "
            "(the complex-quote-aware warm path) -> the new x. Throws if the residual count differs (structural)."},
    {"name": "start_streaming", "cpp": "start_streaming", "args": [("reg", "REG"), ("step_tol", "SCALAR", "0.0")],
     "ret": "VOID", "verb": None,
     "doc": "Anchor the streaming calibrator at the current x (step_tol>0 loosens the corrector tolerance)."},
    {"name": "set_evaluation_date", "cpp": None, "args": [("serial", "INT")], "ret": "INT", "verb": None,
     "body": "sess_.set_evaluation_date(serial); return sess_.n_unresolved();",
     "doc": "Set the evaluation date (Unix-day serial: days since 1970-01-01); re-resolve fixing schedules. Returns n still un-priceable."},
    # CUSTOM irregular shapes (bespoke marshalling; still typed for pyi/excel via ret='CUSTOM')
    {"name": "calibrate", "custom": True,
     "args": [("x0", "VEC", "None"), ("reg", "REG")], "ret": "CUSTOM:dict", "verb": None,
     "doc": "Cold calibrate -> {iterations,rms_residual,stationarity,info,solve_micros,rank_deficiency,x}. rank_deficiency>0 = the instrument set under-determines the curve (null states sit at the seed; add an instrument or enable smoothing)."},
    {"name": "sample", "custom": True, "args": [("times", "VEC")], "ret": "CUSTOM:list", "verb": "sample_times",
     "doc": "Sample every curve on a time grid -> [{currency,t,discount,zero,forward}, ...]."},
    {"name": "set_fixings", "custom": True, "args": [("index", "STR"), ("rows", "CUSTOM:pairs")], "ret": "INT",
     "verb": None, "doc": "Upsert fixings for an index (rows=[(unix_day_serial,rate_decimal),...]; serial = days since 1970-01-01). Returns n un-priceable."},
    {"name": "knot_times", "custom": True, "args": [], "ret": "CUSTOM:list", "verb": None,
     "doc": "Per-curve sorted knot times (meeting+back) as a list of lists."},
    {"name": "currencies", "custom": True, "args": [], "ret": "IVEC", "verb": None,
     "doc": "Per-curve currency indices."},
]

# ---- scalar readonly properties (no args) -----------------------------------------------------------
PROPS = [
    ("n_unresolved", "INT"), ("n_knots", "INT"), ("n_residuals", "INT"), ("n_curves", "INT"),
    ("has_fx", "BOOL"), ("has_modular", "BOOL"), ("has_nonlinear", "BOOL"), ("needs_recalibrate", "BOOL"),
    ("streaming", "BOOL"), ("last_risk_us", "SCALAR"), ("last_solve_us", "SCALAR"),
    ("last_price_us", "SCALAR"), ("stream_avg_us", "SCALAR"), ("stream_ticks", "INT"),
    ("last_newton_steps", "INT"), ("last_refreshes", "INT"), ("last_drift", "SCALAR"),
    ("last_converged", "BOOL"), ("last_rescales", "INT"),
    # WHY the last stream tick ended (calibration::StreamStatus): 0 converged, 1 step cap, 2 refresh cap,
    # 3 band re-scale budget, 4 non-finite, 5 diverged. (last_reason(), the text, needs a STR prop kind --
    # TASKS-API A0.)
    ("last_status", "INT"),
]
# props whose C++ getter lives on BundleSession vs is derived from problem()
PROP_CPP = {
    "n_unresolved": "sess_.n_unresolved()", "n_knots": "sess_.problem().n_knots()",
    "n_residuals": "sess_.problem().n_residuals()", "n_curves": "sess_.problem().n_curves()",
    "has_fx": "sess_.has_fx()", "has_modular": "sess_.has_modular()", "has_nonlinear": "sess_.has_nonlinear()",
    "needs_recalibrate": "sess_.needs_recalibrate()", "streaming": "sess_.streaming()",
    "last_risk_us": "sess_.last_risk_us()", "last_solve_us": "sess_.last_solve_us()",
    "last_price_us": "sess_.last_price_us()", "stream_avg_us": "sess_.stream_avg_us()",
    "stream_ticks": "sess_.stream_ticks()", "last_newton_steps": "sess_.last_newton_steps()",
    "last_refreshes": "sess_.last_refreshes()", "last_drift": "sess_.last_drift()",
    "last_converged": "sess_.last_converged()", "last_rescales": "sess_.last_rescales()",
    "last_status": "sess_.last_status()",
}

# ---- module-level free functions --------------------------------------------------------------------
FREE = [
    {"name": "run_json", "args": [("request", "STR")], "ret": "STR",
     "doc": "One-shot stateless dispatch: {bundle, x0?, regularize?, ...} -> JSON response."},
    {"name": "market_from_x", "args": [("bundle_json", "STR"), ("x", "VEC")], "ret": "VEC",
     "doc": "Model quotes of every instrument priced at knot vector x (self-consistent template markets)."},
]


# ---- run_json dispatch (stateless JSON seam) — consumed by BOTH the engine's run_json generator AND the
# Excel/C UDF generator, so the two never drift. Each generatable verb arm: a request key -> a BundleSession
# call -> a response section. `payload`:
#   "bool"          bare `true` guard, no payload (the call takes just reg)
#   "book"          book_from_json(o.at(key))                      -> the call's first arg
#   "source_bundle" bundle_from_json(o.at(key).at("source_bundle"))-> the call's first arg
# `reg`=True appends the top-level RegSpec; `ret` drives the response serialization (STRUCT:* field list or a
# MATRIX of nested arrays). `payload` "json" passes the serialized sub-object string to a *_json method (for
# ops that parse their own bespoke sub-schema, e.g. the vol cube). The always-on stages
# (bundle/x0/regularize/calibrate/x) and the irregular `price` and `sample_times` arms stay hand-written in
# bundle_api.cpp — their shapes are bespoke.
RUN_JSON = [
    {"key": "portfolio", "resp": "portfolio", "cpp": "price_portfolio", "payload": "book", "reg": False,
     "ret": "STRUCT:PortfolioReprice"},
    {"key": "vol_cube", "resp": "vol_cube", "cpp": "price_vol_cube_json", "payload": "json", "reg": False,
     "ret": "STRUCT:VolCube"},
    {"key": "portfolio_risk", "resp": "portfolio_risk", "cpp": "price_portfolio_risk", "payload": "book",
     "reg": True, "ret": "STRUCT:PortfolioRisk"},
    {"key": "risk", "resp": "risk_operator", "cpp": "risk_operator", "payload": "bool", "reg": True,
     "ret": "MATRIX"},
    {"key": "transform", "resp": "transform", "cpp": "transform_matrix", "payload": "source_bundle",
     "reg": True, "ret": "MATRIX"},
]
