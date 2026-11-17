#!/usr/bin/env python3
"""Bring the deposition .esm files in line with the reference AtmosphericDeposition
package (jialinl6/DepositionMTK.jl, branch superfast-gas-only, commit cbe7d7db).

  wesley_dry_gas.esm                regenerated: upstream DryDepositionGasFractional
                                    (11-class Wesely mosaic, all 132 species)
  wet_deposition.esm                patched: EMEP ratios x 1e6, in-cloud ratios, fixed h_s, no lev
  wesley1989/surface_resistance.esm patched: r_s hot-closure fix, r_ac / r_gsS floors
  dry_aerosol.esm                   patched: k = v * g * rho_a / del_P

(superfast_deposition_sink.esm - NO not deposited, CH2O on k_CH2O - is hand-formatted and was
edited by hand.)

The patches apply to the files as they were before this update (AtmosphericDeposition.jl@e4468c79
era); run them on a clean checkout of those files.

Input: DATADIR/deposition_ref.json from extract_deposition_data.jl (same directory).
Run:
  julia --project=ENV scripts/_archive/migrations/extract_deposition_data.jl DATADIR
  python3 scripts/_archive/migrations/gen_deposition_esm.py DATADIR
"""
import json
import sys
from pathlib import Path

DATA = Path(sys.argv[1])
REF = json.loads((DATA / "deposition_ref.json").read_text())
REPO = Path(__file__).resolve().parents[3]
DEP = REPO / "components" / "atmospheric_deposition"
UPSTREAM = "jialinl6/DepositionMTK.jl@cbe7d7db (branch superfast-gas-only)"
TODAY = "2026-10-28"
INF = 1.0e25          # upstream `inf` (wesley1989.jl); the tables hold Float32(1e25)
CLASSES = [f[2:] for f in REF["fractions"]]   # urban, agricultural, ..., rockyshrubs
TABLES = REF["tables"]


# --------------------------------------------------------------------------- AST
def Op(o, *a):
    return {"op": o, "args": list(a)}


def fn(name, *a):
    return {"op": "fn", "name": name, "args": list(a)}


def C(v):
    return {"op": "const", "args": [], "value": v}


def add(*a):
    return a[0] if len(a) == 1 else Op("+", *a)


def mul(*a):
    return Op("*", *a)


def div(a, b):
    return Op("/", a, b)


def ifelse(c, a, b):
    return Op("ifelse", c, a, b)


class Builder:
    def __init__(self):
        self.vars, self.eqs = {}, []

    def param(self, name, units, default, desc):
        self.vars[name] = {"type": "parameter", "units": units, "default": default, "description": desc}

    def obs(self, name, units, expr, desc):
        assert name not in self.vars, name
        self.vars[name] = {"type": "unknown", "units": units, "description": desc}
        self.eqs.append({"lhs": name, "rhs": expr})


def dumps(o, ind=0):
    """json.dumps(indent=2), with short nodes, arrays of numbers and whole equations kept on one line."""
    pad, sub = "  " * ind, "  " * (ind + 1)
    if isinstance(o, (dict, list)):
        flat = json.dumps(o, ensure_ascii=False)
        # Short nodes, and every equation, stay on one line.
        if len(flat) <= 100 or (isinstance(o, dict) and set(o) == {"lhs", "rhs"}):
            return flat
    if isinstance(o, dict):
        items = [f"{sub}{json.dumps(k, ensure_ascii=False)}: {dumps(v, ind + 1)}" for k, v in o.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(o, list):
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in o):
            return "[" + ", ".join(json.dumps(x) for x in o) + "]"
        return "[\n" + ",\n".join(sub + dumps(x, ind + 1) for x in o) + "\n" + pad + "]"
    return json.dumps(o, ensure_ascii=False)


def write(path, d, indent=None):
    """indent=None: this script's compact layout (new files); an int: the patched file's own indentation."""
    text = (dumps(d) if indent is None else json.dumps(d, indent=indent, ensure_ascii=False)) + "\n"
    assert json.loads(text) == d
    path.write_text(text)
    print(f"wrote {path.relative_to(REPO)} ({path.stat().st_size / 1024:.0f} KB)")


def assertion(var, time, expected, **kw):
    a = {"variable": var, "time": time, "expected": expected}
    a.update(kw)
    return a


# ============================================================== dry deposition (gas)
def season_table(name, c):
    """Wesely table `name`, land-use class c (1-based), as a function of `season`.
    interp.linear on the integer season knots returns the table entry exactly; upstream
    indexes the table at round(Int, season), so season is an integer as there."""
    vals = [TABLES[name][s][c - 1] for s in range(5)]
    if all(v == vals[0] for v in vals):
        return vals[0]
    return fn("interp.linear", C(vals), C([1.0, 2.0, 3.0, 4.0, 5.0]), "season")


def build_dry():
    b = Builder()
    b.param("season", "1", 1, "Wesely (1989) season index: 1 midsummer, 2 autumn, 3 late autumn, 4 winter, "
            "5 transitional spring. An integer, as upstream's Int parameter; upstream's GEOS-FP coupling sets it "
            "from the month (season_at: Jun-Aug 1, Sep-Oct 2, Nov 3, Dec-Feb 4, Mar-May 5).")
    b.param("z", "m", 60.0, "Reference height for the aerodynamic resistance: the mid-point of level 1 [m]. "
            "Upstream's GEOS-FP coupling uses the hypsometric first-level midpoint height.")
    b.param("del_P", "Pa", 1520.0, "Pressure thickness of level 1 [Pa], for k = v*g*rho_A/del_P.")
    b.param("z0", "m", 0.04, "Surface roughness length [m]. Upstream z₀.")
    b.param("u_star", "m/s", 0.44, "Friction velocity [m/s].")
    b.param("L", "m", 0.0, "Monin-Obukhov length [m]; L = 0 is the neutral-stability sentinel.")
    b.param("rho_A", "kg/m^3", 1.2, "Air density [kg/m^3]. Upstream ρA.")
    b.param("G", "W/m^2", 300.0, "Solar irradiation at the surface [W/m^2].")
    b.param("Ts", "K", 298.0, "Surface air temperature [K]; the Wesely resistances use Ts_c = Ts - 273.")
    b.param("theta", "1", 0.0, "Slope of the local terrain [radians]. Upstream θ.")
    b.param("lev", "1", 1, "Model level index; deposition acts only at lev == 1.")
    for name in CLASSES:
        c = CLASSES.index(name) + 1
        b.param(f"f_{name}", "1", 1.0 if name == "mixedforest" else 0.0,
                f"Area fraction of Wesely (1989) land-use class {c} ({name}). Upstream's GEOS-FP coupling binds the "
                "11 fractions to a MODIS MCD12C1 (2016) CONUS 0.05° land-use map conservatively regridded onto the "
                "model grid; they should sum to 1. Upstream default: mixed forest 1, the rest 0.")
    b.param("g_const", "m/s^2", 9.81, "Gravitational acceleration [m/s^2]. Upstream @constants g.")
    b.param("kappa", "1", 0.4, "Von Karman constant. Upstream @constants κ.")

    # Aerodynamic resistance (Seinfeld & Pandis 2006 eqs. 19.13-19.14), upstream ra().
    b.obs("Ts_c", "degC", Op("-", "Ts", 273.0), "Surface air temperature in °C, Ts - 273 (upstream Ts*T_unitless - 273).")
    b.obs("L2_eps", "m^2", add(mul("L", "L"), 1e-300),
          "L^2 + 1e-300: lets zeta = z*L/L2_eps reproduce upstream's ifelse(L == 0, 0, z/L) without dividing by zero.")
    b.obs("zeta", "1", div(mul("z", "L"), "L2_eps"), "Stability parameter z/L (0 when L = 0).")
    b.obs("zeta_0", "1", div(mul("z0", "L"), "L2_eps"), "Stability parameter z0/L (0 when L = 0).")
    inv_ku = div(1, mul("kappa", "u_star"))
    b.obs("ra_neutral_stable", "s/m",
          ifelse(Op("<", 0, "zeta"), mul(inv_ku, add(Op("log", div("z", "z0")), mul(4.7, Op("-", "zeta", "zeta_0")))),
                 mul(inv_ku, Op("log", div("z", "z0")))),
          "Aerodynamic resistance for neutral or stable conditions (upstream rₐ_1).")
    b.obs("i_unstable", "1", ifelse(Op("<", "zeta", 0), 1, 0), "1 when unstable (zeta < 0), else 0.")
    b.obs("eta_0", "1", Op("^", Op("-", 1, mul(15, "zeta_0", "i_unstable")), 0.25), "(1 - 15*zeta_0)^(1/4), unstable only.")
    b.obs("eta", "1", Op("^", Op("-", 1, mul(15, "zeta", "i_unstable")), 0.25), "(1 - 15*zeta)^(1/4), unstable only.")
    psi = Op("log", div(mul(add(Op("^", "eta_0", 2), 1), Op("^", add("eta_0", 1), 2)),
                        mul(add(Op("^", "eta", 2), 1), Op("^", add("eta", 1), 2))))
    b.obs("Ra", "s/m",
          ifelse(Op("<", "zeta", 0),
                 mul(inv_ku, add(Op("log", div("z", "z0")), psi, mul(2, Op("-", Op("atan", "eta"), Op("atan", "eta_0"))))),
                 "ra_neutral_stable"),
          "Aerodynamic resistance [s/m] (upstream ra(z, z₀, u_star, L)).")
    b.obs("mu_air", "kg/(m*s)", div(mul(1.458e-06, Op("^", "Ts", 1.5)), add("Ts", 110.4)),
          "Dynamic viscosity of air [kg/(m*s)] (upstream mu).")
    b.obs("Dh2o", "m^2/s", add(-2.775e-06, mul(4.479e-08, "Ts"), mul(1.656e-10, Op("^", "Ts", 2))),
          "Molecular diffusivity of water vapour in air [m^2/s] (upstream dH2O).")
    b.obs("i_lev", "1", ifelse(Op("==", "lev", 1), 1, 0), "1 at level 1, else 0 (upstream i = ifelse(lev == 1, 1, 0)).")
    b.obs("rdc", "s/m", div(mul(100.0, add(1.0, div(1000.0, add("G", 10.0)))), add(1.0, mul(1000.0, "theta"))),
          "Buoyant-convection mixing resistance [s/m], Wesely (1989) eq. 5 (upstream r_dc).")
    b.obs("correction", "s/m", ifelse(Op("<", "Ts_c", 0.0), mul(1000.0, Op("exp", Op("-", Op("-", "Ts_c"), 4))), 0.0),
          "Cold-surface correction [s/m] added to r_lux, r_clx and r_gsx below 0 °C (Wesely 1989 p. 4).")

    # Per land-use class: Wesely table entries at the current season, and stomatal resistance.
    for c, name in enumerate(CLASSES, start=1):
        for tbl, floor in (("r_i", False), ("r_lu", False), ("r_ac", True), ("r_gsS", True),
                           ("r_gsO", False), ("r_clS", False), ("r_clO", False)):
            e = season_table(tbl, c)
            if floor:
                e = Op("max", e, 1.0)
            b.obs(f"{tbl}_{name}", "s/m", e,
                  f"Wesely (1989) {tbl} for class {c} ({name}) at the current season"
                  + (", floored at 1 s/m as upstream (after GEOS-Chem drydep_mod)." if floor else "."))
        expr = mul(f"r_i_{name}", add(1, Op("^", div(200.0, add("G", 1.0)), 2)),
                   div(400.0, mul("Ts_c", Op("-", 40.0, "Ts_c"))))
        b.obs(f"rs_{name}", "s/m",
              ifelse(Op("<=", "Ts_c", 0.1), INF, ifelse(Op(">=", "Ts_c", 39.9), INF, expr)),
              f"Bulk canopy stomatal resistance [s/m], class {c} ({name}), Wesely (1989) eq. 3: stomata close "
              "(1e25) below 0.1 °C and at or above 39.9 °C (upstream r_s, including its hot-closure fix).")

    # Per species: Ra + Rb, then the area-weighted deposition velocity and rate.
    for sp in REF["species"]:
        n, D, H, fo = sp["name"], sp["Dh2oPerDx"], sp["Hstar"], sp["Fo"]
        sc = div("mu_air", mul("rho_A", div("Dh2o", D)))
        b.obs(f"RaRb_{n}", "s/m", add("Ra", div(mul(5, Op("^", sc, 2.0 / 3.0)), "u_star")),
              f"Ra + Rb for {n}: aerodynamic plus quasi-laminar resistance, Rb = 5*Sc^(2/3)/u_star with "
              f"Sc = mu/(rho_A*Dh2o/{D}) (S&P 2006 eq. 19.17). The same for every land-use class.")
        rmx = 1.0 / (H / 3000.0 + 100.0 * fo)
        lux_den = 1.0e-5 * H + fo
        terms = []
        for c, name in enumerate(CLASSES, start=1):
            rsmx = add(mul(f"rs_{name}", D), rmx)
            rlux = add(div(f"r_lu_{name}", lux_den), "correction")
            if sp["isSO2"]:
                rclx, rgsx = f"r_clS_{name}", f"r_gsS_{name}"
            elif sp["isO3"]:
                rclx, rgsx = f"r_clO_{name}", f"r_gsO_{name}"
            else:
                rclx = div(1.0, add(div(H, mul(1.0e5, f"r_clS_{name}")), div(fo, f"r_clO_{name}")))
                rgsx = div(1.0, add(div(H, mul(1.0e5, f"r_gsS_{name}")), div(fo, f"r_gsO_{name}")))
            rc = div(1.0, add(div(1.0, rsmx), div(1.0, rlux), div(1.0, add("rdc", add(rclx, "correction"))),
                              div(1.0, add(f"r_ac_{name}", add(rgsx, "correction")))))
            rc = Op("min", Op("max", rc, 10.0), 9999.0)
            terms.append(div(f"f_{name}", add(f"RaRb_{n}", rc)))
        kind = "SO2 tables" if sp["isSO2"] else "O3 tables" if sp["isO3"] else f"Hstar = {H:g}, f0 = {fo:g}"
        b.obs(f"v_{n}", "m/s", mul("i_lev", add(*terms)),
              f"{n} dry deposition velocity [m/s] ({sp['data']}: Dh2o/Dx = {D:g}, {kind}): "
              "sum over the 11 classes of f_c / (Ra + Rb + Rc_c), Rc_c the Wesely (1989) surface resistance for "
              "class c clamped to [10, 9999] s/m (upstream FractionalWesleyVd).")
        b.obs(f"k_{n}", "1/s", div(mul(f"v_{n}", "g_const", "rho_A"), "del_P"),
              f"{n} first-order dry deposition rate [1/s] = v_{n}*g*rho_A/del_P.")

    tests = []
    sf = ["O3", "NO2", "HNO3", "H2O2", "CH2O"]
    spot = sf + ["HCHO", "SO2", "NO", "PAN", "N2O5", "NH3"]
    names = [s["name"] for s in REF["species"]]
    for case in REF["dry_cases"]:
        p, o = case["p"], case["out"]
        full = case["id"] == "midwest_mosaic_summer_unstable"
        pick = names if full else [s for s in spot if s in names]
        asserts = [assertion(f"v_{s}", 0.0, o[f"v_{s}"]) for s in pick]
        asserts += [assertion(f"k_{s}", 0.0, o[f"k_{s}"]) for s in (pick if full else sf)]
        mix = ", ".join(f"{k[2:]} {v:g}" for k, v in p.items() if k.startswith("f_") and v)
        tests.append({
            "id": case["id"],
            "description": (f"season {p['season']}, land use {mix}; z = {p['z']} m, z0 = {p['z0']} m, u* = {p['u_star']} m/s, "
                            f"L = {p['L']} m, G = {p['G']} W/m2, Ts = {p['Ts']} K, theta = {p['theta']}, lev = {p['lev']}. "
                            + ("Asserts v and k for all 132 species. " if full else "")
                            + f"Expected values are upstream's DryDepGasFractional at these inputs ({UPSTREAM})."),
            "parameter_overrides": p,
            "time_span": {"start": 0.0, "end": 1.0},
            "assertions": asserts,
        })

    analyses = [{
        "id": "season_sweep",
        "description": ("O3, HNO3 and CH2O deposition velocity by Wesely season over a Midwest-like mosaic (55% "
                        "agricultural, 20% range/agricultural, 10% deciduous, 10% urban, 5% water) at noon-like "
                        "conditions: stomatal and canopy uptake fall from midsummer to winter."),
        "parameters": {"f_mixedforest": 0.0, "f_agricultural": 0.55, "f_rangeag": 0.2, "f_deciduous": 0.1,
                       "f_urban": 0.1, "f_water": 0.05, "G": 600.0, "Ts": 293.0, "u_star": 0.35},
        "time_span": {"start": 0.0, "end": 1.0},
        "parameter_sweep": {"type": "cartesian", "dimensions": [{"parameter": "season", "values": [1, 2, 3, 4, 5]}]},
        "plots": [{"id": f"v_{s}_vs_season", "type": "line", "description": f"v_{s} by season.",
                   "x": {"variable": "season", "label": "Wesely season (1 midsummer ... 5 transitional)"},
                   "y": {"variable": f"v_{s}", "label": f"v_{s} (m/s)"}} for s in ("O3", "HNO3", "CH2O")],
    }]

    desc = ("Gas-phase dry deposition, upstream `DryDepositionGasFractional` (Wesely 1989 surface resistance with "
            "fractional / mosaic land use). For each of 132 species the deposition velocity is the area-weighted "
            "sum over the 11 Wesely land-use classes, v = i_lev * sum_c f_c / (Ra + Rb + Rc_c) (the GEOS-Chem "
            "drydep_mod / CMAQ STAGE convention), and the rate is k = v*g*rho_A/del_P. Inputs: the 11 area "
            "fractions f_<class>, the season index, and surface meteorology (z, z0, u_star, L, rho_A, G, Ts, "
            "theta, lev, del_P). 0-D and stateless.")
    notes = (
        f"source_ref: {UPSTREAM}:src/dry_deposition.jl DryDepositionGasFractional / DryDepGasFractional, "
        "src/wesley1989.jl WesleySurfaceResistance / FractionalWesleyVd.\n\n"
        "What changed against the previous revision (AtmosphericDeposition.jl@e4468c79 DryDepositionGas): the single "
        "`landuse` index is replaced by the 11 area fractions f_<class>; r_s keeps its hot-closure guard (stomata "
        "closed at Ts >= 39.9 °C - the old code re-evaluated eq. 3 and discarded it, so r_s went negative above "
        "40 °C); r_ac and r_gsS are floored at 1 s/m (GEOS-Chem drydep_mod.F90), which matters for water and wetland "
        "where the tables hold 0.\n\n"
        "Structure: the class- and species-independent pieces (Ra, mu_air, Dh2o, rdc, correction) and, per class, "
        "the seven Wesely table entries at the current season and the stomatal resistance rs_<class> are shared "
        "observeds; each species carries RaRb_<s> = Ra + Rb and v_<s> inlines the 11 class surface resistances with "
        "its GasData (Dh2oPerDx, Hstar, Fo) substituted. rain and dew are false, as upstream hardcodes them, so "
        "r_lux = r_lu/(1e-5*Hstar + Fo). SO2 and O3 use the SO2 / O3 table columns for r_cl and r_gs, as upstream. "
        "Table lookups are interp.linear on the integer season knots, which returns the entry exactly; upstream "
        "indexes at round(Int, season), so season must be an integer.\n\n"
        "Not in this component: where the inputs come from. Upstream's GEOS-FP coupling (ext/EarthSciDataExt.jl) "
        "sets z to the hypsometric first-level midpoint height, del_P to the first-level pressure thickness "
        "(-4.804826 Pa + 0.015048*PS), season from the month, and the 11 fractions from the bundled CONUS land-use "
        "map; in EarthSciModels a host model does that wiring.\n\n"
        "Tolerance rationale (rel = 1e-9): expected values are upstream's own DryDepGasFractional evaluated at the "
        "test inputs; the ESM repeats the same Float64 arithmetic on the same Float32-stored tables."
    )
    doc = {
        "esm": "1.0.0",
        "metadata": {
            "name": "DryDepositionGas",
            "description": desc,
            "authors": ["AtmosphericDeposition.jl / DepositionMTK.jl contributors", "EarthSciModels"],
            "created": f"{TODAY}T00:00:00Z",
            "tags": ["atmospheric_deposition", "dry-deposition", "wesley-1989", "mosaic-land-use", "multi-species", "gas-phase"],
            "references": [
                {"citation": "Wesely, M. L. (1989). Parameterization of surface resistances to gaseous dry deposition in "
                             "regional-scale numerical models. Atmospheric Environment 23(6), 1293-1304. "
                             "doi:10.1016/0004-6981(89)90153-4"},
                {"citation": "Seinfeld, J. H., & Pandis, S. N. (2006). Atmospheric Chemistry and Physics (2nd ed.). Wiley. "
                             "Eqs. 19.13-19.14 (aerodynamic resistance), 19.17 (quasi-laminar resistance for gases)."},
                {"citation": "GEOS-Chem drydep_mod.F90: area-weighted deposition velocity over land types and the 1 s/m "
                             "floors on r_ac and r_gs."},
                {"citation": f"Source: {UPSTREAM}, src/dry_deposition.jl, src/wesley1989.jl."},
            ],
        },
        "models": {"DryDepositionGas": {
            "reference": {"notes": notes},
            "tolerance": {"rel": 1e-9},
            "variables": b.vars,
            "equations": b.eqs,
            "tests": tests,
            "analyses": analyses,
        }},
    }
    write(DEP / "wesley_dry_gas.esm", doc)


# ================================================================== wet deposition
def patch_wet():
    path = DEP / "wet_deposition.esm"
    d = json.loads(path.read_text())
    m = d["models"]["WetDeposition"]
    V = m["variables"]
    V.pop("lev")
    V.pop("Dz")
    V["h_s"] = {"type": "parameter", "units": "m", "default": 1000.0,
                "description": "EMEP characteristic scavenging depth [m] (Simpson et al. 2003, EMEP Status Report "
                               "1/2003 Part I, Ch. 9). Upstream module @constant h_s. The gas and in-cloud rates "
                               "divide by this fixed depth, not by the local layer thickness."}
    per_w = lambda w: div(mul(w, "Vdr"), "rho_water")
    rates = {
        "k_particle": mul("i_rain", "qrain", "rho_air",
                          add(mul("A_wd", 0.1), mul("cloudFrac", div(per_w(1.0e6), "h_s")))),
        "k_SO2": div(mul("i_rain", add(per_w(0.15e6), mul("cloudFrac", per_w(0.3e6))), "qrain", "rho_air"), "h_s"),
        "k_othergas": div(mul("i_rain", add(per_w(0.5e6), mul("cloudFrac", per_w(1.4e6))), "qrain", "rho_air"), "h_s"),
    }
    m["equations"] = [e for e in m["equations"] if e["lhs"] not in ("Dz",)]
    for e in m["equations"]:
        if e["lhs"] in rates:
            e["rhs"] = rates[e["lhs"]]
    V["k_particle"]["description"] = (
        "First-order wet deposition rate for particles [1/s]: i_rain*qrain*rho_air*(A_wd*E + cloudFrac*W_in*Vdr/"
        "(rho_water*h_s)) with E = 0.1 and in-cloud ratio W_in = 1.0e6 (EMEP).")
    V["k_SO2"]["description"] = (
        "First-order wet deposition rate for SO2 [1/s]: i_rain*(W_sub + cloudFrac*W_in)*Vdr/rho_water*qrain*rho_air/h_s "
        "with sub-cloud W_sub = 0.15e6 and in-cloud W_in = 0.3e6 (EMEP).")
    V["k_othergas"]["description"] = (
        "First-order wet deposition rate for other soluble gases [1/s]: i_rain*(W_sub + cloudFrac*W_in)*Vdr/rho_water*"
        "qrain*rho_air/h_s with W_sub = 0.5e6 and W_in = 1.4e6 (EMEP).")
    tests = []
    for case in REF["wet_cases"]:
        p, o = case["p"], case["out"]
        tests.append({
            "id": case["id"],
            "description": (f"cloudFrac = {p['cloudFrac']}, qrain = {p['qrain']:g}, rho_air = {p['rho_air']} kg/m3. "
                            f"Expected values are upstream's _WetDeposition at these inputs ({UPSTREAM})."),
            "parameter_overrides": {k: p[k] for k in ("cloudFrac", "qrain", "rho_air")},
            "time_span": {"start": 0.0, "end": 1.0},
            "assertions": [assertion("i_rain", 0.0, 1 if p["qrain"] > 0 else 0)]
                          + [assertion(k, 0.0, o[k]) for k in ("k_particle", "k_SO2", "k_othergas")],
        })
    m["tests"] = tests
    m["analyses"] = [a for a in m.get("analyses", []) if "lev" not in json.dumps(a.get("parameter_sweep", {}))]
    d["metadata"]["description"] = (
        "EMEP wet deposition for particles, SO2 and other soluble gases (upstream WetDeposition). First-order rates "
        "k = W*P/(h_s*rho_water), with precipitation flux P = qrain*Vdr*rho_air, sub-cloud and cloudFrac-weighted "
        "in-cloud scavenging ratios W from EMEP Status Report 1/2003 Table 9.1 (times 1e6, the tabulated unit), and "
        "the fixed EMEP scavenging depth h_s = 1000 m; the particle sub-cloud term is A_wd*E*qrain*rho_air.")
    ref = m.setdefault("reference", {})
    ref["notes"] = (
        f"source_ref: {UPSTREAM}:src/wet_deposition.jl _WetDeposition / WetDeposition.\n\n"
        "Changes against the previous revision (AtmosphericDeposition.jl@e4468c79), all upstream: the EMEP "
        "scavenging ratios are the Table 9.1 entries times 1e6 (the table lists W x 1e-6; using the bare entries made "
        "gas and in-cloud wet deposition a no-op, with lifetimes of years); the in-cloud term uses the in-cloud ratios "
        "(SO2 0.3e6, other gases 1.4e6) where it used the sub-cloud ones; the gas and in-cloud rates divide by the "
        "fixed scavenging depth h_s = 1000 m instead of the local layer thickness, which made every layer remove the "
        "rain's full carrying capacity; and the `lev` input and its 72-level depth table are gone, since nothing "
        "uses them.\n\n"
        "Tolerance rationale (model rel = 1e-9): closed-form arithmetic on Float64 constants; expected values are "
        "upstream's _WetDeposition at the test inputs.")
    write(path, d, indent=2)


# =============================================== WesleySurfaceResistance standalone
def patch_wsr():
    path = DEP / "wesley1989" / "surface_resistance.esm"
    raw = path.read_text()
    d = json.loads(raw)
    m = d["models"]["WesleySurfaceResistance"]
    eq = {e["lhs"]: e for e in m["equations"]}
    old_rs = eq["rs"]["rhs"]
    inner = old_rs["args"][0]                       # ifelse(Ts <= 0.1, 1e25, eq3)
    assert inner["op"] == "ifelse"
    inner["args"][2] = ifelse(Op(">=", "Ts", 39.9), 1e25, inner["args"][2])
    floor = lambda x: Op("max", x, 1.0)
    eq["r_gsx_gen"]["rhs"] = json.loads(json.dumps(eq["r_gsx_gen"]["rhs"]).replace(
        '"r_gsS_val"', json.dumps(floor("r_gsS_val"))))
    rg = eq["rgsx"]["rhs"]
    assert rg["args"][1] == "r_gsS_val"
    rg["args"][1] = floor("r_gsS_val")
    eq["r_c"]["rhs"] = json.loads(json.dumps(eq["r_c"]["rhs"]).replace('"r_ac_val"', json.dumps(floor("r_ac_val"))))
    V = m["variables"]
    V["rs"]["description"] += " Closed (1e25) at Ts >= 39.9 °C as well as at Ts <= 0.1 °C (upstream hot-closure fix)."
    V["r_gsx_gen"]["description"] += " r_gsS floored at 1 s/m (upstream, after GEOS-Chem)."
    V["r_c"]["description"] += " r_ac floored at 1 s/m (upstream, after GEOS-Chem)."
    cases = {c["id"]: c["out"] for c in REF["wsr_cases"]}
    changed = []
    for t in m["tests"]:
        out = cases[t["id"]]
        for a in t["assertions"]:
            if a["variable"] in out and a["expected"] != out[a["variable"]]:
                old = a["expected"]
                if abs(old - out[a["variable"]]) > 1e-9 * abs(out[a["variable"]]):
                    changed.append((t["id"], a["variable"], old, out[a["variable"]]))
                a["expected"] = out[a["variable"]]
    print("  surface_resistance: assertions whose upstream value moved:", changed or "none")
    have = {t["id"] for t in m["tests"]}
    why = {"hot_closure_45C_general": "Ts = 45 °C: the stomata are closed (rs = 1e25) by the hot-closure guard the "
                                      "previous revision discarded.",
           "water_floors_so2": "SO2 over water (class 7), where the r_ac and r_gsS tables hold 0: both are floored "
                               "at 1 s/m.",
           "wetland_floor_soluble_gas": "A very soluble gas over nonforested wetland (class 9, r_gsS = 0): the generic "
                                        "r_gsx uses the floored r_gsS."}
    for c in REF["wsr_cases"]:
        if c["id"] in have or c["id"] not in why:
            continue
        m["tests"].append({
            "id": c["id"],
            "description": why[c["id"]] + f" Expected values are upstream's functions at these inputs ({UPSTREAM}).",
            "parameter_overrides": c["p"],
            "time_span": {"start": 0.0, "end": 1.0},
            "assertions": [assertion(k, 0.0, c["out"][k]) for k in ("rs", "rclx", "rgsx", "r_gsx_gen", "r_c")],
        })
    m["reference"]["notes"] += (
        f"\n\nUpdated {TODAY} to {UPSTREAM}: r_s keeps its hot-closure guard (stomata closed at Ts >= 39.9 °C; the "
        "previous code re-evaluated eq. 3 and discarded the guard, so r_s went negative above 40 °C), and r_ac and "
        "r_gsS are floored at 1 s/m in r_c, r_gsx and the SO2 branch, as GEOS-Chem drydep_mod.F90 does. The "
        "expected values of rs, rlux, rclx, rgsx, r_clx_gen, r_gsx_gen, correction and r_c are upstream's own "
        "functions at each test's inputs.")
    write(path, d, indent=4)


# ============================================================ aerosol dry deposition
def patch_aerosol():
    path = DEP / "dry_aerosol.esm"
    raw = path.read_text()
    d = json.loads(raw)
    m = d["models"]["DryDepositionAerosol"]
    V = m["variables"]
    V["del_P"] = {"type": "parameter", "units": "Pa", "default": 1520.0,
                  "description": "Pressure thickness of level 1 [Pa] (upstream del_P), for k = v*g*rho_a/del_P."}
    for e in m["equations"]:
        if e["lhs"] == "k":
            assert e["rhs"] == div("v", "z"), e["rhs"]
            e["rhs"] = div(mul("v", "g", "rho_a"), "del_P")
    V["k"]["description"] = ("Particle dry deposition rate [1/s] = v*g*rho_a/del_P: the deposition flux v*c spread over "
                             "the level-1 air mass (upstream; it was v/z).")
    cases = {c["id"]: c["out"] for c in REF["aer_cases"]}
    for t in m["tests"]:
        for a in t["assertions"]:
            if a["variable"] in ("v", "k"):
                a["expected"] = cases[t["id"]][a["variable"]]
    m["reference"]["notes"] += (
        f"\n\nUpdated {TODAY} to {UPSTREAM}: the rate is k = v*g*rho_a/del_P (the gas system's form) instead of v/z; "
        "del_P is a new parameter (upstream default 1520 Pa). v is unchanged. The expected v and k are upstream's "
        "DryDepParticle at each test's inputs.")
    write(path, d, indent=4)


# ===================================================== land-use fractions (data source)
LANDUSE_URL = ("https://raw.githubusercontent.com/jialinl6/DepositionMTK.jl/"
               "cbe7d7db0a86bd6945f6b06d1fdc8274399b33de/data/landuse_wesely_conus_0p05.nc")


def build_landuse():
    """Upstream src/landuse_fractions.jl's data, as an esm data source plus the model that
    exposes it as the 11 fractions DryDepositionGas takes. The regridding onto a model grid
    is the host's coupling expression (esm-spec §8.6), as for every other source here."""
    V = {
        "lon": {"type": "parameter", "units": "rad", "default": 0.0,
                "description": "Longitude of the model grid point (radians). Set by the host, as for the other loaders."},
        "lat": {"type": "parameter", "units": "rad", "default": 0.0,
                "description": "Latitude of the model grid point (radians). Set by the host, as for the other loaders."},
    }
    for c, name in enumerate(CLASSES, start=1):
        V[f"f_{name}"] = {
            "type": "parameter", "units": "1", "shape": [],
            "description": (f"Area fraction of Wesely (1989) land-use class {c} ({name}), the native 0.05° cell's "
                            "percentage / 100. Same name as the DryDepositionGas parameter it is meant to drive."),
            "update": {"kind": "data", "source": "WESELY_LANDUSE_CONUS", "from": {
                "file_variable": "percent",
                "select": {"axes": [{"fixed": c - 1}, "all", "all"]},
                "unit_conversion": 0.01,
                "description": f"percent(class = {c}, lat, lon): class {c} ({name}) slice of the UInt8 percentage cube; "
                               "the native dims are (class, lat, lon) and `fixed` is 0-based."}},
        }
    notes = (
        f"source_ref: {UPSTREAM}:src/landuse_fractions.jl and data/landuse_wesely_conus_0p05.nc.\n\n"
        "Upstream reads this file at run time, conservatively area-averages it onto the simulation grid "
        "(_remap_landuse: weights dlon*(sin(lat_hi) - sin(lat_lo)) per overlap, cell edges rebuilt from the "
        "window_* / cell_size attributes because the Float32 lon/lat centres sit ~1e-6° off the 0.05° multiples), "
        "refuses a grid that reaches outside the window, and its GEOS-FP coupling binds DryDepositionGas.f_<class> "
        "to the result at each cell. Here the file is a data source and the 11 fractions are data-fed parameters; "
        "the conservative remap onto a host grid and the f_<class> -> DryDepositionGas.f_<class> edges belong to the "
        "host's coupling (esm-spec §8.6), like every other source in this repo. Every native cell sums to exactly "
        "100%, so the fractions sum to 1.\n\n"
        "The URL is pinned to the reference commit in the jialinl6/DepositionMTK.jl fork; point it elsewhere if the "
        "file moves (it is the same 1,317,215-byte file either way).\n\n"
        "No inline tests: a data-fed model's values come from the fetched file. The class axis was checked against "
        "the file directly: its class_name variable lists the 11 classes in this order, every cell sums to 100, and "
        "the dominant class is urban over Los Angeles, agricultural over Iowa (96%) and water over the Gulf of "
        "Mexico, the same checks as upstream's test/fractional_test.jl.")
    doc = {
        "esm": "1.0.0",
        "metadata": {
            "name": "WeselyLandUse",
            "description": ("Wesely (1989) land-use area fractions for CONUS at 0.05°, from MODIS MCD12C1 v6.1 (2016): "
                            "the 11 class fractions f_<class> that DryDepositionGas weights its mosaic deposition "
                            "velocity by. Data source + loader model; the host regrids it onto its grid."),
            "authors": ["AtmosphericDeposition.jl / DepositionMTK.jl contributors", "EarthSciModels"],
            "created": f"{TODAY}T00:00:00Z",
            "tags": ["atmospheric_deposition", "land-use", "wesley-1989", "modis", "static", "conus", "longlat"],
            "references": [
                {"citation": "Friedl, M., Sulla-Menashe, D. (2022). MODIS/Terra+Aqua Land Cover Type Yearly L3 Global "
                             "0.05Deg CMG V061 (MCD12C1). NASA EOSDIS Land Processes DAAC. doi:10.5067/MODIS/MCD12C1.061"},
                {"citation": "Wesely, M. L. (1989). Atmospheric Environment 23(6), 1293-1304 (the 11 land-use classes)."},
                {"citation": f"Source: {UPSTREAM}, src/landuse_fractions.jl, data/landuse_wesely_conus_0p05.nc "
                             "(built by data/preprocess_landuse.jl)."},
            ],
        },
        "data_sources": {
            "WESELY_LANDUSE_CONUS": {
                "kind": "static",
                "source": {"url_template": LANDUSE_URL},
                "reference": {
                    "citation": "MODIS MCD12C1 v6.1 2016 Land_Cover_Type_1_Percent, IGBP bands summed into the Wesely "
                                "(1989) classes (many-to-one, covering all 17) and windowed to CONUS.",
                    "url": "https://github.com/jialinl6/DepositionMTK.jl/tree/superfast-gas-only/data",
                },
                "metadata": {
                    "tags": ["land-use", "static", "conus", "categorical-fractions", "longlat"],
                    "file_format": "netCDF-4; variable percent (ubyte, 0..100), dims (class, lat, lon) = (11, 700, 1400)",
                    "native_resolution_deg": {"lon": 0.05, "lat": 0.05},
                    "native_extent_edges_deg": {"lon": [-130.0, -60.0], "lat": [20.0, 55.0]},
                    "coordinates": "lon/lat are Float32 cell centres; exact edges are window_*_min + k*cell_size.",
                    "classes": CLASSES,
                },
            }
        },
        "models": {"WeselyLandUse": {"reference": {"notes": notes}, "variables": V, "equations": [], "tests": []}},
    }
    write(DEP / "wesely_landuse.esm", doc)


if __name__ == "__main__":
    build_landuse()
    build_dry()
    patch_wet()
    patch_wsr()
    patch_aerosol()
