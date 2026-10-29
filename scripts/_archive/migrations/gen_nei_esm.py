#!/usr/bin/env python3
"""Bring the NEI2016 and GEOS-FP .esm files in line with EarthSciML/EarthSciData.jl PR #226
(head jialinl6:Oct_merge @87f8da1), as used by the SuperFast gas-only stack.

  components/earthsci_data/nei2016_monthly.esm   patched: NOx/ISOP clocks on round(lon/15), CO weekday factors,
                                                 pinned-flux inline tests
  components/earthsci_data/nei2016_elevated.esm  new: NEI2016ElevatedEmis, the six inline point sectors with
                                                 GEOS-Chem's vertical profiles
  registered_functions/nei_{diurnal_itp_nox,diurnal_itp_isop,dayofweek_itp_nox,dayofweek_itp_co}.esm
                                                 patched the same way, expected values re-evaluated
  couplings/nei2016_superfast.esm                patched: NO converted with MW_NO2
  couplings/earthscidata_spatial_params.esm      patched: GEOSFP / ERA5 lat, lon, lev -> NEI2016ElevatedEmis
  components/earthsci_data/geosfp.esm            patched: Tv_bar removed and δPδlev added, as upstream

Input: DATADIR/nei_ref.json from extract_nei_data.jl (same directory). The patches apply to the
files as they were before this update; run them on a clean checkout of those files.
Run:
  julia --project=ENV scripts/_archive/migrations/extract_nei_data.jl DATADIR
  python3 scripts/_archive/migrations/gen_nei_esm.py DATADIR
"""
import copy
import json
import math
import sys
from pathlib import Path

DATA = Path(sys.argv[1])
REF = json.loads((DATA / "nei_ref.json").read_text())
REPO = Path(__file__).resolve().parents[3]
UPSTREAM = "EarthSciML/EarthSciData.jl PR #226 (head jialinl6:Oct_merge @87f8da1)"
TODAY = "2026-10-28"
T = REF["tables"]
SECTORS = T["ELEVATED_SECTORS"]
SHORT = {s: s.removeprefix("emln_").removesuffix("_12") for s in SECTORS}   # ptegu, ..., cmv_c3, cmv_c1c2
PROFILE_OF = {s: T["SECTOR_PROFILES"].get(s, "surface") for s in SECTORS}
PROFILES = T["PROFILE_NAMES"]


def Op(o, *a):
    return {"op": o, "args": list(a)}


def fn(name, *a):
    return {"op": "fn", "name": name, "args": list(a)}


def C(v):
    return {"op": "const", "args": [], "value": v}


def ifelse(c, a, b):
    return Op("ifelse", c, a, b)


def mul(*a):
    return Op("*", *a)


def div(a, b):
    return Op("/", a, b)


def add(*a):
    return a[0] if len(a) == 1 else Op("+", *a)


def load(rel):
    p = REPO / rel
    return p, json.loads(p.read_text())


def dumps(o, ind=0):
    """json.dumps(indent=2), with short nodes, arrays of numbers and whole equations kept on one line."""
    pad, sub = "  " * ind, "  " * (ind + 1)
    if isinstance(o, (dict, list)):
        flat = json.dumps(o, ensure_ascii=False)
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
    """indent=None: the compact layout (new files); an int: the patched file's own indentation."""
    text = (dumps(d) if indent is None else json.dumps(d, indent=indent, ensure_ascii=False)) + "\n"
    assert json.loads(text) == d
    path.write_text(text)
    print(f"wrote {path.relative_to(REPO)} ({path.stat().st_size / 1024:.0f} KB)")


def round_half_even(x):
    """Julia round(x) (RoundNearest, ties to even) from floor: floor(x + 0.5), less one when
    x + 0.5 is an odd integer, i.e. when y = (x + 0.5)/2 has fractional part exactly 0.5."""
    y = div(add(x, 0.5), 2.0)
    return Op("-", Op("floor", add(x, 0.5)), ifelse(Op("==", Op("-", y, Op("floor", y)), 0.5), 1.0, 0.0))


def rename(expr, old, new):
    return json.loads(json.dumps(expr).replace(json.dumps(old), json.dumps(new)))


# ========================================================================= surface NEI
CLOCK_OF = {"CO": ["dayofweek_co", "diurnal_factor"], "FORM": ["diurnal_factor"],
            "ISOP": ["diurnal_factor_isop"], "NO": ["dayofweek_nox", "diurnal_factor_nox"],
            "NO2": ["dayofweek_nox", "diurnal_factor_nox"]}
ASSERT = ["NO", "NO2", "CO", "FORM", "ISOP", "SO2", "NH3", "ACROLEIN", "PM2_5"]


def patch_surface():
    path, d = load("components/earthsci_data/nei2016_monthly.esm")
    m = d["models"]["NEI2016MonthlyEmis"]
    V, eqs = m["variables"], m["equations"]
    by = {e["lhs"]: e for e in eqs if isinstance(e["lhs"], str)}
    new = {
        "tz_offset_round_h": ("1", round_half_even(div("lon_deg", 15.0)),
                              "Solar time-zone offset in hours, round(lon_deg/15) with ties to even (Julia round): the "
                              "clock of the EDGAR NOx hourly file GEOS-Chem uses, applied upstream to the NOx and "
                              "isoprene factors. CO and formaldehyde keep tz_offset_h = floor(lon_deg/15) (HEMCO)."),
        "t_local_round_utc": ("s", add("t", "t_ref", mul("tz_offset_round_h", 3600.0)),
                              "Local time on the round(lon/15) clock (Unix s): t + t_ref + tz_offset_round_h*3600."),
        "local_hour_round": ("1", add(fn("datetime.hour", "t_local_round_utc"), 1.0),
                             "Local hour 1..24 on the round(lon/15) clock."),
        "day_of_week_round": ("1", rename(by["day_of_week"]["rhs"], "t_local_utc", "t_local_round_utc"),
                              "Local ISO weekday 1..7 (Mon..Sun) on the round(lon/15) clock."),
    }
    at = [i for i, e in enumerate(eqs) if e["lhs"] == "day_of_week"][0] + 1
    for k, (u, rhs, desc) in reversed(list(new.items())):
        V[k] = {"type": "unknown", "units": u, "description": desc}
        eqs.insert(at, {"lhs": k, "rhs": rhs})
    by = {e["lhs"]: e for e in eqs if isinstance(e["lhs"], str)}
    for k in ("diurnal_factor_nox", "diurnal_factor_isop"):
        by[k]["rhs"] = rename(by[k]["rhs"], "local_hour", "local_hour_round")
        V[k]["description"] += " Read at local_hour_round: upstream's NOx and isoprene clock is round(lon/15)."
    by["dayofweek_nox"]["rhs"] = rename(by["dayofweek_nox"]["rhs"], "day_of_week", "day_of_week_round")
    V["dayofweek_nox"]["description"] += " Read at day_of_week_round (round(lon/15) clock, as upstream)."
    delp = by["delp_dry_surface"]["rhs"]["args"]
    assert delp[1]["value"] == T["DELP_LON"] and delp[2]["value"] == T["DELP_LAT"]
    assert max(abs(a - b) for r, u in zip(delp[0]["value"], T["DELP_DRY_SURFACE"]) for a, b in zip(r, u)) < 5.0001e-5
    delp[0]["value"] = T["DELP_DRY_SURFACE"]
    V["delp_dry_surface"]["description"] += (" Table values are upstream's mean_domian_delp_dry_surface.jld2 at full "
                                             "precision (they were rounded to 4 decimals, up to 3.3e-6 relative).")
    co = by["dayofweek_co"]["rhs"]["args"][0]
    assert co["op"] == "const"
    co["value"] = T["DayofWeekFactors_CO"]
    V["dayofweek_co"]["description"] = (
        "CO weekday factor, Monday-first: HEMCO GEIA_DOW_CO with 1.1076 on all five weekdays (upstream; HEMCO's "
        "line repeats the NOx value 1.0706 on Tue-Fri and sums to 6.852, and the previous table here read 1.076 on "
        "Monday). The week sums to 7.")
    names = REF["surf_species"]
    flux = lambda sp: 1.0e-9 * (1 + 0.001 * (names.index(sp) + 1))
    tests = []
    for c in REF["surf_cases"]:
        po = {f"flux_{sp}": flux(sp) for sp in names}
        po.update({"t_ref": c["t_abs"], "lon": c["lon"], "lat": c["lat"], "lev": c["lev"]})
        tests.append({
            "id": c["id"],
            "description": (f"Pinned fluxes (flux_<SP> = 1e-9*(1 + 0.001*i), i the species' 1-based place in sorted "
                            f"order) at t_ref = {c['t_abs']:.0f} s, lon = {math.degrees(c['lon']):.4g} deg, "
                            f"lat = {math.degrees(c['lat']):.4g} deg, lev = {c['lev']}. Expected rates are upstream's "
                            f"per-level factors (nei_layer_fraction, nei_layer_gate, nei_clock_*, nei_inv_delp) "
                            f"combined as its NEI2016MonthlyEmis builder does ({UPSTREAM}); no data is read."),
            "parameter_overrides": po,
            "time_span": {"start": 0.0, "end": 1.0},
            "tolerance": {"rel": 1e-12, "abs": 1e-30},
            "assertions": [{"variable": sp, "time": 0.0, "expected": c["out"][sp]} for sp in ASSERT if sp in names],
        })
    m["tests"] = tests
    m["reference"]["notes"] += (
        f"\n\nUpdated {TODAY} to {UPSTREAM}: the NOx (diurnal and weekday) and isoprene factors use the "
        "round(lon/15) solar clock with ties to even (tz_offset_round_h), as the EDGAR NOx hourly file GEOS-Chem uses; "
        "the floor clock put about half of NEI NOx one hour behind GEOS-Chem. CO and formaldehyde stay on "
        "floor(lon/15), HEMCO's local-time rule. The CO weekday factors are 1.1076 on all five weekdays (the week "
        "sums to 7). The delp_dry_surface table carries upstream's values at full precision; it had been rounded to 4 decimals, which moved every rate by up to 3.3e-6. The per-day normalisation of the monthly totals was already here. Inline tests pin every flux "
        "parameter, so no data is fetched; their expected values are upstream's own per-level factors combined as "
        "its builder does, and agree to roundoff (tolerance rel = 1e-12).")
    write(path, d, 2)
    return d


# ======================================================================== elevated NEI
def build_elevated(surface):
    sm = surface["models"]["NEI2016MonthlyEmis"]
    sV = sm["variables"]
    seq = {e["lhs"]: e for e in sm["equations"] if isinstance(e["lhs"], str)}
    src0 = surface["data_sources"]["NEI2016"]
    sources = {}
    for s in SECTORS:
        src = copy.deepcopy(src0)
        src["source"]["url_template"] = src["source"]["url_template"].replace("{sector}", s)
        src["source"]["mirrors"] = [u.replace("{sector}", s) for u in src["source"].get("mirrors", [])]
        md = {k: v for k, v in src["metadata"].items()
              if k not in ("default_sector", "merged_from", "merge_note", "merged_file_metadata", "former_regridding")}
        md["tags"] = md["tags"] + ["elevated", "point-sources"]
        md["sector"] = s
        md["vertical_profile"] = PROFILE_OF[s]
        md["species"] = REF["species"][s]
        src["metadata"] = md
        sources[f"NEI2016_{s}"] = src

    V, eqs = {}, []

    def obs(name, units, rhs, desc):
        V[name] = {"type": "unknown", "units": units, "description": desc}
        eqs.append({"lhs": name, "rhs": rhs})

    for k in ("t_ref", "g0_100", "Δz", "scale", "lon", "lat", "lev"):
        V[k] = copy.deepcopy(sV[k])
    V["lev"]["description"] = ("Vertical level index; layer k covers k <= lev < k + 1 (values below 1 count as layer 1). "
                               "Coupled from the met provider's lev.")
    V["ktop"] = {"type": "parameter", "units": "1", "default": 72.0,
                 "description": "Top model layer of the domain (upstream reads it off the DomainInfo grid). Profile "
                                "fractions above it are added to it, so the column total is kept."}
    for s in SECTORS:
        for sp in REF["species"][s]:
            V[f"flux_{sp}_{SHORT[s]}"] = {
                "type": "parameter", "units": "kg/m^2/s", "shape": [],
                "description": f"NEI2016 {s} {sp} column emission flux density (kg/m²/s), the monthly file's total.",
                "update": {"kind": "data", "source": f"NEI2016_{s}",
                           "from": {"file_variable": sp, "description": f"{sp} in 2016fh_16j_{s}_12US1_month_MM.ncf."}}}

    shared = ["lon_deg", "lat_deg", "tz_offset_h", "t_local_utc", "local_hour", "day_of_week",
              "tz_offset_round_h", "t_local_round_utc", "local_hour_round", "day_of_week_round",
              "diurnal_factor", "diurnal_factor_nox", "diurnal_factor_isop", "dayofweek_co", "dayofweek_nox",
              "delp_dry_surface", "t_utc", "month_utc", "is_leap_year_utc", "days_in_month"]
    for k in shared:
        V[k] = copy.deepcopy(sV[k])
        eqs.append(copy.deepcopy(seq[k]))

    obs("layer", "1", Op("max", 1.0, Op("floor", "lev")),
        "Model layer of lev, max(1, floor(lev)) (upstream _nei_layer).")
    dap1, dbp1 = T["NEI_DAP"][0], T["NEI_DBP"][0]
    obs("delp_ps", "1", div(Op("-", "delp_dry_surface", dap1), dbp1),
        "Mean surface pressure (hPa) backed out of the layer-1 thickness: (delp_dry_surface - dAp_1)/dBp_1.")
    knots72 = [float(k) for k in range(1, 73)]
    obs("delp_dry", "1",
        ifelse(Op("<=", "layer", 1.0), "delp_dry_surface",
               add(fn("interp.linear", C(T["NEI_DAP"]), C(knots72), "layer"),
                   mul(fn("interp.linear", C(T["NEI_DBP"]), C(knots72), "layer"), "delp_ps"))),
        "Dry pressure thickness (hPa, unitless) of layer `layer`: the surface map for layer 1, else "
        "dAp_k + dBp_k*delp_ps with the GEOS-FP 72-layer hybrid coefficients (upstream delp_dry_itp).")
    for p in sorted(set(PROFILE_OF.values()), key=PROFILES.index):
        f, tail = T["LAYER_FRACTIONS"][PROFILES.index(p)], T["LAYER_TAILS"][PROFILES.index(p)]
        n = len(f)
        kn = [float(k) for k in range(1, n + 1)]
        look = (lambda v: fn("interp.linear", C(v), C(kn), "layer")) if n > 1 else (lambda v: v[0])
        obs(f"frac_{p}", "1",
            ifelse(Op(">", "layer", Op("min", "ktop", float(n))), 0.0,
                   ifelse(Op("==", "layer", "ktop"), look(tail), look(f))),
            f"Fraction of the '{p}' profile's column released into this layer "
            + (f"(GEOS-Chem's {n}-layer profile, Simpson et al. 2003 EMEP Table 4.1, renormalized); " if n > 1 else "(all of it in layer 1); ")
            + "0 above the profile or the domain top, and the remainder at the domain top folded into layer ktop "
              "(upstream nei_layer_fraction).")

    groups = {}
    for s in SECTORS:
        groups.setdefault(PROFILE_OF[s], []).append(s)
    for sp in REF["elev_species"]:
        terms, nmax = [], 0
        for p, secs in groups.items():
            present = [s for s in secs if sp in REF["species"][s]]
            if not present:
                continue
            nmax = max(nmax, len(T["LAYER_FRACTIONS"][PROFILES.index(p)]))
            terms.append(mul(f"frac_{p}", div(add(*[f"flux_{sp}_{SHORT[s]}" for s in present]), "days_in_month")))
        rate = div(mul(add(*terms), div(1.0, "Δz"), "scale"), mul("g0_100", "delp_dry"))
        if sp in CLOCK_OF:
            rate = mul(rate, *CLOCK_OF[sp])
        obs(sp, "1/s", ifelse(Op("<=", "layer", Op("min", "ktop", float(nmax))), rate, 0.0),
            f"NEI2016 elevated-sector {sp} mixing-ratio source rate (kg/kg/s) in this layer: "
            f"sum over profile groups of frac_<p> * flux/days_in_month, /Δz*scale/(g0_100*delp_dry)"
            + (" * " + " * ".join(CLOCK_OF[sp]) if sp in CLOCK_OF else "") + f"; 0 above layer min(ktop, {nmax}).")

    elev = REF["elev_species"]
    flux = lambda sp, s: 1.0e-9 * (1 + 0.1 * (SECTORS.index(s) + 1) + 0.001 * (elev.index(sp) + 1))
    tests = []
    for c in REF["elev_cases"]:
        po = {f"flux_{sp}_{SHORT[s]}": flux(sp, s) for s in SECTORS for sp in REF["species"][s]}
        po.update({"t_ref": c["t_abs"], "lon": c["lon"], "lat": c["lat"], "lev": c["lev"], "ktop": float(c["ktop"])})
        tests.append({
            "id": c["id"],
            "description": (f"Pinned fluxes (flux_<SP>_<sector> = 1e-9*(1 + 0.1*j + 0.001*i), j the sector's place in "
                            f"NEI2016_ELEVATED_SECTORS, i the species' in sorted order) at t_ref = {c['t_abs']:.0f} s, "
                            f"lon = {math.degrees(c['lon']):.4g} deg, lat = {math.degrees(c['lat']):.4g} deg, "
                            f"lev = {c['lev']}, ktop = {c['ktop']}. Expected rates are upstream's per-level factors "
                            f"combined as its builder does ({UPSTREAM}); no data is read."),
            "parameter_overrides": po,
            "time_span": {"start": 0.0, "end": 1.0},
            "tolerance": {"rel": 1e-12, "abs": 1e-30},
            "assertions": [{"variable": sp, "time": 0.0, "expected": c["out"][sp]} for sp in ASSERT],
        })

    notes = (
        f"source_ref: {UPSTREAM}:src/nei2016monthly.jl NEI2016ElevatedEmis = NEI2016MonthlyEmis(NEI2016_ELEVATED_SECTORS).\n\n"
        "The six inline point sectors GEOS-Chem reads besides the merged surface file: power plants (emln_ptegu), other "
        "industrial points (emln_ptnonipm), the inline part of oil and gas (emln_pt_oilgas), Canada/Mexico points "
        "(emln_othpt) and class 1/2 and 3 commercial marine vessels (emln_cmv_c1c2_12, emln_cmv_c3_12), from EPA's "
        "2016fh 12US1 monthly files (one data source per sector). The EPA files are column totals in one layer; as "
        "GEOS-Chem's NEI2016 preprocessing (griddedepa2gc.py) does, each sector's column is spread over the GEOS-FP "
        "layers with one profile - ptegu, ptnonipm, pt_oilgas (11 layers) and cmv (3) - while othpt, which GEOS-Chem "
        "reads as 2-D, stays at the surface. Sectors sharing a profile are summed first.\n\n"
        "Per species: rate = [sum_p frac_p(layer) * sum_{s in p} flux_s/days_in_month] / Δz * scale * clock / "
        "(g0_100 * delp_dry(layer)), zero above layer min(ktop, longest profile among the species' groups). delp_dry "
        "is the surface thickness map for layer 1 and the GEOS-FP hybrid layer thickness above it. The clock is the "
        "surface file's (CO: weekday x diurnal on floor(lon/15); FORM: diurnal; ISOP: diurnal on round(lon/15); NO, "
        "NO2: weekday x diurnal on round(lon/15)).\n\n"
        "Choosing: this component is the elevated half only. A host that wants surface emissions alone mounts "
        "nei2016_monthly.esm; one that wants both mounts this as well and binds each to the NEI -> chemistry coupling "
        "library. The output names and units are the surface file's, so the same library binds either. Upstream's "
        "single-system form NEI2016MonthlyEmis([\"mrggrid_withbeis_withrwc\"; NEI2016_ELEVATED_SECTORS]) is the sum of "
        "the two (regridding and the per-level factors are linear), and mounting both with the same sector twice "
        "counts it twice, as upstream warns.\n\n"
        "Inline tests pin every flux parameter (no data is fetched) and check the per-level factors against upstream: "
        "layers 1, 3, 7.6, 11 and 12, domain-top folding with ktop = 3, the -97.5 deg half-hour meridian (where "
        "round and floor clocks differ), and two other times and places. Tolerance rel = 1e-12: the same arithmetic "
        "on the same constants.")
    doc = {
        "esm": "1.0.0",
        "metadata": {
            "name": "NEI2016ElevatedEmis",
            "description": ("NEI 2016 elevated point-sector emissions (upstream NEI2016ElevatedEmis): the six inline point "
                            "sectors from EPA's 2016fh monthly files, spread over model layers with GEOS-Chem's per-sector "
                            "vertical profiles. Same output species, units and time factors as NEI2016MonthlyEmis, so a "
                            "host can use surface emissions only, or surface plus elevated."),
            "authors": ["EarthSciData.jl contributors", "EarthSciModels"],
            "created": f"{TODAY}T00:00:00Z",
            "tags": ["emissions", "nei", "cmaq", "12us1", "monthly", "elevated", "point-sources", "vertical-profile"],
            "references": [
                {"citation": "US EPA (2020). 2016 National Emissions Inventory, Version 1 (2016fh platform). "
                             "https://gaftp.epa.gov/Air/emismod/2016/"},
                {"citation": "Simpson, D. et al. (2003). Transboundary acidification, eutrophication and ground level "
                             "ozone in Europe, Part I: Unified EMEP model description. EMEP Status Report 1/2003, Table 4.1 "
                             "(the layer profiles)."},
                {"citation": "GEOS-Chem NEI2016 3-D input preprocessing (griddedepa2gc.py, B. Henderson)."},
                {"citation": f"Source: {UPSTREAM}, src/nei2016monthly.jl."},
            ],
        },
        "data_sources": sources,
        "models": {"NEI2016ElevatedEmis": {
            "reference": {"notes": notes}, "tolerance": copy.deepcopy(sm.get("tolerance")),
            "variables": V, "equations": eqs, "tests": tests}},
    }
    write(REPO / "components/earthsci_data/nei2016_elevated.esm", doc)


# ================================================================ registered functions
def floor_lon_to_round(e):
    """Replace every floor(<expression in lon>) - the time-zone offset - by its ties-to-even round.
    Returns (new expression, number replaced)."""
    if isinstance(e, dict):
        inner = json.dumps(e.get("args"))
        if e.get("op") == "floor" and '"lon"' in inner and '"t_utc"' not in inner:
            return round_half_even(e["args"][0]), 1
        n, out = 0, dict(e)
        if "args" in e:
            out["args"] = []
            for a in e["args"]:
                a2, k = floor_lon_to_round(a)
                out["args"].append(a2)
                n += k
        return out, n
    return e, 0


def patch_regfns():
    for file, use_round, table_name in (("nei_diurnal_itp_nox", True, None), ("nei_diurnal_itp_isop", True, None),
                                        ("nei_dayofweek_itp_nox", True, None),
                                        ("nei_dayofweek_itp_co", False, "DayofWeekFactors_CO")):
        path, d = load(f"registered_functions/{file}.esm")
        m = next(iter(d["models"].values()))
        e = m["equations"][0]
        if use_round:
            e["rhs"], n = floor_lon_to_round(e["rhs"])
            assert n >= 1, file
        if table_name:
            assert e["rhs"]["args"][0]["op"] == "const"
            e["rhs"]["args"][0]["value"] = T[table_name]
        vals = {r["id"]: r["value"] for r in REF["regfn"][file]}
        for t in m["tests"]:
            t["assertions"][0]["expected"] = vals[t["id"]]
        key = {"nei_diurnal_itp_nox": "diurnal_NOx", "nei_diurnal_itp_isop": "diurnal_ISOP",
               "nei_dayofweek_itp_nox": "dow_NOx", "nei_dayofweek_itp_co": "dow_CO"}[file]
        if use_round:
            for c in REF["clocks"]:
                # 2016-05-01T18 and 2016-05-02T06 UTC: local noon Sunday and 00:00 Monday on the round
                # clock (UTC-6), 11:00 Sunday and 23:00 Sunday on the floor clock (UTC-7).
                if c["lon_deg"] == -97.5 and c["t"] in (1462125600.0, 1462168800.0):
                    lon = c["lon_deg"] * (math.pi / 180)
                    m["tests"].append({
                        "id": f"halfhour_meridian_m97_5_t{int(c['t'])}",
                        "description": ("lon = -97.5 deg, where lon/15 = -6.5 is a tie: round-half-even gives UTC-6 and "
                                        f"floor UTC-7. Expected value is upstream's function ({UPSTREAM})."),
                        "parameter_overrides": {"t_utc": c["t"], "lon": lon},
                        "time_span": {"start": 0.0, "end": 1.0},
                        "assertions": [{"variable": e["lhs"], "time": 0.0, "expected": c[key]}],
                    })
        ref = m.setdefault("reference", {})
        ref["notes"] = ref.get("notes", "") + (
            f"\n\nUpdated {TODAY} to {UPSTREAM}: "
            + ("the local time uses round(lon/15) with ties to even, the EDGAR NOx hourly clock GEOS-Chem uses, instead of "
               "floor(lon/15)." if use_round else
               "the weekday table is 1.1076 on all five weekdays (it read 1.076 on Monday and 1.0706 Tue-Fri, which summed "
               "to 6.85).")
            + " Expected values are upstream's function at each test's inputs.")
        write(path, d, 2)


# ============================================================================ couplings
def patch_coupling():
    path = REPO / "couplings/nei2016_superfast.esm"
    raw = path.read_text()
    old_no = json.dumps({"op": "*", "args": [{"op": "/", "args": [{"op": "*", "args": [1000000000.0, 0.02897]}, 0.03001]}, "Emissions.NO"]})
    d = json.loads(raw)
    hits = [e for c in d["coupling"] for e in c["connector"]["equations"] if e["from"] == "Emissions.NO"]
    assert len(hits) == 1 and json.dumps(hits[0]["expression"]) == old_no
    assert raw.count("30.01e-3") == 1
    raw = raw.replace("30.01e-3", "46.0055e-3")
    o1 = "Provenance: GasChem.jl ext/EarthSciDataExt.jl couple2(SuperFastCoupler, NEI2016MonthlyEmisCoupler), which performs the same mapping via operator_compose with conv = uconv/MW_X, uconv = nmolpermol*MW_Air."
    n1 = ("Provenance: jialinl6/GasChemMTK.jl@37792b29 (branch superfast-gas-only) ext/EarthSciDataExt.jl "
          "couple2(SuperFastCoupler, NEI2016MonthlyEmisCoupler), which performs the same mapping via operator_compose with "
          "conv = uconv/MW_X, uconv = nmolpermol*MW_Air. NEI's NO is NOx-as-NO2 mass, so NO is converted with MW_NO2 "
          "(46.0055e-3), not MW_NO, as upstream does. NEI2016ElevatedEmis (nei2016_elevated.esm) exposes the same "
          "species, so this library binds it as Emissions too: import it once for surface-only emissions, or once per "
          "NEI component for surface plus elevated.")
    assert raw.count(o1) == 1
    raw = raw.replace(o1, n1)
    json.loads(raw)
    path.write_text(raw)
    print(f"wrote {path.relative_to(REPO)}")


def patch_spatial():
    path, d = load("couplings/earthscidata_spatial_params.esm")
    stub = copy.deepcopy(d["models"]["NEI2016MonthlyEmis"])
    stub["reference"]["notes"] = ("Coupling-interface stub: declares the three spatial parameter variables (lat, lon, lev) used "
                                  "as targets in this coupling file. Full model definition lives in "
                                  f"components/earthsci_data/nei2016_elevated.esm.\n\nUpstream: {UPSTREAM} src/coupling.jl "
                                  "(NEI2016ElevatedEmis shares NEI2016MonthlyEmisCoupler, so _couple_emis_to_met applies).")
    stub["variables"]["lev"]["description"] = ("Vertical level index (1=surface); the elevated sectors emit into layers "
                                               "1..11. Coupled via param_to_var from met provider (GEOSFP or ERA5) lev.")
    d["models"]["NEI2016ElevatedEmis"] = stub
    extra = []
    for c in d["coupling"]:
        if c["to"].startswith("NEI2016MonthlyEmis."):
            e = copy.deepcopy(c)
            e["to"] = e["to"].replace("NEI2016MonthlyEmis", "NEI2016ElevatedEmis")
            e["description"] = e["description"].replace("NEI2016MonthlyEmis", "NEI2016ElevatedEmis").replace(
                "EarthSciData.jl@830435a src/coupling.jl:31 (_couple_emis_to_met)",
                f"{UPSTREAM} src/coupling.jl (_couple_emis_to_met via NEI2016MonthlyEmisCoupler)")
            if e["to"].endswith(".lev"):
                met = e["from"].split(".")[0]
                old = ("NEI applies emissions only at lev < 2 (surface layer); param_to_var ensures the level index matches "
                       "the GEOSFP grid." if met == "GEOSFP" else
                       "NEI applies emissions only at lev < 2; param_to_var ensures the level index matches the ERA5 grid "
                       "(index 1 = 1000 hPa near-surface).")
                new = ("The elevated sectors' vertical profiles are in GEOS-FP layers 1..11, so lev selects the layer "
                       "receiving its share of each column." if met == "GEOSFP" else
                       "The elevated sectors' vertical profiles are defined on GEOS-FP's 72 layers; on ERA5's pressure-level "
                       "grid (index 1 = 1000 hPa) they are applied by level index, as upstream's coupling does.")
                assert old in e["description"], e["description"]
                e["description"] = e["description"].replace(old, new)
            extra.append(e)
    d["coupling"] += extra
    md = d["metadata"]
    md["description"] = md["description"].replace(
        "- NEI2016MonthlyEmis ↔ GEOSFP: lat, lon, lev (3 mappings).",
        "- NEI2016MonthlyEmis ↔ GEOSFP: lat, lon, lev (3 mappings).\n- NEI2016ElevatedEmis ↔ GEOSFP: lat, lon, lev (3 mappings).").replace(
        "- NEI2016MonthlyEmis ↔ ERA5: lat, lon, lev (3 mappings).",
        "- NEI2016MonthlyEmis ↔ ERA5: lat, lon, lev (3 mappings).\n- NEI2016ElevatedEmis ↔ ERA5: lat, lon, lev (3 mappings).").replace(
        "Total: 14 variable_map entries.", f"Total: {len(d['coupling'])} variable_map entries.").replace(
        "NEI2016MonthlyEmis) via param_to_var. All five models", "NEI2016MonthlyEmis, NEI2016ElevatedEmis) via param_to_var. All six models").replace(
        "Source: EarthSciData.jl@830435a src/coupling.jl:24,31,...",
        f"Source: EarthSciData.jl@830435a src/coupling.jl:24,31,...; NEI2016ElevatedEmis: {UPSTREAM}, which couples it "
        "through NEI2016MonthlyEmisCoupler.")
    assert "All six models" in md["description"] and f"Total: {len(d['coupling'])}" in md["description"]
    write(path, d, 4)


def patch_geosfp():
    path, d = load("components/earthsci_data/geosfp.esm")
    m = d["models"]["GEOSFP"]
    G = REF["geosfp"]
    V = m["variables"]
    assert not any("Tv_bar" in json.dumps(e["rhs"]) for e in m["equations"] if e["lhs"] != "Tv_bar")
    m["equations"] = [e for e in m["equations"] if e["lhs"] != "Tv_bar"]
    V.pop("Tv_bar")
    tv = " Used in layer-mean virtual temperature Tv\u0304."
    assert tv in V["Tv_sfc"]["description"]
    V["Tv_sfc"]["description"] = V["Tv_sfc"]["description"].replace(tv, "")
    md, old = d["metadata"], "virtual temperature Tv and Tv_sfc, layer-mean virtual temperature Tv_bar,"
    assert old in md["description"]
    md["description"] = md["description"].replace(old, "virtual temperature Tv and Tv_sfc, the level derivative δPδlev,")
    # P's tables are upstream's _Ap_data / _Bp_data, which upstream builds as hPa*100 (last-bit
    # roundoff apart from the decimal Pa values here).
    P = next(e for e in m["equations"] if e["lhs"] == "P")["rhs"]
    ap, bp = P["args"][0]["args"][0]["value"], P["args"][1]["args"][0]["args"][0]["value"]
    assert len(ap) == len(G["Ap"]) and all(abs(a - b) <= 1e-15 * abs(b) for a, b in zip(ap + bp, G["Ap"] + G["Bp"]))
    n = len(ap) - 1
    knots = C([float(k) for k in range(1, n + 1)])
    seg = Op("max", 1.0, Op("min", float(n), Op("-", mul(-1.0, Op("floor", mul(-1.0, "lev"))), 1.0)))
    V["δPδlev"] = {"type": "unknown", "units": "Pa",
                   "description": "Pressure gradient with respect to the hybrid level, dP/dlev (Pa): the slope of P's "
                                  "piecewise-linear Ap/Bp segment, (Ap[i+1] - Ap[i]) + (Bp[i+1] - Bp[i])*PS with "
                                  "i = clamp(ceil(lev) - 1, 1, 72), so an integer lev takes the segment below it "
                                  "(upstream Ap_deriv/Bp_deriv)."}
    m["equations"].append({"lhs": "δPδlev", "rhs": add(
        fn("interp.linear", C([ap[i + 1] - ap[i] for i in range(n)]), knots, seg),
        mul(fn("interp.linear", C([bp[i + 1] - bp[i] for i in range(n)]), knots, seg), "PS"))})
    # The runner needs a value for every data-fed parameter; all but PS are pinned to 1.
    fed = sorted(k for k, v in V.items() if v.get("update", {}).get("kind") == "data" and "default" not in v)
    assert "PS" in fed
    levs = list(dict.fromkeys(c["lev"] for c in G["cases"]))
    cases = [c for c in G["cases"] if c["PS"] == (101325.0 if levs.index(c["lev"]) % 2 == 0 else 85000.0)]
    assert len(cases) == len(levs)
    m["tests"] = [{
        "id": f"hybrid_pressure_lev_{str(c['lev']).replace('.', '_')}_ps_{int(c['PS'])}",
        "description": (f"P and δPδlev at lev = {c['lev']} with the surface pressure pinned to PS = {c['PS']} Pa and the "
                        f"other data-fed parameters to 1 (no data is read). Expected values are upstream's "
                        f"Ap(lev) + Bp(lev)*PS and Ap_deriv(lev) + Bp_deriv(lev)*PS ({UPSTREAM})."),
        "parameter_overrides": {**{k: 1.0 for k in fed}, "PS": c["PS"], "lev": c["lev"]},
        "time_span": {"start": 0.0, "end": 1.0},
        "tolerance": {"rel": 1e-12, "abs": 1e-9},
        "assertions": [{"variable": "P", "time": 0.0, "expected": c["P"]},
                       {"variable": "δPδlev", "time": 0.0, "expected": c["dPdlev"]}],
    } for c in cases]
    m["reference"]["notes"] += (
        f"\n\nChecked {TODAY} against {UPSTREAM}. δPδlev is now carried: upstream replaced the DataInterpolations "
        "tables with its own piecewise-linear Ap/Bp and their closed-form derivative (the slope of the segment below an "
        "integer lev), which an interp.linear over the 72 segment differences reproduces, so the gt-6ohw gap above no "
        "longer applies to it. δZδlev and Tv_bar (Tv\u0304) are gone upstream and removed here. Z_agl is still not "
        "carried: upstream now sums the hypsometric thickness of every layer from the surface up to lev, which needs T "
        "and QV at each of those layers (a column read the data-fed parameters here, one value per cell, do not give), "
        "and nothing in the SuperFast gas-only stack reads it. Inline tests pin PS and check P and δPδlev against "
        "upstream. Outside 1 <= lev <= 73 upstream's P extrapolates the end segments while interp.linear here clamps; "
        "GEOS-FP levels stay inside that range. EarthSciData #222 (interpolation on the true, non-uniform latitude "
        "knots of the half-polar GEOS-FP grid) and the PR's index-box subset reads are inside the Julia loader, which a "
        "data source here does not encode: an esm host regrids the source itself.")
    write(path, d, 2)


if __name__ == "__main__":
    surface = patch_surface()
    build_elevated(surface)
    patch_regfns()
    patch_coupling()
    patch_spatial()
    patch_geosfp()
