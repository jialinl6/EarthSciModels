#!/usr/bin/env python3
"""Regenerate the two Fast-JX .esm files from the reference GasChem package
(jialinl6/GasChemMTK.jl, branch superfast-gas-only, commit 37792b29).

  components/gaschem/fastjx/fastjx.esm
      == upstream `FastJX(t_ref)` (default fluxes = :scattering)
  components/gaschem/fastjx/fastjx_interp_troposphere.esm
      == upstream `FastJX_interpolation_troposphere(t_ref; mech = :superfast)`

Inputs are written by extract_fastjx_data.jl (same directory) against the reference package:
  DATADIR/fastjx_ref.json     sigma/phi knots, flux tables, reference j-values
  DATADIR/fastjx_direct.json  fixed-column constants of the online direct beam + a check grid

Run:
  julia --project=ENV scripts/_archive/migrations/extract_fastjx_data.jl DATADIR
  python3 scripts/_archive/migrations/gen_fastjx_esm.py DATADIR [OUTDIR]
"""
import json
import math
import sys
from pathlib import Path

DATA = Path(sys.argv[1])
REF = json.loads((DATA / "fastjx_ref.json").read_text())
DIR = json.loads((DATA / "fastjx_direct.json").read_text())
UPSTREAM = "jialinl6/GasChemMTK.jl@37792b29 (branch superfast-gas-only)"
T_REF_DEFAULT = 1462060800.0  # 2016-05-01T00:00:00Z

WL = [int(round(w)) for w in REF["flux"]["WL"]]
FNAME = [f"F_{w}" for w in WL]


# --------------------------------------------------------------------------- AST
def Op(o, *a):
    return {"op": o, "args": list(a)}


def fn(name, *a):
    return {"op": "fn", "name": name, "args": list(a)}


def C(v):
    return {"op": "const", "args": [], "value": v}


def ifelse(c, a, b):
    return Op("ifelse", c, a, b)


def add(*a):
    return a[0] if len(a) == 1 else Op("+", *a)


def mul(*a):
    return Op("*", *a)


DEG2RAD = math.pi / 180.0  # Julia deg2rad(x) = x * (pi/180)


# ------------------------------------------------------------------ variables
class Builder:
    def __init__(self):
        self.vars = {}
        self.eqs = []

    def param(self, name, units, default, desc):
        self.vars[name] = {"type": "parameter", "units": units, "default": default, "description": desc}

    def obs(self, name, units, expr, desc):
        assert name not in self.vars, name
        self.vars[name] = {"type": "unknown", "units": units, "description": desc}
        self.eqs.append({"lhs": name, "rhs": expr})


def common_inputs(b):
    b.param("T", "K", 298.0, "Air temperature (K). Selects the temperature-dependent cross sections "
            "and O(1D) quantum yields and enters adjust_j_o31D. Upstream default 298 K.")
    b.param("P", "Pa", 101325.0, "Air pressure (Pa). Selects the actinic-flux level and enters "
            "adjust_j_o31D; ActAld's cross section is interpolated on P/100 (hPa). Upstream default 101325 Pa.")
    b.param("H2O", "ppb", 450.0, "Water-vapour mixing ratio (ppb); enters adjust_j_o31D only. "
            "Upstream default 450 ppb (the reference couples it from GEOS-FP RH).")
    b.param("lat", "deg", 40.0, "Latitude (degrees north). Upstream default 40.")
    b.param("long", "deg", -97.0, "Longitude (degrees east). Upstream default -97. Named `long` as upstream.")
    b.param("t_ref", "s", T_REF_DEFAULT, "Reference Unix time (s) of t = 0; the absolute time is "
            "t + t_ref. Upstream takes it as the constructor argument (DomainInfo's t_ref); "
            "default 2016-05-01T00:00:00Z.")


def solar_geometry(b):
    """Upstream `cos_solar_zenith_angle(t + t_ref, lat, long)` and `solar_flux_factor(t + t_ref)`."""
    b.obs("t_utc", "s", Op("+", "t", "t_ref"), "Absolute UTC time (Unix s) = t + t_ref.")
    b.obs("DOY", "1", fn("datetime.day_of_year", "t_utc"), "UTC day of year (1 = Jan 1).")
    b.obs("utc_hours", "1", add(fn("datetime.hour", "t_utc"), Op("/", fn("datetime.minute", "t_utc"), 60.0),
                               Op("/", fn("datetime.second", "t_utc"), 3600.0)),
          "UTC hour of day, hour + minute/60 + second/3600 (upstream `hours`).")
    b.obs("year", "1", fn("datetime.year", "t_utc"), "UTC calendar year.")
    b.obs("yearday_temp", "1",
          mul(2.0 * math.pi, add(Op("-", "DOY", 1.0), Op("/", Op("-", "utc_hours", 12.0), 24.0))),
          "2*pi*(DOY - 1 + (hours - 12)/24), the un-normalised fractional-year angle.")
    b.obs("gamma", "1",
          ifelse(Op("==", Op("-", "year", mul(4.0, Op("floor", Op("/", "year", 4.0)))), 0.0),
                 Op("/", "yearday_temp", 366.0), Op("/", "yearday_temp", 365.0)),
          "Fractional-year angle (rad): yearday_temp/366 when mod(year, 4) == 0, else /365 (upstream's leap rule).")
    b.obs("Eot", "1", mul(229.18, add(0.000075, mul(0.001868, Op("cos", "gamma")),
                                     mul(-0.032077, Op("sin", "gamma")),
                                     mul(-0.014615, Op("cos", mul("gamma", 2.0))),
                                     mul(-0.040849, Op("sin", mul("gamma", 2.0))))),
          "Equation of time (minutes).")
    b.obs("timezone", "1", Op("floor", Op("/", "long", 15.0)), "floor(long/15): the nominal time-zone offset (hours).")
    b.obs("time_offset", "1", add("Eot", mul(4.0, Op("-", "long", mul(15.0, "timezone")))),
          "Eot + 4*(long - 15*timezone) (minutes).")
    b.obs("t_local", "s", add("t_utc", mul("timezone", 3600.0)), "Zone-local time (Unix s) = t_utc + timezone*3600.")
    b.obs("tst", "1", add(fn("datetime.hour", "t_local"), Op("/", fn("datetime.minute", "t_local"), 60.0),
                         Op("/", fn("datetime.second", "t_local"), 3600.0), Op("/", "time_offset", 60.0)),
          "True solar time (hours).")
    b.obs("AHR", "rad", mul(15.0 * DEG2RAD, Op("-", "tst", 12.0)), "Hour angle (rad) = deg2rad(15)*(tst - 12).")
    b.obs("LAT", "rad", Op("abs", Op("/", mul("lat", math.pi), 180.0)),
          "abs(lat) in radians. Upstream takes the absolute value (see notes: southern-hemisphere seasons are not flipped).")
    dec_inner = add(mul(360.0 / 365.24, add("DOY", 10.0)),
                    mul(360.0 / math.pi * 0.0167,
                        Op("sin", mul(mul(360.0 / 365.24, Op("-", "DOY", 2.0)), DEG2RAD))))
    b.obs("DEC", "rad", Op("asin", mul(math.sin(-23.44 * DEG2RAD), Op("cos", mul(dec_inner, DEG2RAD)))),
          "Solar declination (rad): asin(sin(deg2rad(-23.44)) * cos(deg2rad(360/365.24*(DOY+10) + "
          "360/pi*0.0167*sin(deg2rad(360/365.24*(DOY-2)))))).")
    b.obs("cosSZA", "1", add(mul(Op("sin", "LAT"), Op("sin", "DEC")),
                             mul(Op("cos", "LAT"), Op("cos", "DEC"), Op("cos", "AHR"))),
          "Cosine of the solar zenith angle (upstream cos_solar_zenith_angle; unclamped).")
    b.obs("solf", "1", Op("-", 1.0, mul(0.034, Op("cos", Op("/", mul(mul(Op("-", "DOY", 172.0), 2.0), math.pi), 365.0)))),
          "Earth-Sun distance factor on the TOA flux, 1 - 0.034*cos((DOY - 172)*2*pi/365) "
          "(upstream solar_flux_factor, after GEOS-Chem SOLAR_JX SOLFX).")


def bilinear(table, x_axis, y_axis, x, y):
    return fn("interp.bilinear", C(table), C(x_axis), C(y_axis), x, y)


def flux_interp(b):
    fl = REF["flux"]
    for k in range(18):
        b.obs(FNAME[k], "1/s",
              mul(bilinear(fl["total"][k], fl["P_axis"], fl["cosSZA_axis"], "P", "cosSZA"), "solf"),
              f"Actinic flux in band {k+1} ({WL[k]} nm): bilinear (P, cosSZA) lookup of upstream's "
              f"total-flux table (direct + diffuse, tropospheric_interpolation_data.bson; flat outside "
              f"the 1000-100000 Pa x -0.2..1.0 grid) times solf.")


def flux_online(b):
    """Upstream flux_sys_scattering: calc_direct_flux(cosSZA, P, k) + diffuse_k(P, cosSZA), times solf."""
    R = DIR["R2"]                 # 147 fine-grid radii (cm), index i = 1..147
    D = DIR["D"]                  # D[k][i-1] = OD_total[div(i+1,2), k]
    PL = DIR["P_levels"]          # 73 edges, surface first
    TF = DIR["top_flux"]
    asc = list(reversed(PL))
    R_edges = [R[2 * j] for j in range(73)]   # fine index 2j-1 (1-based) = edge j
    fl = REF["flux"]

    b.obs("lev_index", "1",
          ifelse(Op(">", "P", PL[0]), 1.0, Op("-", 74.0, fn("interp.searchsorted", "P", C(asc)))),
          "Upstream find_closest_pressure_index(P, P_levels): 1 above the surface edge, else the number of "
          "the 73 GEOS 72-layer edges (at Psurf = 101325 Pa) with P_level >= P. The direct beam is evaluated "
          "at that edge, so it is piecewise constant in P.")
    b.obs("fine_index", "1", Op("-", mul(2.0, "lev_index"), 1.0), "2*lev_index - 1: the edge's index on the 147-point fine grid.")
    b.obs("R_f", "cm", fn("interp.linear", C(R_edges), C([float(j) for j in range(1, 74)]), "lev_index"),
          "Radius (cm) of the evaluation edge, Earth radius 6375 km + ZHL[lev_index] (upstream calcRZ2). "
          "interp.linear at an integer knot returns the table entry exactly.")
    b.obs("Rf_u0", "cm", mul("R_f", "cosSZA"), "R_f*cosSZA.")
    b.obs("b_imp", "cm", mul("R_f", Op("sqrt", Op("max", 0.0, Op("-", 1.0, Op("^", "cosSZA", 2.0))))),
          "Impact parameter of the slant solar ray, R_f*sin(SZA). Upstream's sphere2J iterates "
          "sqrt(1 - (R_i/R_i+1)^2 (1 - mu^2)), which conserves R*sin(theta) = b_imp, so every "
          "level's direction cosine has the closed form used by mu_i.")
    b.obs("dark", "1", ifelse(Op("<", "cosSZA", 0.0), ifelse(Op("<", "b_imp", R[0]), 1.0, 0.0), 0.0),
          "1 when the ray is in the Earth's shadow (upstream shadht test, cosSZA < 0 and b_imp < R_1), else 0.")
    for i in range(1, 148):
        Ri = R[i - 1]
        b.obs(f"mu_{i}", "1",
              Op("/", Op("sqrt", Op("max", 0.0, add(mul(Op("-", Ri, "R_f"), Op("+", Ri, "R_f")), Op("^", "Rf_u0", 2.0)))), Ri),
              f"Direction cosine of the solar ray at fine-grid radius R_{i} = {Ri:.6e} cm: "
              "sqrt(max(0, 1 - (b_imp/R_i)^2)), written to be exact at R_i = R_f (mu = |cosSZA|).")
    for i in range(1, 147):
        Ri, Ri1 = R[i - 1], R[i]
        d = Ri1 - Ri
        up = Op("-", mul(Ri1, f"mu_{i+1}"), mul(Ri, f"mu_{i}"))
        amf = ifelse(Op(">=", float(i), "fine_index"), Op("/", up, d),
                     ifelse(Op("<", "cosSZA", 0.0),
                            ifelse(Op("<", "b_imp", Ri), Op("/", mul(2.0, Op("abs", up)), d),
                                   ifelse(Op("<", "b_imp", Ri1), Op("/", mul(2.0, Ri1, f"mu_{i+1}"), d), 0.0)),
                            0.0))
        b.obs(f"amf_{i}", "1", amf,
              f"Air-mass factor of fine sub-layer {i} (upstream sphere2J): the ascending path above the "
              "evaluation edge, plus - for cosSZA < 0 - the twilight path below it down to the tangent layer.")
    for k in range(18):
        terms = [mul(D[k][i - 1], f"amf_{i}") for i in range(1, 147) if D[k][i - 1] != 0.0]
        b.obs(f"tau_{WL[k]}", "1", mul(0.5, add(*terms)),
              f"Slant optical depth in band {k+1} ({WL[k]} nm): 0.5*sum_i OD_total[div(i+1,2), {k+1}]*amf_i "
              "over upstream's fixed climatological column (Rayleigh + O2 + O3 absorption).")
        fdir = mul(TF[k], ifelse(Op("<", "P", 1.0), 1.0,
                                 ifelse(Op(">", "dark", 0.5), 0.0,
                                        ifelse(Op("<", f"tau_{WL[k]}", 76.0), Op("exp", Op("-", f"tau_{WL[k]}")), 0.0))))
        b.obs(f"Fdir_{WL[k]}", "1/s", fdir,
              f"Direct-beam actinic flux in band {k+1} (upstream calc_direct_flux): top_flux*exp(-tau) "
              "(0 in shadow or for tau >= 76; the bare TOA flux for P < 1 Pa).")
        b.obs(f"Fdif_{WL[k]}", "1/s", bilinear(fl["diffuse"][k], fl["P_axis"], fl["cosSZA_axis"], "P", "cosSZA"),
              f"Clear-sky diffuse actinic flux in band {k+1}: bilinear (P, cosSZA) lookup of upstream's "
              "precomputed Cloud-J Feautrier field (diffuse_flux_data.bson; Rayleigh multiple scattering, "
              "albedo 0.10, AOD 0; flat outside 1000-100000 Pa x -0.2..1.0).")
        b.obs(FNAME[k], "1/s", mul(Op("+", f"Fdir_{WL[k]}", f"Fdif_{WL[k]}"), "solf"),
              f"Actinic flux in band {k+1} ({WL[k]} nm) = (direct + diffuse)*solf (upstream flux_sys_scattering).")


def knot_expr(kn, x):
    if "const" in kn:
        return kn["const"]
    return fn("interp.linear", C(kn["vals"]), C(kn["axis"]), x)


def photolysis(b):
    sig, phi = REF["sigma"], REF["phi"]
    paxis = set(REF["pressure_axis"])
    b.obs("P_hPa", "1", Op("/", "P", 100.0), "P/100: the hPa axis of the pressure-interpolated cross sections (ActAld).")
    for sp in sorted(sig):
        x = "P_hPa" if sp in paxis else "T"
        for k, kn in enumerate(sig[sp]):
            if kn.get("const") == 0.0:
                continue
            how = ("constant" if "const" in kn else
                   f"linear in {'P (hPa)' if x == 'P_hPa' else 'T (K)'} over {kn['axis']}, flat outside")
            b.obs(f"sigma_{sp}_{k+1}", "1", knot_expr(kn, x), f"sigma_{sp} in band {k+1} ({WL[k]} nm): {how}.")
    for k, kn in enumerate(phi["O31D"]):
        how = "constant" if "const" in kn else f"linear in T over {kn['axis']}, flat outside"
        b.obs(f"phi_O31D_{k+1}", "1", knot_expr(kn, "T"), f"O(1D) quantum yield in band {k+1} ({WL[k]} nm): {how}.")

    def jsum(sp, with_phi=False):
        terms = []
        for k, kn in enumerate(sig[sp]):
            if kn.get("const") == 0.0:
                continue
            t = [FNAME[k], f"sigma_{sp}_{k+1}"] + ([f"phi_O31D_{k+1}"] if with_phi else [])
            terms.append(mul(*t))
        return add(*terms)

    for sp in REF["jset"]:
        if sp == "O31D":
            b.obs("j_O31D", "1/s", jsum("O3", True), "j_O31D = sum_k F_k*sigma_O3_k*phi_O31D_k(T) (upstream j_mean_O31D).")
        elif sp in ("NO3a", "NO3b"):
            f = 0.886 if sp == "NO3a" else 0.114
            b.obs(f"j_{sp}", "1/s", mul(jsum("NO3"), f),
                  f"j_{sp} = (sum_k F_k*sigma_NO3_k)*{f} (NO3 -> {'NO2 + O' if sp == 'NO3a' else 'NO + O2'} branch).")
        else:
            b.obs(f"j_{sp}", "1/s", jsum(sp), f"j_{sp} = sum_k F_k*sigma_{sp}_k (upstream j_mean_{sp}; phi = 1).")
    # adjust_j_o31D, verbatim. 6.02e23 is written as 6.02*1e15*1e8 (mdl-ntu: an integer-valued
    # literal > Int64 max is coerced on load by one binding).
    A = mul(6.02, 1.0e15, 1.0e8)
    b.obs("num_density", "1", Op("/", mul(A, "P"), mul(8.314e6, "T")), "Air number density (molec/cm^3, unitless as upstream): A*P/(R*T).")
    b.obs("C_H2O", "1", mul("H2O", 1.0e-9, "num_density"), "H2O number density: H2O[ppb]*1e-9*num_density.")
    b.obs("C_O2", "1", mul(0.2095, "num_density"), "O2 number density: 0.2095*num_density.")
    b.obs("C_N2", "1", mul(0.7808, "num_density"), "N2 number density: 0.7808*num_density.")
    b.obs("C_H2", "1", mul(0.5e-6, "num_density"), "H2 number density: 0.5e-6*num_density.")
    b.obs("RO1DplH2O", "1", mul(1.63e-10, Op("exp", Op("/", 60.0, "T")), "C_H2O"), "O(1D) + H2O loss frequency.")
    b.obs("RO1DplH2", "1", mul(1.2e-10, "C_H2"), "O(1D) + H2 loss frequency.")
    b.obs("RO1DplN2", "1", mul(2.15e-11, Op("exp", Op("/", 110.0, "T")), "C_N2"), "O(1D) + N2 loss frequency.")
    b.obs("RO1DplO2", "1", mul(3.3e-11, Op("exp", Op("/", 55.0, "T")), "C_O2"), "O(1D) + O2 loss frequency.")
    b.obs("RO1D", "1", add("RO1DplH2O", "RO1DplH2", "RO1DplN2", "RO1DplO2"), "Total O(1D) loss frequency.")
    b.obs("j_O31D_adj", "1", Op("/", "RO1DplH2O", "RO1D"), "Fraction of O(1D) that reacts with H2O (upstream adjust_j_o31D).")
    b.obs("j_o32OH", "1/s", mul("j_O31D", "j_O31D_adj"), "Effective O3 + hv -> 2 OH rate: j_O31D*j_O31D_adj (drives SuperFast.jO32OH).")


# ---------------------------------------------------------------------- tests
J_ASSERT = ["j_NO2", "j_O31D", "j_o32OH", "j_H2O2", "j_H2COa", "j_H2COb", "j_CH3OOH",
            "j_O3", "j_NO3a", "j_NO3b", "j_N2O5", "j_ActAld", "j_PAN"]


# The online file asserts what is particular to it - the band fluxes the direct beam feeds - plus
# j_NO2 end to end and j_o32OH once. The cross-section / J / adjust_j_o31D code it shares with the
# interpolated file is asserted in full there. The trim is for the Julia runner: its test-time
# observed reader re-evaluates an observed's whole dependency tree on every read, and one J read
# here walks every band's 146 air-mass factors (~14 s per read, measured), where the interpolated
# file's reads are instant.
F_ONLINE = ["F_295", "F_310", "F_333", "F_380", "F_574"]


def make_tests(kind, rel):
    online = kind == "online"
    tests = []
    for sc in REF["scenarios"]:
        v = sc[kind]
        names = ["cosSZA", "solf"] + (F_ONLINE + ["j_NO2"] + (["j_o32OH"] if sc["id"] == "may_midlat_local_noon" else [])
                                      if online else ["F_310", "F_380", "j_O31D_adj"] + J_ASSERT)
        asserts = [{"variable": n, "time": sc["t"], "expected": v[n]} for n in names]
        tests.append({
            "id": sc["id"],
            "description": (f"Upstream reference at t_ref = {sc['t_ref']:.0f} s, t = {sc['t']:.0f} s, "
                            f"lat = {sc['lat']}, long = {sc['long']}, T = {sc['T']} K, P = {sc['P']:.0f} Pa, "
                            f"H2O = {sc['H2O']:g} ppb (cosSZA = {v['cosSZA']:.4f}). Expected values are the "
                            f"reference package's own functions evaluated at these inputs ({UPSTREAM})."),
            "parameter_overrides": {"t_ref": sc["t_ref"], "lat": sc["lat"], "long": sc["long"],
                                    "T": sc["T"], "P": sc["P"], "H2O": sc["H2O"]},
            "time_span": {"start": 0.0, "end": max(sc["t"], 1.0)},
            "tolerance": {"rel": rel, "abs": 1.0e-22},
            "assertions": asserts,
        })
    d = REF["diurnal"]
    asserts = []
    for t, v in zip(d["t"], d[kind]):
        if int(t) % 10800 == 0:
            names = ["cosSZA"] + (["j_NO2"] if online and int(t) in (21600, 43200, 64800)
                                  else [] if online else ["j_NO2", "j_O31D", "j_H2O2", "j_CH3OOH"])
            asserts += [{"variable": n, "time": t, "expected": v[n]} for n in names]
    tests.append({
        "id": "diurnal_cycle_may_midlat_surface",
        "description": ("24 h at lat 40, long -97, 101325 Pa, 298 K, 1e7 ppb H2O from 2016-05-01T00Z, sampled every "
                        "3 h: t enters through t + t_ref, so this checks the time-varying solar geometry and solf "
                        f"against the reference ({UPSTREAM})."),
        "parameter_overrides": {"t_ref": d["t_ref"], "lat": d["lat"], "long": d["long"],
                                "T": d["T"], "P": d["P"], "H2O": d["H2O"]},
        "time_span": {"start": 0.0, "end": 86400.0},
        "tolerance": {"rel": rel, "abs": 1.0e-22},
        "assertions": asserts,
    })
    return tests


def analyses():
    # The component is stateless, so each analysis is a parameter sweep: every grid point is
    # evaluated over a 1 s window and the renderer plots its endpoint.
    hours = [T_REF_DEFAULT + 3600.0 * h for h in range(25)]
    return [{
        "id": "diurnal_cycle",
        "description": ("Photolysis rates through 1 May 2016 at 40N, 97W, at the surface, one point per UTC hour "
                        "(t_ref = 2016-05-01T00Z + h hours): upstream's solar geometry sets cosSZA from t + t_ref, "
                        "so the rates peak near 18Z (local solar noon) and fall to the twilight diffuse floor, "
                        "then zero, at night."),
        "parameters": {"lat": 40.0, "long": -97.0, "T": 298.0, "P": 101325.0, "H2O": 1.0e7},
        "time_span": {"start": 0.0, "end": 1.0},
        "parameter_sweep": {"type": "cartesian", "dimensions": [{"parameter": "t_ref", "values": hours}]},
        "plots": [
            {"id": "jNO2_vs_time", "type": "line", "description": "j_NO2 through the day.",
             "x": {"variable": "t_ref", "label": "UTC time (Unix s), 2016-05-01"},
             "y": {"variable": "j_NO2", "label": "j_NO2 (1/s)"}},
            {"id": "jO31D_vs_time", "type": "line", "description": "j_O(1D) through the day.",
             "x": {"variable": "t_ref", "label": "UTC time (Unix s), 2016-05-01"},
             "y": {"variable": "j_O31D", "label": "j_O31D (1/s)"}},
        ],
    }, {
        "id": "pressure_profile",
        "description": ("j_NO2 and j_O(1D) against pressure at local solar noon (2016-05-01T18Z, 40N, 97W, 298 K). "
                        "In fastjx.esm the direct beam is evaluated at the GEOS edge at or above P, so the profile "
                        "steps between edges; fastjx_interp_troposphere.esm interpolates between its 23 nodes."),
        "parameters": {"t_ref": T_REF_DEFAULT + 64800.0, "lat": 40.0, "long": -97.0, "T": 298.0, "H2O": 1.0e7},
        "time_span": {"start": 0.0, "end": 1.0},
        "parameter_sweep": {"type": "cartesian", "dimensions": [
            {"parameter": "P", "values": [float(p) for p in range(10000, 101001, 3000)] + [101325.0]}]},
        "plots": [
            {"id": "jNO2_vs_P", "type": "line", "description": "j_NO2 vs pressure at local noon.",
             "x": {"variable": "P", "label": "P (Pa)"}, "y": {"variable": "j_NO2", "label": "j_NO2 (1/s)"}},
            {"id": "jO31D_vs_P", "type": "line", "description": "j_O(1D) vs pressure at local noon.",
             "x": {"variable": "P", "label": "P (Pa)"}, "y": {"variable": "j_O31D", "label": "j_O31D (1/s)"}},
        ],
    }]


COMMON_REFS = [
    {"citation": "Wild, O., Zhu, X., and Prather, M. J. (2000), Fast-J: Accurate simulation of in- and below-cloud "
                 "photolysis in tropospheric chemical models, J. Atmos. Chem., 37, 245-282."},
    {"citation": "Neu, J. L., Prather, M. J., and Penner, J. E. (2007), Global atmospheric chemistry: Integrating "
                 "over fractional cloud cover, J. Geophys. Res., 112, D11306, doi:10.1029/2006JD008007."},
    {"citation": "Cross sections and quantum yields on the 18-bin grid of GEOS-Chem's CHEM_INPUTS/CLOUD_J/v2024-09 "
                 "FJX_spec.dat (bins 17-18 at 380/574 nm); clear-sky diffuse field from a Julia port of the Cloud-J "
                 "Feautrier solver (geoschem/cloud-j@16d18b07)."},
    {"citation": f"Source: {UPSTREAM}, src/Fast-JX.jl, src/direct_flux.jl, src/interpolations_FastJX.jl, "
                 "src/tropospheric_interpolation_data.bson, src/diffuse_flux_data.bson."},
]

SHARED_NOTES = (
    "What is transcribed verbatim from upstream: cos_solar_zenith_angle (its own declination/equation-of-time "
    "formula, NOT lib/solar.esm's NOAA Spencer series, including upstream's abs(lat) - southern-hemisphere "
    "points get the northern-hemisphere seasonal declination term - and its mod(year, 4) leap rule); "
    "solar_flux_factor; every cross section and quantum yield the 13 rates use, knot for knot as the "
    "reference evaluates them (bins whose two temperature rows agree within Float32 isapprox are the "
    "constant upstream returns); ActAld's cross section on the pressure axis (P/100, hPa) as upstream's "
    "_FJX_PRESSURE_AXIS_J requires; adjust_j_o31D. The J set is upstream's `mech = :superfast` list "
    "(the rates SuperFast and Pollu consume); upstream FastJX also exposes ~95 GEOS-Chem channels that are "
    "not carried here. datetime.* decomposes t_utc with integer seconds, where Julia's unix2datetime "
    "rounds to the millisecond; the two differ only within 0.5 ms of a second boundary."
)


def doc(desc, notes, b, tests, tags):
    return {
        "esm": "1.0.0",
        "metadata": {
            "name": "FastJX",
            "description": desc,
            "authors": ["GasChem.jl / GasChemMTK.jl contributors", "EarthSciModels"],
            "created": "2026-10-28T00:00:00Z",
            "tags": tags,
            "references": COMMON_REFS,
        },
        "models": {
            "FastJX": {
                "reference": {"notes": notes},
                "variables": b.vars,
                "equations": b.eqs,
                "tests": tests,
                "analyses": analyses(),
            }
        },
    }


def build_online(rel):
    b = Builder()
    common_inputs(b)
    solar_geometry(b)
    flux_online(b)
    photolysis(b)
    desc = ("Fast-JX photolysis rates, upstream `FastJX(t_ref)` with its default `fluxes = :scattering`: "
            "the direct solar beam is computed online by spherical Beer-Lambert attenuation through a fixed "
            "climatological column (Rayleigh + O2 + O3) and upstream's precomputed clear-sky diffuse field is "
            "added, both scaled by the Earth-Sun distance factor. 0-D and stateless: every output is an observed "
            "function of (t + t_ref, lat, long, T, P, H2O). Exposes the 13 J-rates of upstream's `:superfast` set "
            "(j_H2O2, j_H2COa, j_H2COb, j_O31D, j_o32OH, j_CH3OOH, j_NO2, j_ActAld, j_PAN, j_NO3a, j_NO3b, j_N2O5, "
            "j_O3) and the 18 band fluxes F_<nm>.")
    notes = (
        f"source_ref: {UPSTREAM}:src/Fast-JX.jl FastJX (fluxes = :scattering), src/direct_flux.jl, "
        "src/interpolations_FastJX.jl flux_sys_scattering.\n\n"
        "Direct beam. Upstream's calc_direct_flux(cosSZA, P, k) snaps P to an edge of the GEOS 72-layer grid "
        "(find_closest_pressure_index), builds the 147-point fine-grid air-mass factors with sphere2J and applies "
        "Beer-Lambert to the fixed OD_total column. sphere2J's recurrence mu' = sqrt(1 - (R/R')^2 (1 - mu^2)) "
        "conserves the impact parameter R*sin(theta), so each fine level's direction cosine is the closed form "
        "mu_i here, and the twilight branch's early exit is the tangent layer b_imp < R_i. Checked against "
        "upstream's iterative calc_direct_flux on 4309 (P, cosSZA) points x 18 bands, every level and twilight "
        "included: worst relative difference 4.4e-11. Diffuse field: upstream's diffuse_flux_data.bson table "
        "with the same bilinear, flat-extrapolated lookup upstream uses.\n\n" + SHARED_NOTES +
        f"\n\nTolerance rationale (rel = {rel:g}): the expected values are the reference package's own Julia "
        "functions at the test inputs. The ESM evaluates the same arithmetic in Float64 from the same constants; "
        "what remains is roundoff (upstream forms the cross-section slope (x2 - x1) in Float32) and the "
        "closed-form-vs-iterative direct beam (<= 4.4e-11); measured, no assertion here is off by more than 3.0e-8 relative, and "
        "the worst over 17 outputs at 300 random (t, lat, long, T, P, H2O) points is 3.3e-8. abs = 1e-22 floors rates that "
        "are exactly 0 at night.\n\nTest coverage: each scenario asserts cosSZA, solf, five band fluxes (295-574 nm) and "
        "j_NO2, with j_o32OH at local noon; the 13 J-rates, cross sections and adjust_j_o31D, which this file shares "
        "with fastjx_interp_troposphere.esm, are asserted in full there. The split is for the Julia runner, whose "
        "test-time observed reader re-evaluates an observed's whole dependency tree on every read: one J read here walks "
        "every band's 146 air-mass factors and costs ~14 s, so the full set would take ~30 min."
    )
    tags = ["chemistry", "photolysis", "fast-jx", "scattering", "spherical-geometry", "interp-linear", "interp-bilinear"]
    return doc(desc, notes, b, make_tests("online", rel), tags)


def build_interp(rel):
    b = Builder()
    common_inputs(b)
    solar_geometry(b)
    flux_interp(b)
    photolysis(b)
    desc = ("Fast-JX photolysis rates, upstream `FastJX_interpolation_troposphere(t_ref; mech = :superfast)`: "
            "no radiation is computed at run time - the 18 band actinic fluxes are bilinear (P, cosSZA) lookups "
            "of upstream's offline precompute of the online FastJX (direct beam + clear-sky diffuse field at each "
            "node), scaled by the Earth-Sun distance factor. 0-D and stateless; same inputs, cross sections and "
            "13 J-rates as fastjx.esm.")
    notes = (
        f"source_ref: {UPSTREAM}:src/interpolations_FastJX.jl FastJX_interpolation_troposphere (mech = :superfast).\n\n"
        "Flux table: tropospheric_interpolation_data.bson as regenerated upstream by "
        "scripts/scattering/precompute_table.jl (23 P nodes 1000-100000 Pa x 61 cosSZA nodes -0.2..1.0, total = "
        "calc_direct_flux + Cloud-J diffuse at each node, solf applied at run time). It agrees with the online "
        "fastjx.esm exactly at the nodes. Between them the two differ by what interpolation does to the direct "
        "beam, which the online model holds piecewise constant on the 73 GEOS edges: against upstream's online "
        "FastJX the six SuperFast J's here are within 0.43% at the surface over a full day while cosSZA >= 0.05 "
        "(2.2% in twilight), within 0.7% at 950 hPa, and up to 9.6% at 300 hPa at low sun (cosSZA ~0.05).\n\n"
        "Interface change: cosSZA is computed internally from (t + t_ref, lat, long) as upstream does. The "
        "previous revision took cos_sza as an input parameter; a host that set it now sets lat/long/t_ref "
        "instead.\n\n" + SHARED_NOTES +
        f"\n\nTolerance rationale (rel = {rel:g}): expected values are the reference package's own functions at "
        "the test inputs; the ESM repeats the same Float64 arithmetic on the same tables, leaving roundoff (and "
        "upstream's Float32 cross-section slope); measured, the worst assertion here is off by 3.0e-8 relative, and the worst over "
        "17 outputs at 300 random (t, lat, long, T, P, H2O) points is 3.3e-8. abs = 1e-22 floors rates that are exactly 0 at night."
    )
    tags = ["chemistry", "photolysis", "fast-jx", "interp-linear", "interp-bilinear"]
    return doc(desc, notes, b, make_tests("interp", rel), tags)


def dumps(o, ind=0):
    """json.dumps(indent=2), except that an array of numbers (a flux-table row, a knot vector)
    and any node whose compact form is short stay on one line, which keeps the 23 x 61 tables
    and the expression trees legible and the files small."""
    pad, sub = "  " * ind, "  " * (ind + 1)
    if isinstance(o, (dict, list)):
        flat = json.dumps(o, ensure_ascii=False)
        if len(flat) <= 100:
            return flat
    if isinstance(o, dict):
        if not o:
            return "{}"
        items = [f"{sub}{json.dumps(k, ensure_ascii=False)}: {dumps(v, ind + 1)}" for k, v in o.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(o, list):
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in o):
            return "[" + ", ".join(json.dumps(x) for x in o) + "]"
        return "[\n" + ",\n".join(sub + dumps(x, ind + 1) for x in o) + "\n" + pad + "]"
    return json.dumps(o, ensure_ascii=False)


def write(path, d):
    text = dumps(d) + "\n"
    assert json.loads(text) == d
    path.write_text(text)
    print(f"wrote {path} ({path.stat().st_size/1024:.0f} KB, {len(d['models']['FastJX']['variables'])} vars)")


def check_direct_closed_form():
    """The closed-form direct beam of flux_online, in plain Python, against upstream's
    iterative calc_direct_flux on the check grid in fastjx_direct.json."""
    R, PL, TF = DIR["R2"], DIR["P_levels"], DIR["top_flux"]
    D = [[DIR["D"][k][i] for k in range(18)] for i in range(147)]
    asc = list(reversed(PL))
    worst = 0.0
    for g in DIR["grid"]:
        u0, P = g["u0"], g["P"]
        idx = 1 if P > PL[0] else 74 - next((j + 1 for j, a in enumerate(asc) if a >= P), 74)
        f = 2 * idx - 1
        Rf = R[f - 1]
        b = Rf * math.sqrt(max(0.0, 1 - u0 * u0))
        mu = [math.sqrt(max(0.0, (Ri - Rf) * (Ri + Rf) + (Rf * u0) ** 2)) / Ri for Ri in R]
        amf = []
        for i in range(1, 147):
            Ri, Ri1, d = R[i - 1], R[i], R[i] - R[i - 1]
            up = Ri1 * mu[i] - Ri * mu[i - 1]
            if i >= f:
                a = up / d
            elif u0 >= 0:
                a = 0.0
            elif b < Ri:
                a = 2 * abs(up) / d
            elif b < Ri1:
                a = 2 * Ri1 * mu[i] / d
            else:
                a = 0.0
            amf.append(a)
        dark = u0 < 0 and b < R[0]
        for k in range(18):
            tau = 0.5 * sum(D[i][k] * amf[i] for i in range(146))
            fac = 1.0 if P < 1.0 else 0.0 if dark else (math.exp(-tau) if tau < 76 else 0.0)
            mine, ref = TF[k] * fac, g["F"][k]
            if ref == 0.0 and mine == 0.0:
                continue
            worst = max(worst, abs(mine - ref) / (abs(ref) if abs(ref) >= 1e-30 * TF[k] else TF[k]))
    print(f"closed-form direct beam vs upstream calc_direct_flux: worst relative difference {worst:.2e} "
          f"over {len(DIR['grid'])} (P, cosSZA) points x 18 bands")
    assert worst < 1e-9, worst


if __name__ == "__main__":
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[3] / "components" / "gaschem" / "fastjx"
    rel = 1.0e-6
    check_direct_closed_form()
    write(outdir / "fastjx.esm", build_online(rel))
    write(outdir / "fastjx_interp_troposphere.esm", build_interp(rel))
