# Extract dry/wet deposition data and reference values from the reference
# AtmosphericDeposition package (jialinl6/DepositionMTK.jl, branch superfast-gas-only,
# commit cbe7d7db) for gen_deposition_esm.py.
#
# Needs an environment with that checkout plus JSON3, e.g.
#   julia --project=ENV -e 'using Pkg; Pkg.develop(path="/path/to/DepositionMTK.jl"); Pkg.add("JSON3")'
# Run:
#   julia --project=ENV scripts/_archive/migrations/extract_deposition_data.jl OUTDIR
# Writes OUTDIR/deposition_ref.json.
using AtmosphericDeposition, ModelingToolkit, JSON3, Random
using ModelingToolkit: t
const AD = AtmosphericDeposition
outdir = ARGS[1]

# ---- species of DryDepositionGasFractional, in builder order ---------------------
# The builder pairs `depvel` / `deprate` variables with a local `datas` vector by
# position; read both lists from the source so the pairing is upstream's own.
src = read(joinpath(pkgdir(AD), "src", "dry_deposition.jl"), String)
body = src[findfirst("function DryDepositionGasFractional(; name", src).start:end]
blk(open_pat) = (i = findfirst(open_pat, body); j = findnext(r"\n    (end|\])\n", body, i.stop); body[i.stop:j.start])
vnames = [m.captures[1] for m in eachmatch(r"\bv_([A-Za-z0-9_]+)\(t\)", blk("depvel = @variables begin"))]
knames = [m.captures[1] for m in eachmatch(r"\bk_([A-Za-z0-9_]+)\(t\)", blk("deprate = @variables begin"))]
dnames = [m.captures[1] for m in eachmatch(r"\b([A-Za-z0-9_]+Data)\b", blk("datas = ["))]
@assert vnames == knames "depvel/deprate order differs"
@assert length(vnames) == length(dnames) "$(length(vnames)) species vs $(length(dnames)) GasData"
isSO2 = [m.captures[1] for m in eachmatch(r"isSO2\[(\d+)\] = true", body)]
isO3 = [m.captures[1] for m in eachmatch(r"isO3\[(\d+)\] = true", body)]
species = [Dict("name" => vnames[i], "data" => dnames[i],
    "Dh2oPerDx" => getfield(AD, Symbol(dnames[i])).Dh2oPerDx,
    "Hstar" => getfield(AD, Symbol(dnames[i])).Hstar,
    "Fo" => getfield(AD, Symbol(dnames[i])).Fo,
    "isSO2" => string(i) in isSO2, "isO3" => string(i) in isO3) for i in eachindex(vnames)]
println(length(species), " species; SO2 = ", vnames[parse.(Int, isSO2)], ", O3 = ", vnames[parse.(Int, isO3)])

# Wesely (1989) tables as the reference evaluates them (Float32 storage -> Float64).
tables = Dict{String, Any}(String(n) => [[Float64(getfield(AD, n)[s, c]) for c in 1:11] for s in 1:5]
    for n in (:r_i, :r_lu, :r_ac, :r_gsS, :r_gsO, :r_clS, :r_clO))
tables["inf"] = Float64(AD.inf)

# ---- reference values: upstream's own builder functions ---------------------------
# Upstream's tests evaluate DryDepGas / _WetDeposition this way (test/drydep_test.jl,
# test/wetdep_test.jl): call the builder function with numbers, then substitute the
# package's constants (`AtmosphericDeposition.defaults`, `wd_defaults`). The assembled
# DryDepositionGasFractional() system does not mtkcompile standalone (its unit
# constants are not unknowns), and these functions are exactly what it broadcasts.
using Symbolics
function numeric(x)
    x = Symbolics.substitute(x, Dict(AD.defaults..., AD.wd_defaults...))
    rest = Symbolics.get_variables(x)
    isempty(rest) || (x = Symbolics.substitute(x, Dict(v => ModelingToolkit.getdefault(v) for v in rest)))
    v = Symbolics.value(x)
    # Constant-folded but unevaluated terms (ifelse, exp of literals) - evaluate them as
    # upstream's test helper `to_float` does.
    return v isa Number ? Float64(v) : Float64(eval(Symbolics.toexpr(Symbolics.unwrap(x))))
end
const FR = ["f_urban", "f_agricultural", "f_range", "f_deciduous", "f_coniferous", "f_mixedforest",
    "f_water", "f_barren", "f_wetland", "f_rangeag", "f_rockyshrubs"]
const g0 = 9.81   # AtmosphericDeposition `g`, used in deprate = depvel * g * ρA / del_P
function dry_eval(p)
    out = Dict{String, Float64}()
    f = [Float64(p[k]) for k in FR]
    for sp in species
        gd = getfield(AD, Symbol(sp["data"]))
        v = numeric(AD.DryDepGasFractional(p["lev"], p["z"], p["z0"], p["u_star"], p["L"], p["rho_A"], gd,
            p["G"], p["Ts"], p["theta"], p["season"], f..., false, false, sp["isSO2"], sp["isO3"]))
        out["v_" * sp["name"]] = v
        out["k_" * sp["name"]] = v * g0 * p["rho_A"] / p["del_P"]
    end
    return out
end
dry_case(id, p) = Dict("id" => id, "p" => p, "out" => dry_eval(p))
base = Dict{String, Any}("season" => 1, "z" => 60.0, "del_P" => 1520.0, "z0" => 0.04, "u_star" => 0.44,
    "L" => 0.0, "rho_A" => 1.2, "G" => 300.0, "Ts" => 298.0, "theta" => 0.0, "lev" => 1)
fr(; kw...) = Dict{String, Any}(f => Float64(get(kw, Symbol(f), 0.0)) for f in FR)
dry_cases = [
    dry_case("defaults_mixedforest", merge(base, fr(f_mixedforest = 1.0))),
    dry_case("midwest_mosaic_summer_unstable", merge(base, fr(f_agricultural = 0.55, f_rangeag = 0.2,
        f_deciduous = 0.1, f_urban = 0.1, f_water = 0.05), Dict("L" => -40.0, "u_star" => 0.35, "G" => 750.0,
        "Ts" => 303.0, "z" => 58.0, "z0" => 0.25))),
    dry_case("forest_autumn_stable", merge(base, fr(f_coniferous = 0.6, f_mixedforest = 0.3, f_wetland = 0.1),
        Dict("season" => 2, "L" => 120.0, "u_star" => 0.2, "G" => 120.0, "Ts" => 285.0, "z0" => 1.0))),
    dry_case("winter_subfreezing", merge(base, fr(f_range = 0.5, f_barren = 0.3, f_rockyshrubs = 0.2),
        Dict("season" => 4, "Ts" => 265.0, "G" => 200.0, "L" => 300.0, "u_star" => 0.3))),
    dry_case("hot_closure_above_40C", merge(base, fr(f_agricultural = 1.0), Dict("Ts" => 315.0, "G" => 900.0))),
    dry_case("water_late_autumn_night", merge(base, fr(f_water = 0.9, f_urban = 0.1),
        Dict("season" => 3, "G" => 0.0, "Ts" => 280.0, "L" => -500.0))),
    dry_case("transitional_slope", merge(base, fr(f_deciduous = 0.4, f_range = 0.6),
        Dict("season" => 5, "theta" => 0.1, "G" => 500.0))),
    dry_case("aloft_level_zero", merge(base, fr(f_mixedforest = 1.0), Dict("lev" => 2))),
]
# Broader sweep for the verification script (not written into the .esm).
rng = MersenneTwister(1028)
dry_random = []
for n in 1:200
    f = rand(rng, 11) .^ 3; f ./= sum(f)
    p = Dict{String, Any}("season" => rand(rng, 1:5), "z" => 10 + 90 * rand(rng), "del_P" => 1000 + 1000 * rand(rng),
        "z0" => 10^(-3 + 3.3 * rand(rng)), "u_star" => 0.05 + 0.9 * rand(rng),
        "L" => rand(rng) < 0.15 ? 0.0 : (rand(rng) < 0.5 ? -1 : 1) * 10^(1 + 2.5 * rand(rng)),
        "rho_A" => 1.0 + 0.3 * rand(rng), "G" => 1000 * rand(rng), "Ts" => 255 + 60 * rand(rng),
        "theta" => 0.2 * rand(rng), "lev" => rand(rng) < 0.1 ? 2 : 1)
    for (i, k) in enumerate(FR); p[k] = f[i]; end
    push!(dry_random, dry_case("rand$n", p))
end

function wet_eval(p)
    k = AD._WetDeposition(p["cloudFrac"], p["qrain"], p["rho_air"], 0.0)   # Δz no longer enters
    return Dict("k_particle" => numeric(k[1]), "k_SO2" => numeric(k[2]), "k_othergas" => numeric(k[3]))
end
wbase = Dict{String, Any}("cloudFrac" => 0.5, "qrain" => 0.5, "rho_air" => 1.204)   # WetDeposition() defaults
wet_cases = [Dict("id" => id, "p" => merge(wbase, p), "out" => wet_eval(merge(wbase, p))) for (id, p) in [
    ("defaults", Dict{String, Any}()),
    ("light_rain_partly_cloudy", Dict{String, Any}("cloudFrac" => 0.3, "qrain" => 2.0e-5, "rho_air" => 1.15)),
    ("heavy_rain_overcast", Dict{String, Any}("cloudFrac" => 1.0, "qrain" => 4.0e-4, "rho_air" => 1.2)),
    ("rain_no_cloud", Dict{String, Any}("cloudFrac" => 0.0, "qrain" => 1.0e-4, "rho_air" => 1.1)),
    ("no_rain", Dict{String, Any}("cloudFrac" => 0.8, "qrain" => 0.0, "rho_air" => 1.2)),
]]

# The existing standalone-ESM tests, re-evaluated by upstream at the same inputs.
esm(path) = JSON3.read(read(joinpath(@__DIR__, "..", "..", "..", path), String))
aer_m = esm("components/atmospheric_deposition/dry_aerosol.esm").models.DryDepositionAerosol
aer_cases = []
for tst in aer_m.tests
    o = Dict(String(k) => v for (k, v) in pairs(tst.parameter_overrides))
    q(k, d) = Float64(get(o, k, d))
    # DryDepositionAerosol() defaults; the ESM never exposed WesleySeason/WesleyLandUse.
    v = numeric(AD.DryDepParticle(q("lev", 1), q("z", 50), q("z0", 0.04), q("u_star", 0.44), q("L", 0),
        q("Dp", 0.8e-6), q("Ts", 298), q("P", 101325), q("rho_particle", 1000.0), q("rho_a", 1.2),
        Int(q("SeinfeldSeason", 1)), 1, Int(q("SeinfeldLandUse", 3)), 10))
    push!(aer_cases, Dict("id" => tst.id, "out" => Dict("v" => v, "k" => v * g0 * q("rho_a", 1.2) / q("del_P", 1520))))
end

wsr_m = esm("components/atmospheric_deposition/wesley1989/surface_resistance.esm").models.WesleySurfaceResistance
# Three more inputs that reach the code upstream changed (none of the existing tests did):
# the r_s hot closure, and the 1 s/m floors on r_ac / r_gsS where the tables hold 0.
wsr_extra = [
    ("hot_closure_45C_general", Dict("Dh2oPerDx" => 1.6, "Hstar" => 0.01, "Fo" => 0.1, "G" => 800.0, "Ts" => 45.0,
        "theta" => 0.0, "iSeason" => 1, "iLandUse" => 2, "rain" => 0, "dew" => 0, "isSO2" => 0, "isO3" => 0)),
    ("water_floors_so2", Dict("Dh2oPerDx" => 1.9, "Hstar" => 1.0e5, "Fo" => 0.0, "G" => 300.0, "Ts" => 20.0,
        "theta" => 0.0, "iSeason" => 1, "iLandUse" => 7, "rain" => 0, "dew" => 0, "isSO2" => 1, "isO3" => 0)),
    ("wetland_floor_soluble_gas", Dict("Dh2oPerDx" => 1.9, "Hstar" => 1.0e14, "Fo" => 0.0, "G" => 300.0, "Ts" => 20.0,
        "theta" => 0.0, "iSeason" => 1, "iLandUse" => 9, "rain" => 0, "dew" => 0, "isSO2" => 0, "isO3" => 0)),
]
wsr_inputs = vcat([(String(tst.id), tst.parameter_overrides) for tst in wsr_m.tests],
    [(id, (; (Symbol(k) => v for (k, v) in p)...)) for (id, p) in wsr_extra])
wsr_cases = []
for (id, o) in wsr_inputs
    gd = AD.GasData(Float64(o.Dh2oPerDx), Float64(o.Hstar), Float64(o.Fo))
    G, Ts, θ, s, l = Float64(o.G), Float64(o.Ts), Float64(o.theta), o.iSeason, o.iLandUse
    rain, dew, so2, o3 = o.rain > 0.5, o.dew > 0.5, o.isSO2 > 0.5, o.isO3 > 0.5
    ov(tbl) = AD.obtain_value(s, l, tbl)
    out = Dict(
        "rs" => AD.r_s(G, Ts, s, l, rain || dew),
        "rlux" => AD.r_lux(gd.Hstar, gd.Fo, s, l, rain, dew, so2, o3),
        "r_clx_gen" => AD.r_clx(gd.Hstar, gd.Fo, s, l),
        "r_gsx_gen" => AD.r_gsx(gd.Hstar, gd.Fo, s, l),
        "rclx" => so2 ? ov(AD.r_clS) : o3 ? ov(AD.r_clO) : AD.r_clx(gd.Hstar, gd.Fo, s, l),
        "rgsx" => so2 ? max(ov(AD.r_gsS), 1.0) : o3 ? ov(AD.r_gsO) : AD.r_gsx(gd.Hstar, gd.Fo, s, l),
        "correction" => Ts < 0 ? 1000.0 * exp(-Ts - 4) : 0.0,
        "r_c" => AD.WesleySurfaceResistance(gd, G, Ts, θ, s, l, rain, dew, so2, o3),
    )
    push!(wsr_cases, Dict("id" => id, "p" => Dict(String(k) => v for (k, v) in pairs(o)),
        "out" => Dict(k => Float64(v) for (k, v) in out)))
end

open(joinpath(outdir, "deposition_ref.json"), "w") do io
    JSON3.write(io, Dict("species" => species, "tables" => tables, "fractions" => FR,
        "dry_cases" => dry_cases, "dry_random" => dry_random, "wet_cases" => wet_cases, "aer_cases" => aer_cases, "wsr_cases" => wsr_cases,
))
end
println("wrote ", joinpath(outdir, "deposition_ref.json"))
