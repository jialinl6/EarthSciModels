# Extract NEI2016 time factors, vertical profiles and reference values from the reference
# EarthSciData package (EarthSciML/EarthSciData.jl PR #226, head jialinl6:Oct_merge @87f8da1)
# for gen_nei_esm.py, plus GEOS-FP's hybrid pressure P and δPδlev.
#
# Needs an environment with that checkout plus JSON3, e.g.
#   julia --project=ENV -e 'using Pkg; Pkg.develop(path="/path/to/EarthSciData.jl"); Pkg.add("JSON3")'
# and the EPA 2016fh May sector files (for their variable lists) in EarthSciData's cache.
# Run:
#   julia --project=ENV scripts/_archive/migrations/extract_nei_data.jl OUTDIR
# Writes OUTDIR/nei_ref.json.
using EarthSciData, Dates, JSON3, NCDatasets, Random
const E = EarthSciData
outdir = ARGS[1]

# ---- tables ----------------------------------------------------------------------
tables = Dict(
    "DIURNAL_FACTORS" => E.DIURNAL_FACTORS, "DIURNAL_FACTORS_NOx" => E.DIURNAL_FACTORS_NOx,
    "DIURNAL_FACTORS_ISOP" => E.DIURNAL_FACTORS_ISOP,
    "DayofWeekFactors_CO" => E.DayofWeekFactors_CO, "DayofWeekFactors_NOx" => E.DayofWeekFactors_NOx,
    "PROFILE_NAMES" => E.NEI2016_PROFILE_NAMES, "LAYER_FRACTIONS" => E.NEI2016_LAYER_FRACTIONS,
    "LAYER_TAILS" => E.NEI2016_LAYER_TAILS, "SECTOR_PROFILES" => E.NEI2016_SECTOR_PROFILES,
    "ELEVATED_SECTORS" => E.NEI2016_ELEVATED_SECTORS,
    "NEI_DAP" => E.NEI_DAP, "NEI_DBP" => E.NEI_DBP,
    # the surface layer's dry pressure thickness, (lon, lat) rows on its knots
    "DELP_DRY_SURFACE" => [collect(E.DELP_DRY_SURFACE_ITP.itp.coefs[i, :]) for i in 1:size(E.DELP_DRY_SURFACE_ITP.itp.coefs, 1)],
    "DELP_LON" => collect(E.DELP_DRY_SURFACE_ITP.itp.knots[1]), "DELP_LAT" => collect(E.DELP_DRY_SURFACE_ITP.itp.knots[2]),
)

# ---- species of each elevated sector (from the EPA file headers) -----------------
cache = joinpath(first(DEPOT_PATH), "scratchspaces", "a293c155-435f-439d-9c11-a083b6b47337", "earthsci_data",
    "https_gaftp.epa.gov", "Air", "emismod", "2016", "v1", "gridded", "monthly_netCDF")
species = Dict{String, Vector{String}}()
for s in E.NEI2016_ELEVATED_SECTORS
    NCDataset(joinpath(cache, "2016fh_16j_$(s)_12US1_month_05.ncf")) do ds
        species[s] = sort([k for k in keys(ds) if k != "TFLAG" && ndims(ds[k]) == 4])
    end
end

# ---- time factors at sample instants, including half-hour meridians -------------
rng = MersenneTwister(226)
lons = vcat([-97.5, -112.5, -82.5, -67.5, -7.5, 7.5, -97.0, -120.3, -75.0, 0.0], -130 .+ 70 .* rand(rng, 20))
times = vcat([datetime2unix(DateTime(2016, 5, 1)) + h * 3600 for h in 0:3:48],
    datetime2unix(DateTime(2016, 1, 1)) .+ 366 * 86400 .* rand(rng, 30))
clocks = []
for lo in lons, tt in times
    l = deg2rad(lo)
    push!(clocks, Dict("t" => tt, "lon_deg" => lo,
        "diurnal" => E.diurnal_itp(tt, l), "diurnal_NOx" => E.diurnal_itp_NOx(tt, l),
        "diurnal_ISOP" => E.diurnal_itp_ISOP(tt, l), "dow_CO" => E.dayofweek_itp_CO(tt, l),
        "dow_NOx" => E.dayofweek_itp_NOx(tt, l)))
end

# ---- the existing registered-function tests, re-evaluated -----------------------
esm(path) = JSON3.read(read(joinpath(@__DIR__, "..", "..", "..", path), String))
regfn = Dict{String, Any}()
for (file, f) in (("nei_diurnal_itp", E.diurnal_itp), ("nei_diurnal_itp_nox", E.diurnal_itp_NOx),
    ("nei_diurnal_itp_isop", E.diurnal_itp_ISOP), ("nei_dayofweek_itp_co", E.dayofweek_itp_CO),
    ("nei_dayofweek_itp_nox", E.dayofweek_itp_NOx))
    m = first(values(esm("registered_functions/$file.esm").models))
    regfn[file] = [Dict("id" => tst.id, "value" => f(Float64(tst.parameter_overrides.t_utc), Float64(tst.parameter_overrides.lon)))
        for tst in m.tests]
end

# ---- layer fractions, layer thickness and the full per-level factor ---------------
fracs = [Dict("pid" => pid, "ktop" => kt, "lev" => lv, "value" => E.nei_layer_fraction(pid, kt, lv))
    for pid in 1:length(E.NEI2016_PROFILE_NAMES), kt in (3, 10, 11, 15, 72), lv in (0.5, 1.0, 1.5, 2.0, 2.99, 3.0, 5.2, 9.0, 10.0, 10.5, 11.0, 11.9, 12.0, 15.0, 30.0)]
pts = [(-97.5, 40.0), (-120.0, 35.0), (-75.0, 41.0), (-90.0, 30.0), (-125.0, 25.0), (-60.0, 52.0)]
delp = [Dict("lon_deg" => lo, "lat_deg" => la, "lev" => lv, "value" => E.delp_dry_itp(deg2rad(lo), deg2rad(la), lv))
    for (lo, la) in pts, lv in (1.0, 1.7, 2.0, 3.0, 5.5, 8.0, 11.0, 11.9)]

# ---- expected emission rates for pinned-flux tests --------------------------------
# Each data-fed flux gets a distinct value, by the same rule gen_nei_esm.py writes into
# the tests; the expected rate combines upstream's own per-level factors exactly as its
# builder does: sum(frac_g * flux_g) / Δz * scale * clock * nei_inv_delp / g0_100, with
# the monthly totals divided by the days in the month as loadslice! does.
const SECTORS = E.NEI2016_ELEVATED_SECTORS
elev_species = sort(unique(vcat(values(species)...)))
surf_species = sort([String(k)[6:end] for (k, v) in pairs(esm("components/earthsci_data/nei2016_monthly.esm").models.NEI2016MonthlyEmis.variables)
    if startswith(String(k), "flux_")])
flux_elev(sp, s) = 1.0e-9 * (1 + 0.1 * findfirst(==(s), SECTORS) + 0.001 * findfirst(==(sp), elev_species))
flux_surf(sp) = 1.0e-9 * (1 + 0.001 * findfirst(==(sp), surf_species))
const CLOCK = Dict("CO" => E.nei_clock_CO, "FORM" => E.nei_clock_FORM, "ISOP" => E.nei_clock_ISOP,
    "NO" => E.nei_clock_NOx, "NO2" => E.nei_clock_NOx)
g0_100 = 100.0 / 9.80665
function rate(groups, sp, t_abs, lon, lat, lev, ktop)
    # groups: [(profile id, total flux of the species in that group)]
    days = Dates.daysinmonth(Dates.unix2datetime(t_abs))
    kemit = maximum(min(ktop, length(E.NEI2016_LAYER_FRACTIONS[pid])) for (pid, _) in groups)
    gate = E.nei_layer_gate(lev, kemit)
    clock = haskey(CLOCK, sp) ? CLOCK[sp](t_abs, lon, gate) : 1
    terms = sum(E.nei_layer_fraction(pid, ktop, lev) * (f / days) for (pid, f) in groups)
    return terms / 1.0 * 1.0 * clock * E.nei_inv_delp(lon, lat, lev, gate) / g0_100
end
pid(name) = findfirst(==(name), E.NEI2016_PROFILE_NAMES)
function elev_groups(sp)
    g = Dict{Int, Float64}()
    for s in SECTORS
        sp in species[s] || continue
        p = pid(get(E.NEI2016_SECTOR_PROFILES, s, "surface"))
        g[p] = get(g, p, 0.0) + flux_elev(sp, s)
    end
    return collect(g)
end
cases(spec) = [Dict("id" => id, "t_abs" => datetime2unix(dt), "lon" => deg2rad(lo), "lat" => deg2rad(la),
    "lev" => lv, "ktop" => kt) for (id, dt, lo, la, lv, kt) in spec]
elev_cases = cases([
    ("local_noon_layer1_halfhour_meridian", DateTime(2016, 5, 1, 18), -97.5, 40.0, 1.0, 72),
    ("local_noon_layer3", DateTime(2016, 5, 1, 18), -97.5, 40.0, 3.0, 72),
    ("local_noon_fractional_lev_7_6", DateTime(2016, 5, 1, 18), -97.5, 40.0, 7.6, 72),
    ("top_of_profiles_layer11", DateTime(2016, 5, 1, 18), -97.5, 40.0, 11.0, 72),
    ("above_profiles_layer12", DateTime(2016, 5, 1, 18), -97.5, 40.0, 12.0, 72),
    ("domain_top_folding_ktop3", DateTime(2016, 5, 1, 18), -97.5, 40.0, 3.0, 3),
    ("july_night_west_lev2_5", DateTime(2016, 7, 4, 2, 30), -120.3, 35.0, 2.5, 72),
    ("december_morning_east_layer1", DateTime(2016, 12, 15, 13), -75.0, 41.0, 1.0, 72),
])
surf_cases = cases([
    ("local_noon_halfhour_meridian", DateTime(2016, 5, 1, 18), -97.5, 40.0, 1.0, 72),
    ("above_surface_layer2", DateTime(2016, 5, 1, 18), -97.5, 40.0, 2.0, 72),
    ("december_morning_east", DateTime(2016, 12, 15, 13), -75.0, 41.0, 1.0, 72),
    ("july_night_west", DateTime(2016, 7, 4, 2, 30), -120.3, 35.0, 1.0, 72),
])
for c in elev_cases
    c["out"] = Dict(sp => rate(elev_groups(sp), sp, c["t_abs"], c["lon"], c["lat"], c["lev"], c["ktop"]) for sp in elev_species)
end
for c in surf_cases
    c["out"] = Dict(sp => rate([(1, flux_surf(sp))], sp, c["t_abs"], c["lon"], c["lat"], c["lev"], c["ktop"]) for sp in surf_species)
end

# ---- GEOS-FP hybrid pressure and its level derivative ------------------------------
geosfp = Dict("Ap" => E._Ap_data, "Bp" => E._Bp_data,
    "cases" => [Dict("lev" => lv, "PS" => ps, "P" => E.Ap(lv) + E.Bp(lv) * ps,
        "dPdlev" => E.Ap_deriv(lv) + E.Bp_deriv(lv) * ps)
        for ps in (101325.0, 85000.0), lv in (1.0, 1.25, 2.0, 2.5, 3.0, 36.7, 71.0, 72.0, 72.5, 73.0)])

open(joinpath(outdir, "nei_ref.json"), "w") do io
    JSON3.write(io, Dict("tables" => tables, "species" => species, "clocks" => clocks,
        "regfn" => regfn, "fracs" => fracs, "delp" => delp, "elev_species" => elev_species,
        "surf_species" => surf_species, "elev_cases" => elev_cases, "surf_cases" => surf_cases,
        "geosfp" => geosfp))
end
println("wrote ", joinpath(outdir, "nei_ref.json"), ": ", length(clocks), " clock samples, ",
    Dict(s => length(v) for (s, v) in species))
