# Extract Fast-JX data and reference values from the reference GasChem package
# (jialinl6/GasChemMTK.jl, branch superfast-gas-only, commit 37792b29) for
# gen_fastjx_esm.py, which writes components/gaschem/fastjx/{fastjx,fastjx_interp_troposphere}.esm.
#
# Needs an environment with that GasChem checkout plus JSON3 and BSON, e.g.
#   julia --project=ENV -e 'using Pkg; Pkg.develop(path="/path/to/GasChemMTK.jl"); Pkg.add(["JSON3", "BSON"])'
# Run:
#   julia --project=ENV scripts/_archive/migrations/extract_fastjx_data.jl OUTDIR
# Writes OUTDIR/fastjx_ref.json and OUTDIR/fastjx_direct.json.
using GasChem, Dates, JSON3, BSON

const G = GasChem
outdir = ARGS[1]

# ---- per-band sigma / phi closures -> their knots ----------------------------
# create_fjx_interp returns, per band, interp2/interp3 closures (fields T1.., x1..) or a
# constant closure (the x1 ≈ x2 Float32-isapprox shortcut, or `(T) -> σ[i]`).
function knots(f)
    fn = fieldnames(typeof(f))
    if :T3 in fn
        return Dict("axis" => Float64[f.T1, f.T2, f.T3], "vals" => Float64[f.x1, f.x2, f.x3])
    elseif :T2 in fn
        return Dict("axis" => Float64[f.T1, f.T2], "vals" => Float64[f.x1, f.x2])
    end
    v1 = f(200.0)
    @assert v1 == f(300.0)
    return Dict("const" => Float64(v1))
end
table(interps) = [knots(f) for f in interps]

jset = [(sp, String(sp) in String.(G._FJX_PRESSURE_AXIS_J)) for (sp, _) in G._FJX_INTERP_J_SUPERFAST]
const jfun = Dict(sp => f for (sp, f) in G._FJX_INTERP_J_SUPERFAST)
sigma = Dict(
    "H2O2" => table(G.σ_H2O2_interp), "H2COa" => table(G.σ_H2COa_interp),
    "H2COb" => table(G.σ_H2COb_interp), "O3" => table(G.σ_O3_interp),
    "CH3OOH" => table(G.σ_CH3OOH_interp), "NO2" => table(G.σ_NO2_interp),
    "ActAld" => table(G.σ_ActAld_interp), "PAN" => table(G.σ_PAN_interp),
    "NO3" => table(G.σ_NO3_interp), "N2O5" => table(G.σ_N2O5_interp),
)
phi_scalars = Dict(String(s) => Float64(getfield(G, Symbol("ϕ_", s, "_jx")))
    for s in (:H2O2, :H2COa, :H2COb, :O3, :CH3OOH, :NO2, :ActAld, :PAN, :NO3, :N2O5))
@assert all(==(1.0), values(phi_scalars))   # gen_fastjx_esm.py relies on this

# ---- flux tables ------------------------------------------------------------
src = joinpath(pkgdir(G), "src")
tro = BSON.load(joinpath(src, "tropospheric_interpolation_data.bson"))
dif = BSON.load(joinpath(src, "diffuse_flux_data.bson"))
@assert collect(tro[:tropospheric_P]) == collect(dif[:tropospheric_P])
@assert collect(tro[:cosSZA_vals]) == collect(dif[:cosSZA_vals])
rowmajor(Z) = [[Float64(Z[ip, ic]) for ic in 1:size(Z, 2)] for ip in 1:size(Z, 1)]
flux = Dict(
    "P_axis" => Float64.(collect(tro[:tropospheric_P])),
    "cosSZA_axis" => Float64.(collect(tro[:cosSZA_vals])),
    "total" => [rowmajor(tro[:Z_all][i]) for i in 1:18],
    "diffuse" => [rowmajor(dif[:Z_all][i]) for i in 1:18],
    "WL" => Float64.(collect(G.WL)),
)

# ---- reference values from upstream's own functions -------------------------
function adjust(T, P, H2O)   # adjust_j_o31D, scalar form
    nd = 6.02e23 * P / (8.314e6 * T)
    a = 1.63e-10 * exp(60.0 / T) * (H2O * 1.0e-9 * nd)
    b = 1.2e-10 * (0.5e-6 * nd)
    c = 2.15e-11 * exp(110.0 / T) * (0.7808 * nd)
    d = 3.3e-11 * exp(55.0 / T) * (0.2095 * nd)
    return a / (a + b + c + d)
end
function jvals(t_abs, lat, long, T, P, H2O; online = false)
    csa = G.cos_solar_zenith_angle(t_abs, lat, long)
    solf = G.solar_flux_factor(t_abs)
    F = if online   # FastJX (fluxes = :scattering): online direct beam + diffuse field
        [(G.calc_direct_flux(csa, P, i) + G.diffuse_funcs[i](P, csa)) * solf for i in 1:18]
    else            # FastJX_interpolation_troposphere
        [G.interpolations_18_const[i](P, csa) * solf for i in 1:18]
    end
    r = Dict{String, Float64}("cosSZA" => csa, "solf" => solf)
    for i in 1:18
        r["F_$(Int(round(G.WL[i])))"] = F[i]
    end
    for (sp, isP) in jset
        r["j_$(sp)"] = jfun[sp](isP ? P / 100 : T, F)
    end
    r["j_O31D_adj"] = adjust(T, P, H2O)
    r["j_o32OH"] = r["j_O31D"] * r["j_O31D_adj"]
    return r
end

tref_may = datetime2unix(DateTime(2016, 5, 1))
scen = [   # id, t_ref (Unix s), t (s), lat, long, T, P, H2O
    ("may_midlat_local_noon", tref_may, 64800.0, 40.0, -97.0, 298.0, 101325.0, 450.0),
    ("may_midlat_morning", tref_may, 50400.0, 40.0, -97.0, 290.0, 101325.0, 1.0e7),
    ("may_midlat_night", tref_may, 21600.0, 40.0, -97.0, 285.0, 101325.0, 1.0e7),
    ("jun_eq_noon_500hPa", datetime2unix(DateTime(2016, 6, 21)), 43200.0, 0.0, 0.0, 255.0, 50000.0, 2.0e6),
    ("dec_midlat_noon_cold", datetime2unix(DateTime(2016, 12, 21)), 64800.0, 40.0, -97.0, 273.0, 95000.0, 3.0e6),
    ("jan_twilight_upper", datetime2unix(DateTime(2016, 1, 15)), 50400.0, 60.0, 10.0, 220.0, 20000.0, 1.0e4),
    ("unix_epoch_default", 0.0, 0.0, 40.0, -97.0, 298.0, 101325.0, 450.0),
]
scenarios = [Dict("id" => id, "t_ref" => tr, "t" => tt, "lat" => la, "long" => lo, "T" => T, "P" => P, "H2O" => H,
    "interp" => jvals(tr + tt, la, lo, T, P, H), "online" => jvals(tr + tt, la, lo, T, P, H; online = true))
    for (id, tr, tt, la, lo, T, P, H) in scen]

diurnal = Dict{String, Any}("t_ref" => tref_may, "lat" => 40.0, "long" => -97.0, "T" => 298.0,
    "P" => 101325.0, "H2O" => 1.0e7, "t" => Float64[], "interp" => [], "online" => [])
for tt in 0.0:900.0:86400.0
    push!(diurnal["t"], tt)
    push!(diurnal["interp"], jvals(tref_may + tt, 40.0, -97.0, 298.0, 101325.0, 1.0e7))
    push!(diurnal["online"], jvals(tref_may + tt, 40.0, -97.0, 298.0, 101325.0, 1.0e7; online = true))
end

# How far upstream's own interpolated table sits from its online model (quoted in
# fastjx_interp_troposphere.esm's notes).
println("max |interp/online - 1| for the SuperFast J's:")
for (label, Ps) in (("surface, diurnal", nothing), ("950 hPa", [95000.0]), ("300 hPa", [30000.0]))
    worst = 0.0
    if Ps === nothing
        for (a, b) in zip(diurnal["interp"], diurnal["online"]), j in ("j_NO2", "j_H2O2", "j_H2COa", "j_H2COb", "j_CH3OOH", "j_O31D")
            b[j] > 0 && (worst = max(worst, abs(a[j] / b[j] - 1)))
        end
    else
        for P in Ps, csa in 0.05:0.05:1.0, (sp, isP) in jset
            sp in (:NO2, :H2O2, :H2COa, :H2COb, :CH3OOH, :O31D) || continue
            Fi = [G.interpolations_18_const[i](P, csa) for i in 1:18]
            Fo = [G.calc_direct_flux(csa, P, i) + G.diffuse_funcs[i](P, csa) for i in 1:18]
            worst = max(worst, abs(jfun[sp](298.0, Fi) / jfun[sp](298.0, Fo) - 1))
        end
    end
    println("  ", label, ": ", round(100 * worst, digits = 2), "%")
end

open(joinpath(outdir, "fastjx_ref.json"), "w") do io
    JSON3.write(io, Dict("jset" => [String(s) for (s, _) in jset],
        "pressure_axis" => [String(s) for (s, p) in jset if p],
        "sigma" => sigma, "phi" => Dict("O31D" => table(G.ϕ_O31D_interp), "scalars" => phi_scalars),
        "flux" => flux, "scenarios" => scenarios, "diurnal" => diurnal))
end

# ---- fixed-column constants behind the online direct beam (calc_direct_flux) ----
R2 = [G.calcRZ2(G.z_profile, i) for i in 1:147]   # fine-grid radii (cm)
D = [Float64(G.OD_total[div(i + 1, 2), k]) for i in 1:147, k in 1:18]
Ps = vcat([101325.0, 100000.0, 99000.0, 95000.0, 90000.0, 85000.0, 80000.0, 70000.0, 60000.0,
    50000.0, 40000.0, 30000.0, 25000.0, 20000.0, 15000.0, 10000.0, 5000.0, 2000.0, 1000.0,
    500.0, 100.0, 10.0, 1.5, 0.5], collect(G.P_levels[[1, 2, 10, 30, 50, 72, 73]]))
U0s = vcat(collect(-0.3:0.01:1.0), [-0.15, -0.105, -0.0501, -0.0012, 0.0, 1.0e-4, 0.0123, 0.987])
grid = [Dict("P" => P, "u0" => u0, "F" => [G.calc_direct_flux(u0, P, k) for k in 1:18]) for P in Ps for u0 in U0s]
open(joinpath(outdir, "fastjx_direct.json"), "w") do io
    JSON3.write(io, Dict("R2" => R2, "D" => [D[:, k] for k in 1:18],
        "P_levels" => Float64.(collect(G.P_levels)), "top_flux" => Float64.(collect(G.top_flux)),
        "grid" => grid))
end
println("wrote ", joinpath(outdir, "fastjx_ref.json"), " and fastjx_direct.json")
