# Compare the right-hand side dumped by verify_superfast_rhs.py (superfast.esm through the
# Python binding) with the reference SuperFast(; aerosol = false) from
# jialinl6/GasChemMTK.jl@37792b29, at the same states and parameters.
#
# Needs an environment with that GasChem checkout plus JSON3.
#   julia --project=ENV scripts/_archive/migrations/verify_superfast_rhs.jl OUT.json
using GasChem, ModelingToolkit, JSON3
using ModelingToolkit: variable_index

cases = JSON3.read(read(ARGS[1], String))
sys = mtkcompile(SuperFast(; aerosol = false))
worst = 0.0
for c in cases
    op = Dict{Any, Any}()
    for (k, v) in pairs(c.p)
        op[getproperty(sys, k)] = Float64(v)
    end
    for (k, v) in pairs(c.u)
        op[getproperty(sys, k)] = Float64(v)
    end
    prob = ODEProblem(sys, op, (0.0, 1.0))
    du = similar(prob.u0)
    prob.f(du, prob.u0, prob.p, 0.0)
    for (k, v) in pairs(c.du)
        i = variable_index(sys, getproperty(sys, k))
        global worst = max(worst, abs(du[i] - v) / max(abs(du[i]), 1.0e-300))
    end
end
println("max relative RHS difference over ", length(cases), " random points: ", worst)
