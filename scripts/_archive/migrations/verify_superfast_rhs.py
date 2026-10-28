#!/usr/bin/env python3
"""Evaluate components/gaschem/superfast.esm's ODE right-hand side at random points
through the Python binding, for verify_superfast_rhs.jl to compare against the
reference SuperFast(; aerosol = false) (jialinl6/GasChemMTK.jl@37792b29).

Run:
  python3 scripts/_archive/migrations/verify_superfast_rhs.py OUT.json
  julia --project=ENV scripts/_archive/migrations/verify_superfast_rhs.jl OUT.json
"""
import json
import sys

import numpy as np
from earthsci_ast.problem import esm_problem

rng = np.random.default_rng(1)
cases = []
for _ in range(8):
    p = {"T": float(rng.uniform(230, 310)), "P": float(rng.uniform(2e4, 1.05e5)),
         "H2O": float(10 ** rng.uniform(4, 7.5))}
    p.update({n: float(10 ** rng.uniform(-6, -2)) for n in ("jO32OH", "jH2O2", "jNO2", "jH2COa", "jH2COb", "jCH3OOH")})
    p["CH4"] = float(rng.uniform(1600, 2000))
    sb = esm_problem("components/gaschem/superfast.esm", (0.0, 1.0), p=p, compiler="sympy").scalar_build
    names = [n.split(".")[-1] for n in sb.state_names]
    y = np.array([float(10 ** rng.uniform(-6, 2)) for _ in names])
    dy = np.asarray(sb.rhs_function(0.0, y))
    cases.append({"p": p, "u": dict(zip(names, y.tolist())), "du": dict(zip(names, dy.tolist()))})
json.dump(cases, open(sys.argv[1], "w"))
print(f"wrote {len(cases)} cases over states {names}")
