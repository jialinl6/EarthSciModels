#!/usr/bin/env python3
"""
render_example_plots — evaluate `.esm` analyses and emit PNG plots.

Walks `components/**/*.esm`, and for each top-level component with analyses
carrying a `parameter_sweep` + `plots`, evaluates the sweep and writes one
PNG per plot under

    <esm_dir>/<esm_stem>.plots/<analysis_id>-<plot_id>.png

`analyses` is the esm 1.0.0 spelling (esm-spec §6.7) of the block the 0.x
corpus spelled `examples`; the module keeps its own name so the CI job, the
docs generator's artifact convention and this repo's docs keep resolving.

`tools/esm_to_docs.py` already understands this convention and will inline
the artifacts on the rendered Hugo page (mdl-f42).

Expression evaluation is delegated to the `earthsci_ast` Python binding
(ESS) — this file imports `load` from the package root and the scalar AST
evaluator `fold_constant_expr` from `earthsci_ast.numpy_interpreter`,
walking each sweep grid point through the ESS evaluator. That keeps op
semantics single-sourced with the rest of the toolchain (Rust ndarray
runtime, Julia SymbolicUtils path, conformance fixtures).

Variable classification is delegated the same way. esm 1.0.0 declares only
`unknown` and `parameter`; which unknowns are ODE states, which are observed
and which are algebraic is DERIVED from the equations by
`earthsci_ast.classification` (esm-spec §6.3.1), and this renderer asks that
module rather than re-deriving it — an observed quantity in particular no
longer carries an `expression` field, it is defined by a bare-variable-LHS
equation like any other row.

Renderable components:
- Algebraic models (no `D` op in equations) drive the cartesian-sweep path:
  WaterEquilibrium, CloudAlbedo, etc. evaluate at every grid point and
  produce one PNG per plot spec.
- ODE models (one or more `D(state)/dt = rhs` equations) drive the
  time-series path when the analysis carries `initial_state` (per_variable
  form). Each analysis integrates via the canonical Python runner
  (`earthsci_ast.esm_problem` + `solve`) and plots state/algebraic
  trajectories vs `t`. A 1-D `parameter_sweep` is allowed and produces a
  family of curves on one axes (one integration per grid point).
  DiameterGrowthRate's Fig. 13.2 analyses drive this path.

Components with `D` ops but no `initial_state` are skipped — there's
nothing to integrate.

Entry points:
    python3 tools/render_example_plots.py                   # from repo root
    python3 tools/render_example_plots.py --components-dir <path>
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Big reaction systems (e.g. geoschem_fullchem with 819 reactions × 272 species)
# build per-species rhs trees hundreds of `+` ops deep, which the recursive
# `fold_constant_expr` walks once per integration step. Default 1000 is too tight.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))

# Headless backend so this works in CI without a display.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from earthsci_ast import (  # noqa: E402
    EsmFile,
    ExprNode,
    Model,
    free_variables,
    from_sympy,
    load_path,
    substitute,
    to_sympy,
)
from earthsci_ast.classification import (  # noqa: E402
    algebraic_unknowns,
    ode_states,
    parameters as declared_parameters,
)
from earthsci_ast.esm_types import (  # noqa: E402
    Analysis,
    Equation,
    Metadata,
    ModelVariable,
    ParameterSweep,
    Plot,
    PlotAxis,
    PlotValue,
    SweepDimension,
)
from earthsci_ast.flatten import FlattenedSystem, flatten  # noqa: E402
from earthsci_ast.numpy_interpreter import (  # noqa: E402
    NumpyInterpreterError,
    fold_constant_expr,
)
from earthsci_ast import (  # noqa: E402
    ReturnCode,
    UnsupportedDimensionalityError,
    esm_problem,
    solve,
)


class UnsupportedExpression(Exception):
    """Raised when an analysis or expression cannot be rendered here.

    Wraps both renderer-side problems (missing variable bindings, unsupported
    plot types, sweep shapes) and ESS-side errors propagated from
    `earthsci_ast.numpy_interpreter.fold_constant_expr` (unbound symbols,
    unsupported ops including `call` and time-derivatives reached at
    evaluation time).
    """


# ---------------------------------------------------------------------------
# Sweep + grid evaluation (delegates op semantics to ESS)
# ---------------------------------------------------------------------------


def _has_time_derivative(node: Any) -> bool:
    """True if the expression tree contains a `D` (time-derivative) op anywhere."""
    if isinstance(node, ExprNode):
        if node.op == "D":
            return True
        return any(_has_time_derivative(a) for a in node.args)
    return False


def _component_has_dynamics(model: Model) -> bool:
    """A model has dynamics if any equation has a time derivative on either side.

    Scans both lhs and rhs because canonical ODE form puts `D(state, wrt=t)`
    on the lhs (`D(D_p) = I_D`), while expression-shaped equations may bury
    a `D` op inside the rhs tree.
    """
    for eq in model.equations or []:
        if _has_time_derivative(eq.lhs) or _has_time_derivative(eq.rhs):
            return True
    return False


def _baseline_bindings(model: Model, analysis: Analysis) -> dict[str, float]:
    """Initial bindings: parameter defaults, then the analysis's overrides.

    `parameter` is one of the two declared types in esm 1.0.0; the 0.x
    `constant` spelling was never in any version's enum, and a constant is now
    derived (a parameter with neither `distribution` nor `update`), so it is
    already covered here.
    """
    bindings: dict[str, float] = {}
    variables = model.variables
    for name in declared_parameters(model):
        mv = variables[name]
        if mv.default is not None:
            bindings[name] = float(mv.default)
    for name, val in (analysis.parameters or {}).items():
        bindings[name] = float(val)
    return bindings


def _build_sweep_grid(
    sweep: ParameterSweep,
) -> tuple[list[str], list[np.ndarray], list[str]]:
    """Return (names, values, scales) for a 1D or 2D cartesian sweep.

    `scales` parallels `names`: "linear" or "log" per dimension. Dimensions
    given as an explicit `values` list are treated as "linear" unless every
    value is positive and they span >1 decade, in which case "log" is a
    reasonable default for axis presentation.
    """
    if sweep.type != "cartesian":
        raise UnsupportedExpression(f"unsupported sweep type: {sweep.type!r}")
    if not sweep.dimensions:
        raise UnsupportedExpression("parameter_sweep has no dimensions")
    names: list[str] = []
    values: list[np.ndarray] = []
    scales: list[str] = []
    for d in sweep.dimensions:
        if not isinstance(d, SweepDimension):
            raise UnsupportedExpression("dimension is not a SweepDimension")
        if d.values is not None:
            arr = np.asarray(d.values, dtype=float)
            scale = "linear"
            if arr.size > 1 and np.all(arr > 0):
                if float(arr.max()) / float(arr.min()) >= 10.0:
                    scale = "log"
        elif d.range is not None:
            rng = d.range
            scale = rng.scale or "linear"
            if scale == "log":
                arr = np.logspace(math.log10(rng.start), math.log10(rng.stop), rng.count)
            else:
                arr = np.linspace(rng.start, rng.stop, rng.count)
        else:
            raise UnsupportedExpression(
                f"dimension {d.parameter!r} has neither values nor range"
            )
        names.append(d.parameter)
        values.append(arr)
        scales.append(scale)
    return names, values, scales


def _solve_for_unbound(lhs: Any, rhs: Any, target_name: str) -> Any | None:
    """Symbolically solve `lhs = rhs` for `target_name` (must appear in
    exactly one side as a free variable). Returns the ESS expression that
    computes `target_name`, or None if the solver couldn't find a closed
    form.
    """
    import sympy as sp

    lhs_sym = to_sympy(lhs) if not isinstance(lhs, str) else sp.Symbol(lhs)
    rhs_sym = to_sympy(rhs) if not isinstance(rhs, str) else sp.Symbol(rhs)
    try:
        sols = sp.solve(sp.Eq(lhs_sym, rhs_sym), sp.Symbol(target_name))
    except (NotImplementedError, ValueError, TypeError):
        return None
    if not sols:
        return None
    return from_sympy(sols[0])


def _build_resolution_plan(
    model: Model,
    base_bound_names: set[str],
    equations: list[Any] | None = None,
) -> list[tuple[str, Any]]:
    """Build a dependency-ordered list of `(target, expr)` pairs.

    Each step in the plan defines one variable from a closed-form expression
    over previously-bound symbols (parameters + sweep axes + earlier targets).

    Two sources contribute targets:
    - Equations where `lhs` is a bare name not yet bound: forward as
      `(lhs, rhs)`. From esm 1.0.0 this is also where OBSERVED quantities
      come from — an observed unknown is exactly an unknown some equation
      defines with a bare-variable LHS (esm-spec §6.3.1), and there is no
      longer a `variables[v].expression` field to read separately. Because
      they arrive through the same fixpoint loop as everything else, they are
      now ordered by their real dependencies rather than assumed forward.
    - Equations where `lhs` is bound (constraint): symbolically solve for
      the single unbound variable in `rhs` and add as a target.

    `equations` defaults to `model.equations`. Callers that want to scope
    the resolver to a subset (e.g. an algebraic-only slice) may pass their
    own list — `_extract_ode_equations` produces such a slice by stripping
    `D(state)/dt = rhs` rows.

    Targets that can't be resolved (cyclic deps, multi-unbound constraints,
    constraint with no closed form) raise `UnsupportedExpression`.
    """
    bound: set[str] = set(base_bound_names)
    plan: list[tuple[str, Any]] = []

    eq_source = equations if equations is not None else (model.equations or [])
    pending = list(eq_source)
    while pending:
        progress = False
        still_pending = []
        for eq in pending:
            if not isinstance(eq.lhs, str):
                still_pending.append(eq)
                continue
            rhs_vars = free_variables(eq.rhs)
            unbound_in_rhs = rhs_vars - bound
            if eq.lhs not in bound:
                if not unbound_in_rhs:
                    plan.append((eq.lhs, eq.rhs))
                    bound.add(eq.lhs)
                    progress = True
                else:
                    still_pending.append(eq)
            else:
                if len(unbound_in_rhs) == 1:
                    target = next(iter(unbound_in_rhs))
                    solved = _solve_for_unbound(eq.lhs, eq.rhs, target)
                    if solved is None:
                        raise UnsupportedExpression(
                            f"could not symbolically solve {eq.lhs}={eq.rhs} "
                            f"for {target!r}"
                        )
                    plan.append((target, solved))
                    bound.add(target)
                    progress = True
                else:
                    still_pending.append(eq)
        if not progress:
            missing = []
            for eq in still_pending:
                if isinstance(eq.lhs, str) and eq.lhs not in bound:
                    missing.append(eq.lhs)
            raise UnsupportedExpression(
                f"could not resolve equations (cyclic or under-determined): "
                f"{sorted(set(missing))!r}"
            )
        pending = still_pending

    return plan


def _evaluate_grid(
    model: Model,
    base_bindings: dict[str, float],
    sweep_names: list[str],
    sweep_values: list[np.ndarray],
    t: float | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate every observed/algebraic target across the cartesian sweep.

    Returns a dict mapping every known name (parameters, sweep axes,
    resolved targets) to a numpy array shaped like the sweep grid (or to
    a 0-d array for parameters that don't vary).

    The resolution plan is built once symbolically (handling forward
    definitions and constraint equations alike) and then evaluated per
    grid point through ESS's scalar `fold_constant_expr()`.

    `t`, when given, binds the independent variable, so an observed that
    reads it (Fast-JX's solar geometry on `t + t_ref`) resolves; the caller
    passes the analysis window's end, the instant a stateless sweep point
    is reported at.
    """
    grids = np.meshgrid(*sweep_values, indexing="ij")
    shape = grids[0].shape
    flat_grids = [g.ravel() for g in grids]
    n = int(np.prod(shape))

    bound_names = set(base_bindings.keys()) | set(sweep_names)
    if t is not None:
        bound_names.add("t")
    plan = _build_resolution_plan(model, bound_names)
    if t is not None:
        # `fold_constant_expr` evaluates in a context whose time is fixed at 0
        # and does not read `t` from its bindings, so fold the instant in.
        plan = [(name, substitute(expr, {"t": float(t)})) for name, expr in plan]
    target_names = [name for name, _ in plan]
    point_results: dict[str, np.ndarray] = {
        name: np.empty(n, dtype=float) for name in target_names
    }

    for i in range(n):
        env: dict[str, float] = dict(base_bindings)
        for name, fg in zip(sweep_names, flat_grids):
            env[name] = float(fg[i])
        for name, expr in plan:
            try:
                value = float(fold_constant_expr(expr, env))
            except (ValueError, TypeError, NumpyInterpreterError) as exc:
                raise UnsupportedExpression(
                    f"could not evaluate {name!r} at grid point {i}: {exc}"
                ) from exc
            except ZeroDivisionError:
                # Retained as a belt-and-braces guard; see the non-finite
                # normalisation below for why it no longer fires.
                value = math.nan
            # A sweep that crosses a denominator-is-zero point yields a
            # non-finite value rather than an exception: CONFORMANCE_SPEC §5.7
            # rule 6 makes division IEEE in every binding ("x/0 is ±Inf, 0/0 is
            # NaN, never an exception"), and ESS's `ifelse` is EAGER, so a
            # `ifelse(W > 0, A/W, 0)` guard in the .esm still evaluates the
            # division at W=0. Normalise every non-finite result to NaN so it
            # propagates through the rest of the plan and matplotlib draws a
            # gap at that grid point — better than losing the whole analysis to
            # one degenerate point (the snow_melt_no_layers.esm failure mode,
            # W_sno_np1 / W_sno_n at the W_sno_n=0 endpoint).
            env[name] = value if math.isfinite(value) else math.nan
        for name in target_names:
            point_results[name][i] = env[name]

    out: dict[str, np.ndarray] = {}
    for name, val in base_bindings.items():
        out[name] = np.asarray(val)
    for name, grid in zip(sweep_names, grids):
        out[name] = grid
    for name, arr in point_results.items():
        out[name] = arr.reshape(shape)
    return out


# ---------------------------------------------------------------------------
# ODE integration (time_span + initial_state path)
# ---------------------------------------------------------------------------


def _extract_ode_equations(model: Model) -> tuple[dict[str, Any], list[Any]]:
    """Split a model's equations into ODEs (`D(state)/dt = rhs`) and algebraics.

    Returns `(ode_map, alg_eqs)`:
    - `ode_map`: `{state_name: rhs_expr}` for every `D(state, wrt=t) = rhs` row.
    - `alg_eqs`: every other equation, preserving order, for the algebraic
      resolution plan to consume.

    Raises `UnsupportedExpression` for D-on-lhs forms we can't interpret
    (non-string state, missing args).
    """
    ode_map: dict[str, Any] = {}
    alg_eqs: list[Any] = []
    for eq in model.equations or []:
        lhs = eq.lhs
        if isinstance(lhs, ExprNode) and lhs.op == "D":
            if not lhs.args or not isinstance(lhs.args[0], str):
                raise UnsupportedExpression(
                    f"unsupported D() lhs (need bare state name): {lhs!r}"
                )
            ode_map[lhs.args[0]] = eq.rhs
        else:
            alg_eqs.append(eq)
    return ode_map, alg_eqs


def _wrap_model_as_esm(model: Any) -> EsmFile:
    """Wrap a Model or _AdapterModel into a minimal EsmFile for `esm_problem`.

    The renderer iterates per top-level component (one Model or one
    ReactionSystem-derived _AdapterModel at a time), but the canonical
    Python runner consumes an EsmFile (which it flattens internally). This
    helper materialises a single-component EsmFile around the renderer's
    in-hand model so the esm_problem call has the shape it expects, without
    making the renderer track the originating EsmFile through every layer.
    """
    if isinstance(model, _AdapterModel):
        real_variables = {
            name: ModelVariable(
                type=v.type,  # type: ignore[arg-type]
                default=v.default,
                units=v.units,
            )
            for name, v in model.variables.items()
        }
        wrapped: Model = Model(
            name=model.name,
            variables=real_variables,
            equations=list(model.equations),
            metadata={},
            subsystems={},
            tests=[],
            analyses=[],
            initialization_equations=[],
            guesses={},
        )
    else:
        wrapped = model
    title = getattr(model, "name", None) or "Renderer"
    return EsmFile(
        version="1.0.0",
        metadata=Metadata(
            title=title,
            authors=[],
            references=[],
            keywords=[],
            custom={},
        ),
        models={wrapped.name: wrapped},
        reaction_systems={},
        events=[],
        # esm 1.0.0: `data_sources` is a document-scoped ingest registry, not a
        # component list. The synthetic wrapper consumes no external data.
        data_sources={},
        operators=[],
        registered_functions={},
        coupling=[],
        enums={},
        function_tables={},
    )


def _solve_time_series(
    model: Any,
    base_bindings: dict[str, float],
    initial_state_values: dict[str, float],
    time_span: Any,
    n_points: int = 200,
    flat: FlattenedSystem | None = None,
) -> dict[str, np.ndarray]:
    """Integrate the model's ODE system over `time_span` and return trajectories.

    Routes through `earthsci_ast.esm_problem` + `solve` — the canonical
    Python ESS runner — per CLAUDE.md "Simulation Pathway — ABSOLUTE Rule".
    The runner handles ODE compilation (sympy lambdify with shared CSE),
    integration, dense output, and algebraic-state trajectory recovery
    internally. A Solution's `vars` only surface state variables; observed
    variables (e.g. fastjx's `j_NO2 = Σ F_i·σ_i`) are recovered here by
    walking the model's resolution plan against each saved time sample,
    using `earthsci_ast.numpy_interpreter.fold_constant_expr` — the
    canonical Python scalar AST evaluator — at every point. That's a
    post-integration consumer of the integrated state, not a parallel ODE
    pipeline.

    `flat` is an optional pre-built :class:`FlattenedSystem`. Passing it
    lets sweep loops reuse the `_simulate_compile_cache` (attached
    to the flat instance) across every sweep point, which dominates wall
    time on large mechanisms (geoschem_fullchem: ~30 s lambdify per
    integration). When `flat` is None — the test path — this function
    wraps `model` into a fresh single-component EsmFile and lets the build
    flatten internally.

    Returns a dict mapping every name (parameters as 0-d arrays, ODE
    states, algebraic-recovered states, observed-variable trajectories,
    and `t`) to a 1-D numpy array of length `n_points` sampled uniformly
    across `[time_span.start, time_span.end]`. solve produces a dense
    ~10k-point grid via `dense_output=True`; we resample to `n_points` so
    plotters / fixtures can keep their current grid expectations.
    """
    sim_input: Any = flat if flat is not None else _wrap_model_as_esm(model)
    t0, t1 = float(time_span.start), float(time_span.end)
    # EarthSciAST phase 4: `simulate` is gone. Build once with `esm_problem`,
    # then `solve`. The tolerances stay explicit and tight -- the library
    # defaults are now 1e-4/1e-6, which would visibly coarsen a plotted curve.
    try:
        problem = esm_problem(
            sim_input,
            (t0, t1),
            p=dict(base_bindings),
            u0=dict(initial_state_values),
        )
    except UnsupportedDimensionalityError as exc:
        # A document whose flattened system carries a SPATIAL independent
        # variable is a PDE: `esm_problem` builds time-only systems, and this
        # renderer has no spatial plot form to put the solution in. Report it
        # the way every other un-renderable analysis is reported -- a skip
        # line -- instead of ending the whole docs build on the first PDE
        # component in the corpus.
        raise UnsupportedExpression(f"PDE component (not a time-only system): {exc}") from exc
    result = solve(problem, alg="LSODA", reltol=1e-8, abstol=1e-12)
    if result.retcode is not ReturnCode.Success:
        raise UnsupportedExpression(
            f"ODE integration failed ({result.retcode.name}): {result.message}"
        )

    t_dense = np.asarray(result.t, dtype=float)
    t_out = np.linspace(t0, t1, n_points)
    out: dict[str, np.ndarray] = {"t": t_out}
    for i, name in enumerate(result.vars):
        bare = name.rsplit(".", 1)[-1]
        y_dense = np.asarray(result.y[i], dtype=float)
        out[bare] = np.interp(t_out, t_dense, y_dense)
        out.setdefault(name, out[bare])

    for name, val in base_bindings.items():
        if name not in out:
            out[name] = np.asarray(val)

    # Observed-variable recovery. solve returns only state-variable
    # trajectories, but the renderer's plot specs may name observed vars
    # (e.g. fastjx's `j_NO2`). Build a resolution plan over the algebraic
    # subset of model equations and run the canonical AST evaluator at
    # each saved time sample.
    _, alg_eqs = _extract_ode_equations(model)
    bound = set(out.keys()) | set(base_bindings.keys()) | {"t"}
    try:
        plan = _build_resolution_plan(model, bound, equations=alg_eqs)
    except UnsupportedExpression:
        plan = []
    if plan:
        n = len(t_out)
        plan_arrs = {name: np.empty(n, dtype=float) for name, _ in plan}
        for k in range(n):
            env: dict[str, Any] = dict(base_bindings)
            env["t"] = float(t_out[k])
            for name in out:
                v = out[name]
                if isinstance(v, np.ndarray) and v.ndim == 1 and v.size == n:
                    env[name] = float(v[k])
                else:
                    env[name] = float(np.asarray(v))
            for name, expr in plan:
                try:
                    env[name] = float(fold_constant_expr(expr, env))
                except (ValueError, TypeError, KeyError, ZeroDivisionError, NumpyInterpreterError):
                    env[name] = float("nan")
                plan_arrs[name][k] = env[name]
        for name, arr in plan_arrs.items():
            out.setdefault(name, arr)

    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _axis_scale(
    var_name: str | None,
    sweep_names: list[str],
    sweep_scales: list[str],
) -> str:
    """Determine axis scale from the sweep dimension scale for `var_name`,
    or "linear" if the variable isn't a swept parameter.

    (`PlotAxis` itself doesn't carry a `scale` field in the current ESS
    schema; if/when it adds one this fn should consult it first.)
    """
    if var_name and var_name in sweep_names:
        return sweep_scales[sweep_names.index(var_name)]
    return "linear"


def _axis_label(axis: PlotAxis | PlotValue | None, model: Model) -> str:
    """Compose an axis label. Honors explicit `axis.label`; if absent, falls
    back to the variable name plus its `units` if the model declares them."""
    if axis is None:
        return ""
    var_name = axis.variable
    label = getattr(axis, "label", None)
    if label:
        return label
    units = None
    if var_name and var_name in model.variables:
        units = model.variables[var_name].units
    if units:
        return f"{var_name} [{units}]"
    return var_name or ""


def _require_var(env: dict[str, np.ndarray], name: str | None, role: str) -> np.ndarray:
    if name is None or name not in env:
        raise UnsupportedExpression(
            f"{role} references unbound variable: {name!r}"
        )
    return np.asarray(env[name])


def _render_line_plot(
    plot: Plot,
    sweep_names: list[str],
    sweep_values: list[np.ndarray],
    sweep_scales: list[str],
    env: dict[str, Any],
    model: Model,
    out_path: Path,
) -> None:
    if len(sweep_values) != 1:
        raise UnsupportedExpression(
            f"line plot needs 1D sweep, got {len(sweep_values)}D"
        )
    x_vals = _require_var(env, plot.x.variable, "line plot x")
    y_vals = _require_var(env, plot.y.variable, "line plot y")
    if y_vals.ndim == 0:
        y_vals = np.full_like(x_vals, float(y_vals))

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(x_vals, y_vals, color="#1f4d8a", linewidth=2)
    ax.set_xlabel(_axis_label(plot.x, model))
    ax.set_ylabel(_axis_label(plot.y, model))
    if _axis_scale(plot.x.variable, sweep_names, sweep_scales) == "log":
        ax.set_xscale("log")
    if _axis_scale(plot.y.variable, sweep_names, sweep_scales) == "log":
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    title = plot.description or plot.id or ""
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _render_scatter_plot(
    plot: Plot,
    sweep_names: list[str],
    sweep_values: list[np.ndarray],
    sweep_scales: list[str],
    env: dict[str, Any],
    model: Model,
    out_path: Path,
) -> None:
    if len(sweep_values) != 1:
        raise UnsupportedExpression(
            f"scatter plot needs 1D sweep, got {len(sweep_values)}D"
        )
    x_vals = _require_var(env, plot.x.variable, "scatter plot x")
    y_vals = _require_var(env, plot.y.variable, "scatter plot y")
    if y_vals.ndim == 0:
        y_vals = np.full_like(x_vals, float(y_vals))

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.scatter(x_vals, y_vals, color="#1f4d8a", s=24, alpha=0.85)
    ax.set_xlabel(_axis_label(plot.x, model))
    ax.set_ylabel(_axis_label(plot.y, model))
    if _axis_scale(plot.x.variable, sweep_names, sweep_scales) == "log":
        ax.set_xscale("log")
    if _axis_scale(plot.y.variable, sweep_names, sweep_scales) == "log":
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    title = plot.description or plot.id or ""
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _render_heatmap_plot(
    plot: Plot,
    sweep_names: list[str],
    sweep_values: list[np.ndarray],
    sweep_scales: list[str],
    env: dict[str, Any],
    model: Model,
    out_path: Path,
) -> None:
    if len(sweep_values) != 2:
        raise UnsupportedExpression(
            f"heatmap plot needs 2D sweep, got {len(sweep_values)}D"
        )
    if plot.value is None:
        raise UnsupportedExpression("heatmap plot is missing 'value' axis")
    x_name = plot.x.variable
    y_name = plot.y.variable
    v_name = plot.value.variable
    if x_name not in env or y_name not in env or v_name not in env:
        raise UnsupportedExpression(
            f"heatmap references unbound: x={x_name} y={y_name} value={v_name}"
        )
    x_vals = sweep_values[sweep_names.index(x_name)]
    y_vals = sweep_values[sweep_names.index(y_name)]
    z = np.asarray(env[v_name])
    # Orient: rows = y, cols = x. _evaluate_grid uses meshgrid(indexing="ij")
    # so the result is shaped (len(sweep[0]), len(sweep[1])); transpose when
    # the first axis isn't the y axis.
    if sweep_names.index(x_name) == 0:
        z_yx = z.T
    else:
        z_yx = z

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    extent = (float(x_vals[0]), float(x_vals[-1]), float(y_vals[0]), float(y_vals[-1]))
    im = ax.imshow(
        z_yx,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
    )
    ax.set_xlabel(_axis_label(plot.x, model))
    ax.set_ylabel(_axis_label(plot.y, model))
    if _axis_scale(x_name, sweep_names, sweep_scales) == "log":
        ax.set_xscale("log")
    if _axis_scale(y_name, sweep_names, sweep_scales) == "log":
        ax.set_yscale("log")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(_axis_label(plot.value, model))
    title = plot.description or plot.id or ""
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


_PLOT_RENDERERS = {
    "line": _render_line_plot,
    "scatter": _render_scatter_plot,
    "heatmap": _render_heatmap_plot,
}


def _render_time_series_line_plot(
    plot: Plot,
    trajectories: list[tuple[str | None, dict[str, np.ndarray]]],
    model: Model,
    out_path: Path,
) -> None:
    """Render `y vs t` for one or more ODE trajectories on a single axes.

    `trajectories` is a list of `(label, env)` pairs — one entry for the
    no-sweep case, N entries for a parameter sweep where each grid point
    drives a separate integration. `label` annotates the line in the legend
    (e.g. `"k=0.5"`); `None` suppresses the legend entry.
    """
    if plot.type != "line":
        raise UnsupportedExpression(
            f"time-series plot type {plot.type!r} not supported "
            f"(use 'line' with x.variable='t')"
        )
    if plot.x.variable != "t":
        raise UnsupportedExpression(
            f"time-series line plot requires x.variable='t', got "
            f"{plot.x.variable!r}"
        )
    y_name = plot.y.variable
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    cmap = plt.get_cmap("viridis")
    n = max(1, len(trajectories))
    for i, (label, env) in enumerate(trajectories):
        if y_name not in env:
            raise UnsupportedExpression(
                f"time-series line plot y references unbound variable: {y_name!r}"
            )
        t_arr = np.asarray(env["t"])
        y_arr = np.asarray(env[y_name])
        if y_arr.ndim == 0:
            y_arr = np.full_like(t_arr, float(y_arr))
        color = cmap(i / max(1, n - 1)) if n > 1 else "#1f4d8a"
        ax.plot(t_arr, y_arr, color=color, linewidth=2, label=label)
    ax.set_xlabel(_axis_label(plot.x, model))
    ax.set_ylabel(_axis_label(plot.y, model))
    ax.grid(True, which="both", alpha=0.3)
    if any(label is not None for label, _ in trajectories) and len(trajectories) > 1:
        ax.legend(fontsize=8, loc="best")
    title = plot.description or plot.id or ""
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _initial_state_values(
    analysis: Analysis,
) -> tuple[dict[str, float] | None, list[str]]:
    """Extract IC values from ``analysis.initial_state``.

    Returns ``(scalar_values, expression_vars)`` where:

    - ``scalar_values``: ``{name: float}`` dict for ``per_variable`` ICs, or
      None when no scalar IC is available.
    - ``expression_vars``: names of variables whose ICs are expression-typed
      (PDE spatial fields the renderer cannot collapse to a scalar).

    The schema admits two spellings (esm-spec §11.4): a flat scalar-override
    map ``{name: number}``, and the discriminated union ``{type:
    "per_variable"|"expression", values: {...}}``. Both are live in this
    corpus. The 1.0.0 binding no longer wraps either in a dataclass — the
    former ``InitialCondition`` / ``InitialConditionType`` types were removed
    along with domain-level initial conditions, and `earthsci_ast.parse` hands
    back the JSON object verbatim — so we discriminate on the object here.

    We never rely on the ``type`` tag alone: whichever spelling is used, a
    value that is not a number is an expression-typed field the renderer
    cannot collapse to a scalar, and it is reported as such.
    """
    ic = analysis.initial_state
    if not ic:
        return None, []
    values = ic["values"] if set(ic) == {"type", "values"} else ic
    if not isinstance(values, dict) or not values:
        return None, []
    expr_vars = [name for name, v in values.items() if not isinstance(v, (int, float))]
    if expr_vars:
        return None, expr_vars
    return {name: float(v) for name, v in values.items()} or None, []


def _state_defaults(model: Any) -> dict[str, float]:
    """Initial-condition fallback: every solved-for unknown that declares a default.

    Used when an analysis has no `initial_state` but the model carries enough
    species/state defaults to seed an integration.

    "Solved for" is the ODE states plus the algebraic unknowns — exactly the
    complement of the observed unknowns within `unknowns`, by the esm-spec
    §6.3.1 partition, and exactly what 0.x spelled `type: "state"`. Observed
    unknowns are deliberately excluded: their values follow from their defining
    equations, so seeding them would over-determine the system. Parameters are
    picked up by `_baseline_bindings` instead.
    """
    seedable = set(ode_states(model)) | set(algebraic_unknowns(model))
    out: dict[str, float] = {}
    for name, mv in model.variables.items():
        if name in seedable and mv.default is not None:
            out[name] = float(mv.default)
    return out


def _render_expression_ic_placeholder(
    esm_path: Path,
    component_name: str,
    model: Any,
    analysis: Analysis,
    expression_vars: list[str],
    plots_dir: Path,
    stats: _RenderStats,
) -> None:
    """Emit a placeholder PNG for each plot in an analysis with expression-typed ICs.

    Per esm-spec §11.4, a renderer that cannot evaluate spatial-field ICs MUST
    NOT crash and MUST visibly distinguish the output from a scalar-IC plot.
    Each placeholder PNG contains a text annotation naming the expression-typed
    variables instead of integration output that would require a PDE solver.
    """
    stats.analyses_seen += 1
    analysis_id = analysis.id or "analysis"
    var_label = ", ".join(expression_vars)
    for plot in (analysis.plots or []):
        plot_id = plot.id or "plot"
        out_path = plots_dir / f"{analysis_id}-{plot_id}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.set_axis_off()
        ax.text(
            0.5, 0.5,
            f"⟨expression IC: {var_label}⟩\n"
            "(rendered with state defaults)",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=11, color="#555555",
        )
        title = plot.description or plot.id or ""
        if title:
            ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        stats.plots_rendered += 1
        rel = (
            out_path.relative_to(esm_path.parent.parent.parent.parent)
            if str(out_path).startswith(str(esm_path.parent.parent.parent.parent))
            else out_path
        )
        print(f"[ok]   {rel} (expression IC: {var_label})")


def _render_final_state_vs_sweep_plot(
    plot: Plot,
    sweep_name: str,
    sweep_values: np.ndarray,
    sweep_scale: str,
    trajectories: list[tuple[str | None, dict[str, np.ndarray]]],
    model: Any,
    out_path: Path,
) -> None:
    """Plot the final-time value of `plot.y.variable` against the swept parameter.

    Used for analyses that combine `time_span` with a 1D `parameter_sweep`
    and whose plot's x-axis is the swept parameter (not `t`) — e.g. an
    "O3 vs jNO2" steady-state-style sweep where each grid point drives an
    independent integration and only the endpoint matters.
    """
    if plot.type != "line":
        raise UnsupportedExpression(
            f"final-state-vs-sweep plot type {plot.type!r} not supported "
            f"(use 'line')"
        )
    y_name = plot.y.variable
    if y_name is None:
        raise UnsupportedExpression("plot y has no variable")
    if len(trajectories) != len(sweep_values):
        raise UnsupportedExpression(
            f"final-state plot expected {len(sweep_values)} trajectories, "
            f"got {len(trajectories)}"
        )
    y_finals: list[float] = []
    for _, env in trajectories:
        if y_name not in env:
            raise UnsupportedExpression(
                f"final-state plot y references unbound variable: {y_name!r}"
            )
        y_arr = np.asarray(env[y_name])
        if y_arr.ndim == 0:
            y_finals.append(float(y_arr))
        else:
            y_finals.append(float(y_arr[-1]))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(sweep_values, y_finals, color="#1f4d8a", linewidth=2, marker="o")
    ax.set_xlabel(_axis_label(plot.x, model))
    ax.set_ylabel(_axis_label(plot.y, model))
    if sweep_scale == "log":
        ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    title = plot.description or plot.id or ""
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reaction-system adapter (treat ReactionSystem as a Model for rendering)
# ---------------------------------------------------------------------------


@dataclass
class _AdapterVariable:
    """Lightweight stand-in for ModelVariable when rendering reaction_systems.

    The rendering helpers query a small subset of ModelVariable's surface:
    `.type`, `.default`, `.units`. This dataclass mirrors that surface so the
    existing baseline-bindings / axis-labeling / resolution-plan code paths run
    unchanged on a ReactionSystem. There is no `.expression`: esm 1.0.0 removed
    that field, and `earthsci_ast.classification` reads `.type` off this
    dataclass exactly as it does off a real ModelVariable.
    """
    type: str
    default: Any | None = None
    units: str | None = None


@dataclass
class _AdapterModel:
    """Model-shaped adapter wrapping a ReactionSystem.

    Synthesizes one `D(species, wrt=t) = rhs` equation per non-constant
    species using mass-action kinetics: rate_r = k_r * prod_{(s,n) in
    reactants_r} s^n, with each species's net contribution from reaction r
    being `(products_r[s] - reactants_r[s]) * rate_r`.
    """
    name: str
    variables: dict[str, _AdapterVariable]
    equations: list[Any]
    analyses: list[Any]


def _build_rate_expression(rate_constant: Any, reactants: dict[str, float]) -> Any:
    """Build `k * prod_{(s,n) in reactants} s^n` as a left-folded ExprNode tree.

    Reactant stoichiometries are typically 1 (omit the `^1` factor) and
    occasionally 2 (e.g. HO2+HO2 → ... uses `HO2^2`). Fractional reactant
    stoichiometry is rare in real mechanisms but handled for completeness.

    When `rate_constant` references a reactant species (e.g. GEOS-Chem's
    aqueous channels with rate `k/SO2` × mass-action `SO2·HMS·OH` to express
    a zero-order-in-SO2 channel), the species cancels symbolically. The
    raw left-folded form would compute `k/SO2` first and divide by zero
    whenever `[SO2] = 0` along the trajectory; route through sympy so the
    cancellation happens at build time.
    """
    factors: list[Any] = [rate_constant]
    for sp, n in reactants.items():
        if n == 1:
            factors.append(sp)
        elif float(n).is_integer():
            factors.append(ExprNode(op="^", args=[sp, int(n)]))
        else:
            factors.append(ExprNode(op="^", args=[sp, float(n)]))
    if len(factors) == 1:
        return factors[0]
    expr = factors[0]
    for f in factors[1:]:
        expr = ExprNode(op="*", args=[expr, f])

    rate_free = free_variables(rate_constant) if isinstance(rate_constant, ExprNode) else set()
    if rate_free & set(reactants.keys()):
        try:
            expr = from_sympy(to_sympy(expr))
        except (NotImplementedError, ValueError, TypeError):
            pass
    return expr


def _build_species_ode_rhs(species_name: str, reactions: list[Any]) -> Any | None:
    """Sum net-stoichiometry-weighted rate terms over every reaction touching the species.

    Returns None if the species appears in no reaction (its derivative is
    identically zero — the caller skips emitting an equation rather than
    forcing a trivial constraint into the resolution plan).
    """
    terms: list[Any] = []
    for r in reactions:
        net = float(r.products.get(species_name, 0.0)) - float(
            r.reactants.get(species_name, 0.0)
        )
        if net == 0.0:
            continue
        rate_expr = _build_rate_expression(r.rate_constant, r.reactants)
        if net == 1.0:
            terms.append(rate_expr)
        elif net == -1.0:
            terms.append(ExprNode(op="*", args=[-1.0, rate_expr]))
        else:
            terms.append(ExprNode(op="*", args=[net, rate_expr]))
    if not terms:
        return None
    expr = terms[0]
    for t in terms[1:]:
        expr = ExprNode(op="+", args=[expr, t])
    return expr


def _reaction_system_to_model(rs: Any) -> _AdapterModel:
    """Convert a ReactionSystem into a renderable Model adapter.

    Mapping:
    - Parameters and constant species (`constant=True`) become `parameter`
      variables seeded with their declared defaults.
    - Non-constant species become `unknown` variables. The synthesized
      `D(species, wrt=t)` equation below is what makes each of them an ODE
      STATE under esm-spec §6.3.1 — the classification is derived from the
      equations here exactly as it is for a real model. Their declared defaults
      are surfaced as initial-state fallback when an analysis omits
      `initial_state` (a common shape for reaction_system analyses).
    - One `D(species, wrt=t) = rhs` equation is synthesized per non-constant
      species, with rhs summing net-stoichiometry-weighted rate terms over
      the reactions list.
    """
    variables: dict[str, _AdapterVariable] = {}
    for p in rs.parameters or []:
        variables[p.name] = _AdapterVariable(
            type="parameter",
            default=p.value,
            units=getattr(p, "units", None),
        )
    for sp in rs.species or []:
        if sp.constant:
            variables[sp.name] = _AdapterVariable(
                type="parameter",
                default=sp.default,
                units=sp.units,
            )
        else:
            variables[sp.name] = _AdapterVariable(
                type="unknown",
                default=sp.default,
                units=sp.units,
            )
    reactions = rs.reactions or []
    equations: list[Any] = []
    for sp in rs.species or []:
        if sp.constant:
            continue
        rhs = _build_species_ode_rhs(sp.name, reactions)
        if rhs is None:
            continue
        equations.append(
            Equation(
                lhs=ExprNode(op="D", args=[sp.name], wrt="t"),
                rhs=rhs,
            )
        )
    return _AdapterModel(
        name=rs.name,
        variables=variables,
        equations=equations,
        analyses=list(rs.analyses or []),
    )


# ---------------------------------------------------------------------------
# Per-file driver
# ---------------------------------------------------------------------------


@dataclass
class _RenderStats:
    files_seen: int = 0
    analyses_seen: int = 0
    plots_rendered: int = 0
    plots_skipped: int = 0
    elapsed_s: float = 0.0


def render_analyses_for_file(esm_path: Path, stats: _RenderStats) -> None:
    try:
        esm: EsmFile = load_path(str(esm_path))
    except Exception as exc:
        # Stay liberal on parse errors — surface and move on.
        print(f"[skip] {esm_path.name}: ESS load failed: {exc}", file=sys.stderr)
        return
    plots_dir = esm_path.parent / (esm_path.stem + ".plots")

    # Build the FlattenedSystem once per top-level component and pass it to
    # every analysis on that component. The build caches its compiled rhs on
    # the FlattenedSystem (`_simulate_compile_cache`); reusing the same flat
    # across all analyses on geoschem_fullchem cuts ~30 s of sympy.lambdify
    # work per extra analysis after the first (3 analyses × ~30 s saved).
    for cname, model in (esm.models or {}).items():
        if not model.analyses:
            continue
        flat = flatten(_wrap_model_as_esm(model)) if _component_has_dynamics(model) else None
        for analysis in model.analyses:
            _render_one_analysis(
                esm_path, cname, model, analysis, plots_dir, stats, flat=flat
            )

    for cname, rs in (esm.reaction_systems or {}).items():
        if not getattr(rs, "analyses", None):
            continue
        adapter = _reaction_system_to_model(rs)
        flat = flatten(_wrap_model_as_esm(adapter)) if _component_has_dynamics(adapter) else None
        for analysis in adapter.analyses:
            _render_one_analysis(
                esm_path, cname, adapter, analysis, plots_dir, stats, flat=flat
            )


def _render_one_analysis(
    esm_path: Path,
    component_name: str,
    model: Model,
    analysis: Analysis,
    plots_dir: Path,
    stats: _RenderStats,
    flat: FlattenedSystem | None = None,
) -> None:
    plots = analysis.plots or []
    if not plots:
        return

    has_dynamics = _component_has_dynamics(model)
    explicit_ic, expression_vars = _initial_state_values(analysis)

    # Expression-typed ICs (PDE spatial fields) cannot be reduced to scalars
    # for ODE integration. Emit a placeholder plot per esm-spec §11.4 instead
    # of silently falling through to state defaults.
    if has_dynamics and expression_vars:
        _render_expression_ic_placeholder(
            esm_path, component_name, model, analysis, expression_vars,
            plots_dir, stats,
        )
        return

    initial_values = None
    if has_dynamics:
        # Reaction-system species typically declare reservoir defaults that
        # serve as plausible t=0 initial conditions; merge any explicit
        # `initial_state.values` over the defaults so an analysis only has to
        # name the species it actually cares about (the GEOS-Chem fullchem
        # mechanism has 272 states — listing them all per analysis is not
        # workable).
        defaults = _state_defaults(model)
        if defaults or explicit_ic:
            merged = dict(defaults or {})
            if explicit_ic:
                merged.update(explicit_ic)
            initial_values = merged

    # Route ODE/time-series analyses (`time_span` + `initial_state`) through
    # the integration path. Falls back to the algebraic skip below if the
    # component has dynamics but no initial state was supplied.
    if has_dynamics and initial_values is not None:
        _render_time_series_analysis(
            esm_path,
            component_name,
            model,
            analysis,
            plots_dir,
            stats,
            initial_values,
            flat=flat,
        )
        return

    if analysis.parameter_sweep is None:
        for _ in plots:
            stats.plots_skipped += 1
            print(
                f"[skip] {esm_path.name}::{component_name} analysis "
                f"{analysis.id!r}: no parameter_sweep and no initial_state "
                f"(nothing to evaluate)"
            )
        return

    if has_dynamics:
        for _ in plots:
            stats.plots_skipped += 1
            print(
                f"[skip] {esm_path.name}::{component_name} analysis "
                f"{analysis.id!r}: component has time-derivative dynamics "
                f"but analysis has no initial_state for ODE integration"
            )
        return

    stats.analyses_seen += 1
    base_bindings = _baseline_bindings(model, analysis)

    try:
        sweep_names, sweep_values, sweep_scales = _build_sweep_grid(
            analysis.parameter_sweep
        )
        span = getattr(analysis, "time_span", None)
        t_eval = float(span.end) if span is not None else None
        env = _evaluate_grid(model, base_bindings, sweep_names, sweep_values, t=t_eval)
    except UnsupportedExpression as exc:
        for _ in plots:
            stats.plots_skipped += 1
            print(
                f"[skip] {esm_path.name}::{component_name} analysis "
                f"{analysis.id!r}: {exc}"
            )
        return

    analysis_id = analysis.id or "analysis"
    for plot in plots:
        plot_id = plot.id or "plot"
        renderer = _PLOT_RENDERERS.get(plot.type)
        if renderer is None:
            stats.plots_skipped += 1
            print(
                f"[skip] {esm_path.name}::{component_name} analysis "
                f"{analysis_id} plot {plot_id}: unsupported plot type {plot.type!r}"
            )
            continue
        out_path = plots_dir / f"{analysis_id}-{plot_id}.png"
        try:
            renderer(plot, sweep_names, sweep_values, sweep_scales, env, model, out_path)
        except UnsupportedExpression as exc:
            stats.plots_skipped += 1
            print(
                f"[skip] {esm_path.name}::{component_name} analysis "
                f"{analysis_id} plot {plot_id}: {exc}"
            )
            continue
        stats.plots_rendered += 1
        rel = out_path.relative_to(esm_path.parent.parent.parent.parent) \
            if str(out_path).startswith(str(esm_path.parent.parent.parent.parent)) \
            else out_path
        print(f"[ok]   {rel}")


def _render_time_series_analysis(
    esm_path: Path,
    component_name: str,
    model: Model,
    analysis: Analysis,
    plots_dir: Path,
    stats: _RenderStats,
    initial_values: dict[str, float],
    flat: FlattenedSystem | None = None,
) -> None:
    """Render plots for a `time_span` + `initial_state` analysis via ODE integration.

    No sweep: integrate once, plot one curve per plot spec (y vs t).
    With sweep: each grid point overrides parameters in `base_bindings`,
    integrate per-point, draw the family of curves on a single axes labeled
    by the sweep value.

    `flat` is the per-component :class:`FlattenedSystem` materialised by the
    file driver (`render_analyses_for_file`). Passing it through lets every
    sweep-point integration share the `_simulate_compile_cache`,
    avoiding a ~30 s lambdify per sweep step on large mechanisms. When None
    (e.g. test entry points exercising `_render_time_series_analysis` in
    isolation), `_solve_time_series` falls back to wrapping the model into a
    fresh EsmFile per call.
    """
    plots = analysis.plots or []
    stats.analyses_seen += 1
    base_bindings = _baseline_bindings(model, analysis)

    sweep_name: str | None = None
    sweep_axis_values: np.ndarray | None = None
    sweep_axis_scale: str = "linear"
    sweep_runs: list[tuple[str | None, dict[str, float]]]
    if analysis.parameter_sweep is None:
        sweep_runs = [(None, base_bindings)]
    else:
        try:
            sweep_names, sweep_values, sweep_scales = _build_sweep_grid(
                analysis.parameter_sweep
            )
        except UnsupportedExpression as exc:
            for _ in plots:
                stats.plots_skipped += 1
                print(
                    f"[skip] {esm_path.name}::{component_name} analysis "
                    f"{analysis.id!r}: {exc}"
                )
            return
        if len(sweep_values) != 1:
            for _ in plots:
                stats.plots_skipped += 1
                print(
                    f"[skip] {esm_path.name}::{component_name} analysis "
                    f"{analysis.id!r}: time-series + sweep needs 1D sweep, "
                    f"got {len(sweep_values)}D"
                )
            return
        sweep_name = sweep_names[0]
        sweep_axis_values = sweep_values[0]
        sweep_axis_scale = sweep_scales[0]
        runs: list[tuple[str | None, dict[str, float]]] = []
        for v in sweep_axis_values:
            b = dict(base_bindings)
            b[sweep_name] = float(v)
            runs.append((f"{sweep_name}={float(v):.3g}", b))
        sweep_runs = runs

    trajectories: list[tuple[str | None, dict[str, np.ndarray]]] = []
    try:
        for label, bindings in sweep_runs:
            env = _solve_time_series(
                model, bindings, initial_values, analysis.time_span, flat=flat
            )
            trajectories.append((label, env))
    except UnsupportedExpression as exc:
        for _ in plots:
            stats.plots_skipped += 1
            print(
                f"[skip] {esm_path.name}::{component_name} analysis "
                f"{analysis.id!r}: {exc}"
            )
        return

    analysis_id = analysis.id or "analysis"
    for plot in plots:
        plot_id = plot.id or "plot"
        out_path = plots_dir / f"{analysis_id}-{plot_id}.png"
        try:
            # Dispatch on plot.x.variable: `t` ⇒ classic time-series; the
            # swept parameter ⇒ collapse each trajectory to its endpoint and
            # plot final-state vs sweep value (steady-state-style sweeps).
            if (
                sweep_name is not None
                and plot.x.variable == sweep_name
                and sweep_axis_values is not None
            ):
                _render_final_state_vs_sweep_plot(
                    plot,
                    sweep_name,
                    sweep_axis_values,
                    sweep_axis_scale,
                    trajectories,
                    model,
                    out_path,
                )
            else:
                _render_time_series_line_plot(plot, trajectories, model, out_path)
        except UnsupportedExpression as exc:
            stats.plots_skipped += 1
            print(
                f"[skip] {esm_path.name}::{component_name} analysis "
                f"{analysis_id} plot {plot_id}: {exc}"
            )
            continue
        stats.plots_rendered += 1
        rel = out_path.relative_to(esm_path.parent.parent.parent.parent) \
            if str(out_path).startswith(str(esm_path.parent.parent.parent.parent)) \
            else out_path
        print(f"[ok]   {rel}")


# ---------------------------------------------------------------------------
# Discovery + CLI
# ---------------------------------------------------------------------------


def discover_esm_files(components_root: Path) -> list[Path]:
    return sorted(p for p in components_root.rglob("*.esm") if p.is_file())


# Defensive memory budget for the doc-build path. esm_problem() compiles each
# mechanism via sympy.lambdify(..., cse=True), which has a known memory
# cliff on very large reaction systems. The CI doc-build runner has ~7 GB
# available; aborting here at 6 GB lets us emit a diagnostic instead of
# being SIGKILLed by the OOM killer mid-plot. Only enforced on Linux where
# /proc/self/status is available.
_RSS_HARD_ABORT_GB = 6.0


def _read_rss_gb() -> float | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0 / 1024.0
    except OSError:
        return None
    return None


def _check_rss_budget(label: str) -> None:
    rss = _read_rss_gb()
    if rss is None:
        return
    if rss > _RSS_HARD_ABORT_GB:
        print(
            f"\n[render_example_plots] HARD ABORT after {label}: RSS={rss:.2f} GB "
            f"exceeds {_RSS_HARD_ABORT_GB:.1f} GB budget. esm_problem() likely hit "
            f"its lambdify memory cliff on a large mechanism — file a bead to "
            f"extend esm_problem's API (e.g. cse=False knob) rather than re-adding "
            f"a homebrew solver path.",
            file=sys.stderr,
        )
        os._exit(99)


def run(components_root: Path) -> int:
    if not components_root.exists():
        print(
            f"error: components/ not found at {components_root}", file=sys.stderr
        )
        return 2
    files = discover_esm_files(components_root)
    if not files:
        print(f"warning: no .esm files under {components_root}", file=sys.stderr)
        return 0
    return run_files(files)


def run_files(files: list[Path]) -> int:
    """Render analyses for an explicit list of .esm files. Used by --files."""
    if not files:
        print("warning: no .esm files to render", file=sys.stderr)
        return 0
    stats = _RenderStats()
    t0 = time.time()
    for f in files:
        stats.files_seen += 1
        render_analyses_for_file(f, stats)
        _check_rss_budget(f.name)
    stats.elapsed_s = time.time() - t0
    print()
    print(
        f"render_example_plots: {stats.plots_rendered} plot(s) rendered, "
        f"{stats.plots_skipped} skipped, {stats.analyses_seen} analyses seen "
        f"across {stats.files_seen} .esm file(s) in {stats.elapsed_s:.2f}s"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render analysis plots from .esm files.")
    parser.add_argument(
        "--components-dir",
        type=Path,
        default=None,
        help="Components root (default: <repo-root>/components). "
             "Mutually exclusive with --files.",
    )
    parser.add_argument(
        "--files", nargs="+", default=None,
        help="Explicit list of .esm files to render analyses for (instead "
             "of walking --components-dir). Useful for pre-merge gates that "
             "only want to validate files changed in the diff. Mutually "
             "exclusive with --components-dir.",
    )
    args = parser.parse_args(argv)
    if args.files and args.components_dir:
        parser.error("--files and --components-dir are mutually exclusive")
    repo_root = Path(__file__).resolve().parent.parent
    if args.files:
        files: list[Path] = []
        for f in args.files:
            p = Path(f)
            if not p.is_absolute():
                p = repo_root / p
            if not p.exists():
                print(f"ERROR: --files: not found: {p}", file=sys.stderr)
                return 2
            if p.suffix != ".esm":
                print(f"ERROR: --files: not a .esm file: {p}", file=sys.stderr)
                return 2
            files.append(p.resolve())
        return run_files(sorted(set(files)))
    components_root = (args.components_dir or (repo_root / "components")).resolve()
    return run(components_root)


if __name__ == "__main__":
    raise SystemExit(main())
