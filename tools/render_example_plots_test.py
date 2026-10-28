"""
Unit tests for tools/render_example_plots.py.

Run locally with: python3 -m unittest tools/render_example_plots_test.py

Coverage of expression-evaluation op semantics intentionally lives in the
ESS conformance fixtures (esm-4aw) rather than here — the renderer only
covers binding orchestration, plot dispatch, and skip-behavior.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import render_example_plots as mod  # noqa: E402
from earthsci_ast import ExprNode, Model, ModelVariable  # noqa: E402
from earthsci_ast.classification import (  # noqa: E402
    observed_definitions,
    ode_states,
)
from earthsci_ast.esm_types import Equation  # noqa: E402


def _model(variables: dict, equations: list | None = None) -> Model:
    """Build a Model from the esm 1.0.0 two-type variable vocabulary.

    Only `unknown` and `parameter` may be declared. An unknown becomes an ODE
    state, an observed quantity or an algebraic one purely by what `equations`
    say about it (esm-spec §6.3.1) — there is no `expression` field to set.
    """
    return Model(
        name="Test",
        variables={
            name: ModelVariable(
                type=spec.get("type", "parameter"),
                default=spec.get("default"),
                units=spec.get("units"),
            )
            for name, spec in variables.items()
        },
        equations=[Equation(lhs=eq["lhs"], rhs=eq["rhs"]) for eq in (equations or [])],
    )


class HasTimeDerivativeTest(unittest.TestCase):
    def test_no_d_op(self):
        self.assertFalse(mod._has_time_derivative(3.5))
        self.assertFalse(mod._has_time_derivative("x"))
        self.assertFalse(mod._has_time_derivative(ExprNode(op="+", args=["x", 1])))

    def test_d_at_root(self):
        n = ExprNode(op="D", args=["y"], wrt="t")
        self.assertTrue(mod._has_time_derivative(n))

    def test_d_nested(self):
        inner = ExprNode(op="D", args=["y"], wrt="t")
        outer = ExprNode(op="*", args=[2, inner])
        self.assertTrue(mod._has_time_derivative(outer))


class ComponentHasDynamicsTest(unittest.TestCase):
    def test_observed_only_no_dynamics(self):
        # `y` is observed because a bare-variable-LHS equation defines it, not
        # because anything declares it so (esm-spec §6.3.1).
        m = _model(
            {
                "x": {"type": "parameter", "default": 1.0},
                "y": {"type": "unknown"},
            },
            equations=[{"lhs": "y", "rhs": ExprNode(op="*", args=["x", 2])}],
        )
        self.assertEqual(ode_states(m), [])
        self.assertFalse(mod._component_has_dynamics(m))

    def test_algebraic_equation_no_dynamics(self):
        # WaterEquilibrium-shape: lhs is a state variable, rhs is algebraic.
        m = _model(
            {
                "T": {"type": "parameter", "default": 298.0},
                "K_w": {"type": "unknown"},
            },
            equations=[{"lhs": "K_w", "rhs": ExprNode(op="*", args=["T", 1e-10])}],
        )
        self.assertFalse(mod._component_has_dynamics(m))

    def test_ode_equation_has_dynamics(self):
        # Canonical form: D(state, wrt=t) on lhs (DiameterGrowthRate shape).
        m = _model(
            {"y": {"type": "unknown"}, "k": {"type": "parameter", "default": 1.0}},
            equations=[
                {"lhs": ExprNode(op="D", args=["y"], wrt="t"), "rhs": ExprNode(op="*", args=[-1, "y"])}
            ],
        )
        self.assertTrue(mod._component_has_dynamics(m))
        # D buried inside rhs expression tree (less common but valid).
        m2 = _model(
            {"y": {"type": "unknown"}},
            equations=[{"lhs": "y_dot", "rhs": ExprNode(op="D", args=["y"], wrt="t")}],
        )
        self.assertTrue(mod._component_has_dynamics(m2))


class BuildSweepGridTest(unittest.TestCase):
    def _sweep(self, dims: list) -> "ParameterSweep":
        from earthsci_ast.esm_types import ParameterSweep, SweepDimension, SweepRange

        out_dims = []
        for d in dims:
            if "values" in d:
                out_dims.append(SweepDimension(parameter=d["parameter"], values=d["values"]))
            else:
                out_dims.append(
                    SweepDimension(
                        parameter=d["parameter"],
                        range=SweepRange(
                            start=d["start"],
                            stop=d["stop"],
                            count=d["count"],
                            scale=d.get("scale"),
                        ),
                    )
                )
        return ParameterSweep(type="cartesian", dimensions=out_dims)

    def test_linear_range(self):
        names, values, scales = mod._build_sweep_grid(
            self._sweep([{"parameter": "x", "start": 0.0, "stop": 1.0, "count": 5}])
        )
        self.assertEqual(names, ["x"])
        np.testing.assert_array_almost_equal(values[0], np.linspace(0.0, 1.0, 5))
        self.assertEqual(scales, ["linear"])

    def test_log_range(self):
        names, values, scales = mod._build_sweep_grid(
            self._sweep(
                [
                    {
                        "parameter": "x",
                        "start": 0.01,
                        "stop": 100.0,
                        "count": 5,
                        "scale": "log",
                    }
                ]
            )
        )
        self.assertEqual(scales, ["log"])
        self.assertAlmostEqual(values[0][0], 0.01)
        self.assertAlmostEqual(values[0][-1], 100.0)

    def test_explicit_values_log_inferred(self):
        # values list spanning >1 decade ⇒ log axis presentation.
        _, values, scales = mod._build_sweep_grid(
            self._sweep([{"parameter": "Dp", "values": [1e-8, 1e-6, 1e-4]}])
        )
        np.testing.assert_array_equal(values[0], np.array([1e-8, 1e-6, 1e-4]))
        self.assertEqual(scales, ["log"])

    def test_explicit_values_linear(self):
        _, values, scales = mod._build_sweep_grid(
            self._sweep([{"parameter": "T", "values": [273.0, 293.0, 313.0]}])
        )
        np.testing.assert_array_equal(values[0], np.array([273.0, 293.0, 313.0]))
        self.assertEqual(scales, ["linear"])

    def test_2d_cartesian(self):
        names, values, _ = mod._build_sweep_grid(
            self._sweep(
                [
                    {"parameter": "x", "start": 0.0, "stop": 1.0, "count": 3},
                    {"parameter": "y", "start": 0.0, "stop": 2.0, "count": 4},
                ]
            )
        )
        self.assertEqual(names, ["x", "y"])
        self.assertEqual(values[0].shape, (3,))
        self.assertEqual(values[1].shape, (4,))


class EvaluateGridTest(unittest.TestCase):
    """Smoke-tests that delegating evaluation to ESS produces correct numeric
    outputs for the renderer's two binding sources: observed variables and
    algebraic equations (forward and constraint forms)."""

    def test_observed_variable(self):
        # esm 1.0.0: the observed quantity's defining expression lives in
        # `equations`, and the resolution plan picks it up from there.
        m = _model(
            {
                "x": {"type": "parameter", "default": 1.0},
                "y": {"type": "unknown"},
            },
            equations=[{"lhs": "y", "rhs": ExprNode(op="*", args=["x", 2])}],
        )
        self.assertEqual(set(observed_definitions(m)), {"y"})
        env = mod._evaluate_grid(
            m,
            base_bindings={"x": 1.0},
            sweep_names=["x"],
            sweep_values=[np.array([1.0, 2.0, 3.0])],
        )
        np.testing.assert_array_equal(env["y"], np.array([2.0, 4.0, 6.0]))

    def test_forward_algebraic_equation(self):
        # WaterEquilibrium-shape: state defined by algebraic equation.
        m = _model(
            {"T": {"type": "parameter", "default": 298.0}, "y": {"type": "unknown"}},
            equations=[{"lhs": "y", "rhs": ExprNode(op="+", args=["T", 1])}],
        )
        env = mod._evaluate_grid(
            m,
            base_bindings={"T": 298.0},
            sweep_names=["T"],
            sweep_values=[np.array([10.0, 20.0])],
        )
        np.testing.assert_array_equal(env["y"], np.array([11.0, 21.0]))

    def test_constraint_equation_solved(self):
        # `K_w = H_plus * OH_minus` with K_w forward-defined → solve for OH_minus.
        m = _model(
            {
                "T": {"type": "parameter", "default": 298.0},
                "H_plus": {"type": "parameter", "default": 1e-4},
                "K_w": {"type": "unknown"},
                "OH_minus": {"type": "unknown"},
            },
            equations=[
                {"lhs": "K_w", "rhs": ExprNode(op="*", args=["T", 1e-10])},
                {"lhs": "K_w", "rhs": ExprNode(op="*", args=["H_plus", "OH_minus"])},
            ],
        )
        env = mod._evaluate_grid(
            m,
            base_bindings={"T": 298.0, "H_plus": 1e-4},
            sweep_names=["H_plus"],
            sweep_values=[np.array([1e-4, 1e-3])],
        )
        # K_w = T * 1e-10 = 298e-10 ≈ 2.98e-8 (constant under the H_plus sweep)
        # OH_minus = K_w / H_plus
        np.testing.assert_allclose(env["K_w"], np.full(2, 298.0 * 1e-10))
        np.testing.assert_allclose(env["OH_minus"], np.array([2.98e-4, 2.98e-5]))

    def test_zero_division_substitutes_nan(self):
        # Sweep includes the divisor=0 boundary point. ESS's ifelse() does
        # not short-circuit, so even when the .esm guards the division
        # behind `ifelse(x > 0, a/x, 0)`, the evaluator hits 0/0 and
        # raises Python's native ZeroDivisionError. The renderer must
        # substitute NaN at that grid point rather than abandoning the
        # whole analysis — matches the snow_melt_no_layers.esm failure
        # mode (W_sno_np1 / W_sno_n at the W_sno_n=0 endpoint).
        m = _model(
            {"x": {"type": "parameter", "default": 1.0}, "y": {"type": "unknown"}},
            equations=[{"lhs": "y", "rhs": ExprNode(op="/", args=[1.0, "x"])}],
        )
        env = mod._evaluate_grid(
            m,
            base_bindings={"x": 1.0},
            sweep_names=["x"],
            sweep_values=[np.array([0.0, 1.0, 2.0])],
        )
        self.assertTrue(np.isnan(env["y"][0]))
        np.testing.assert_array_equal(env["y"][1:], np.array([1.0, 0.5]))

    def test_zero_division_propagates_nan_to_downstream(self):
        # If a NaN-substituted grid point feeds a later expression in the
        # plan, the downstream value also becomes NaN via standard IEEE-754
        # NaN propagation — no second exception. Verifies the catch leaves
        # the rest of the plan in a numerically-safe state.
        m = _model(
            {
                "x": {"type": "parameter", "default": 1.0},
                "y": {"type": "unknown"},
                "z": {"type": "unknown"},
            },
            equations=[
                {"lhs": "y", "rhs": ExprNode(op="/", args=[1.0, "x"])},
                {"lhs": "z", "rhs": ExprNode(op="*", args=["y", 2.0])},
            ],
        )
        env = mod._evaluate_grid(
            m,
            base_bindings={"x": 1.0},
            sweep_names=["x"],
            sweep_values=[np.array([0.0, 1.0])],
        )
        self.assertTrue(np.isnan(env["y"][0]))
        self.assertTrue(np.isnan(env["z"][0]))
        np.testing.assert_array_equal(env["y"][1:], np.array([1.0]))
        np.testing.assert_array_equal(env["z"][1:], np.array([2.0]))

    def test_observed_reading_the_independent_variable(self):
        # Fast-JX shape: an observed of `t + t_ref`. Without `t` bound the plan
        # cannot resolve it; with it, every grid point reads the same instant.
        m = _model(
            {
                "t_ref": {"type": "parameter", "default": 0.0},
                "t_utc": {"type": "unknown"},
            },
            equations=[{"lhs": "t_utc", "rhs": ExprNode(op="+", args=["t", "t_ref"])}],
        )
        with self.assertRaises(mod.UnsupportedExpression):
            mod._evaluate_grid(
                m, base_bindings={}, sweep_names=["t_ref"], sweep_values=[np.array([0.0, 10.0])]
            )
        env = mod._evaluate_grid(
            m, base_bindings={}, sweep_names=["t_ref"], sweep_values=[np.array([0.0, 10.0])], t=1.0
        )
        np.testing.assert_array_equal(env["t_utc"], np.array([1.0, 11.0]))


class ExtractOdeEquationsTest(unittest.TestCase):
    def test_separates_ode_from_algebraic(self):
        m = _model(
            {"D_p": {"type": "unknown"}, "I_D": {"type": "unknown"}, "A": {"type": "unknown"}},
            equations=[
                {"lhs": ExprNode(op="D", args=["D_p"], wrt="t"), "rhs": "I_D"},
                {"lhs": "A", "rhs": ExprNode(op="*", args=[1.0, 2.0])},
                {"lhs": "I_D", "rhs": ExprNode(op="/", args=["A", "D_p"])},
            ],
        )
        ode_map, alg_eqs = mod._extract_ode_equations(m)
        self.assertEqual(set(ode_map.keys()), {"D_p"})
        self.assertEqual(ode_map["D_p"], "I_D")
        self.assertEqual([eq.lhs for eq in alg_eqs], ["A", "I_D"])

    def test_no_ode_returns_empty_map(self):
        m = _model({"y": {"type": "unknown"}}, equations=[{"lhs": "y", "rhs": 1.0}])
        ode_map, alg_eqs = mod._extract_ode_equations(m)
        self.assertEqual(ode_map, {})
        self.assertEqual(len(alg_eqs), 1)


class SolveTimeSeriesTest(unittest.TestCase):
    """Smoke-tests for the ODE integration path."""

    def test_exponential_decay(self):
        # dy/dt = -k*y, y(0)=1 → y(t) = exp(-k*t).
        from earthsci_ast.esm_types import TimeSpan

        m = _model(
            {"k": {"type": "parameter", "default": 0.5}, "y": {"type": "unknown"}},
            equations=[
                {
                    "lhs": ExprNode(op="D", args=["y"], wrt="t"),
                    "rhs": ExprNode(op="*", args=[-1, "k", "y"]),
                }
            ],
        )
        env = mod._solve_time_series(
            m,
            base_bindings={"k": 0.5},
            initial_state_values={"y": 1.0},
            time_span=TimeSpan(start=0.0, end=4.0),
            n_points=101,
        )
        np.testing.assert_allclose(env["t"][[0, -1]], [0.0, 4.0])
        np.testing.assert_allclose(
            env["y"], np.exp(-0.5 * env["t"]), rtol=1e-5, atol=1e-7
        )

    def test_algebraic_observed_trajectory(self):
        # DiameterGrowthRate-shape: D(D_p)/dt = I_D, A = const, I_D = A/D_p
        # ⇒ D_p² = D_p0² + 2A*t. Verifies algebraic-equation post-processing.
        from earthsci_ast.esm_types import TimeSpan

        m = _model(
            {
                "A": {"type": "unknown"},
                "D_p": {"type": "unknown"},
                "I_D": {"type": "unknown"},
                "k": {"type": "parameter", "default": 1.0e-16},
            },
            equations=[
                {"lhs": ExprNode(op="D", args=["D_p"], wrt="t"), "rhs": "I_D"},
                {"lhs": "A", "rhs": "k"},
                {"lhs": "I_D", "rhs": ExprNode(op="/", args=["A", "D_p"])},
            ],
        )
        env = mod._solve_time_series(
            m,
            base_bindings={"k": 1.0e-16},
            initial_state_values={"D_p": 2.0e-7},
            time_span=TimeSpan(start=0.0, end=1200.0),
            n_points=51,
        )
        # Analytical: D_p(t) = sqrt(D_p0² + 2*A*t)
        expected = np.sqrt(2.0e-7**2 + 2.0 * 1.0e-16 * env["t"])
        np.testing.assert_allclose(env["D_p"], expected, rtol=1e-4)
        # I_D trajectory comes from algebraic post-processing
        np.testing.assert_allclose(env["I_D"], 1.0e-16 / env["D_p"], rtol=1e-4)


class IntegrationTest(unittest.TestCase):
    """Render plots from a synthetic .esm file end-to-end."""

    def _write_esm(self, root: Path, body: dict, subdir: str) -> Path:
        path = root / "components" / subdir / f"{subdir}.esm"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(body), encoding="utf-8")
        return path

    def _toy_observed_body(self) -> dict:
        return {
            "esm": "1.0.0",
            "metadata": {"name": "Toy"},
            "models": {
                "Toy": {
                    "variables": {
                        "x": {"type": "parameter", "default": 1.0, "units": "m"},
                        # esm 1.0.0: `y` is observed by virtue of the
                        # bare-variable-LHS equation below, not a declared type.
                        "y": {"type": "unknown", "units": "m"},
                    },
                    "equations": [{"lhs": "y", "rhs": {"op": "*", "args": ["x", 2]}}],
                    "analyses": [
                        {
                            "id": "y_vs_x",
                            "description": "y over x",
                            "time_span": {"start": 0.0, "end": 1.0},
                            "parameter_sweep": {
                                "type": "cartesian",
                                "dimensions": [
                                    {
                                        "parameter": "x",
                                        "range": {
                                            "start": 0.0,
                                            "stop": 1.0,
                                            "count": 10,
                                        },
                                    }
                                ],
                            },
                            "plots": [
                                {
                                    "id": "line",
                                    "type": "line",
                                    "x": {"variable": "x", "label": "x"},
                                    "y": {"variable": "y", "label": "y"},
                                    "description": "y = 2x",
                                }
                            ],
                        }
                    ],
                }
            },
        }

    def test_run_emits_line_plot(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_esm(root, self._toy_observed_body(), "toy")
            rc = mod.run(root / "components")
            self.assertEqual(rc, 0)
            png = root / "components" / "toy" / "toy.plots" / "y_vs_x-line.png"
            self.assertTrue(png.is_file(), f"expected {png} to exist")
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_run_emits_scatter_plot(self):
        body = self._toy_observed_body()
        body["models"]["Toy"]["analyses"][0]["plots"][0]["type"] = "scatter"
        body["models"]["Toy"]["analyses"][0]["plots"][0]["id"] = "scat"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_esm(root, body, "toy")
            rc = mod.run(root / "components")
            self.assertEqual(rc, 0)
            png = root / "components" / "toy" / "toy.plots" / "y_vs_x-scat.png"
            self.assertTrue(png.is_file())
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_algebraic_equations_component_renders(self):
        # Mirrors WaterEquilibrium shape: state variable defined by an
        # algebraic equation. Pre-mdl-qz4 the renderer would skip these.
        body = {
            "esm": "1.0.0",
            "metadata": {"name": "Algebraic"},
            "models": {
                "Algebraic": {
                    "variables": {
                        "T": {"type": "parameter", "default": 298.0, "units": "K"},
                        "K": {"type": "unknown"},
                    },
                    "equations": [
                        {
                            "lhs": "K",
                            "rhs": {"op": "*", "args": ["T", 1e-3]},
                        }
                    ],
                    "analyses": [
                        {
                            "id": "k_vs_t",
                            "time_span": {"start": 0.0, "end": 1.0},
                            "parameter_sweep": {
                                "type": "cartesian",
                                "dimensions": [
                                    {
                                        "parameter": "T",
                                        "range": {
                                            "start": 273.0,
                                            "stop": 318.0,
                                            "count": 5,
                                        },
                                    }
                                ],
                            },
                            "plots": [
                                {
                                    "id": "k_curve",
                                    "type": "line",
                                    "x": {"variable": "T"},
                                    "y": {"variable": "K"},
                                }
                            ],
                        }
                    ],
                }
            },
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_esm(root, body, "alg")
            rc = mod.run(root / "components")
            self.assertEqual(rc, 0)
            png = root / "components" / "alg" / "alg.plots" / "k_vs_t-k_curve.png"
            self.assertTrue(png.is_file())

    def test_skips_analyses_without_sweep(self):
        body = {
            "esm": "1.0.0",
            "models": {
                "T": {
                    "variables": {
                        "x": {"type": "parameter", "default": 1.0},
                        "y": {"type": "unknown"},
                    },
                    "equations": [{"lhs": "y", "rhs": {"op": "*", "args": ["x", 2]}}],
                    "analyses": [
                        {
                            "id": "no_sweep",
                            "time_span": {"start": 0.0, "end": 1.0},
                            "plots": [
                                {
                                    "id": "p",
                                    "type": "line",
                                    "x": {"variable": "x"},
                                    "y": {"variable": "y"},
                                }
                            ],
                        }
                    ],
                }
            },
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_esm(root, body, "t")
            rc = mod.run(root / "components")
            self.assertEqual(rc, 0)
            self.assertFalse((root / "components" / "t" / "t.plots").exists())

    def test_time_series_ode_renders(self):
        # ODE component with time_span + initial_state should render
        # one curve per plot via the integration path (mirrors
        # DiameterGrowthRate's figure_13_2 analyses).
        body = {
            "esm": "1.0.0",
            "metadata": {"name": "Decay"},
            "models": {
                "Decay": {
                    "variables": {
                        "k": {"type": "parameter", "default": 0.5},
                        "y": {"type": "unknown"},
                    },
                    "equations": [
                        {
                            "lhs": {"op": "D", "wrt": "t", "args": ["y"]},
                            "rhs": {"op": "*", "args": [-1, "k", "y"]},
                        }
                    ],
                    "analyses": [
                        {
                            "id": "decay",
                            "time_span": {"start": 0.0, "end": 4.0},
                            "initial_state": {
                                "type": "per_variable",
                                "values": {"y": 1.0},
                            },
                            "plots": [
                                {
                                    "id": "y_curve",
                                    "type": "line",
                                    "x": {"variable": "t", "label": "t"},
                                    "y": {"variable": "y", "label": "y"},
                                }
                            ],
                        }
                    ],
                }
            },
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_esm(root, body, "decay")
            rc = mod.run(root / "components")
            self.assertEqual(rc, 0)
            png = root / "components" / "decay" / "decay.plots" / "decay-y_curve.png"
            self.assertTrue(png.is_file(), f"expected {png} to exist")
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_time_series_ode_with_sweep_renders(self):
        # ODE + 1-D parameter_sweep: one integration per grid point, one
        # plot with multiple curves overlaid.
        body = {
            "esm": "1.0.0",
            "metadata": {"name": "Decay"},
            "models": {
                "Decay": {
                    "variables": {
                        "k": {"type": "parameter", "default": 0.5},
                        "y": {"type": "unknown"},
                    },
                    "equations": [
                        {
                            "lhs": {"op": "D", "wrt": "t", "args": ["y"]},
                            "rhs": {"op": "*", "args": [-1, "k", "y"]},
                        }
                    ],
                    "analyses": [
                        {
                            "id": "decay_sweep",
                            "time_span": {"start": 0.0, "end": 4.0},
                            "initial_state": {
                                "type": "per_variable",
                                "values": {"y": 1.0},
                            },
                            "parameter_sweep": {
                                "type": "cartesian",
                                "dimensions": [
                                    {"parameter": "k", "values": [0.25, 0.5, 1.0]}
                                ],
                            },
                            "plots": [
                                {
                                    "id": "family",
                                    "type": "line",
                                    "x": {"variable": "t"},
                                    "y": {"variable": "y"},
                                }
                            ],
                        }
                    ],
                }
            },
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_esm(root, body, "decay")
            rc = mod.run(root / "components")
            self.assertEqual(rc, 0)
            png = root / "components" / "decay" / "decay.plots" / "decay_sweep-family.png"
            self.assertTrue(png.is_file())
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_reaction_system_renders_time_series_with_default_initial_state(self):
        # First-order isomerization A -> B, k = 0.5/s. Tests three things at once:
        # (1) reaction_systems are discovered alongside models;
        # (2) species defaults seed the integration when no initial_state present;
        # (3) plot.x='t' on a reaction_system analysis draws a time-series curve.
        body = {
            "esm": "1.0.0",
            "metadata": {"name": "AB"},
            "reaction_systems": {
                "AB": {
                    "species": {
                        "A": {"default": 1.0, "units": "M"},
                        "B": {"default": 0.0, "units": "M"},
                    },
                    "parameters": {"k": {"default": 0.5, "units": "1/s"}},
                    "reactions": [
                        {
                            "id": "R1",
                            "substrates": [{"species": "A", "stoichiometry": 1}],
                            "products": [{"species": "B", "stoichiometry": 1}],
                            "rate": "k",
                        }
                    ],
                    "analyses": [
                        {
                            "id": "decay",
                            "time_span": {"start": 0.0, "end": 4.0},
                            "plots": [
                                {
                                    "id": "A_vs_t",
                                    "type": "line",
                                    "x": {"variable": "t"},
                                    "y": {"variable": "A"},
                                }
                            ],
                        }
                    ],
                }
            },
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_esm(root, body, "ab")
            rc = mod.run(root / "components")
            self.assertEqual(rc, 0)
            png = root / "components" / "ab" / "ab.plots" / "decay-A_vs_t.png"
            self.assertTrue(png.is_file())
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_reaction_system_rate_constant_referencing_reactant_species(self):
        # A + B -> C with rate_constant `k/A` (mass-action multiplier of A
        # cancels symbolically — a pattern used by GEOS-Chem fullchem's R1,
        # R4, R12 for SO2/SALAAL/SALCAL aqueous channels). Without symbolic
        # simplification at adapter-build time the renderer would compute
        # `(k/A) * A * B = k * B` by evaluating `k/A` first and dividing
        # by zero whenever [A] hits 0 along the trajectory. This test
        # exercises that exact case with A starting at 0 — integration
        # must succeed and the plot must render.
        body = {
            "esm": "1.0.0",
            "metadata": {"name": "ABC"},
            "reaction_systems": {
                "ABC": {
                    "species": {
                        "A": {"default": 0.0},
                        "B": {"default": 1.0},
                        "C": {"default": 0.0},
                    },
                    "parameters": {"k": {"default": 0.1}},
                    "reactions": [
                        {
                            "id": "R1",
                            "substrates": [
                                {"species": "A", "stoichiometry": 1},
                                {"species": "B", "stoichiometry": 1},
                            ],
                            "products": [{"species": "C", "stoichiometry": 1}],
                            "rate": {"op": "/", "args": ["k", "A"]},
                        }
                    ],
                    "analyses": [
                        {
                            "id": "cancel",
                            "time_span": {"start": 0.0, "end": 4.0},
                            "plots": [
                                {
                                    "id": "C_vs_t",
                                    "type": "line",
                                    "x": {"variable": "t"},
                                    "y": {"variable": "C"},
                                }
                            ],
                        }
                    ],
                }
            },
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_esm(root, body, "abc")
            rc = mod.run(root / "components")
            self.assertEqual(rc, 0)
            png = root / "components" / "abc" / "abc.plots" / "cancel-C_vs_t.png"
            self.assertTrue(png.is_file())

    def test_reaction_system_initial_state_merges_with_defaults(self):
        # A reaction system with two species — only one is named in the
        # analysis's `initial_state.values`. The other should fall back to
        # its declared species default rather than triggering a "missing
        # ODE state" error. Lets fullchem-scale analyses (272 species) name
        # only the few they care about (mdl-dtm).
        body = {
            "esm": "1.0.0",
            "metadata": {"name": "AB"},
            "reaction_systems": {
                "AB": {
                    "species": {
                        "A": {"default": 1.0},
                        "B": {"default": 0.0},
                    },
                    "parameters": {"k": {"default": 0.5}},
                    "reactions": [
                        {
                            "id": "R1",
                            "substrates": [{"species": "A", "stoichiometry": 1}],
                            "products": [{"species": "B", "stoichiometry": 1}],
                            "rate": "k",
                        }
                    ],
                    "analyses": [
                        {
                            "id": "partial_ic",
                            "time_span": {"start": 0.0, "end": 4.0},
                            "initial_state": {
                                "type": "per_variable",
                                "values": {"A": 2.0},
                            },
                            "plots": [
                                {
                                    "id": "A_vs_t",
                                    "type": "line",
                                    "x": {"variable": "t"},
                                    "y": {"variable": "A"},
                                }
                            ],
                        }
                    ],
                }
            },
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_esm(root, body, "ab2")
            rc = mod.run(root / "components")
            self.assertEqual(rc, 0)
            png = root / "components" / "ab2" / "ab2.plots" / "partial_ic-A_vs_t.png"
            self.assertTrue(png.is_file())

    def test_reaction_system_renders_final_state_vs_sweep(self):
        # A -> B with k swept; plot final-state B vs k. Exercises the new
        # final-state-vs-sweep dispatch path: plot.x is the swept parameter,
        # not 't', so each grid point's endpoint becomes one data point.
        body = {
            "esm": "1.0.0",
            "metadata": {"name": "AB"},
            "reaction_systems": {
                "AB": {
                    "species": {
                        "A": {"default": 1.0},
                        "B": {"default": 0.0},
                    },
                    "parameters": {"k": {"default": 0.5}},
                    "reactions": [
                        {
                            "id": "R1",
                            "substrates": [{"species": "A", "stoichiometry": 1}],
                            "products": [{"species": "B", "stoichiometry": 1}],
                            "rate": "k",
                        }
                    ],
                    "analyses": [
                        {
                            "id": "k_sweep",
                            "time_span": {"start": 0.0, "end": 4.0},
                            "parameter_sweep": {
                                "type": "cartesian",
                                "dimensions": [
                                    {"parameter": "k", "values": [0.25, 0.5, 1.0]}
                                ],
                            },
                            "plots": [
                                {
                                    "id": "B_vs_k",
                                    "type": "line",
                                    "x": {"variable": "k"},
                                    "y": {"variable": "B"},
                                }
                            ],
                        }
                    ],
                }
            },
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_esm(root, body, "ab")
            rc = mod.run(root / "components")
            self.assertEqual(rc, 0)
            png = root / "components" / "ab" / "ab.plots" / "k_sweep-B_vs_k.png"
            self.assertTrue(png.is_file())
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_skips_components_with_time_derivatives(self):
        body = {
            "esm": "1.0.0",
            "models": {
                "Ode": {
                    "variables": {
                        "y": {"type": "unknown", "default": 1.0},
                        "k": {"type": "parameter", "default": 1.0},
                    },
                    "equations": [
                        {
                            "lhs": "y_dot",
                            "rhs": {
                                "op": "*",
                                "args": [-1, {"op": "D", "args": ["y"], "wrt": "t"}],
                            },
                        }
                    ],
                    "analyses": [
                        {
                            "id": "sweep",
                            "time_span": {"start": 0.0, "end": 1.0},
                            "parameter_sweep": {
                                "type": "cartesian",
                                "dimensions": [
                                    {
                                        "parameter": "k",
                                        "range": {
                                            "start": 0.1,
                                            "stop": 1.0,
                                            "count": 3,
                                        },
                                    }
                                ],
                            },
                            "plots": [
                                {
                                    "id": "p",
                                    "type": "line",
                                    "x": {"variable": "k"},
                                    "y": {"variable": "y"},
                                }
                            ],
                        }
                    ],
                }
            },
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_esm(root, body, "ode")
            rc = mod.run(root / "components")
            self.assertEqual(rc, 0)
            self.assertFalse((root / "components" / "ode" / "ode.plots").exists())


    def test_expression_ic_renders_placeholder(self):
        # Regression for esm-8m7w9: a component whose initial_state is
        # type:"expression" (PDE spatial field) must render a placeholder PNG
        # rather than crashing or silently using state defaults. The placeholder
        # must be visibly distinct from a scalar-IC plot (different file bytes).
        body = {
            "esm": "1.0.0",
            "metadata": {"name": "PdeTest"},
            "models": {
                "PdeTest": {
                    "variables": {
                        "psi": {"type": "unknown"},
                        "k": {"type": "parameter", "default": 0.1},
                    },
                    "equations": [
                        {
                            "lhs": {"op": "D", "wrt": "t", "args": ["psi"]},
                            "rhs": {"op": "*", "args": ["k", "psi"]},
                        }
                    ],
                    "analyses": [
                        {
                            "id": "pde_ic",
                            "time_span": {"start": 0.0, "end": 1.0},
                            "initial_state": {
                                "type": "expression",
                                "values": {
                                    "psi": {"op": "-", "args": ["x", 0.5]},
                                },
                            },
                            "plots": [
                                {
                                    "id": "psi_t",
                                    "type": "line",
                                    "x": {"variable": "t"},
                                    "y": {"variable": "psi"},
                                    "description": "psi vs t",
                                }
                            ],
                        },
                        {
                            "id": "scalar_ic",
                            "time_span": {"start": 0.0, "end": 1.0},
                            "initial_state": {
                                "type": "per_variable",
                                "values": {"psi": 1.0},
                            },
                            "plots": [
                                {
                                    "id": "psi_scalar",
                                    "type": "line",
                                    "x": {"variable": "t"},
                                    "y": {"variable": "psi"},
                                }
                            ],
                        },
                    ],
                }
            },
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_esm(root, body, "pde")
            rc = mod.run(root / "components")
            self.assertEqual(rc, 0)
            plots_dir = root / "components" / "pde" / "pde.plots"
            # Expression IC: placeholder PNG must exist and be a valid PNG
            placeholder = plots_dir / "pde_ic-psi_t.png"
            self.assertTrue(placeholder.is_file(), f"expected placeholder at {placeholder}")
            self.assertEqual(placeholder.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            # Scalar IC: normal rendered PNG must exist
            scalar_png = plots_dir / "scalar_ic-psi_scalar.png"
            self.assertTrue(scalar_png.is_file(), f"expected rendered PNG at {scalar_png}")
            self.assertEqual(scalar_png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            # Placeholder and scalar PNG must be visibly different (different bytes)
            self.assertNotEqual(placeholder.read_bytes(), scalar_png.read_bytes())


if __name__ == "__main__":
    unittest.main()
