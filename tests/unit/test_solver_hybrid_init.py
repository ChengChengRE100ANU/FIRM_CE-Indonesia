import numpy as np
from scipy.optimize import OptimizeResult

import firm_ce.optimisation.solver as solver_module
from firm_ce.model import Model
from firm_ce.optimisation.broad_optimum import create_groups_dict
from firm_ce.optimisation.solver import Solver


def _build_solver() -> Solver:
    model = Model(
        config_directory="tests/inputs/test_1hr_config_data_diversify/config",
        data_directory="tests/inputs/test_1hr_config_data_diversify/data",
        logging_flag=False,
    )
    scenario = model.scenarios["gas"]
    return Solver(
        model.config,
        scenario.x0,
        scenario.static,
        scenario.fleet,
        scenario.network,
        scenario.logger,
        scenario.name,
        scenario.initial_population,
    )


def test_build_hybrid_initial_population_shape_and_bounds():
    solver = _build_solver()
    center_x = 0.5 * (solver.lower_bounds + solver.upper_bounds)
    init_population = solver.build_hybrid_initial_population(center_x)

    free_dimensions = int(np.sum(np.abs(solver.upper_bounds - solver.lower_bounds) > 1e-12))
    expected_size = max(5, solver.config.population * max(free_dimensions, 1))

    assert init_population.shape == (expected_size, center_x.size)
    assert np.all(init_population >= solver.lower_bounds - 1e-12)
    assert np.all(init_population <= solver.upper_bounds + 1e-12)
    assert np.allclose(init_population[0], np.clip(center_x, solver.lower_bounds, solver.upper_bounds))
    assert np.any(np.any(np.abs(init_population[1:] - init_population[0]) > 1e-12, axis=1))


def test_find_near_optimal_band_uses_hybrid_initial_population(monkeypatch):
    solver = _build_solver()
    groups = create_groups_dict(solver.broad_optimum_var_info)
    assert len(groups) > 0

    solver.decision_x0 = 0.5 * (solver.lower_bounds + solver.upper_bounds)

    init_calls = []

    monkeypatch.setattr(solver, "get_band_lcoe_max", lambda: 1.0)
    monkeypatch.setattr(solver_module, "write_broad_optimum_records", lambda *args, **kwargs: None)
    monkeypatch.setattr(solver_module, "write_broad_optimum_bands", lambda *args, **kwargs: None)

    def fake_run_differential_evolution(objective_function, args, init_override=None):
        init_calls.append(init_override)
        return OptimizeResult(x=solver.decision_x0.copy())

    monkeypatch.setattr(solver, "run_differential_evolution", fake_run_differential_evolution)

    solver.find_near_optimal_band()

    expected_calls = 2 * len(groups)
    assert len(init_calls) == expected_calls
    for init_population in init_calls:
        assert isinstance(init_population, np.ndarray)
        assert init_population.shape[1] == solver.decision_x0.size
        assert np.all(init_population >= solver.lower_bounds - 1e-12)
        assert np.all(init_population <= solver.upper_bounds + 1e-12)
        assert np.allclose(
            init_population[0],
            np.clip(solver.decision_x0, solver.lower_bounds, solver.upper_bounds),
        )


def test_diversify_uses_multi_anchor_hybrid_initial_population(monkeypatch):
    solver = _build_solver()
    solver.decision_x0 = 0.5 * (solver.lower_bounds + solver.upper_bounds)
    solver.optimal_lcoe = 1.0

    x_band_min = np.clip(solver.decision_x0 * 0.95, solver.lower_bounds, solver.upper_bounds)
    x_band_max = np.clip(solver.decision_x0 * 1.05, solver.lower_bounds, solver.upper_bounds)
    x_record_a = np.clip(solver.decision_x0 * 0.98, solver.lower_bounds, solver.upper_bounds)
    x_record_b = np.clip(solver.decision_x0 * 1.02, solver.lower_bounds, solver.upper_bounds)

    solver.near_optimal_bands = {"PV": (x_band_min, x_band_max)}
    solver.near_optimal_records = [
        ("PV", "min", "N/A", 1.0, 0.0, 0.0, x_record_a),
        ("PV", "max", "N/A", 1.0, 0.0, 0.0, x_record_b),
    ]

    monkeypatch.setattr(solver_module, "write_diversify_records", lambda *args, **kwargs: None)

    init_calls = []

    def fake_run_differential_evolution(objective_function, args, init_override=None):
        init_calls.append(init_override)
        return OptimizeResult(x=solver.decision_x0.copy())

    monkeypatch.setattr(solver, "run_differential_evolution", fake_run_differential_evolution)

    solver.diversify()

    assert len(init_calls) == 1
    init_population = init_calls[0]
    assert isinstance(init_population, np.ndarray)
    assert init_population.shape[1] == solver.decision_x0.size
    assert np.all(init_population >= solver.lower_bounds - 1e-12)
    assert np.all(init_population <= solver.upper_bounds + 1e-12)
    assert np.allclose(
        init_population[0],
        np.clip(solver.decision_x0, solver.lower_bounds, solver.upper_bounds),
    )

    expected_anchors = solver._deduplicate_anchor_points(
        solver.collect_diversify_anchor_points(solver.decision_x0)
    )
    for anchor in expected_anchors:
        assert np.any(np.all(np.isclose(init_population, anchor, atol=1e-12), axis=1))
