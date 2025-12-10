import csv
import os
from itertools import chain
from logging import Logger
from typing import Dict, List, Tuple, Union, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, differential_evolution

from firm_ce.common.constants import SAVE_POPULATION
from firm_ce.optimisation.broad_optimum import (
    broad_optimum_objective,
    build_broad_optimum_var_info,
    create_groups_dict,
    render_candidates,
    diversify_objective,
    write_broad_optimum_bands,
    write_broad_optimum_records,
    write_diversify_records,
)
from firm_ce.optimisation.single_time import Solution, evaluate_vectorised_xs
from firm_ce.system.components import Fleet_InstanceType, Generator_InstanceType, Storage_InstanceType
from firm_ce.system.parameters import ModelConfig, ScenarioParameters_InstanceType
from firm_ce.system.topology import Line_InstanceType, Network_InstanceType


class Solver:
    def __init__(
        self,
        config: ModelConfig,
        initial_x_candidate: NDArray[np.float64],
        parameters_static: ScenarioParameters_InstanceType,
        fleet_static: Fleet_InstanceType,
        network_static: Network_InstanceType,
        scenario_logger: Logger,
        scenario_name: str,
        initial_population: Union[NDArray[np.float64], str] = "latinhypercube",
    ) -> None:
        self.config = config
        self.decision_x0 = initial_x_candidate if len(initial_x_candidate) > 0 else None
        self.parameters_static = parameters_static
        self.fleet_static = fleet_static
        self.network_static = network_static
        self.logger = scenario_logger
        self.lower_bounds, self.upper_bounds = self.get_bounds()
        self.broad_optimum_var_info = build_broad_optimum_var_info(fleet_static, network_static)
        self.scenario_name = scenario_name
        self.result = None
        self.optimal_lcoe = None
        self.initial_population = initial_population
        self.iterations = config.iterations

    def get_bounds(self) -> NDArray[np.float64]:
        def power_capacity_bounds(
            asset_list: Union[List[Generator_InstanceType], List[Storage_InstanceType], List[Line_InstanceType]],
            build_cap_constraint: str,
        ) -> List[float]:
            return [getattr(asset, build_cap_constraint) for asset in asset_list]

        def energy_capacity_bounds(storage_list: List[Storage_InstanceType], build_cap_constraint: str) -> List[float]:
            bounds = []
            for s in storage_list:
                if s.duration == 0:
                    bounds.append(getattr(s, build_cap_constraint))
                else:
                    if "min" in build_cap_constraint:
                        bounds.append(s.min_build_p * s.duration)
                    else:
                        bounds.append(s.max_build_p * s.duration)
            return bounds

        generators = list(self.fleet_static.generators.values())
        storages = list(self.fleet_static.storages.values())
        lines = list(self.network_static.major_lines.values())

        lower_bounds = np.array(
            list(
                chain(
                    power_capacity_bounds(generators, "min_build"),
                    power_capacity_bounds(storages, "min_build_p"),
                    energy_capacity_bounds(storages, "min_build_e"),
                    power_capacity_bounds(lines, "min_build"),
                )
            )
        )

        upper_bounds = np.array(
            list(
                chain(
                    power_capacity_bounds(generators, "max_build"),
                    power_capacity_bounds(storages, "max_build_p"),
                    energy_capacity_bounds(storages, "max_build_e"),
                    power_capacity_bounds(lines, "max_build"),
                )
            )
        )

        return lower_bounds, upper_bounds

    def initialise_callback(self) -> None:
        temp_dir = os.path.join("results", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        with open(os.path.join(temp_dir, "callback.csv"), "w", newline="") as csvfile:
            csv.writer(csvfile)
        with open(os.path.join(temp_dir, "population.csv"), "w", newline="") as csvfile:
            csv.writer(csvfile)
        with open(os.path.join(temp_dir, "population_energies.csv"), "w", newline="") as csvfile:
            csv.writer(csvfile)

    def get_differential_evolution_args(
        self,
    ) -> Tuple[ScenarioParameters_InstanceType, Fleet_InstanceType, Network_InstanceType, str, float]:
        args = (
            self.parameters_static,
            self.fleet_static,
            self.network_static,
            self.config.balancing_type,
            self.config.fixed_costs_threshold,
        )
        return args

    def run_differential_evolution(self, objective_function: Callable, args: Tuple) -> OptimizeResult:
        result = differential_evolution(
            x0=self.decision_x0,
            func=objective_function,
            bounds=list(zip(self.lower_bounds, self.upper_bounds)),
            args=args,
            tol=0,
            maxiter=self.iterations,
            popsize=self.config.population,
            init=self.initial_population,
            mutation=(0.2, self.config.mutation),
            recombination=self.config.recombination,
            disp=True,
            polish=False,
            updating="deferred",
            callback=callback,
            workers=1,
            vectorized=True,
        )
        return result

    def _optimal_x_path(self) -> str:
        return os.path.join("results", "temp", f"optimal_x_{self.scenario_name}.csv")

    def write_optimal_x(self, x_vec: NDArray[np.float64]) -> None:
        path = self._optimal_x_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savetxt(path, x_vec.reshape(1, -1), delimiter=",")

    def load_optimal_x(self) -> NDArray[np.float64] | None:
        path = self._optimal_x_path()
        if os.path.isfile(path):
            try:
                return np.loadtxt(path, delimiter=",").reshape(-1)
            except Exception:
                return None
        return None

    def single_time(self) -> None:
        self.initialise_callback()
        self.result = self.run_differential_evolution(evaluate_vectorised_xs, self.get_differential_evolution_args())
        if self.result is not None and getattr(self.result, "x", None) is not None:
            self.write_optimal_x(self.result.x)

    def get_band_lcoe_max(self) -> float:
        # Use the best available feasible point to set the LCOE band cap.
        def _evaluate(x_vec):
            sol = Solution(x_vec, *self.get_differential_evolution_args())
            sol.evaluate()
            return sol

        x_candidate = self.decision_x0 if self.decision_x0 is not None else self.load_optimal_x()
        solution = _evaluate(x_candidate)

        if solution.penalties > 1e-6 or solution.lcoe <= 0:
            self.logger.warning(
                f"Initial guess used for band cap has penalties={solution.penalties} and lcoe={solution.lcoe}."
                f" Attempting to derive band cap from the optimiser result instead."
            )
            if self.result is not None and getattr(self.result, "x", None) is not None:
                alt_solution = _evaluate(self.result.x)
                if alt_solution.penalties <= 1e-6 and alt_solution.lcoe > 0:
                    solution = alt_solution
                else:
                    self.logger.warning(
                        f"Optimiser best solution also penalised (penalties={alt_solution.penalties}, "
                        f"lcoe={alt_solution.lcoe}). Using lcoe+penalties for band cap."
                    )
                    solution.lcoe = alt_solution.lcoe + alt_solution.penalties
            else:
                # Fall back to using lcoe + penalties to avoid zero band
                solution.lcoe = solution.lcoe + solution.penalties

        self.optimal_lcoe = solution.lcoe
        band_lcoe_max = max(self.optimal_lcoe, 1e-6) * (1 + self.config.near_optimal_tol)

        return band_lcoe_max

    def find_near_optimal_band(self) -> Dict[str, Tuple[float]]:
        if self.decision_x0 is None:
            loaded = self.load_optimal_x()
            if loaded is not None:
                self.decision_x0 = loaded
        band_lcoe_max = self.get_band_lcoe_max()
        evaluation_records = []
        bands = {}
        groups = create_groups_dict(self.broad_optimum_var_info)

        for group_key, idx_list in groups.items():
            self.logger.info(f"[near_optimum] exploring group '{group_key}'")

            bands_record = []
            for band_type in ("min", "max"):
                match band_type:
                    case "min":
                        self.logger.info(f"[near_optimum] finding MIN for group '{group_key}'")
                    case "max":
                        self.logger.info(f"[near_optimum] finding MAX for group '{group_key}'")

                args = (
                    self.get_differential_evolution_args(),
                    self.broad_optimum_var_info,
                    group_key,
                    band_lcoe_max,
                    idx_list,
                    evaluation_records,
                    band_type,
                )

                result = self.run_differential_evolution(broad_optimum_objective, args)

                bands_record.append(result.x.copy())

            bands[group_key] = tuple(bands_record)

        write_broad_optimum_records(self.scenario_name, evaluation_records, self.broad_optimum_var_info)
        write_broad_optimum_bands(
            self.scenario_name,
            self.broad_optimum_var_info,
            bands,
            self.get_differential_evolution_args(),
            band_lcoe_max,
            groups,
        )
        self.near_optimal_records = evaluation_records
        self.near_optimal_bands = bands
        return bands

    def diversify(self) -> None:
        optimal_x = self.decision_x0 if self.decision_x0 is not None else getattr(self.result, "x", None)
        if optimal_x is None:
            raise ValueError("Diversify requires an optimal solution from a prior single_time run.")

        if self.optimal_lcoe is None:
            self.optimal_lcoe = Solution(optimal_x, *self.get_differential_evolution_args()).evaluate().lcoe
        band_lcoe_max = self.optimal_lcoe * (1 + self.config.near_optimal_tol)

        rendered_optimal = render_candidates(self.broad_optimum_var_info, optimal_x.reshape(-1, 1))[:, 0]
        archive_scaled = [np.zeros_like(rendered_optimal)]
        diversify_records = []
        scale_floor = 1e-3
        scale = np.maximum(np.abs(rendered_optimal), scale_floor)
        novelty_threshold = self.config.novelty_threshold
        k_neighbors = self.config.novelty_k

        # Seed archive with feasible points from near_optimum if available
        if getattr(self, "near_optimal_records", None):
            for _, _, _, _, _, _, candidate_x in self.near_optimal_records:
                scaled = (candidate_x - rendered_optimal) / scale
                archive_scaled.append(scaled)

        args = (
            self.get_differential_evolution_args(),
            self.broad_optimum_var_info,
            optimal_x,
            rendered_optimal,
            scale_floor,
            archive_scaled,
            diversify_records,
            band_lcoe_max,
            novelty_threshold,
            k_neighbors,
        )

        self.run_differential_evolution(diversify_objective, args)
        write_diversify_records(self.scenario_name, self.broad_optimum_var_info, diversify_records)

    def capacity_expansion(self):
        pass

    def evaluate(self) -> None:
        if self.config.type == "single_time":
            self.single_time()
        elif self.config.type == "near_optimum":
            self.find_near_optimal_band()
        elif self.config.type == "diversify":
            self.single_time()
            if self.result is not None and getattr(self.result, "x", None) is not None:
                self.decision_x0 = self.result.x
            if self.decision_x0 is None:
                loaded = self.load_optimal_x()
                if loaded is not None:
                    self.decision_x0 = loaded
            self.find_near_optimal_band()
            self.diversify()
        elif self.config.type == "capacity_expansion":
            self.capacity_expansion()
        else:
            raise Exception(
                "Model type in config must be 'single_time' or 'capacity_expansion' or 'near_optimum' or 'diversify'"
            )


def callback(intermediate_result: OptimizeResult) -> None:
    results_dir = os.path.join("results", "temp")
    os.makedirs(results_dir, exist_ok=True)

    # Save best solution from last iteration
    with open(os.path.join(results_dir, "callback.csv"), "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(intermediate_result.x))

    if SAVE_POPULATION:
        # Save population from last iteration
        if hasattr(intermediate_result, "population"):
            with open(os.path.join(results_dir, "population.csv"), "a", newline="") as f:
                writer = csv.writer(f)
                for individual in intermediate_result.population:
                    writer.writerow(list(individual))

            with open(os.path.join(results_dir, "latest_population.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                for individual in intermediate_result.population:
                    writer.writerow(list(individual))

        # Save population energies from last iteration
        if hasattr(intermediate_result, "population_energies"):
            with open(os.path.join(results_dir, "population_energies.csv"), "a", newline="") as f:
                writer = csv.writer(f)
                for energy in intermediate_result.population_energies:
                    writer.writerow([energy])
