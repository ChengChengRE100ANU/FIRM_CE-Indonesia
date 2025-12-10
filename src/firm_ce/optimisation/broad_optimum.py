import csv
import os
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from firm_ce.common.constants import PENALTY_MULTIPLIER
from firm_ce.common.typing import (
    BandCandidates_Type,
    BroadOptimumVars_Type,
    EvaluationRecord_Type,
)
from firm_ce.optimisation.single_time import Solution, parallel_wrapper
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.topology import Network_InstanceType


def near_optimum_path(root: str, scenario_name: str):
    base = os.path.join("results", root, scenario_name)
    os.makedirs(base, exist_ok=True)
    return base


def create_broad_optimum_vars_record(
    candidate_x_idx: int,
    asset_name: str,
    near_optimum_check: bool,
    near_optimum_group: str,
    var_kind: str,
    duration: float,
    initial_power: float,
    linked_power_idx: int,
) -> BroadOptimumVars_Type:
    return (
        candidate_x_idx,
        asset_name,
        near_optimum_check,
        near_optimum_group,
        var_kind,
        duration,
        initial_power,
        linked_power_idx,
    )


def build_broad_optimum_var_info(
    fleet: Fleet_InstanceType, network: Network_InstanceType
) -> List[BroadOptimumVars_Type]:
    """create a list of records mapping each decision variable index to:
    - its name
    - near_optimum on or off
    - its group key (to aggregate)"""

    broad_optimum_var_info = []

    for generator in fleet.generators.values():
        broad_optimum_var_info.append(
            create_broad_optimum_vars_record(
                generator.candidate_x_idx,
                generator.name,
                generator.near_optimum_check,
                generator.group,
                "power",
                0.0,
                0.0,
                -1,
            )
        )

    for storage in fleet.storages.values():
        power_name = f"{storage.name}_power"
        power_group = f"{storage.group}_power" if storage.group else power_name
        broad_optimum_var_info.append(
            create_broad_optimum_vars_record(
                storage.candidate_p_x_idx,
                power_name,
                storage.near_optimum_check,
                power_group,
                "power",
                float(storage.duration),
                float(storage.initial_power_capacity),
                -1,
            )
        )

    for storage in fleet.storages.values():
        energy_name = f"{storage.name}_energy"
        energy_group = f"{storage.group}_energy" if storage.group else energy_name
        broad_optimum_var_info.append(
            create_broad_optimum_vars_record(
                storage.candidate_e_x_idx,
                energy_name,
                storage.near_optimum_check,
                energy_group,
                "energy",
                float(storage.duration),
                float(storage.initial_power_capacity),
                storage.candidate_p_x_idx,
            )
        )

    for line in network.major_lines.values():
        broad_optimum_var_info.append(
            create_broad_optimum_vars_record(
                line.candidate_x_idx,
                line.name,
                line.near_optimum_check,
                line.group,
                "power",
                0.0,
                0.0,
                -1,
            )
        )
    return broad_optimum_var_info


def create_evaluation_record(
    group_key: str,
    band_type: str,
    population_lcoes: List[float],
    de_population_penalties: List[float],
    band_population_penalties: List[float],
    band_population_candidates: List[List[float]],
    solution_index: int,
    target_group_var_sum: float | None = None,
) -> EvaluationRecord_Type:
    return (
        group_key,
        band_type,
        target_group_var_sum if target_group_var_sum else "N/A",
        float(population_lcoes[solution_index]),
        float(de_population_penalties[solution_index]),
        float(band_population_penalties[solution_index]),
        band_population_candidates[:, solution_index].copy(),
    )


def broad_optimum_objective(
    band_population_candidates: List[List[float]],  # 2-D array to allow vectorized DE
    differential_evolution_args,
    broad_optimum_var_info: List[BroadOptimumVars_Type],
    group_key: str,
    band_lcoe_max: float,
    bo_group_orders: List[int],
    evaluation_records: List[EvaluationRecord_Type],
    band_type: str,
) -> float:

    _, population_lcoes, de_population_penalties = parallel_wrapper(
        band_population_candidates, *differential_evolution_args
    )
    rendered_candidates = render_candidates(broad_optimum_var_info, np.asarray(band_population_candidates))
    group_var_sums = rendered_candidates[bo_group_orders, :].sum(axis=0)
    band_population_penalties = np.maximum(0, population_lcoes - band_lcoe_max) * PENALTY_MULTIPLIER

    match band_type:
        case "min" | "max":
            for candidate_x in range(band_population_candidates.shape[1]):
                if not (
                    de_population_penalties[candidate_x] <= 0.001 and band_population_penalties[candidate_x] <= 0.001
                ):
                    continue
                evaluation_records.append(
                    create_evaluation_record(
                        group_key,
                        band_type,
                        population_lcoes,
                        de_population_penalties,
                        band_population_penalties,
                        rendered_candidates,
                        candidate_x,
                    )
                )

    match band_type:
        case "min":
            return band_population_penalties + de_population_penalties + group_var_sums
        case "max":
            return band_population_penalties + de_population_penalties - group_var_sums
        case _:
            return None


def write_broad_optimum_records(
    scenario_name: str,
    evaluation_records: List[EvaluationRecord_Type],
    broad_optimum_var_info: List[BroadOptimumVars_Type],
) -> None:
    space_dir = near_optimum_path("near_optimum", scenario_name)

    space_path = os.path.join(space_dir, "near_optimal_space.csv")
    with open(space_path, "w", newline="") as f_space:
        writer_space = csv.writer(f_space)
        writer_space.writerow(
            [
                "Group",
                "Band_Type",
                "LCOE [$/MWh]",
                "Operational_Penalty",
                "Band_Penalty",
                *[f"{asset_name}" for _, asset_name, _, _, *_ in broad_optimum_var_info],
            ]
        )
        for group, band_type, _, lcoe, de_penalty, band_penalty, candidate_x in evaluation_records:
            writer_space.writerow([group, band_type, lcoe, de_penalty, band_penalty, *candidate_x])
    return None


def get_broad_optimum_bands_path(scenario_name: str) -> str:
    space_dir = near_optimum_path("near_optimum", scenario_name)
    return os.path.join(space_dir, "near_optimal_bands.csv")


def write_broad_optimum_bands(
    scenario_name: str,
    broad_optimum_var_info: List[BroadOptimumVars_Type],
    bands: BandCandidates_Type,
    de_args,
    band_lcoe_max: float,
    groups: Dict[str, List[int]],
) -> None:
    bands_path = get_broad_optimum_bands_path(scenario_name)
    near_optimal_asset_names = [
        asset_name for _, asset_name, near_optimum_check, _, *_ in broad_optimum_var_info if near_optimum_check
    ]
    names_to_columns = {name: col for col, name in enumerate(near_optimal_asset_names)}

    with open(bands_path, "w", newline="") as f_bands:
        writer_bands = csv.writer(f_bands)
        header = [
            "Group",
            "Band_Type",
            "LCOE [$/MWh]",
            "Operational_Penalty",
            "Band_Penalty",
        ] + near_optimal_asset_names
        writer_bands.writerow(header)

        for group, (candidate_x_min, candidate_x_max) in bands.items():
            for band_type, candidate_x in (("min", candidate_x_min), ("max", candidate_x_max)):
                solution = Solution(candidate_x, *de_args)
                solution.evaluate()
                band_penalty = max(0, solution.lcoe - band_lcoe_max) * PENALTY_MULTIPLIER
                row = [group, band_type, solution.lcoe, solution.penalties, band_penalty]
                vals = [""] * len(near_optimal_asset_names)
                rendered = render_candidates(
                    broad_optimum_var_info, np.asarray(candidate_x, dtype=np.float64).reshape(-1, 1)
                )[:, 0]
                for candidate_x_idx in groups[group]:
                    _, asset_name, _, _, *_ = broad_optimum_var_info[candidate_x_idx]
                    col = names_to_columns[asset_name]
                    vals[col] = rendered[candidate_x_idx]
                writer_bands.writerow(row + vals)
    return None


def create_groups_dict(broad_optimum_var_info):
    groups = {}
    for record in broad_optimum_var_info:
        candidate_x_idx, _, near_optimum_check, group, *_ = record

        if not near_optimum_check:
            continue
        key = group or candidate_x_idx
        groups.setdefault(key, []).append(candidate_x_idx)
    return groups


def render_candidates(
    broad_optimum_var_info: List[BroadOptimumVars_Type], candidates: NDArray[np.float64]
) -> NDArray[np.float64]:
    rendered = np.zeros_like(candidates)
    for record in broad_optimum_var_info:
        idx, _, _, _, var_kind, duration, initial_power, linked_power_idx = record
        if var_kind == "energy" and duration > 0 and linked_power_idx >= 0:
            rendered[idx, :] = (initial_power + candidates[linked_power_idx, :]) * duration
        else:
            rendered[idx, :] = candidates[idx, :]
    return rendered


# ---------------- Diversify search (distance-based) ---------------- #

DiversifyRecord = Tuple[float, float, float, float, NDArray[np.float64]]


def get_diversify_csv_path(scenario_name: str) -> str:
    diversify_dir = near_optimum_path("diversify", scenario_name)
    return os.path.join(diversify_dir, "diversify_space.csv")


def write_diversify_records(
    scenario_name: str,
    broad_optimum_var_info: List[BroadOptimumVars_Type],
    diversify_records: List[DiversifyRecord],
) -> None:
    csv_path = get_diversify_csv_path(scenario_name)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "LCOE [$/MWh]",
                "Operational_Penalty",
                "Band_Penalty",
                "Scaled_Novelty",
                *[f"{asset_name}" for _, asset_name, _, _, *_ in broad_optimum_var_info],
            ]
        )
        for lcoe, op_pen, band_pen, distance, candidate_x in diversify_records:
            writer.writerow([lcoe, op_pen, band_pen, distance, *candidate_x])
    return None


def _scale_candidates(
    candidates: NDArray[np.float64],
    optimal_x: NDArray[np.float64],
    scale_floor: float,
) -> NDArray[np.float64]:
    scale = np.maximum(np.abs(optimal_x), scale_floor)
    return (candidates - optimal_x[:, None]) / scale[:, None]


def _novelty_score(
    archive_scaled: List[NDArray[np.float64]],
    candidate_scaled: NDArray[np.float64],
    k_neighbors: int,
) -> float:
    if not archive_scaled:
        return float(np.linalg.norm(candidate_scaled))
    stacked = np.stack(archive_scaled)
    distances = np.linalg.norm(stacked - candidate_scaled, axis=1)
    k = min(k_neighbors, len(distances))
    nearest = np.partition(distances, k - 1)[:k]
    return float(np.mean(nearest))


def diversify_objective(
    band_population_candidates: List[List[float]],
    differential_evolution_args,
    broad_optimum_var_info: List[BroadOptimumVars_Type],
    optimal_x: NDArray[np.float64],
    rendered_optimal_x: NDArray[np.float64],
    scale_floor: float,
    archive_scaled: List[NDArray[np.float64]],
    diversify_records: List[DiversifyRecord],
    band_lcoe_max: float,
    novelty_threshold: float,
    k_neighbors: int,
) -> float:
    candidates = np.asarray(band_population_candidates)
    rendered_candidates = render_candidates(broad_optimum_var_info, candidates)
    _, population_lcoes, de_population_penalties = parallel_wrapper(candidates, *differential_evolution_args)
    band_population_penalties = np.maximum(0, population_lcoes - band_lcoe_max) * PENALTY_MULTIPLIER

    scaled_candidates = _scale_candidates(rendered_candidates, rendered_optimal_x, scale_floor)
    distances = np.zeros(scaled_candidates.shape[1], dtype=np.float64)

    for idx in range(scaled_candidates.shape[1]):
        band_penalty = band_population_penalties[idx]
        op_penalty = de_population_penalties[idx]
        if op_penalty > 0.001 or band_penalty > 0.001:
            continue

        candidate_scaled = scaled_candidates[:, idx]
        novelty = _novelty_score(archive_scaled, candidate_scaled, k_neighbors)
        distances[idx] = novelty

        if novelty < novelty_threshold:
            continue

        archive_scaled.append(candidate_scaled.copy())
        diversify_records.append(
            (
                float(population_lcoes[idx]),
                float(op_penalty),
                float(band_penalty),
                float(novelty),
                rendered_candidates[:, idx].copy(),
            )
        )

    return band_population_penalties + de_population_penalties - distances
