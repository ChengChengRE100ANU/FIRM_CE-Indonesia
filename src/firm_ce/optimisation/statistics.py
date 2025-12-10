import os
import re
import shutil
import time

import numpy as np
from numpy.typing import NDArray

from firm_ce.common.constants import SAVE_POPULATION
from firm_ce.common.helpers import safe_divide
from firm_ce.fast_methods import fleet_m, generator_m, ltcosts_m, network_m, static_m
from firm_ce.io.file_manager import ResultFile
from firm_ce.optimisation.single_time import Solution
from firm_ce.system.components import Fleet_InstanceType
from firm_ce.system.parameters import ScenarioParameters_InstanceType
from firm_ce.system.topology import Network_InstanceType


class Statistics:
    @staticmethod
    def initialise_header_rows(col_count: int, row_labels_flag: bool = True) -> NDArray[object]:
        if row_labels_flag:
            header = np.empty((5, col_count + 1), dtype=object)
            header[:, 0] = np.array(
                [
                    "Asset Name",
                    "Asset Type",
                    "Asset ID",
                    "Column Name",
                    "Column Units",
                ]
            )
        else:
            header = np.empty((5, col_count), dtype=object)
        return header

    def __init__(
        self,
        x_candidate: NDArray[np.float64],
        parameters_static: ScenarioParameters_InstanceType,
        fleet_static: Fleet_InstanceType,
        network_static: Network_InstanceType,
        solution_results_directory: str,
        scenario_name: str,
        balancing_type: str,
        fixed_costs_threshold: float,
        copy_callback: bool = True,
    ):
        self.solution = Solution(
            x_candidate, parameters_static, fleet_static, network_static, balancing_type, fixed_costs_threshold
        )
        start_time = time.time()
        self.solution.evaluate()
        end_time = time.time()
        print(f"Statistics solution evaluation time: {end_time - start_time:.4f} seconds")
        print(f"{scenario_name} LCOE: {self.solution.lcoe} [$/MWh], " f"Penalties: {self.solution.penalties}")

        self.results_directory = self.create_solution_directory(
            solution_results_directory, scenario_name + "_" + balancing_type
        )
        self.copy_temp_files(copy_callback)
        self.result_files = None

        self.full_intervals_count = self.solution.static.block_lengths.sum()
        self.block_first_intervals, self.block_last_intervals = static_m.get_block_intervals(
            self.solution.static.block_lengths
        )

    def create_solution_directory(self, result_directory: str, solution_name: str) -> str:
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", solution_name)
        solution_dir = os.path.join(result_directory, safe_name)
        os.makedirs(solution_dir, exist_ok=True)
        return solution_dir

    def copy_temp_files(self, copy_callback: bool) -> None:
        if copy_callback:
            temp_dir = os.path.join("results", "temp")
            shutil.copy(os.path.join(temp_dir, "callback.csv"), os.path.join(self.results_directory, "callback.csv"))

            if SAVE_POPULATION:
                shutil.copy(
                    os.path.join(temp_dir, "latest_population.csv"),
                    os.path.join(self.results_directory, "latest_population.csv"),
                )
                shutil.copy(
                    os.path.join(temp_dir, "population.csv"), os.path.join(self.results_directory, "population.csv")
                )
                shutil.copy(
                    os.path.join(temp_dir, "population_energies.csv"),
                    os.path.join(self.results_directory, "population_energies.csv"),
                )
        return None

    def expand_block_data(self, block_array: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.full_intervals_count == self.solution.static.intervals_count:
            return block_array

        expanded_array = np.zeros(self.full_intervals_count, dtype=np.float64)
        for block in range(self.solution.static.intervals_count):
            for idx in range(self.block_first_intervals[block], self.block_last_intervals[block]):
                expanded_array[idx] = block_array[block]
        return expanded_array

    def generate_result_files(self) -> None:
        self.result_files = {
            "capacities": self.generate_capacities_file(),
            "component_costs": self.generate_component_costs_file(),
            "energy_balance_ASSETS": self.generate_energy_balance_file("assets"),
            "energy_balance_NODES": self.generate_energy_balance_file("nodes"),
            "energy_balance_NETWORK": self.generate_energy_balance_file("network"),
            "levelised_costs": self.generate_levelised_costs_file(),
            "summary": self.generate_summary_file(),
            "x": self.generate_x_file(),
        }
        return None

    def write_results(self) -> None:
        if not self.solution.evaluated:
            print("WARNING: Solution must be evaluated before writing statistics.")
        for result_file in self.result_files.values():
            result_file.write()
        return None

    def get_col_count(self, result_file: str) -> int:
        col_count = 0
        match result_file:
            case "capacities":
                col_count = (
                    len(self.solution.fleet.generators)
                    + 2 * len(self.solution.fleet.storages)
                    + len(self.solution.network.major_lines)
                    + len(self.solution.network.minor_lines)
                )
            case "component_costs":
                col_count = (
                    len(self.solution.fleet.generators)
                    + len(self.solution.fleet.storages)
                    + len(self.solution.network.major_lines)
                    + len(self.solution.network.minor_lines)
                )
            case "energy_balance_ASSETS":
                col_count = (
                    3 * len(self.solution.network.nodes)
                    + len(self.solution.fleet.generators)
                    + 2 * len(self.solution.fleet.storages)
                    + len(self.solution.network.major_lines)
                    + fleet_m.count_generator_unit_type(self.solution.fleet, "flexible")
                )
            case "energy_balance_NODES":
                col_count = 10 * len(self.solution.network.nodes) + len(self.solution.network.major_lines)
            case "energy_balance_NETWORK":
                col_count = 10
            case "levelised_costs":
                col_count = (
                    6
                    + 2 * len(self.solution.fleet.generators)
                    + 2 * len(self.solution.fleet.storages)
                    + 2 * len(self.solution.network.major_lines)
                    + 2 * len(self.solution.network.minor_lines)
                )
            case "summary":
                col_count = (
                    3 * len(self.solution.network.nodes)
                    + len(self.solution.fleet.generators)
                    + len(self.solution.fleet.storages)
                    + len(self.solution.network.major_lines)
                )
        return col_count

    def generate_capacities_file(self) -> ResultFile:
        col_count = self.get_col_count("capacities")
        header = self.initialise_header_rows(col_count)

        data_array = np.empty((4, col_count + 1), dtype=object)
        data_array[:, 0] = np.array(
            [
                "Total Capacity",
                "New Build Capacity",
                "Min Build",
                "Max Build",
            ],
            dtype=object,
        )

        col = 1
        for generator in self.solution.fleet.generators.values():
            header[:, col] = np.array(
                [generator.name, "Generator", str(generator.id), "Power Capacity", "[GW]"], dtype=object
            )
            data_array[0, col] = round(generator.capacity, 3)
            data_array[1, col] = round(generator.new_build, 3)
            data_array[2, col] = round(generator.min_build, 3)
            data_array[3, col] = round(generator.max_build, 3)
            col += 1

        for storage in self.solution.fleet.storages.values():
            header[:, col] = np.array(
                [storage.name, "Storage", str(storage.id), "Power Capacity", "[GW]"], dtype=object
            )
            data_array[0, col] = round(storage.power_capacity, 3)
            data_array[1, col] = round(storage.new_build_p, 3)
            data_array[2, col] = round(storage.min_build_p, 3)
            data_array[3, col] = round(storage.max_build_p, 3)
            col += 1

        for storage in self.solution.fleet.storages.values():
            header[:, col] = np.array(
                [storage.name, "Storage", str(storage.id), "Energy Capacity", "[GWh]"], dtype=object
            )
            data_array[0, col] = round(storage.energy_capacity, 3)
            data_array[1, col] = round(storage.new_build_e, 3)
            data_array[2, col] = round(storage.min_build_e, 3)
            data_array[3, col] = round(storage.max_build_e, 3)
            col += 1

        for line in self.solution.network.major_lines.values():
            header[:, col] = np.array([line.name, "Major Line", str(line.id), "Power Capacity", "[GW]"], dtype=object)
            data_array[0, col] = round(line.capacity, 3)
            data_array[1, col] = round(line.new_build, 3)
            data_array[2, col] = round(line.min_build, 3)
            data_array[3, col] = round(line.max_build, 3)
            col += 1

        for line in self.solution.network.minor_lines.values():
            header[:, col] = np.array([line.name, "Minor Line", str(line.id), "Power Capacity", "[GW]"], dtype=object)
            data_array[0, col] = round(line.capacity, 3)
            data_array[1, col] = round(line.new_build, 3)
            data_array[2, col] = round(line.min_build, 3)
            data_array[3, col] = round(line.max_build, 3)
            col += 1

        result_file = ResultFile("capacities", self.results_directory, header, data_array, decimals=None)
        return result_file

    def generate_component_costs_file(self) -> ResultFile:
        col_count = self.get_col_count("component_costs")
        header = self.initialise_header_rows(col_count)

        data_array = np.zeros((4, col_count + 1), dtype=object)
        data_array[:, 0] = np.array(["Annualised Build", "Fixed O&M", "Variable O&M", "Fuel"], dtype=object)

        col = 1
        for generator in self.solution.fleet.generators.values():
            header[:, col] = np.array(
                [generator.name, "Generator", str(generator.id), "Total Cost", "[$]"], dtype=object
            )
            data_array[0, col] = generator.lt_costs.annualised_build
            data_array[1, col] = generator.lt_costs.fom
            data_array[2, col] = generator.lt_costs.vom
            data_array[3, col] = generator.lt_costs.fuel
            col += 1

        for storage in self.solution.fleet.storages.values():
            header[:, col] = np.array([storage.name, "Storage", str(storage.id), "Total Cost", "[$]"], dtype=object)
            data_array[0, col] = storage.lt_costs.annualised_build
            data_array[1, col] = storage.lt_costs.fom
            data_array[2, col] = storage.lt_costs.vom
            data_array[3, col] = storage.lt_costs.fuel
            col += 1

        for line in self.solution.network.major_lines.values():
            header[:, col] = np.array([line.name, "Major Line", str(line.id), "Total Cost", "[$]"], dtype=object)
            data_array[0, col] = line.lt_costs.annualised_build
            data_array[1, col] = line.lt_costs.fom
            data_array[2, col] = line.lt_costs.vom
            data_array[3, col] = line.lt_costs.fuel
            col += 1

        for line in self.solution.network.minor_lines.values():
            header[:, col] = np.array([line.name, "Minor Line", str(line.id), "Total Cost", "[$]"], dtype=object)
            data_array[0, col] = line.lt_costs.annualised_build
            data_array[1, col] = line.lt_costs.fom
            data_array[2, col] = line.lt_costs.vom
            data_array[3, col] = line.lt_costs.fuel
            col += 1

        result_file = ResultFile("component_costs", self.results_directory, header, data_array, decimals=None)
        return result_file

    def generate_energy_balance_file(self, aggregation_type: str) -> ResultFile:
        match aggregation_type:
            case "assets":
                col_count = self.get_col_count("energy_balance_ASSETS")
            case "nodes":
                col_count = self.get_col_count("energy_balance_NODES")
            case "network":
                col_count = self.get_col_count("energy_balance_NETWORK")

        header = self.initialise_header_rows(col_count, row_labels_flag=False)
        data_array = np.zeros((self.full_intervals_count, col_count), dtype=np.float64)

        col = 0
        match aggregation_type:
            case "assets":
                for node in self.solution.network.nodes.values():
                    header[:, col] = np.array([node.name, "Node", str(node.id), "Demand", "[MW]"], dtype=object)
                    data_array[:, col] = self.expand_block_data(node.data * 1000)
                    col += 1

                for generator in self.solution.fleet.generators.values():
                    header[:, col] = np.array(
                        [generator.name, "Generator", str(generator.id), "Dispatch", "[MW]"], dtype=object
                    )
                    match generator.unit_type:
                        case "flexible":
                            data_array[:, col] = self.expand_block_data(generator.dispatch_power * 1000)
                        case _:
                            data_array[:, col] = self.expand_block_data(generator.data * generator.capacity * 1000)
                    col += 1

                for storage in self.solution.fleet.storages.values():
                    header[:, col] = np.array(
                        [storage.name, "Storage", str(storage.id), "Dispatch", "[MW]"], dtype=object
                    )
                    data_array[:, col] = self.expand_block_data(storage.dispatch_power * 1000)
                    col += 1

                for generator in self.solution.fleet.generators.values():
                    if generator.unit_type == "flexible":
                        header[:, col] = np.array(
                            [generator.name, "Generator", str(generator.id), "Remaining Energy", "[MWh]"], dtype=object
                        )
                        data_array[:, col] = self.expand_block_data(generator.remaining_energy * 1000)
                        col += 1

                for storage in self.solution.fleet.storages.values():
                    header[:, col] = np.array(
                        [storage.name, "Storage", str(storage.id), "Stored Energy", "[MWh]"], dtype=object
                    )
                    data_array[:, col] = self.expand_block_data(storage.stored_energy * 1000)
                    col += 1

                for node in self.solution.network.nodes.values():
                    header[:, col] = np.array([node.name, "Node", str(node.id), "Spillage", "[MW]"], dtype=object)
                    data_array[:, col] = self.expand_block_data(node.spillage * 1000)
                    col += 1

                for node in self.solution.network.nodes.values():
                    header[:, col] = np.array([node.name, "Node", str(node.id), "Deficit", "[MW]"], dtype=object)
                    data_array[:, col] = self.expand_block_data(node.deficits * 1000)
                    col += 1

                for line in self.solution.network.major_lines.values():
                    header[:, col] = np.array([line.name, "Major Line", str(line.id), "Flow", "[MW]"], dtype=object)
                    data_array[:, col] = self.expand_block_data(line.flows * 1000)
                    col += 1

            case "nodes":
                for node in self.solution.network.nodes.values():
                    header[:, col] = np.array([node.name, "Node", str(node.id), "Demand", "[MW]"], dtype=object)
                    data_array[:, col] = self.expand_block_data(node.data * 1000)
                    col += 1

                for header_item, units in [
                    ["Solar", "[MW]"],
                    ["Wind", "[MW]"],
                    ["Baseload", "[MW]"],
                    ["Flexible Dispatch", "[MW]"],
                    ["Storage Dispatch", "[MW]"],
                    ["Flexible Remaining", "[MWh]"],
                    ["Stored Energy", "[MWh]"],
                ]:
                    for node in self.solution.network.nodes.values():
                        header[:, col] = np.array([node.name, "Node", str(node.id), header_item, units], dtype=object)
                        col += 1

                for generator in self.solution.fleet.generators.values():
                    match generator.unit_type:
                        case "solar":
                            column_idx = len(self.solution.network.nodes) + generator.node.order
                            data_array[:, column_idx] += self.expand_block_data(
                                generator.data * generator.capacity * 1000
                            )
                        case "wind":
                            column_idx = 2 * len(self.solution.network.nodes) + generator.node.order
                            data_array[:, column_idx] += self.expand_block_data(
                                generator.data * generator.capacity * 1000
                            )
                        case "baseload":
                            column_idx = 3 * len(self.solution.network.nodes) + generator.node.order
                            data_array[:, column_idx] += self.expand_block_data(
                                generator.data * generator.capacity * 1000
                            )
                        case "flexible":
                            column_idx = 4 * len(self.solution.network.nodes) + generator.node.order
                            data_array[:, column_idx] += self.expand_block_data(generator.dispatch_power * 1000)

                for storage in self.solution.fleet.storages.values():
                    column_idx = 5 * len(self.solution.network.nodes) + storage.node.order
                    data_array[:, column_idx] += self.expand_block_data(storage.dispatch_power * 1000)

                for generator in self.solution.fleet.generators.values():
                    if generator.unit_type == "flexible":
                        column_idx = 6 * len(self.solution.network.nodes) + generator.node.order
                        data_array[:, column_idx] += self.expand_block_data(generator.remaining_energy * 1000)

                for storage in self.solution.fleet.storages.values():
                    column_idx = 7 * len(self.solution.network.nodes) + storage.node.order
                    data_array[:, column_idx] += self.expand_block_data(storage.stored_energy * 1000)
                for node in self.solution.network.nodes.values():
                    header[:, col] = np.array([node.name, "Node", str(node.id), "Spillage", "[MW]"], dtype=object)
                    data_array[:, col] += self.expand_block_data(node.spillage * 1000)
                    col += 1

                for node in self.solution.network.nodes.values():
                    header[:, col] = np.array([node.name, "Node", str(node.id), "Deficit", "[MW]"], dtype=object)
                    data_array[:, col] += self.expand_block_data(node.deficits * 1000)
                    col += 1

                for line in self.solution.network.major_lines.values():
                    header[:, col] = np.array([line.name, "Major Line", str(line.id), "Flow", "[MW]"], dtype=object)
                    data_array[:, col] += self.expand_block_data(line.flows * 1000)
                    col += 1

            case "network":
                for header_item, units in [
                    ["Demand", "[MW]"],
                    ["Solar", "[MW]"],
                    ["Wind", "[MW]"],
                    ["Baseload", "[MW]"],
                    ["Flexible Dispatch", "[MW]"],
                    ["Storage Dispatch", "[MW]"],
                    ["Flexible Remaining", "[MWh]"],
                    ["Stored Energy", "[MWh]"],
                    ["Spillage", "[MW]"],
                    ["Deficit", "[MW]"],
                ]:
                    header[:, col] = np.array(["Network", "Network", 0, header_item, units], dtype=object)
                    col += 1
                for node in self.solution.network.nodes.values():
                    data_array[:, 0] += self.expand_block_data(node.data * 1000)
                    data_array[:, 8] += self.expand_block_data(node.spillage * 1000)
                    data_array[:, 9] += self.expand_block_data(node.deficits * 1000)

                for generator in self.solution.fleet.generators.values():
                    match generator.unit_type:
                        case "solar":
                            data_array[:, 1] += self.expand_block_data(generator.data * generator.capacity * 1000)
                        case "wind":
                            data_array[:, 2] += self.expand_block_data(generator.data * generator.capacity * 1000)
                        case "baseload":
                            data_array[:, 3] += self.expand_block_data(generator.data * generator.capacity * 1000)
                        case "flexible":
                            data_array[:, 4] += self.expand_block_data(generator.dispatch_power * 1000)
                            data_array[:, 6] += self.expand_block_data(generator.remaining_energy * 1000)

                for storage in self.solution.fleet.storages.values():
                    data_array[:, 5] += self.expand_block_data(storage.dispatch_power * 1000)
                    data_array[:, 7] += self.expand_block_data(storage.stored_energy * 1000)

        result_file = ResultFile(
            f"energy_balance_{aggregation_type.upper()}", self.results_directory, header, data_array, decimals=0
        )
        return result_file

    def generate_levelised_costs_file(self) -> ResultFile:
        col_count = self.get_col_count("levelised_costs")
        header = self.initialise_header_rows(col_count)
        header[:, 1:7] = np.array(
            [
                ["Scenario Total", "", "", "LCOE", "[$/MWh]"],
                ["Scenario Total", "", "", "LCOG", "[$/MWh]"],
                ["Scenario Total", "", "", "LCOB", "[$/MWh]"],
                ["Scenario Total", "", "", "LCOB_storage", "[$/MWh]"],
                ["Scenario Total", "", "", "LCOB_transmission", "[$/MWh]"],
                ["Scenario Total", "", "", "LCOB_losses_spillage", "[$/MWh]"],
            ]
        ).T

        data_array = np.zeros((1, col_count + 1), dtype=object)
        data_array[:, 0] = np.array(["Levelised Cost"], dtype=object)
        data_array[:, 1:7] = 0.0

        total_energy = (
            np.abs(
                sum(self.solution.static.year_energy_demand) - network_m.calculate_lt_line_losses(self.solution.network)
            )
            * 1000
        )
        total_generation = (
            sum(
                sum(generator.dispatch_power)
                for generator in self.solution.fleet.generators.values()
                if generator.unit_type == "flexible"
            )
            * self.solution.static.resolution
            * 1000
        )
        total_generation += (
            sum(
                sum(generator.data * generator.capacity)
                for generator in self.solution.fleet.generators.values()
                if generator.unit_type != "flexible"
            )
            * self.solution.static.resolution
            * 1000
        )

        # LCOE and Total Levelised Values
        col = 7
        for generator in self.solution.fleet.generators.values():
            header[:, col] = np.array([generator.name, "Generator", str(generator.id), "LCOE", "[$/MWh]"], dtype=object)
            if generator_m.check_unit_type(generator, "flexible"):
                data_array[0, col] = round(ltcosts_m.get_total(generator.lt_costs) / total_energy, 2)
            else:
                data_array[0, col] = round(ltcosts_m.get_total(generator.lt_costs) / total_energy, 2)
            data_array[0, 1] += ltcosts_m.get_total(generator.lt_costs)
            data_array[0, 2] += ltcosts_m.get_total(generator.lt_costs)
            col += 1

        for storage in self.solution.fleet.storages.values():
            header[:, col] = np.array([storage.name, "Storage", str(storage.id), "LCOE", "[$/MWh]"], dtype=object)
            data_array[0, col] = round(ltcosts_m.get_total(storage.lt_costs) / total_energy, 2)
            data_array[0, 1] += ltcosts_m.get_total(storage.lt_costs)
            data_array[0, 4] += ltcosts_m.get_total(storage.lt_costs)
            col += 1

        for line in self.solution.network.major_lines.values():
            header[:, col] = np.array([line.name, "Major Line", str(line.id), "LCOE", "[$/MWh]"], dtype=object)
            data_array[0, col] = round(ltcosts_m.get_total(line.lt_costs) / total_energy, 2)
            data_array[0, 1] += ltcosts_m.get_total(line.lt_costs)
            data_array[0, 5] += ltcosts_m.get_total(line.lt_costs)
            col += 1

        for line in self.solution.network.minor_lines.values():
            header[:, col] = np.array([line.name, "Minor Line", str(line.id), "LCOE", "[$/MWh]"], dtype=object)
            data_array[0, col] = (ltcosts_m.get_total(line.lt_costs) / total_energy, 2)
            data_array[0, 1] += ltcosts_m.get_total(line.lt_costs)
            data_array[0, 5] += ltcosts_m.get_total(line.lt_costs)
            col += 1

        data_array[0, 1] = round(data_array[0, 1] / total_energy, 2)  # LCOE
        data_array[0, 2] = round(data_array[0, 2] / total_generation, 2)  # LCOG
        data_array[0, 3] = round(data_array[0, 1] - data_array[0, 2], 2)  # LCOB
        data_array[0, 4] = round(data_array[0, 4] / total_energy, 2)  # LCOB_storage
        data_array[0, 5] = round(data_array[0, 5] / total_energy, 2)  # LCOB_transmission
        data_array[0, 6] = round(data_array[0, 3] - data_array[0, 4] - data_array[0, 5], 2)  # LCOB_spillage_losses

        # LCOG, LCOS and LCOT
        for generator in self.solution.fleet.generators.values():
            header[:, col] = np.array([generator.name, "Generator", str(generator.id), "LCOG", "[$/MWh]"], dtype=object)
            if generator_m.check_unit_type(generator, "flexible"):
                data_array[0, col] = round(
                    safe_divide(
                        ltcosts_m.get_total(generator.lt_costs),
                        sum(generator.dispatch_power) * self.solution.static.resolution * 1000,
                    ),
                    2,
                )
            else:
                data_array[0, col] = round(
                    safe_divide(
                        ltcosts_m.get_total(generator.lt_costs),
                        sum(generator.data * generator.capacity) * self.solution.static.resolution * 1000,
                    ),
                    2,
                )
            col += 1

        for storage in self.solution.fleet.storages.values():
            header[:, col] = np.array(
                [storage.name, "Storage", str(storage.id), "LCOS", "[$/MWh_discharged]"], dtype=object
            )
            data_array[0, col] = round(
                safe_divide(
                    ltcosts_m.get_total(storage.lt_costs),
                    sum(np.maximum(storage.dispatch_power, 0)) * self.solution.static.resolution * 1000,
                ),
                2,
            )
            col += 1

        for line in self.solution.network.major_lines.values():
            header[:, col] = np.array([line.name, "Major Line", str(line.id), "LCOT", "[$/MWh_flow]"], dtype=object)
            data_array[0, col] = round(
                safe_divide(
                    ltcosts_m.get_total(line.lt_costs), sum(np.abs(line.flows)) * self.solution.static.resolution * 1000
                ),
                2,
            )
            col += 1

        for line in self.solution.network.minor_lines.values():
            header[:, col] = np.array([line.name, "Minor Line", str(line.id), "LCOT", "[$/MWh_flow]"], dtype=object)
            data_array[0, col] = round(
                safe_divide(
                    ltcosts_m.get_total(line.lt_costs), sum(np.abs(line.flows)) * self.solution.static.resolution * 1000
                ),
                2,
            )
            col += 1

        result_file = ResultFile("levelised_costs", self.results_directory, header, data_array, decimals=None)
        return result_file

    def calculate_annual_energies(self, arr: NDArray[np.float64], decimals: int = 3) -> float:
        annual_energies_arr = np.zeros(self.solution.static.year_count + 1, dtype=np.float64)
        for year in range(self.solution.static.year_count):
            first_t, last_t = static_m.get_year_t_boundaries(self.solution.static, year)
            annual_energies_arr[year] = round(
                sum(arr[first_t:last_t] * self.solution.static.interval_resolutions[first_t:last_t]),
                decimals,
            )
        annual_energies_arr[-1] = round(sum(arr * self.solution.static.interval_resolutions), decimals)
        return annual_energies_arr

    def generate_summary_file(self) -> ResultFile:
        col_count = self.get_col_count("summary")
        header = self.initialise_header_rows(col_count)

        row_labels = np.array(
            list(range(self.solution.static.first_year, self.solution.static.final_year + 1)) + ["Total"], dtype=object
        )

        data_array = np.zeros((len(row_labels), col_count + 1), dtype=object)
        data_array[:, 0] = row_labels

        col = 1
        for node in self.solution.network.nodes.values():
            header[:, col] = np.array([node.name, "Node", str(node.id), "Annual Demand", "[GWh]"], dtype=object)
            data_array[:, col] = self.calculate_annual_energies(node.data)
            col += 1

        for generator in self.solution.fleet.generators.values():
            header[:, col] = np.array(
                [generator.name, "Generator", str(generator.id), "Annual Generation", "[GWh]"], dtype=object
            )
            if generator_m.check_unit_type(generator, "flexible"):
                data_array[:, col] = self.calculate_annual_energies(generator.dispatch_power)
            else:
                data_array[:, col] = self.calculate_annual_energies(generator.data * generator.capacity)
            col += 1

        for storage in self.solution.fleet.storages.values():
            header[:, col] = np.array(
                [storage.name, "Storage", str(storage.id), "Annual Discharge", "[GWh]"], dtype=object
            )
            data_array[:, col] = self.calculate_annual_energies(np.maximum(storage.dispatch_power, 0))
            col += 1

        for node in self.solution.network.nodes.values():
            header[:, col] = np.array([node.name, "Node", str(node.id), "Annual Spillage", "[GWh]"], dtype=object)
            data_array[:, col] = self.calculate_annual_energies(node.spillage)
            col += 1

        for node in self.solution.network.nodes.values():
            header[:, col] = np.array([node.name, "Node", str(node.id), "Annual Deficit", "[GWh]"], dtype=object)
            data_array[:, col] = self.calculate_annual_energies(node.deficits)
            col += 1

        for line in self.solution.network.major_lines.values():
            header[:, col] = np.array([line.name, "Major Line", str(line.id), "Annual Flows", "[GWh]"], dtype=object)
            data_array[:, col] = self.calculate_annual_energies(np.abs(line.flows))
            col += 1

        result_file = ResultFile("summary", self.results_directory, header, data_array, decimals=None)
        return result_file

    def generate_x_file(self) -> ResultFile:
        result_file = ResultFile("x", self.results_directory, [], [self.solution.x], decimals=None)
        return result_file

    def dump(self):
        residual_load_header = [node.name for node in self.solution.network.nodes.values()]
        residual_load_data = np.array(
            [node.residual_load for node in self.solution.network.nodes.values()], dtype=np.float64
        ).T
        ResultFile("residual_load", self.results_directory, residual_load_header, residual_load_data).write()
        ResultFile(
            "block_lengths",
            self.results_directory,
            ["Intervals per Block"],
            self.solution.static.block_lengths.reshape(-1, 1),
        ).write()
