import os
import shutil
import time
from datetime import datetime
from typing import Dict, List

import pandas as pd

from firm_ce.common.constants import DEBUG
from firm_ce.common.helpers import parse_comma_separated
from firm_ce.common.exceptions import ValidationError
from firm_ce.io.validate import ModelData
from firm_ce.optimisation.statistics import Statistics
from firm_ce.system.parameters import ModelConfig
from firm_ce.system.scenario import Scenario
from firm_ce.common.constants import DEFAULT_PV_CAGR_CAP, DEFAULT_PV_FIRST_YEAR_CAP_GW


class Model:
    """
    Primary interface for performing a long-term energy planning optimisation using the FIRM framework.

    Notes:
    -------
    - Input configuration files are loaded into the ModelData dataclass and validated before constructing the Scenarios.
    - Time-series data files are loaded after calling the solve() method. Data files are only loaded for one Scenario at a time
    in order to manage memory. After the optimisation for a Scenario has completed and the results saved, data files are unloaded
    from the Scenario.
    - Loggers are handled separately for each Scenario. All results, including log files, are saved in the `results` folder. A
    sub-directory for the Model instance contains separate sub-directories for each Scenario instances.

    Attributes:
    -------
    config_directory (str): Filesystem path to the configuration directory. Defaults to the example `inputs/config` folder.
    data_directory (str): Filesystem path to the directory containing input data files. Defaults to the example `inputs/data`
        folder.
    config (ModelConfig): Data class containing validated model configuration used for the optimisation settings and model-level
        metadata.
    datafile_filenames_dict (Dict[str, Dict[str, str]]): Raw data imported from the `datafiles.csv` config file for the Model.
        Each row of the config file is associated with an item in all_datafiles.
    scenarios (Dict[str, Scenario]): Mapping from scenario name to initialised Scenario instances constructed from the validated
        ModelData.
    """

    def __init__(
        self, config_directory: str = "inputs/config", data_directory: str = "inputs/data", logging_flag: bool = True
    ) -> None:
        """
        Initialises a Model instance.

        Parameters:
        -------
        config_directory (str): Filesystem path to the directory containing input data files. Defaults to the example
            `inputs/data` folder.
        data_directory (str): Filesystem path to the directory containing input data files. Defaults to the example
            `inputs/data` folder.
        logging_flag (bool): If True, creates a model-level folder in `results` containing the Scenario sub-directories and
            log files. When set to false, no model-level results folder is created and the log is stored in `results/temp`
            instead (useful when generating Statistic instance directly using an initial guess).
        """
        self.config_directory = config_directory
        self.data_directory = data_directory
        self.logging_flag = logging_flag
        model_data = ModelData(
            config_directory=self.config_directory, logging_flag=logging_flag, data_directory=self.data_directory
        )

        if not model_data.validate():
            raise ValidationError(
                "Model failed validation. Check the `log.txt` and modify the config and data files to resolve errors."
            )

        self.config = ModelConfig(model_data.config)
        self.datafile_filenames_dict = model_data.datafiles
        self.scenarios = {
            model_data.scenarios[scenario_idx].get("scenario_name"): Scenario(model_data, scenario_idx)
            for scenario_idx in model_data.scenarios
        }

    def solve(self, pv_prev_total: float = 0.0, expansion_interval_years: int = 1) -> None:
        """
        Execute an optimisation for each Scenario: load datafiles, run the optimisation, generate and write results,
        then unload data before moving to the next Scenario.

        Parameters:
        -------
        None.

        Returns:
        -------
        None.

        Side-effects:
        -------
        Modification of the Scenario objects, primarily through loading exogenous time-series data files (modifying
        the Scenario.fleet.generators, Scenario.network.nodes, and Scenario.static) and the creation of a Solver instance in
        Scenario.solver. The optimisation is managed through the Solver, with jitclass attributes for the Scenario remaining
        *static* unmodified instances throughout the optimisation process. These static instances are copied in separate worker
        processes for the optimisation to create dynamic instances that are safe to modify during the optimisation. The dynamic
        instances are not actually contained in the Model instance.
        """
        if self.config.type == "capacity_expansion":
            self.capacity_expansion()
            return None

        for scenario in self.scenarios.values():
            start_time = time.time()
            start_time_str = datetime.fromtimestamp(start_time).strftime("%d/%m/%Y %H:%M:%S")
            scenario.logger.info(f"Started scenario {scenario.name} at {start_time_str}.")

            scenario.load_datafiles(self.datafile_filenames_dict, self.data_directory)
            datafile_loadtime = time.time()
            datafile_loadtime_str = datetime.fromtimestamp(datafile_loadtime).strftime("%d/%m/%Y %H:%M:%S")
            scenario.logger.info(
                f"Datafiles loaded at {datafile_loadtime_str} ({datafile_loadtime - start_time:.4f} seconds)."
            )

            de_result = scenario.solve(
                self.config,
                pv_prev_total=pv_prev_total,
                pv_cagr_cap=self.config.pv_cagr_cap,
                pv_first_year_cap_gw=self.config.pv_first_year_cap_gw,
                expansion_interval_years=expansion_interval_years,
            )

            solve_time = time.time()
            solve_time_str = datetime.fromtimestamp(solve_time).strftime("%d/%m/%Y %H:%M:%S")
            scenario.logger.info(
                f"Optimisation completed at {solve_time_str} ({(solve_time - datafile_loadtime)/(60*60):.4f} hours)."
            )

            if self.config.type in ("single_time", "diversify"):
                scenario.statistics = Statistics(
                    de_result.x,
                    scenario.static,
                    scenario.fleet,
                    scenario.network,
                    scenario.results_dir,
                    scenario.name,
                    self.config.balancing_type,
                    self.config.fixed_costs_threshold,
                    True,
                    pv_prev_total=pv_prev_total,
                    pv_cagr_cap=self.config.pv_cagr_cap,
                    pv_first_year_cap_gw=self.config.pv_first_year_cap_gw,
                    expansion_interval_years=expansion_interval_years,
                )
                scenario.statistics.generate_result_files()
                scenario.statistics.write_results()
                if DEBUG:
                    scenario.statistics.dump()
                results_time = time.time()
                results_time_str = datetime.fromtimestamp(results_time).strftime("%d/%m/%Y %H:%M:%S")
                scenario.logger.info(f"Results saved at {results_time_str} ({results_time - solve_time:.4f} seconds).")

            scenario.unload_datafiles()

            end_time = time.time()
            end_time_str = datetime.fromtimestamp(end_time).strftime("%d/%m/%Y %H:%M:%S")
            scenario.logger.info(
                f"Scenario completed at {end_time_str} (Total time taken: {(end_time - start_time)/(60*60):.4f} hours)."
            )

        return None

    def capacity_expansion(self) -> None:
        """
        Run sequential single_time optimisations over the modelling horizon in user-defined intervals,
        updating capacities and build limits on disk between runs and recording the pathway.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_model_name = self.config.model_name
        root_dir = os.path.join("results", f"{base_model_name}_capacity_expansion_{timestamp}")
        os.makedirs(root_dir, exist_ok=True)

        base_config_dir = os.path.abspath(self.config_directory)
        base_data_dir = os.path.abspath(self.data_directory)

        scenarios_df = pd.read_csv(os.path.join(base_config_dir, "scenarios.csv"))
        config_df_base = pd.read_csv(os.path.join(base_config_dir, "config.csv"))
        generators_df_base = pd.read_csv(os.path.join(base_config_dir, "generators.csv"))
        storages_df_base = pd.read_csv(os.path.join(base_config_dir, "storages.csv"))
        lines_df_base = pd.read_csv(os.path.join(base_config_dir, "lines.csv"))
        fuels_df_base = pd.read_csv(os.path.join(base_config_dir, "fuels.csv"))
        datafiles_df_base = pd.read_csv(os.path.join(base_config_dir, "datafiles.csv"))
        initial_guess_df_base = pd.read_csv(os.path.join(base_config_dir, "initial_guess.csv"))

        retrofit_group_map: Dict[str, int] = {}
        retrofit_caps_by_year: Dict[int, Dict[int, float]] = {}
        if "retrofit_group" in generators_df_base.columns and "retrofit_cap" in generators_df_base.columns:
            for _, row in generators_df_base.iterrows():
                group_val = row.get("retrofit_group")
                cap_val = row.get("retrofit_cap")
                year_val = row.get("Year")
                name_val = str(row.get("name", ""))
                if pd.isna(group_val) or pd.isna(cap_val) or pd.isna(year_val):
                    continue
                try:
                    group = int(float(group_val))
                    year = int(float(year_val))
                    cap = float(cap_val)
                except (TypeError, ValueError):
                    continue
                if group <= 0:
                    continue
                if name_val and name_val not in retrofit_group_map:
                    retrofit_group_map[name_val] = group
                retrofit_caps_by_year.setdefault(year, {})
                retrofit_caps_by_year[year][group] = max(retrofit_caps_by_year[year].get(group, 0.0), cap)

        for _, scenario_row in scenarios_df.iterrows():
            scenario_name = str(scenario_row.get("scenario_name"))
            if not scenario_name:
                continue

            fixed_capacity_prev: Dict[str, float] = {}

            scenario_root = os.path.join(root_dir, scenario_name)
            work_config_dir = os.path.join(scenario_root, "inputs", "config")
            work_data_dir = os.path.join(scenario_root, "inputs", "data")
            shutil.copytree(base_config_dir, work_config_dir, dirs_exist_ok=True)
            os.makedirs(work_data_dir, exist_ok=True)

            scenario_mask = scenarios_df["scenario_name"] == scenario_name
            scenario_only_df = scenarios_df.loc[scenario_mask].copy()
            if scenario_only_df.empty:
                continue

            scenario_files: List[str] = []
            for _, row in datafiles_df_base.iterrows():
                scenarios = parse_comma_separated(row.get("scenarios", ""))
                if scenario_name not in scenarios:
                    continue
                filename = str(row.get("filename", "")).strip()
                if filename and filename not in scenario_files:
                    scenario_files.append(filename)

            datafile_frames: Dict[str, pd.DataFrame] = {}
            for filename in scenario_files:
                datafile_frames[filename] = pd.read_csv(os.path.join(base_data_dir, filename))

            first_year = int(scenario_only_df["firstyear"].iloc[0])
            final_year = int(scenario_only_df["finalyear"].iloc[0])
            total_years = final_year - first_year + 1
            interval_years = int(self.config.expansion_interval_years) if self.config.expansion_interval_years else 0
            if interval_years <= 0 or interval_years > total_years:
                interval_years = total_years

            generators_year_df = self._select_year_rows(generators_df_base, first_year, "generators.csv")
            storages_year_df = self._select_year_rows(storages_df_base, first_year, "storages.csv")
            lines_year_df = self._select_year_rows(lines_df_base, first_year, "lines.csv")

            generator_names = self._asset_names_for_scenario(generators_year_df, scenario_name)
            storage_names = self._asset_names_for_scenario(storages_year_df, scenario_name)
            line_names = self._asset_names_for_scenario(lines_year_df, scenario_name)

            records = self._init_pathway_records(
                generator_names, storage_names, line_names
            )

            generator_state = self._init_generator_state(generators_df_base, scenario_name, first_year)
            storage_state = self._init_storage_state(storages_df_base, scenario_name, first_year)
            line_state = self._init_line_state(lines_df_base, scenario_name, first_year)

            weighted_lcoe_new_sum = 0.0
            weighted_lcoe_existing_sum = 0.0
            weighted_lcoe_total_sum = 0.0
            demand_sum = 0.0

            for start_year in range(first_year, final_year + 1, interval_years):
                end_year = start_year  # model a single year snapshot at each interval step

                scenario_iter_df = scenario_only_df.copy()
                scenario_iter_df.loc[:, "firstyear"] = start_year
                scenario_iter_df.loc[:, "finalyear"] = end_year
                scenario_iter_df.to_csv(os.path.join(work_config_dir, "scenarios.csv"), index=False)

                config_df = config_df_base.copy()
                config_df.loc[config_df["name"] == "type", "value"] = "single_time"
                config_df.loc[config_df["name"] == "model_name", "value"] = (
                    f"{base_model_name}_capexp_{scenario_name}_{start_year}_{end_year}"
                )
                config_df.to_csv(os.path.join(work_config_dir, "config.csv"), index=False)

                initial_guess_df = initial_guess_df_base.copy()
                initial_guess_df = initial_guess_df[initial_guess_df["scenario"] == scenario_name]
                if start_year != first_year:
                    initial_guess_df.loc[:, "x_0"] = ""
                initial_guess_df.to_csv(os.path.join(work_config_dir, "initial_guess.csv"), index=False)

                self._apply_retirements(generator_state, start_year)
                self._apply_storage_retirements(storage_state, start_year)
                self._apply_retirements(line_state, start_year)
                pv_prev_total = Model._sum_pv_capacity(generator_state)

                generators_df = self._select_year_rows(generators_df_base, start_year, "generators.csv")
                storages_df = self._select_year_rows(storages_df_base, start_year, "storages.csv")
                lines_df = self._select_year_rows(lines_df_base, start_year, "lines.csv")
                fuels_df = self._select_year_rows(fuels_df_base, start_year, "fuels.csv")

                self._apply_generator_state(generators_df, scenario_name, generator_state)
                self._apply_storage_state(storages_df, scenario_name, storage_state)
                self._apply_line_state(lines_df, scenario_name, line_state)

                if retrofit_group_map and "retrofit_group" in generators_df.columns:
                    group_used: Dict[int, float] = {}
                    for name, state in generator_state.items():
                        group = retrofit_group_map.get(name)
                        if group is None:
                            continue
                        group_used[group] = group_used.get(group, 0.0) + Model._sum_vintages(
                            state.get("vintages", [])
                        )
                    year_caps = retrofit_caps_by_year.get(start_year, {})
                    for idx, row in generators_df.iterrows():
                        group_val = row.get("retrofit_group")
                        if pd.isna(group_val):
                            continue
                        try:
                            group = int(float(group_val))
                        except (TypeError, ValueError):
                            continue
                        if group <= 0:
                            continue
                        cap = year_caps.get(group, 0.0)
                        remaining = max(cap - group_used.get(group, 0.0), 0.0)
                        generators_df.loc[idx, "max_build"] = remaining

                fixed_names = self._apply_fixed_capacity_gw(
                    generators_df, scenario_name, generator_state, fixed_capacity_prev, start_year
                )

                fuels_df.to_csv(os.path.join(work_config_dir, "fuels.csv"), index=False)
                datafiles_df_base.to_csv(os.path.join(work_config_dir, "datafiles.csv"), index=False)
                generators_df.to_csv(os.path.join(work_config_dir, "generators.csv"), index=False)
                storages_df.to_csv(os.path.join(work_config_dir, "storages.csv"), index=False)
                lines_df.to_csv(os.path.join(work_config_dir, "lines.csv"), index=False)

                self._write_data_slice(datafile_frames, work_data_dir, start_year, end_year)

                interval_model = Model(
                    config_directory=work_config_dir,
                    data_directory=work_data_dir,
                    logging_flag=self.logging_flag,
                )
                interval_model.solve(pv_prev_total=pv_prev_total, expansion_interval_years=interval_years)

                interval_scenario = next(iter(interval_model.scenarios.values()))
                stats = interval_scenario.statistics
                if stats is None:
                    raise RuntimeError("Statistics not generated for capacity expansion run.")

                gen_builds = {g.name: float(g.new_build) for g in stats.solution.fleet.generators.values()}
                gen_builds_state = {name: build for name, build in gen_builds.items() if name not in fixed_names}
                stor_builds_p = {s.name: float(s.new_build_p) for s in stats.solution.fleet.storages.values()}
                stor_builds_e = {s.name: float(s.new_build_e) for s in stats.solution.fleet.storages.values()}
                line_builds = {l.name: float(l.new_build) for l in stats.solution.network.major_lines.values()}

                gen_lifetimes = {
                    str(row.get("name")): int(float(row.get("lifetime", 0) or 0))
                    for _, row in generators_df.iterrows()
                    if scenario_name in parse_comma_separated(row.get("scenarios", ""))
                }
                stor_lifetimes = {
                    str(row.get("name")): int(float(row.get("lifetime", 0) or 0))
                    for _, row in storages_df.iterrows()
                    if scenario_name in parse_comma_separated(row.get("scenarios", ""))
                }
                line_lifetimes = {
                    str(row.get("name")): int(float(row.get("lifetime", 0) or 0))
                    for _, row in lines_df.iterrows()
                    if scenario_name in parse_comma_separated(row.get("scenarios", ""))
                }

                self._update_generator_state(generator_state, gen_builds_state, gen_lifetimes, start_year)
                self._update_storage_state(storage_state, stor_builds_p, stor_builds_e, stor_lifetimes, start_year)
                self._update_line_state(line_state, line_builds, line_lifetimes, start_year)
                pv_prev_total = Model._sum_pv_capacity(generator_state)

                self._append_pathway_records_from_state(
                    records,
                    start_year,
                    end_year,
                    gen_builds,
                    stor_builds_p,
                    stor_builds_e,
                    line_builds,
                    generator_state,
                    storage_state,
                    line_state,
                )

                summary_path = os.path.join(stats.results_directory, "summary.csv")
                interval_demand = self._read_interval_demand_gwh(summary_path, start_year, end_year)
                existing_annualised_cost = self._calculate_existing_annualised_build_cost(stats)
                lcoe_new_build = stats.solution.lcoe
                lcoe_existing_capacity = (
                    existing_annualised_cost / (interval_demand * 1000) if interval_demand > 0 else 0.0
                )
                lcoe_total = lcoe_new_build + lcoe_existing_capacity

                if interval_demand > 0:
                    weighted_lcoe_new_sum += lcoe_new_build * interval_demand
                    weighted_lcoe_existing_sum += lcoe_existing_capacity * interval_demand
                    weighted_lcoe_total_sum += lcoe_total * interval_demand
                    demand_sum += interval_demand

                records["metrics"].append(
                    {
                        "start_year": start_year,
                        "end_year": end_year,
                        "lcoe_new_build": lcoe_new_build,
                        "lcoe_existing_capacity": lcoe_existing_capacity,
                        "lcoe_total": lcoe_total,
                        "penalties": stats.solution.penalties,
                        "demand_gwh": interval_demand,
                    }
                )

            weighted_lcoe_new = weighted_lcoe_new_sum / demand_sum if demand_sum > 0 else 0.0
            weighted_lcoe_existing = weighted_lcoe_existing_sum / demand_sum if demand_sum > 0 else 0.0
            weighted_lcoe_total = weighted_lcoe_total_sum / demand_sum if demand_sum > 0 else 0.0
            records["metrics"].append(
                {
                    "start_year": "Total",
                    "end_year": "Total",
                    "lcoe_new_build": weighted_lcoe_new,
                    "lcoe_existing_capacity": weighted_lcoe_existing,
                    "lcoe_total": weighted_lcoe_total,
                    "penalties": "",
                    "demand_gwh": demand_sum,
                }
            )

            self._write_pathway_records(records, scenario_root)

        return None

    @staticmethod
    def _asset_names_for_scenario(df: pd.DataFrame, scenario_name: str) -> List[str]:
        names = []
        seen = set()
        for _, row in df.iterrows():
            scenarios = parse_comma_separated(row.get("scenarios", ""))
            if scenario_name in scenarios:
                name = str(row.get("name"))
                if name and name not in seen:
                    names.append(name)
                    seen.add(name)
        return names

    @staticmethod
    def _init_pathway_records(generator_names: List[str], storage_names: List[str], line_names: List[str]):
        return {
            "generators_build": [],
            "generators_capacity": [],
            "storages_power_build": [],
            "storages_power_capacity": [],
            "storages_energy_build": [],
            "storages_energy_capacity": [],
            "lines_build": [],
            "lines_capacity": [],
            "metrics": [],
            "generator_names": generator_names,
            "storage_names": storage_names,
            "line_names": line_names,
        }

    @staticmethod
    def _select_year_rows(df: pd.DataFrame, year: int, label: str) -> pd.DataFrame:
        if "Year" in df.columns:
            years = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
            df_year = df[years == year].copy()
            if df_year.empty:
                raise ValueError(f"No rows in {label} for year {year}.")
            return df_year
        return df.copy()

    @staticmethod
    def _write_data_slice(
        datafile_frames: Dict[str, pd.DataFrame], work_data_dir: str, start_year: int, end_year: int
    ) -> None:
        for filename, df in datafile_frames.items():
            df_out = df
            if "Year" in df.columns:
                df_out = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)]
                if df_out.empty:
                    raise ValueError(f"No data in {filename} for years {start_year}-{end_year}.")
            elif "Year-month" in df.columns or "Year_month" in df.columns:
                col = "Year-month" if "Year-month" in df.columns else "Year_month"
                year_series = df[col].astype(str).str.replace(r"\D", "", regex=True).str.slice(0, 4)
                years = pd.to_numeric(year_series, errors="coerce")
                df_out = df[(years >= start_year) & (years <= end_year)]
                if df_out.empty:
                    raise ValueError(f"No data in {filename} for years {start_year}-{end_year}.")
            df_out.to_csv(os.path.join(work_data_dir, filename), index=False)

    @staticmethod
    def _read_interval_demand_gwh(summary_path: str, start_year: int, end_year: int) -> float:
        if not os.path.isfile(summary_path):
            return 0.0
        df = pd.read_csv(summary_path, header=None)
        label_col = df.iloc[:, 0].astype(str)
        col_name_row = df[label_col == "Column Name"]
        if col_name_row.empty:
            return 0.0
        demand_cols = col_name_row.iloc[0][col_name_row.iloc[0] == "Annual Demand"].index
        if demand_cols.empty:
            return 0.0
        years = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        year_mask = (years >= start_year) & (years <= end_year)
        demand_data = df.loc[year_mask, demand_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return float(demand_data.to_numpy().sum())

    @staticmethod
    def _present_value_factor(discount_rate: float, lifetime: int) -> float:
        if discount_rate <= 1e-6:
            return 0.0
        return (1 - (1 + discount_rate) ** (-1 * lifetime)) / discount_rate

    @staticmethod
    def _calculate_existing_annualised_build_cost(stats: Statistics) -> float:
        year_count = int(stats.solution.static.year_count)
        total_cost = 0.0

        for generator in stats.solution.fleet.generators.values():
            existing_power = max(generator.capacity - generator.new_build, 0.0)
            if existing_power <= 0:
                continue
            cost = generator.cost
            pv = Model._present_value_factor(cost.discount_rate, cost.lifetime)
            if pv > 1e-6:
                total_cost += year_count * (existing_power * 1e6 * cost.capex_p) / pv

        for storage in stats.solution.fleet.storages.values():
            existing_power = max(storage.power_capacity - storage.new_build_p, 0.0)
            if existing_power <= 0:
                continue
            if storage.duration > 0:
                existing_energy = existing_power * storage.duration
            else:
                existing_energy = max(storage.energy_capacity - storage.new_build_e, 0.0)
            cost = storage.cost
            pv = Model._present_value_factor(cost.discount_rate, cost.lifetime)
            if pv > 1e-6:
                total_cost += year_count * (
                    existing_energy * 1e6 * cost.capex_e + existing_power * 1e6 * cost.capex_p
                ) / pv

        for line in stats.solution.network.major_lines.values():
            existing_power = max(line.capacity - line.new_build, 0.0)
            if existing_power <= 0:
                continue
            cost = line.cost
            pv = Model._present_value_factor(cost.discount_rate, cost.lifetime)
            if pv > 1e-6:
                total_cost += year_count * (
                    existing_power * 1e3 * line.length * cost.capex_p + existing_power * 1e3 * cost.transformer_capex
                ) / pv

        for line in stats.solution.network.minor_lines.values():
            existing_power = max(line.capacity - line.new_build, 0.0)
            if existing_power <= 0:
                continue
            cost = line.cost
            pv = Model._present_value_factor(cost.discount_rate, cost.lifetime)
            if pv > 1e-6:
                total_cost += year_count * (
                    existing_power * 1e3 * line.length * cost.capex_p + existing_power * 1e3 * cost.transformer_capex
                ) / pv

        return total_cost

    @staticmethod
    def _init_generator_state(df: pd.DataFrame, scenario_name: str, start_year: int) -> Dict[str, dict]:
        df_year = Model._select_year_rows(df, start_year, "generators.csv")
        state: Dict[str, dict] = {}
        for _, row in df_year.iterrows():
            if scenario_name not in parse_comma_separated(row.get("scenarios", "")):
                continue
            name = str(row.get("name"))
            if not name:
                continue
            initial_capacity = float(row.get("initial_capacity", 0) or 0)
            max_build = float(row.get("max_build", 0) or 0)
            lifetime = int(float(row.get("lifetime", 0) or 0))
            initial_lifetime = row.get("initial_capacity_lifetime", lifetime)
            try:
                initial_lifetime = int(float(initial_lifetime))
            except (TypeError, ValueError):
                initial_lifetime = lifetime

            vintages = []
            if initial_capacity > 0:
                vintages.append(
                    {
                        "build_year": start_year,
                        "capacity": initial_capacity,
                        "lifetime": initial_lifetime,
                        "counts_against_max_build": False,
                    }
                )
            state[name] = {"max_build": max_build, "vintages": vintages}
        return state

    @staticmethod
    def _init_storage_state(df: pd.DataFrame, scenario_name: str, start_year: int) -> Dict[str, dict]:
        df_year = Model._select_year_rows(df, start_year, "storages.csv")
        state: Dict[str, dict] = {}
        for _, row in df_year.iterrows():
            if scenario_name not in parse_comma_separated(row.get("scenarios", "")):
                continue
            name = str(row.get("name"))
            if not name:
                continue
            duration = float(row.get("duration", 0) or 0)
            initial_power = float(row.get("initial_power_capacity", 0) or 0)
            initial_energy = float(row.get("initial_energy_capacity", 0) or 0)
            max_build_p = float(row.get("max_build_p", 0) or 0)
            max_build_e = float(row.get("max_build_e", 0) or 0)
            lifetime = int(float(row.get("lifetime", 0) or 0))
            initial_lifetime = row.get("initial_capacity_lifetime", lifetime)
            try:
                initial_lifetime = int(float(initial_lifetime))
            except (TypeError, ValueError):
                initial_lifetime = lifetime

            power_vintages = []
            energy_vintages = []
            if initial_power > 0:
                power_vintages.append(
                    {
                        "build_year": start_year,
                        "capacity": initial_power,
                        "lifetime": initial_lifetime,
                        "counts_against_max_build": False,
                    }
                )
            if duration <= 0 and initial_energy > 0:
                energy_vintages.append(
                    {
                        "build_year": start_year,
                        "capacity": initial_energy,
                        "lifetime": initial_lifetime,
                        "counts_against_max_build": False,
                    }
                )

            state[name] = {
                "duration": duration,
                "max_build_p": max_build_p,
                "max_build_e": max_build_e,
                "power_vintages": power_vintages,
                "energy_vintages": energy_vintages,
            }
        return state

    @staticmethod
    def _init_line_state(df: pd.DataFrame, scenario_name: str, start_year: int) -> Dict[str, dict]:
        df_year = Model._select_year_rows(df, start_year, "lines.csv")
        state: Dict[str, dict] = {}
        for _, row in df_year.iterrows():
            if scenario_name not in parse_comma_separated(row.get("scenarios", "")):
                continue
            name = str(row.get("name"))
            if not name:
                continue
            initial_capacity = float(row.get("initial_capacity", 0) or 0)
            max_build = float(row.get("max_build", 0) or 0)
            lifetime = int(float(row.get("lifetime", 0) or 0))
            initial_lifetime = row.get("initial_capacity_lifetime", lifetime)
            try:
                initial_lifetime = int(float(initial_lifetime))
            except (TypeError, ValueError):
                initial_lifetime = lifetime

            vintages = []
            if initial_capacity > 0:
                vintages.append(
                    {
                        "build_year": start_year,
                        "capacity": initial_capacity,
                        "lifetime": initial_lifetime,
                        "counts_against_max_build": False,
                    }
                )
            state[name] = {"max_build": max_build, "vintages": vintages}
        return state

    @staticmethod
    def _apply_retirements(state: Dict[str, dict], current_year: int) -> None:
        for asset in state.values():
            retired_capacity = 0.0
            remaining = []
            for vintage in asset["vintages"]:
                if (current_year - vintage["build_year"]) < vintage["lifetime"]:
                    remaining.append(vintage)
                else:
                    if vintage.get("counts_against_max_build", True):
                        retired_capacity += float(vintage.get("capacity", 0.0) or 0.0)
            asset["vintages"] = remaining
            if retired_capacity > 0.0:
                asset["max_build"] = max(0.0, asset.get("max_build", 0.0) + retired_capacity)

    @staticmethod
    def _apply_storage_retirements(state: Dict[str, dict], current_year: int) -> None:
        for asset in state.values():
            retired_power = 0.0
            remaining_power = []
            for vintage in asset["power_vintages"]:
                if (current_year - vintage["build_year"]) < vintage["lifetime"]:
                    remaining_power.append(vintage)
                else:
                    if vintage.get("counts_against_max_build", True):
                        retired_power += float(vintage.get("capacity", 0.0) or 0.0)
            asset["power_vintages"] = remaining_power
            if retired_power > 0.0:
                asset["max_build_p"] = max(0.0, asset.get("max_build_p", 0.0) + retired_power)
            if asset.get("duration", 0) <= 0:
                retired_energy = 0.0
                remaining_energy = []
                for vintage in asset["energy_vintages"]:
                    if (current_year - vintage["build_year"]) < vintage["lifetime"]:
                        remaining_energy.append(vintage)
                    else:
                        if vintage.get("counts_against_max_build", True):
                            retired_energy += float(vintage.get("capacity", 0.0) or 0.0)
                asset["energy_vintages"] = remaining_energy
                if retired_energy > 0.0:
                    asset["max_build_e"] = max(0.0, asset.get("max_build_e", 0.0) + retired_energy)

    @staticmethod
    def _sum_vintages(vintages: List[dict]) -> float:
        return float(sum(v["capacity"] for v in vintages))

    @staticmethod
    def _sum_pv_capacity(generator_state: Dict[str, dict]) -> float:
        pv_total = 0.0
        for name, state in generator_state.items():
            if name.startswith("pv_"):
                pv_total += Model._sum_vintages(state.get("vintages", []))
        return pv_total

    @staticmethod
    def _apply_generator_state(df: pd.DataFrame, scenario_name: str, state: Dict[str, dict]) -> None:
        for idx, row in df.iterrows():
            if scenario_name not in parse_comma_separated(row.get("scenarios", "")):
                continue
            name = str(row.get("name"))
            if name not in state:
                continue
            df.loc[idx, "initial_capacity"] = Model._sum_vintages(state[name]["vintages"])
            df.loc[idx, "max_build"] = state[name]["max_build"]

    @staticmethod
    def _apply_storage_state(df: pd.DataFrame, scenario_name: str, state: Dict[str, dict]) -> None:
        for idx, row in df.iterrows():
            if scenario_name not in parse_comma_separated(row.get("scenarios", "")):
                continue
            name = str(row.get("name"))
            if name not in state:
                continue
            duration = float(row.get("duration", 0) or 0)
            state[name]["duration"] = duration
            power_cap = Model._sum_vintages(state[name]["power_vintages"])
            df.loc[idx, "initial_power_capacity"] = power_cap
            df.loc[idx, "max_build_p"] = state[name]["max_build_p"]
            if duration > 0:
                df.loc[idx, "initial_energy_capacity"] = power_cap * duration
            else:
                energy_cap = Model._sum_vintages(state[name]["energy_vintages"])
                df.loc[idx, "initial_energy_capacity"] = energy_cap
                df.loc[idx, "max_build_e"] = state[name]["max_build_e"]

    @staticmethod
    def _apply_line_state(df: pd.DataFrame, scenario_name: str, state: Dict[str, dict]) -> None:
        for idx, row in df.iterrows():
            if scenario_name not in parse_comma_separated(row.get("scenarios", "")):
                continue
            name = str(row.get("name"))
            if name not in state:
                continue
            df.loc[idx, "initial_capacity"] = Model._sum_vintages(state[name]["vintages"])
            df.loc[idx, "max_build"] = state[name]["max_build"]

    @staticmethod
    def _apply_fixed_capacity_gw(
        df: pd.DataFrame,
        scenario_name: str,
        state: Dict[str, dict],
        prev_caps: Dict[str, float],
        start_year: int,
    ) -> set:
        if "fixed_capacity_gw" not in df.columns:
            return set()

        fixed_names = set()
        for idx, row in df.iterrows():
            if scenario_name not in parse_comma_separated(row.get("scenarios", "")):
                continue

            fixed_val = row.get("fixed_capacity_gw")
            if fixed_val is None or pd.isna(fixed_val):
                continue

            try:
                fixed_cap = float(fixed_val)
            except (TypeError, ValueError):
                continue
            if fixed_cap < 0.0:
                fixed_cap = 0.0

            name = str(row.get("name"))
            prev = prev_caps.get(name, fixed_cap)
            delta = max(fixed_cap - prev, 0.0)
            if abs(delta) < 1e-9:
                delta = 0.0

            df.loc[idx, "initial_capacity"] = fixed_cap
            df.loc[idx, "min_build"] = delta
            df.loc[idx, "max_build"] = delta

            lifetime = row.get("lifetime", 0)
            try:
                lifetime_int = int(float(lifetime))
            except (TypeError, ValueError):
                lifetime_int = 0

            if name not in state:
                state[name] = {"max_build": 0.0, "vintages": []}

            if fixed_cap > 0.0:
                state[name]["vintages"] = [
                    {
                        "build_year": start_year,
                        "capacity": fixed_cap,
                        "lifetime": lifetime_int,
                        "counts_against_max_build": False,
                    }
                ]
            else:
                state[name]["vintages"] = []
            state[name]["max_build"] = 0.0

            prev_caps[name] = fixed_cap
            fixed_names.add(name)

        return fixed_names

    @staticmethod
    def _update_generator_state(
        state: Dict[str, dict], builds: Dict[str, float], lifetimes: Dict[str, int], build_year: int
    ) -> None:
        for name, new_build in builds.items():
            if name not in state or new_build <= 0:
                continue
            lifetime = lifetimes.get(name)
            if lifetime is None:
                continue
            state[name]["vintages"].append(
                {
                    "build_year": build_year,
                    "capacity": new_build,
                    "lifetime": lifetime,
                    "counts_against_max_build": True,
                }
            )
            state[name]["max_build"] = max(0.0, state[name]["max_build"] - new_build)

    @staticmethod
    def _update_storage_state(
        state: Dict[str, dict],
        builds_p: Dict[str, float],
        builds_e: Dict[str, float],
        lifetimes: Dict[str, int],
        build_year: int,
    ) -> None:
        for name, new_build_p in builds_p.items():
            if name not in state or new_build_p <= 0:
                continue
            lifetime = lifetimes.get(name)
            if lifetime is None:
                continue
            state[name]["power_vintages"].append(
                {
                    "build_year": build_year,
                    "capacity": new_build_p,
                    "lifetime": lifetime,
                    "counts_against_max_build": True,
                }
            )
            state[name]["max_build_p"] = max(0.0, state[name]["max_build_p"] - new_build_p)

        for name, new_build_e in builds_e.items():
            if name not in state or new_build_e <= 0:
                continue
            if state[name].get("duration", 0) > 0:
                continue
            lifetime = lifetimes.get(name)
            if lifetime is None:
                continue
            state[name]["energy_vintages"].append(
                {
                    "build_year": build_year,
                    "capacity": new_build_e,
                    "lifetime": lifetime,
                    "counts_against_max_build": True,
                }
            )
            state[name]["max_build_e"] = max(0.0, state[name]["max_build_e"] - new_build_e)

    @staticmethod
    def _update_line_state(
        state: Dict[str, dict], builds: Dict[str, float], lifetimes: Dict[str, int], build_year: int
    ) -> None:
        for name, new_build in builds.items():
            if name not in state or new_build <= 0:
                continue
            lifetime = lifetimes.get(name)
            if lifetime is None:
                continue
            state[name]["vintages"].append(
                {
                    "build_year": build_year,
                    "capacity": new_build,
                    "lifetime": lifetime,
                    "counts_against_max_build": True,
                }
            )
            state[name]["max_build"] = max(0.0, state[name]["max_build"] - new_build)

    @staticmethod
    def _append_pathway_records_from_state(
        records: dict,
        start_year: int,
        end_year: int,
        gen_builds: Dict[str, float],
        stor_builds_p: Dict[str, float],
        stor_builds_e: Dict[str, float],
        line_builds: Dict[str, float],
        generator_state: Dict[str, dict],
        storage_state: Dict[str, dict],
        line_state: Dict[str, dict],
    ) -> None:
        base = {"start_year": start_year, "end_year": end_year}

        gen_build_row = {**base, **{name: gen_builds.get(name, 0.0) for name in records["generator_names"]}}
        gen_cap_row = {
            **base,
            **{
                name: Model._sum_vintages(generator_state.get(name, {}).get("vintages", []))
                for name in records["generator_names"]
            },
        }

        stor_p_build_row = {**base, **{name: stor_builds_p.get(name, 0.0) for name in records["storage_names"]}}
        stor_p_cap_row = {}
        stor_e_build_row = {}
        stor_e_cap_row = {}

        for name in records["storage_names"]:
            state = storage_state.get(name, {})
            duration = float(state.get("duration", 0) or 0)
            power_cap = Model._sum_vintages(state.get("power_vintages", []))
            stor_p_cap_row[name] = power_cap

            if duration > 0:
                stor_e_build_row[name] = stor_builds_p.get(name, 0.0) * duration
                stor_e_cap_row[name] = power_cap * duration
            else:
                stor_e_build_row[name] = stor_builds_e.get(name, 0.0)
                stor_e_cap_row[name] = Model._sum_vintages(state.get("energy_vintages", []))

        stor_p_cap_row = {**base, **stor_p_cap_row}
        stor_e_build_row = {**base, **stor_e_build_row}
        stor_e_cap_row = {**base, **stor_e_cap_row}

        line_build_row = {**base, **{name: line_builds.get(name, 0.0) for name in records["line_names"]}}
        line_cap_row = {
            **base,
            **{
                name: Model._sum_vintages(line_state.get(name, {}).get("vintages", []))
                for name in records["line_names"]
            },
        }

        records["generators_build"].append(gen_build_row)
        records["generators_capacity"].append(gen_cap_row)
        records["storages_power_build"].append(stor_p_build_row)
        records["storages_power_capacity"].append(stor_p_cap_row)
        records["storages_energy_build"].append(stor_e_build_row)
        records["storages_energy_capacity"].append(stor_e_cap_row)
        records["lines_build"].append(line_build_row)
        records["lines_capacity"].append(line_cap_row)

    @staticmethod
    def _write_pathway_records(records: dict, scenario_root: str) -> None:
        out_dir = os.path.join(scenario_root, "pathway")
        os.makedirs(out_dir, exist_ok=True)

        pd.DataFrame(records["generators_build"]).to_csv(os.path.join(out_dir, "generators_new_build.csv"), index=False)
        pd.DataFrame(records["generators_capacity"]).to_csv(
            os.path.join(out_dir, "generators_cumulative_capacity.csv"), index=False
        )
        pd.DataFrame(records["storages_power_build"]).to_csv(
            os.path.join(out_dir, "storages_power_new_build.csv"), index=False
        )
        pd.DataFrame(records["storages_power_capacity"]).to_csv(
            os.path.join(out_dir, "storages_power_cumulative_capacity.csv"), index=False
        )
        pd.DataFrame(records["storages_energy_build"]).to_csv(
            os.path.join(out_dir, "storages_energy_new_build.csv"), index=False
        )
        pd.DataFrame(records["storages_energy_capacity"]).to_csv(
            os.path.join(out_dir, "storages_energy_cumulative_capacity.csv"), index=False
        )
        pd.DataFrame(records["lines_build"]).to_csv(os.path.join(out_dir, "lines_new_build.csv"), index=False)
        pd.DataFrame(records["lines_capacity"]).to_csv(
            os.path.join(out_dir, "lines_cumulative_capacity.csv"), index=False
        )
        pd.DataFrame(records["metrics"]).to_csv(os.path.join(out_dir, "capacity_expansion_metrics.csv"), index=False)
