from typing import Dict

import numpy as np

from firm_ce.common.constants import JIT_ENABLED
from firm_ce.common.jit_overload import jitclass
from firm_ce.common.typing import float64, int64

if JIT_ENABLED:
    scenario_parameters_spec = [
        ("resolution", float64),
        ("interval_resolutions", float64[:]),
        ("allowance", float64),
        ("first_year", int64),
        ("final_year", int64),
        ("year_count", int64),
        ("leap_year_count", int64),
        ("year_first_t", int64[:]),
        ("month_first_t", int64[:]),
        ("interval_month", int64[:]),
        ("month_count", int64),
        ("intervals_count", int64),
        ("block_lengths", int64[:]),
        ("node_count", int64),
        ("fom_scalar", float64),
        ("year_energy_demand", float64[:]),
    ]
else:
    scenario_parameters_spec = []


@jitclass(scenario_parameters_spec)
class ScenarioParameters:
    """
    Represents the static parameters for a model scenario.

    Attributes:
    -------
    resolution (float64): Length of a single time interval in hours (based on input time-series data).
    interval_resolutions (float64[:]): Array of interval lengths in hours, one per interval.
    allowance (float64): Annual allowance for unserved energy, as a percentage of annual demand.
    first_year (int64): First year in modelling horizon (YYYY).
    final_year (int64): Final year in modelling horizon (YYYY).
    year_count (int64): Total number of years in the scenario.
    leap_year_count (int64): Number of leap years in the modelling horizon.
    year_first_t (int64[:]): Array mapping each year to its first time interval index.
    month_first_t (int64[:]): Array mapping each month to its first time interval index.
    interval_month (int64[:]): Array mapping each interval to its month index across the modelling horizon.
    month_count (int64): Total number of months in the modelling horizon.
    intervals_count (int64): Total number of intervals in the modelling horizon.
    block_lengths (int64[:]): Array of block lengths in intervals, defaults to 1 interval per block. Allows for
        variable length blocks for simplified balancing method in future.
    node_count (int64): Number of Nodes in the Network.
    fom_scalar (float64): Scaling factor to adjust fixed O&M to account for leap years (PLEXOS consistency).
    year_energy_demand (float64[:]): Array of total annual energy demand for each simulated year, units GWh.
    """

    def __init__(
        self,
        resolution: float64,
        allowance: float64,
        first_year: int64,
        final_year: int64,
        year_count: int64,
        leap_year_count: int64,
        year_first_t: int64[:],
        month_first_t: int64[:],
        interval_month: int64[:],
        month_count: int64,
        intervals_count: int64,
        node_count: int64,
    ):
        """
        Initialise a ScenarioParameters instance.

        Parameters:
        -------
        resolution (float64): Length of a single time interval in hours (based on input time-series data).
        allowance (float64): Annual allowance for unserved energy, as a percentage of annual demand.
        first_year (int64): First year in modelling horizon (YYYY).
        final_year (int64): Final year in modelling horizon (YYYY).
        year_count (int64): Total number of years in the scenario.
        leap_year_count (int64): Number of leap years in the modelling horizon.
        year_first_t (int64[:]): Array mapping each year to its first time interval index.
        intervals_count (int64): Total number of intervals in the modelling horizon.
        node_count (int64): Number of Nodes in the Network.
        """
        self.resolution = resolution  # length of time interval in hours
        self.interval_resolutions = resolution * np.ones(
            intervals_count, dtype=np.float64
        )  # length of blocks in hours, for future 'simple' balancing_method
        self.allowance = allowance  # % annual demand allowed as unserved energy
        self.first_year = first_year  # YYYY
        self.final_year = final_year  # YYYY
        self.year_count = year_count
        self.leap_year_count = leap_year_count
        self.year_first_t = year_first_t
        self.month_first_t = month_first_t
        self.interval_month = interval_month
        self.month_count = month_count
        self.intervals_count = intervals_count
        self.block_lengths = np.ones(intervals_count, dtype=np.int64)
        self.node_count = node_count
        self.fom_scalar = (
            year_count + leap_year_count / 365
        ) / year_count  # Scale average annual fom to account for leap days for PLEXOS consistency
        self.year_energy_demand = np.zeros(self.year_count, dtype=np.float64)  # GWh


if JIT_ENABLED:
    ScenarioParameters_InstanceType = ScenarioParameters.class_type.instance_type
else:
    ScenarioParameters_InstanceType = ScenarioParameters


class ModelConfig:
    """
    Data class for model configuration parameters loaded from CSV input.

    Attributes:
    -------
    type (str): Type of model (e.g., 'single_time', 'broad_optimum').
    model_name (str): User-defined name of the model run.
    iterations (int): Maximum number of differential evolution iterations.
    population (int): Population size multiplier for the differential evolution algorithm.
    mutation (float): Mutation rate for the differential evolution algorithm.
    recombination (float): Recombination rate for the differential evolution algorithm.
    near_optimal_tol (float): Tolerance for near-optimal optimisation (default: 0.0). For example, a tolerance of
        0.1 will search for solutions that are within 10% of the cost of the least-cost solution.
    midpoint_count (int): Number of midpoint samples for the broad optimum midpoint optimisation (default: 0).
    balancing_type (str): Method used for system balancing (e.g., 'full'). Allows for simplified balancing to be
        implemented in future.
    fixed_costs_threshold (float): Maximum ratio of fixed costs to total energy demand ($/MWh) (default: 500.0).
        Allows for high cost solutions to be rapidly discarded using a penalty function before beginning unit
        committment processes.
    """

    def __init__(self, config_dict: Dict[str, str]) -> None:
        """
        Initialise ModelConfig data class instance.

        Parameters:
        -------
        config_dict (Dict[str, str]): Data imported from the `config.csv` input file, keyed by variable name.
        """
        config_dict = {item["name"]: item["value"] for item in config_dict.values()}
        self.type = config_dict["type"]
        self.model_name = config_dict["model_name"]
        self.iterations = int(config_dict["iterations"])
        self.population = int(config_dict["population"])
        self.mutation = float(config_dict["mutation"])
        self.recombination = float(config_dict["recombination"])
        self.near_optimal_tol = float(config_dict.get("near_optimal_tol", 0.0))
        self.midpoint_count = int(config_dict.get("midpoint_count", 0))
        self.balancing_type = str(config_dict["balancing_type"])
        self.fixed_costs_threshold = float(config_dict.get("fixed_costs_threshold", 500.0))
        self.novelty_k = int(config_dict.get("novelty_k", 5))
        self.novelty_threshold = float(config_dict.get("novelty_threshold", 0.03))
