import calendar
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from firm_ce.system.parameters import ScenarioParameters, ScenarioParameters_InstanceType


def determine_interval_parameters(
    first_year: int,
    year_count: int,
    resolution: float,
) -> Tuple[int, NDArray, NDArray, NDArray, int, int]:
    """
    Calculate parameters associated with time intervals, accounting for leap years. The first_year
    and last_year in `config/scenarios.csv` determines whether or not an interval is considered
    a leap year

    Parameters:
    -------
    first_year (int): The first year of the scenario, specified in `config/scenarios.csv`.
    year_count (int): The total number of years in the scenario.
    resolution (float): The time resolution of each interval for the input data [hours/interval].

    Returns:
    -------
    Tuple[int, NDArray, NDArray, NDArray, int, int]: A tuple containing the number of leap days in the scenario,
        a numpy array specifying the first time interval of each year, an array of the first interval of
        each month, a mapping of interval index to month index, the total number of months, and the total number
        of time intervals in the scenario.
    """
    year_first_t = np.zeros(year_count, dtype=np.int64)
    month_first_t = np.empty(year_count * 12, dtype=np.int64)

    interval_month = None
    leap_days = 0
    month_idx = 0
    intervals_so_far = 0

    interval_month = []

    for i in range(year_count):
        year = first_year + i
        year_first_t[i] = intervals_so_far

        for month in range(1, 13):
            days_in_month = calendar.monthrange(year, month)[1]
            intervals_in_month = int(days_in_month * 24 // resolution)
            month_first_t[month_idx] = intervals_so_far
            interval_month.extend([month_idx] * intervals_in_month)

            intervals_so_far += intervals_in_month
            month_idx += 1

        leap_days += calendar.leapdays(year, year + 1)

    intervals_count = intervals_so_far
    return leap_days, year_first_t, month_first_t, np.array(interval_month, dtype=np.int64), month_idx, intervals_count


def construct_ScenarioParameters_object(
    scenario_data_dict: Dict[str, str],
    node_count: int,
) -> ScenarioParameters_InstanceType:
    """
    Takes data required to initialise the ScenarioParameters object, casts values into Numba-compatible
    types, and returns an instance of the ScenarioParameters jitclass. The ScenarioParameters are static
    data referenced by the unit committment model.

    Parameters:
    -------
    scenario_data_dict (Dict[str, str]): A dictionary containing data for a single scenario,
        imported from `config/scenarios.csv`.
    node_count (int): The number of nodes (buses) in the network for the scenario.

    Returns:
    -------
    ScenarioParameters_InstanceType: A static instance of the ScenarioParameters jitclass.
    """
    resolution = float(scenario_data_dict.get("resolution", 0.0))
    allowance = float(scenario_data_dict.get("allowance", 0.0))
    first_year = int(scenario_data_dict.get("firstyear", 0))
    final_year = int(scenario_data_dict.get("finalyear", 0))
    year_count = final_year - first_year + 1
    (
        leap_year_count,
        year_first_t,
        month_first_t,
        interval_month,
        month_count,
        intervals_count,
    ) = determine_interval_parameters(
        first_year,
        year_count,
        resolution,
    )

    return ScenarioParameters(
        resolution,
        allowance,
        first_year,
        final_year,
        year_count,
        leap_year_count,
        year_first_t,
        month_first_t,
        interval_month,
        month_count,
        intervals_count,
        node_count,
    )
