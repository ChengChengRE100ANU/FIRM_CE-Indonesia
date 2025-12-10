from firm_ce.common.constants import FASTMATH
from firm_ce.common.jit_overload import njit
from firm_ce.common.typing import float64, int64, unicode_type
from firm_ce.system.costs import LTCosts_InstanceType, UnitCost_InstanceType


@njit(fastmath=FASTMATH)
def get_total(ltcosts_instance: LTCosts_InstanceType) -> float64:
    """
    Get the total cost of the asset over the modelling horizon ($), based upon the sum of constituent parts.

    Parameters:
    -------
    ltcosts_instance (LTCosts_InstanceType): An instance of the LTCosts jitclass.

    Returns:
    -------
    float64: Total cost of the asset over the long-term modelling horizon.
    """
    return ltcosts_instance.annualised_build + ltcosts_instance.fom + ltcosts_instance.vom + ltcosts_instance.fuel


@njit(fastmath=FASTMATH)
def get_variable(ltcosts_instance: LTCosts_InstanceType) -> float64:
    """
    Get the variable cost of the asset over the modelling horizon ($), based upon the sum of VO&M and fuel costs.
    These costs require completion of unit committment.

    Parameters:
    -------
    ltcosts_instance (LTCosts_InstanceType): An instance of the LTCosts jitclass.

    Returns:
    -------
    float64: Variable cost of the asset over the long-term modelling horizon.
    """
    return ltcosts_instance.vom + ltcosts_instance.fuel


@njit(fastmath=FASTMATH)
def get_fixed(ltcosts_instance: LTCosts_InstanceType) -> float64:
    """
    Get the fixed cost of the asset over the modelling horizon ($), based upon the sum of FO&M and annualised build costs.
    These costs are independent of unit committment, but require completion of all investment (build) actions.

    Parameters:
    -------
    ltcosts_instance (LTCosts_InstanceType): An instance of the LTCosts jitclass.

    Returns:
    -------
    float64: Fixed cost of the asset over the long-term modelling horizon.
    """
    return ltcosts_instance.annualised_build + ltcosts_instance.fom


@njit(fastmath=FASTMATH)
def get_present_value(discount_rate: float64, lifetime: float64) -> float64:
    """
    Net present value factor used to annualise the build costs. Based upon the discount factor and lifetime of the asset.
    Assumes that fully depreciated assets (at the end of economic life) are replaced. This allows build costs to be considered
    when the modelling horizon is shorter than the economic life of assets.

    Parameters:
    -------
    discount_rate (float64): A float in range (0,1] defining the discount rate for the asset.
    lifetime (float64): Economic lifetime of the asset.

    Returns:
    -------
    float64: Net present value factor.
    """
    return (1 - (1 + discount_rate) ** (-1 * lifetime)) / discount_rate


@njit(fastmath=FASTMATH)
def calculate_annualised_build(
    ltcosts_instance: LTCosts_InstanceType,
    energy_capacity: float64,
    power_capacity: float64,
    line_length: float64,
    unit_costs: UnitCost_InstanceType,
    year_count: int64,
    asset_type: unicode_type,
) -> None:
    """
    Calculates the build costs for an asset, annualised using the net present value factor. Each year
    over the modelling horizon is assumed to incur the annualised build cost of each asset. Existing
    assets in the system are assumed to be fully depreciated by the start of the modelling period.

    The build costs for Line instances are dependent upon the length of the line. The energy capacity capital
    cost of Generator objects is assumed to be $0/kWh.

    Parameters:
    -------
    ltcosts_instance (LTCosts_InstanceType): An instance of the LTCosts jitclass.
    energy_capacity (float64): The energy capacity of the asset (GWh). Assumed to be 0 GWh for Generator
        and Line objects.
    power_capacity (float64): The power capacity of the asset (GW).
    line_length (float64): Length of the transmission line (km). Assumed to be 0 km for Generator and Storage
        assets.
    unit_costs (UnitCost_InstanceType): A UnitCost jitclass instance that defines the exogenous cost assumptions
        for the asset.
    year_count (int64): Total number of years over the modelling horizon. Annualised build costs are assumed to
        be incurred each year.
    asset_type (unicode_type): A string value specifying whether the asset is a "generator", "storage" or "line".

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the LTCosts instance: annualised_build.
    """
    present_value = get_present_value(unit_costs.discount_rate, unit_costs.lifetime)
    if asset_type == "generator" or asset_type == "storage":
        ltcosts_instance.annualised_build = (
            year_count
            * (energy_capacity * 1e6 * unit_costs.capex_e + power_capacity * 1e6 * unit_costs.capex_p)
            / present_value
            if present_value > 1e-6
            else 0
        )
    elif asset_type == "line":
        ltcosts_instance.annualised_build = (
            year_count
            * (
                power_capacity * 1e3 * line_length * unit_costs.capex_p
                + power_capacity * 1e3 * unit_costs.transformer_capex
            )
            / present_value
            if present_value > 1e-6
            else 0
        )
    return None


@njit(fastmath=FASTMATH)
def calculate_fom(
    ltcosts_instance: LTCosts_InstanceType,
    power_capacity: float64,
    years_float: float64,
    line_length: float64,
    unit_costs: UnitCost_InstanceType,
    asset_type: unicode_type,
) -> None:
    """
    Calculates the fixed operation and maintenance (FO&M) costs for an asset. Existing
    assets in the system are assumed to be fully depreciated by the start of the modelling period.
    Fully depreciated assets still incur FO&M.

    The FO&M for Line instances are dependent upon the length of the line.

    Parameters:
    -------
    ltcosts_instance (LTCosts_InstanceType): An instance of the LTCosts jitclass.
    power_capacity (float64): The power capacity of the asset (GW).
    years_float (float64): Total number of non-leap years over the modelling horizon. Leap days provide an
        additional fractional value. Ensures FO&M accounts for a small amount of additional costs in leap years.
    line_length (float64): Length of the transmission line (km). Assumed to be 0 km for Generator and Storage
        assets.
    unit_costs (UnitCost_InstanceType): A UnitCost jitclass instance that defines the exogenous cost assumptions
        for the asset.
    asset_type (unicode_type): A string value specifying whether the asset is a "generator", "storage" or "line".

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the LTCosts instance: fom.
    """
    if asset_type == "generator" or asset_type == "storage":
        ltcosts_instance.fom = power_capacity * 1e6 * unit_costs.fom * years_float
    elif asset_type == "line":
        ltcosts_instance.fom = power_capacity * 1e3 * line_length * unit_costs.fom * years_float
    return None


@njit(fastmath=FASTMATH)
def calculate_vom(
    ltcosts_instance: LTCosts_InstanceType, generation: float64, unit_costs: UnitCost_InstanceType
) -> None:
    """
    Calculates the variable operation and maintenance (VO&M) costs for an asset. Requires completion
    of unit committment.

    Parameters:
    -------
    ltcosts_instance (LTCosts_InstanceType): An instance of the LTCosts jitclass.
    generation (float64): Total energy (GWh) generated by a Generator, discharged by a Storage, or transmitted by a Line
        over the modelling horizon.
    unit_costs (UnitCost_InstanceType): A UnitCost jitclass instance that defines the exogenous cost assumptions
        for the asset.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the LTCosts instance: vom.
    """
    ltcosts_instance.vom = generation * 1e3 * unit_costs.vom
    return None


@njit(fastmath=FASTMATH)
def calculate_fuel(
    ltcosts_instance: LTCosts_InstanceType, generation: float64, unit_hours: float64, unit_costs: UnitCost_InstanceType
) -> None:
    """
    Calculates the fuel costs for an asset. Requires completion of unit committment. Line and Storage objects are assumed
    to have fuel costs of $0/hour and $0/MWh. Fuel costs are based upon a first order heat rate function, assuming each
    unit of the asset consumes fuel as a function of hours of operation and MWh of electricity generated.

    Parameters:
    -------
    ltcosts_instance (LTCosts_InstanceType): An instance of the LTCosts jitclass.
    generation (float64): Total energy (GWh) generated by a Generator.
    unit_hours (float64): Total generation unit-hours (i.e., hours that each unit of the Generator was generating) over
        the modelling horizon.
    unit_costs (UnitCost_InstanceType): A UnitCost jitclass instance that defines the exogenous cost assumptions
        for the asset.

    Returns:
    -------
    None.

    Side-effects:
    -------
    Attributes modified for the LTCosts instance: fuel.
    """
    ltcosts_instance.fuel = generation * 1e3 * unit_costs.fuel_cost_mwh + unit_hours * unit_costs.fuel_cost_h
    return None
