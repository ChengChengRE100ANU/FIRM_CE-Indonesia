from firm_ce.common.constants import JIT_ENABLED
from firm_ce.common.jit_overload import jitclass
from firm_ce.common.typing import float64, int64

if JIT_ENABLED:
    unitcost_spec = [
        ("capex_p", float64),
        ("fom", float64),
        ("vom", float64),
        ("lifetime", int64),
        ("discount_rate", float64),
        ("fuel_cost_mwh", float64),
        ("fuel_cost_h", float64),
        ("capex_e", float64),
        ("transformer_capex", float64),
    ]
else:
    unitcost_spec = []


@jitclass(unitcost_spec)
class UnitCost:
    """
    Exogenous cost assumptions for a Generator, Storage, or Line object.

    Attributes:
    -------
    capex_p (float64): Power capacity capital cost, units $/kW for Generator/Storage, $/MW-km for Line.
    capex_e (float64): Energy capacity capital cost, units $/kWh (for storage only).
    fom (float64): Fixed O&M cost, units $/kW/year for Generator/Storage, $/MW/km/year for Line.
    vom (float64): Variable O&M cost, units $/MWh.
    lifetime (int64): Asset economic lifetime in years.
    discount_rate (float64): Annual discount rate in range (0,1].
    fuel_cost_mwh (float64): First order marginal fuel cost term, $/MWh.
    fuel_cost_h (float64): Constant fuel cost term, $/h.
    transformer_capex (float64): Transformer-specific capital cost, units $/MW.
    """

    def __init__(
        self,
        capex_p: float64,
        fom: float64,
        vom: float64,
        lifetime: int64,
        discount_rate: float64,
        fuel_cost_mwh: float64,
        fuel_cost_h: float64,
        capex_e: float64,
        transformer_capex: float64,
    ) -> None:
        """
        Initialise UnitCost instance.

        Parameters:
        -------
        capex_p (float64): Power capacity capital cost, units $/kW for Generator/Storage, $/MW-km for Line.
        fom (float64): Fixed O&M cost, units $/kW/year for Generator/Storage, $/MW/km/year for Line.
        vom (float64): Variable O&M cost, units $/MWh.
        lifetime (int64): Asset economic lifetime in years.
        discount_rate (float64): Annual discount rate in range (0,1].
        fuel_cost_mwh (float64): First order marginal fuel cost term, $/MWh.
        fuel_cost_h (float64): Constant fuel cost term, $/h.
        capex_e (float64): Energy capacity capital cost, units $/kWh (for storage only).
        transformer_capex (float64): Transformer-specific capital cost, units $/MW.
        """

        self.capex_p = capex_p  # $/kW
        self.capex_e = capex_e  # $/kWh, non-zero for energy storage
        self.fom = fom  # $/kW/year
        self.vom = vom  # $/MWh
        self.lifetime = lifetime  # years
        self.discount_rate = discount_rate  # [0,1]

        self.fuel_cost_mwh = fuel_cost_mwh  # $/MWh = $/GJ * GJ/MWh
        self.fuel_cost_h = fuel_cost_h  # $/h/unit = $/GJ * GJ/h/unit

        self.transformer_capex = transformer_capex  # $/kW, non-zero for lines


if JIT_ENABLED:
    UnitCost_InstanceType = UnitCost.class_type.instance_type
else:
    UnitCost_InstanceType = UnitCost

if JIT_ENABLED:
    ltcosts_spec = [
        ("annualised_build", float64),
        ("fom", float64),
        ("vom", float64),
        ("fuel", float64),
    ]
else:
    ltcosts_spec = []


@jitclass(ltcosts_spec)
class LTCosts:
    """
    Endogenously derived total long-term costs over the modelling horizon.

    Attributes:
    -------
    annualised_build (float64): The total build costs for an asset over the entire modelling horizon, annualised
        using the net present value factor. Each year over the modelling horizon is assumed to incur the annualised
        build cost of each asset. Existing assets in the system are assumed to be fully depreciated by the start of
        the modelling period. Units of $/modelling period.
    fom (float64): The total fixed operation and maintenance (FO&M) costs for an asset. Existing assets in the system are
        assumed to be fully depreciated by the start of the modelling period. Fully depreciated assets still incur FO&M.
        Units of $/modelling period.
    vom (float64): The total variable operation and maintenance (VO&M) costs for an asset. Requires completion
        of unit committment. Units of $/modelling period.
    fuel (float64): The fuel costs for an asset. Requires completion of unit committment. Line and Storage objects are assumed
        to have fuel costs of $0. Fuel costs are based upon a first order heat rate function, assuming each
        unit of the asset consumes fuel as a function of hours of operation and MWh of electricity generated. Units of
        $/modelling period.
    """

    def __init__(self):
        """
        Initialise the LTCosts instance.

        All attributes are initialised to 0.0 since they are calculated at the end of the unit committment process.
        """
        self.annualised_build = 0.0
        self.fom = 0.0
        self.vom = 0.0
        self.fuel = 0.0


if JIT_ENABLED:
    LTCosts_InstanceType = LTCosts.class_type.instance_type
else:
    LTCosts_InstanceType = LTCosts
