"""
Manuel Sage, October 2023

Class to model electricity grid and its interaction with the plant (modified from previous projects)
"""
from typing import Optional


class GridModel:
    """
    Class to model the grid and its interaction with the plant.

    :param demand_profile: A string that specifies the type of demand. Valid values are 'industry', 'grid', or None
    (no demand).
    :param sell_surplus: A boolean that indicates whether surplus power can be sold to the grid (only with demand).
    :param buy_deficit: A boolean that indicates whether deficient power can be bought from grid (only with demand).
    :param spread: A float that represents the amount added to the price of bought power.
    :param penalty: A float that represents the penalty for deficient power (only with demand).
    """
    def __init__(self,
                 demand_profile: Optional[str] = None,  # if None no demand is used, else specify demand type
                 sell_surplus: Optional[bool] = False,  # if set to True, surplus power is sold (only with demand)
                 buy_deficit: Optional[bool] = True,  # if True, deficit power is bought from grid (only with demand)
                 spread: float = 0.0,  # in $/MWh, added to price of bought power
                 penalty: Optional[float] = None,  # in $/MWh, added to deficient power (only with demand)
                 ):
        """
        Initialize a new GridModel object.

        :param demand_profile: A string that specifies the type of demand. Valid values are 'industry', 'grid', or None
        (no demand).
        :param sell_surplus: A boolean that indicates whether surplus power can be sold to the grid (only with demand).
        :param buy_deficit: A boolean that indicates whether deficient power can be bought from grid (only with demand).
        :param spread: A float that represents the amount added to the price of bought power.
        :param penalty: A float that represents the penalty for deficient power (only with demand).
        """
        assert demand_profile in [None, 'industry', 'grid'], "Invalid demand_profile."
        assert spread >= 0, "Spread must be greater than or equal to 0."
        assert penalty is None or penalty >= 0, "Penalty must be greater than or equal to 0."

        self.demand_profile = demand_profile
        self.sell_surplus = sell_surplus
        self.buy_deficit = buy_deficit
        self.spread = spread
        self.penalty = penalty

    def get_grid_interaction(self,
                             e_flow: float,
                             pool_price: Optional[float] = None,
                             demand: Optional[float] = None) -> float:
        """
        Models grid response to produced electricity.

        :param e_flow: A float that represents the electricity flow.
        :param pool_price: A float that represents the pool price.
        :param demand: A float that represents the power demand.
        :return: A float that represents the cash flow.
        """
        # Free interaction with the grid, no quantity limits
        if self.demand_profile is None:
            return self._get_free_interaction(e_flow, pool_price)
        else:
            raise NotImplementedError('This project only handles case studies without demand profile.')

    def _get_free_interaction(self, e_flow: float, pool_price: float) -> float:
        """
        Calculate the cash flow for free interaction with the grid.

        :param e_flow: A float that represents the electricity flow (negative = purchase power).
        :param pool_price: A float that represents the pool price in $/MWh.
        :return: A float that represents the cash flow.
        """
        if e_flow < 0:  # Purchase power (e.g. energy arbitrage with a battery)
            return e_flow * (pool_price + self.spread)
        else:  # sell power
            return e_flow * pool_price
