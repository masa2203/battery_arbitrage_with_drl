"""
Manuel Sage, January 2024

Environment classes
"""
from typing import Optional
import numpy as np
from gymnasium import spaces

from envs.base_envs import BatteryBaseEnv


class FreeBatteryEnv(BatteryBaseEnv):
    """
    An environment class based on gym, with a battery as a dispatched component.

    This is a single battery working on energy arbitrage.

    :param env_name: A string that represents the name of the environment.
    :param data_file: A string that represents the path to the data file.
    :param state_vars: A list that represents the state variables.
    :param storage: A dictionary that represents the storage configuration of the environment.
    :param grid: A dictionary that represents the grid configuration of the environment.
    :param resolution_h: A float that represents the resolution of the simulation (in hours). Default is 1.0.
    :param modeling_period_h: An integer that represents the modeling period (in hours). Default is 8760.
    :param tracking: A boolean that indicates whether to track the variables. Default is True.
    :param debug: A boolean that indicates whether to show debug information. Default is False.
    :param precision_level: A string that represents the precision level of the variables. Must be either 'low',
    'medium', or 'high'. Default is 'low'.
    """
    def __init__(self,
                 env_name: str,
                 data_file: str,
                 state_vars: list,
                 storage: dict,
                 grid: dict,
                 resolution_h: float = 1.0,
                 modeling_period_h: int = 8760,
                 tracking: bool = True,
                 debug: bool = False,
                 precision_level: Optional[str] = "low",
                 ):
        """
        Initializes the FreeBatteryEnv class.

        :param env_name: A string that represents the name of the environment.
        :param data_file: A string that represents the path to the data file.
        :param state_vars: A list that represents the state variables.
        :param storage: A dictionary that represents the storage configuration of the environment.
        :param grid: A dictionary that represents the grid configuration of the environment.
        :param resolution_h: A float that represents the resolution of the simulation (in hours). Default is 1.0.
        :param modeling_period_h: An integer that represents the modeling period (in hours). Default is 8760.
        :param tracking: A boolean that indicates whether to track the variables. Default is True.
        :param debug: A boolean that indicates whether to show debug information. Default is False.
        :param precision_level: A string that represents the precision level of the variables. Must be either 'low',
        'medium', or 'high'. Default is 'low'.
        """
        super().__init__(env_name=env_name,
                         storage=storage,
                         grid=grid,
                         resolution_h=resolution_h,
                         modeling_period_h=modeling_period_h,
                         tracking=tracking,
                         precision_level=precision_level,
                         )

        self.data_file = data_file
        self.state_vars = state_vars
        self.debug = debug

        # PREPARE MAIN DATA FILE
        self.data = self._init_data(data_file=data_file,
                                    state_vars=state_vars,
                                    demand_file=None,
                                    num_wind_turbines=0,
                                    pv_capacity_mw=0)

        # DEFINE OBS SPACE
        self.observation_space = spaces.Dict(
            {
                'soc': spaces.Box(low=0, high=1, shape=(1,)),
            }
        )
        for clm in self.state_vars:  # add variables from data file to observation space
            low, high = self.data[clm].min(), self.data[clm].max()
            self.observation_space[clm] = spaces.Box(low=low, high=high, shape=(1,))

        # DEFINE ACTION SPACE
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=self.precision['float'])

    def step(self, action: np.array):
        """
        Runs one timestep of the environment's dynamics.

        :param action: A numpy array that represents the action to take.
        :return: A tuple of (observation, reward, terminated, truncated, info).
        """
        action = action.astype(self.precision["float"])
        self._action_checker(action)  # Check for bounds and NaNs

        # Conduct one step with the storage model
        self.storage_flow, bat_degr_cost = self.storage.step(action=action[0], avail_power=np.inf)

        # Compute sales of electricity
        e_sales = self.grid.get_grid_interaction(e_flow=self.storage_flow, pool_price=self.obs['pool_price'].item())

        # Subtract battery degradation cost
        reward = e_sales - bat_degr_cost

        if np.isnan(reward):
            raise ValueError('Reward is NAN!')

        if self.debug:
            print('#####################################')
            print(f'Time-step: {self.count}')
            print(f'\tObservation: {self.obs}')
            print(f'\tAction: {action} (Battery)')
            print(f'\tStorage flow: {round(self.storage_flow, 3)} | '
                  f'\tDegradation cost: {round(bat_degr_cost, 3)}')
            print(f'\tGrid sales: {round(e_sales, 3)} | '      
                  f'\tReward: {round(reward, 3)}')
            print()

        if self.tracking:
            self._tracking(
                action=list(map(lambda x: round(float(x), 3), action.tolist())),  # Convert to list of rounded floats
                reward=round(reward, 3),
            )

        self.count += 1
        next_obs, done = self._get_obs()

        return next_obs, reward, done, False, self._get_info()  # False for truncated

    def _get_obs(self):
        """
        Returns the observation from the current timestep.

        :return: A tuple of (observation, done).
        """
        if self.count == self.data.shape[0]:  # Check termination
            self.env_log = self._get_episode_info()
            return self.obs, True

        row = self.data.iloc[self.count]
        obs = {
            'soc': np.array([self.storage.soc], dtype=self.precision["float"])
        }
        for i in self.state_vars:
            obs[i] = np.array([row[i]], dtype=self.precision["float"])

        self.obs = obs
        return obs, False
