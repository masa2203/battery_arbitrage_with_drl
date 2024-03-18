"""
Manuel Sage, January 2024

Base classes of environments.
"""
import abc
from typing import Optional

import numpy as np
import gymnasium as gym
import pandas as pd

from envs.grid_model import GridModel
from envs.storage_model import BESS, DODDegradingBESS


class BaseEnv(gym.Env):
    """
    A parent class for all environment classes based on gym.

    :param env_name: A string that represents the name of the environment.
    :param grid: A dictionary that represents the grid configuration of the environment.
    :param resolution_h: A float that represents the resolution of the simulation (in hours). Default is 1.0.
    :param modeling_period_h: An integer that represents the modeling period (in hours). Default is 8760.
    :param tracking: A boolean that indicates whether to track the variables. Default is True.
    :param precision_level: A string that represents the precision level of the variables. Must be either 'low', 'medium',
    or 'high'. Default is 'low'.
    """

    def __init__(self,
                 env_name: str,
                 grid: dict,
                 resolution_h: Optional[float] = 1.0,
                 modeling_period_h: Optional[int] = 8760,
                 tracking: Optional[bool] = True,
                 precision_level: Optional[str] = "low"
                 ):
        """
        Initializes the BaseEnv class.

        :param env_name: A string that represents the name of the environment.
        :param grid: A dictionary that represents the grid configuration of the environment.
        :param resolution_h: A float that represents the resolution of the simulation (in hours). Default is 1.0.
        :param modeling_period_h: An integer that represents the modeling period (in hours). Default is 8760.
        :param tracking: A boolean that indicates whether to track the variables. Default is True.
        :param precision_level: A string that represents the precision level of the variables. Must be either 'low', 'medium',
        or 'high'. Default is 'low'.
        """
        super(BaseEnv, self).__init__()

        self.env_name = env_name
        self.grid_dict = grid
        self.resolution_h = resolution_h
        self.modeling_period_h = modeling_period_h
        self.tracking = tracking
        self.precision = precision_level

        # Grid
        self.grid = self._init_grid()

        self.count = 0
        self.obs = None

        # Tracking
        self.tracked_vars = {}
        self.env_log = None

    @property
    def precision(self):
        """
        A property that returns the precision of the variables as a dictionary with keys 'float' and 'int'.

        :return: A dictionary that represents the precision of the variables.
        """
        return {"float": self._precision_float, "int": self._precision_int}

    @precision.setter
    def precision(self, value):
        """
        A setter method for the precision of the variables.

        :param value: A string that represents the precision level of the variables. Must be either 'low', 'medium', or
        'high'.
        """
        if value == "low":
            self._precision_float = np.float32
            self._precision_int = np.int8
        elif value == "medium":
            self._precision_float = np.float64
            self._precision_int = np.int16
        elif value == "high":
            self._precision_float = np.float128
            self._precision_int = np.int32

    @abc.abstractmethod
    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.

        :param seed: An integer that represents the random seed. Default is None.
        :param options: A dictionary of options. Default is None.
        """
        super().reset(seed=seed)  # Defines np's random generator

        # Reset counter
        self.count = 0

        # Reset trackers
        self.tracked_vars = {k: [] for k in self.tracked_vars}

    @abc.abstractmethod
    def step(self, action):
        raise NotImplementedError("Subclasses of BaseEnv must implement step-method!")

    def render(self, **kwargs):
        raise NotImplementedError("Rendering not implemented!")

    @abc.abstractmethod
    def _get_obs(self):
        raise NotImplementedError("Subclasses of BaseEnv must implement _get_obs-method!")

    def _init_grid(self):
        """
        Initializes the grid model of the environment.

        :return: A GridModel object that represents the grid model of the environment.
        """
        return GridModel(**self.grid_dict)

    def _get_episode_info(self):
        """
        Returns the tracked variables for the current episode.

        :return: A dictionary that represents the tracked variables for the current episode.
        """
        return self.tracked_vars

    def return_episode_info(self):
        """
        Returns the logged information for the current episode.

        :return: A dictionary that represents the logged information for the current episode.
        """
        return self.env_log

    def start_tracking(self):
        """
        Enables tracking of environment variables for the last evaluation episode.
        """
        self.tracking = True

    def _get_info(self):
        """
        Returns the tracked variables for the current time-step.

        :return: A dictionary that represents the tracked variables for the current time-step.
        """
        if self.tracking and isinstance(self.tracked_vars, dict):
            # Return last value for each tracked variable
            return {k: v[-1] if v else None for k, v in self.tracked_vars.items()}
        else:
            return {}

    @staticmethod
    def _init_data(data_file: str,
                   state_vars: list,
                   demand_file: Optional[str],
                   num_wind_turbines: int = 0,
                   pv_capacity_mw: float = 0):
        """
        Initializes the data for the environment.

        :param data_file: A string that represents the path to the data file.
        :param state_vars: A list of strings that represents the state variables of the environment.
        :param demand_file: A string that represents the path to the demand file. Default is None.
        :param num_wind_turbines: An integer that represents the number of wind turbines. Default is 0.
        :param pv_capacity_mw: A float that represents the PV capacity (in MW). Default is 0.
        :return: A pandas DataFrame that represents the data for the environment.
        """
        # Open main data file
        data = pd.read_csv(data_file, index_col=0)

        if num_wind_turbines != 0 or pv_capacity_mw != 0:
            data['re_power'] = np.nan
            if 'wind_power' in data.columns and 'pv_power' in data.columns:
                data['wind_power'] *= num_wind_turbines
                data['pv_power'] *= pv_capacity_mw
                data['re_power'] = data['pv_power'] + data['wind_power'] / 1000
            elif 'wind_power' in data.columns and 'pv_power' not in data.columns:
                data['wind_power'] *= num_wind_turbines
                data['re_power'] = data['wind_power'] / 1000
            elif 'pv_power' in data.columns and 'wind_power' not in data.columns:
                data['pv_power'] *= pv_capacity_mw
                data['re_power'] = data['pv_power']
            else:
                raise ValueError(f'Passed num_wind_turbines = {num_wind_turbines} and pv_capacity = {pv_capacity_mw}, '
                                 f'but "wind_power" and/or "pv_power" columns not in data file.')

        # Remove all columns except the 'Date' column that are not in the observation space
        data.drop([i for i in data.columns if i != 'Date' and i not in state_vars],
                  axis=1, inplace=True)

        # Open demand file and add demand (in MW) to data file
        if demand_file is not None:
            demand = pd.read_csv(demand_file, index_col=0)
            data['demand'] = demand['demand'] / 1000  # add demand column to data-file, convert to MW

        return data

    def _action_checker(self, action: np.array):
        """
        Checks if the action is within the bounds of the action space and does not contain NaN values.

        :param action: A numpy array that represents the action to check.
        """
        if np.isnan(action).any():
            raise gym.error.InvalidAction(f'Action must not contain NaN values. Passed action: {action}')
        if not self.action_space.contains(action):
            raise gym.error.InvalidAction(f'Action must be within the action space bounds. Passed action: {action}')


class BatteryBaseEnv(BaseEnv):
    """
    A parent class for environments operating a battery.

    :param env_name: A string that represents the name of the environment.
    :param storage: A dictionary that represents the storage configuration of the environment.
    :param grid: A dictionary that represents the grid configuration of the environment.
    :param resolution_h: A float that represents the resolution of the simulation (in hours). Default is 1.0.
    :param modeling_period_h: An integer that represents the modeling period (in hours). Default is 8760.
    :param tracking: A boolean that indicates whether to track the variables. Default is True.
    :param precision_level: A string that represents the precision level of the variables. Must be either 'low', 'medium',
    or 'high'. Default is 'low'.
    """

    def __init__(self,
                 env_name: str,
                 storage: dict,
                 grid: dict,
                 resolution_h: Optional[float] = 1.0,
                 modeling_period_h: Optional[int] = 8760,
                 tracking: Optional[bool] = True,
                 precision_level: Optional[str] = "low"):
        """
        Initializes the BatteryBaseEnv class.

        :param env_name: A string that represents the name of the environment.
        :param storage: A dictionary that represents the storage configuration of the environment.
        :param grid: A dictionary that represents the grid configuration of the environment.
        :param resolution_h: A float that represents the resolution of the simulation (in hours). Default is 1.0.
        :param modeling_period_h: An integer that represents the modeling period (in hours). Default is 8760.
        :param tracking: A boolean that indicates whether to track the variables. Default is True.
        :param precision_level: A string that represents the precision level of the variables. Must be either 'low', 'medium',
        or 'high'. Default is 'low'.
        """
        BaseEnv.__init__(self,
                         env_name=env_name,
                         grid=grid,
                         resolution_h=resolution_h,
                         modeling_period_h=modeling_period_h,
                         tracking=tracking,
                         precision_level=precision_level)

        # Storage
        self.storage_dict = storage
        self.storage = self._init_storage()
        self.storage_flow = None  # required for ActionCorrectionPenalty wrapper

        # Tracking
        self.tracked_vars = {
            'actions': [],
            'rewards': [],
            'e_balances': [],  # deficit or surplus power (if demand)
        }

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.

        :param seed: An integer that represents the random seed. Default is None.
        :param options: A dictionary of options. Default is None.
        :return: A tuple that contains the observation and the logged information for the current episode.
        """
        super().reset(seed=seed)

        # Reset storage and get first observation
        self.storage.reset(rng=self.np_random, options={'tracking': self.tracking})
        obs, done = self._get_obs()
        return obs, self.return_episode_info()

    def partial_reset(self, n: int):
        """
        Partially resets the environment.

        :param n: An integer that represents the number of time-steps to reset.
        :return: The observation after the partial reset.
        """
        assert self.tracking, "Cannot partial reset without tracking states"

        if self.count > n:
            self.count -= n
            # Partially reset trackers
            self.tracked_vars = {k: v[:-n] for k, v in self.tracked_vars.items()}

            self.storage.partial_reset(n)
            obs, done = self._get_obs()
            return obs
        else:
            return self.reset()

    @abc.abstractmethod
    def step(self, action):
        raise NotImplementedError("Subclasses of BatteryBaseEnv must implement step-method!")

    @abc.abstractmethod
    def _get_obs(self):
        raise NotImplementedError("Subclasses of BatteryBaseEnv must implement _get_obs-method!")

    def _init_storage(self):
        """
        Initializes the storage model of the environment.

        :return: A Battery Energy Storage System (BESS) object that represents the storage model of the environment.
        """
        # Simple storage without degradation
        if self.storage_dict['degradation'] is None:
            storage = BESS(**self.storage_dict,
                           resolution_h=self.resolution_h,
                           tracking=self.tracking)

        # Storage with degradation based on depth of discharge (DOD), cyclic ageing only
        elif self.storage_dict['degradation']['type'] == 'DOD':
            storage = DODDegradingBESS(**self.storage_dict,
                                       resolution_h=self.resolution_h,
                                       tracking=self.tracking)
        else:
            raise ValueError('Unknown storage type called!')
        return storage

    def _tracking(self, action, reward, e_balance=None):
        """
        Keeps track of the environment's behavior over time.

        :param action: An object that represents the action taken.
        :param reward: A float that represents the reward.
        :param e_balance: A float that represents the energy balance (if demand exists). Default is None.
        """
        self.tracked_vars['actions'].append(action)
        self.tracked_vars['rewards'].append(reward)
        if e_balance is not None:
            self.tracked_vars['e_balances'].append(e_balance)  # e-delivered - demand

    def _get_episode_info(self):
        """
        Returns the information about the current episode.

        :return: A dictionary containing information about the current episode.
        """
        self.tracked_vars.update({
            'socs': self.storage.socs,
            'bes_energy_flows': self.storage.energy_flows,
            'degr_costs': self.storage.degr_costs,
        })
        return self.tracked_vars
