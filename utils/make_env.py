from typing import Optional, Dict, Any

from gymnasium.wrappers import FlattenObservation

from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from utils.wrappers import *


def make_env(env,
             env_kwargs: dict,
             tracking: bool = False,
             allow_early_resets: bool = True,
             path: Optional[str] = None,
             perfect_forecasts: Optional[list] = None,
             forecasts: Optional[Dict[str, Any]] = None,
             flatten_obs: bool = True,
             discrete_actions: Optional[list] = None,
             norm_obs: bool = True,
             norm_reward: bool = True,
             gamma: float = 0.99,
             ):
    """
    Creates a gym environment and applies a set of wrappers.

    :param env: A subclass of gym.Env that represents the environment.
    :param env_kwargs: A dictionary that represents the keyword arguments to pass to the environment.
    :param tracking: A boolean that indicates whether to track the variables. Default is False.
    :param allow_early_resets: A boolean that indicates whether to allow early resets. Default is True.
    :param path: A string that represents the path to save the monitor. Default is None.
    :param perfect_forecasts: A list of perfect forecasting horizons. Default is None.
    :param forecasts: A dictionary with paths pointing to trained forecasters and data files. Default is None.
    :param flatten_obs: A boolean that indicates whether to flatten the observation. Default is True.
    :param discrete_actions: A list that represents the discrete actions. Default is None.
    :param norm_obs: A boolean that indicates whether to normalize the observation. Default is True.
    :param norm_reward: A boolean that indicates whether to normalize the reward. Default is True.
    :param gamma: A float that represents the gamma value. Default is 0.99.
    :return: A stable_baselines3.common.vec_env.VecNormalize object that represents the wrapped environment.
    """
    e = Monitor(env=env(**env_kwargs, tracking=tracking),
                allow_early_resets=allow_early_resets,  # allow finish rollout for PPO -> throws error otherwise
                filename=path)

    if perfect_forecasts is not None:
        e = PerfectPriceForecasts(e, forecasts=perfect_forecasts)

    if forecasts is not None:
        for f in forecasts['log_folder_paths']:
            e = PriceForecasts(e, log_folder_path=f, path_datafile=forecasts['path_datafile'])

    if flatten_obs:
        e = FlattenObservation(e)

    # Add discrete action wrapper
    if discrete_actions is not None:
        e = DiscreteActions(e, discrete_actions)

    e = DummyVecEnv([lambda: e])

    e = VecNormalize(e, norm_obs=norm_obs, norm_reward=norm_reward, gamma=gamma)

    return e
