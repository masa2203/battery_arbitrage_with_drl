import os
import json
from typing import Any

import numpy as np

from envs.environments import FreeBatteryEnv
from envs.env_params import al4_bat_ea
from utils.net_design import activation_fn_dict, net_arch_dict
from utils.scheduler import linear_schedule
from utils.utilities import create_stats_file, src_dir
from train.train import train_rl_agent

# PLANT PARAMS
ENV = FreeBatteryEnv
ENV_KWARGS = al4_bat_ea

# LOG
CREATE_LOG = False
VERBOSE = 0
LOGGER_TYPE = ["csv"]
SAVE_PATH = os.path.join(src_dir, 'log', ENV_KWARGS['env_name'], 'dqn', 'run', input('Save in folder: ')) \
    if CREATE_LOG else None

# ACTIONS
DISCRETE_ACTIONS = [np.array([-1]), np.array([0]), np.array([1])]

# EXP PARAMS
EXP_PARAMS = {
    'n_runs': 5,
    'n_episodes': 50,
    'seed': 22,
    # Env
    'flatten_obs': True,
    # Normalization
    'norm_obs': True,
    'norm_reward': True,
    # Evaluation
    'eval_while_training': True,
    'eval_freq': 8760 * 1,
    # Perfect forecasts
    # 'perfect_forecasts': [1, 2, 3, 6, 12, 18, 24],
    'perfect_forecasts': None,
    # Actual forecasts
    # 'forecasts': None,
    'forecasts': {'log_folder_paths': [
        os.path.join(src_dir, 'forecasters/trained_models/lstm1hour'),
        os.path.join(src_dir, 'forecasters/trained_models/hybrid2hours'),
        os.path.join(src_dir, 'forecasters/trained_models/cnn3hours'),
        os.path.join(src_dir, 'forecasters/trained_models/cnn6hours'),
        os.path.join(src_dir, 'forecasters/trained_models/cnn8hours'),
        os.path.join(src_dir, 'forecasters/trained_models/cnn12hours'),
        os.path.join(src_dir, 'forecasters/trained_models/cnn18hours'),
        os.path.join(src_dir, 'forecasters/trained_models/cnn24hours'),
    ],
        'path_datafile': os.path.join(src_dir, 'data/alberta3/ab_2018-2022_electricity_time_climate.csv')},
}

# DQN PARAMS
RL_PARAMS: dict[str, Any] = {
    'policy': "MlpPolicy" if EXP_PARAMS['flatten_obs'] else 'MultiInputPolicy',
    # 'learning_rate': 0.00176746728919149,  # Default: 1e-4
    'learning_rate': linear_schedule(0.00176746728919149),  # Default: 1e-4
    'buffer_size': 500_000,  # Default: 1e6
    'learning_starts': 255,  # Default: 50_000
    'batch_size': 256,  # Default: 32
    'tau': 0.5016120493544259,  # Default: 1.0
    'gamma': 0.9999812912592504,  # Default: 0.99
    'train_freq': 84,  # Default: 4
    'gradient_steps': -1,  # Default: 1
    'target_update_interval': 10_000,  # Default: 1e4
    'exploration_fraction': 0.5,  # Default: 0.1
    'exploration_initial_eps': 1.0,  # Default: 1.0
    'exploration_final_eps': 0.005,  # Default: 0.05
    'max_grad_norm': 3.266151433390378,  # Default: 10

    'policy_kwargs': {
        # Defaults reported for MultiInputPolicy
        'net_arch': 'extra_large',  # Default: None
        'activation_fn': 'leaky_relu',  # Default: tanh
    }
}

if SAVE_PATH is not None:
    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_PATH, 'inputs.json'), 'w') as f:
        json.dump({
            'DISCRETE_ACTIONS': str(DISCRETE_ACTIONS),
            'EXP_PARAMS': EXP_PARAMS,
            'RL_PARAMS': str(RL_PARAMS),
            'PLANT_PARAMS': ENV_KWARGS,
        }, f)

RL_PARAMS['policy_kwargs']['net_arch'] = net_arch_dict[RL_PARAMS['policy_kwargs']['net_arch']]['qf']
RL_PARAMS['policy_kwargs']['activation_fn'] = activation_fn_dict[RL_PARAMS['policy_kwargs']['activation_fn']]

for run in range(EXP_PARAMS['n_runs']):
    train_rl_agent(
        agent='dqn',
        run=run,
        path=SAVE_PATH,
        exp_params=EXP_PARAMS,
        env_id=ENV,
        env_kwargs=ENV_KWARGS,
        discrete_actions=DISCRETE_ACTIONS,
        rl_params=RL_PARAMS,
        verbose=VERBOSE,
        logger_type=LOGGER_TYPE,
    )

# GET STATISTICS FROM MULTIPLE RUNS
if CREATE_LOG and EXP_PARAMS['n_runs'] > 1:
    create_stats_file(SAVE_PATH, EXP_PARAMS)
