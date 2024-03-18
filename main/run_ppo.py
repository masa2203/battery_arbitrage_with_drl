import os
import json
from typing import Any

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
CREATE_LOG = True
VERBOSE = 0
LOGGER_TYPE = ["csv"]
SAVE_PATH = os.path.join(src_dir, 'log', ENV_KWARGS['env_name'], 'ppo', 'run', input('Save in folder: ')) \
    if CREATE_LOG else None

# ACTIONS
DISCRETE_ACTIONS = None

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

# PPO PARAMS
RL_PARAMS: dict[str, Any] = {
    'policy': "MlpPolicy" if EXP_PARAMS['flatten_obs'] else 'MultiInputPolicy',
    'device': 'cpu',
    # 'learning_rate': 3e-4,  # Default: 3e-4
    'learning_rate': linear_schedule(0.000183691611011775),  # Default: 3e-4
    'n_steps': 1024,  # Default: 2048
    'batch_size': 128,  # Default: 64
    'n_epochs': 20,  # Default: 10
    'gamma': 0.9999793305361806,  # Default: 0.99
    'gae_lambda': 0.9361350566791324,  # Default: 0.95
    'clip_range': 0.0917813271790492,  # Default: 0.2
    'clip_range_vf': None,  # Default: None
    'normalize_advantage': True,  # Default: True
    'ent_coef': 0.2869209453877102,  # Default: 0.0
    'vf_coef': 0.7889303144294002,  # Default: 0.5
    'max_grad_norm': 0.8844030872811149,  # Default: 0.5
    'use_sde': False,  # Default: False
    'target_kl': None,

    'policy_kwargs': {
        # Defaults reported for MultiInputPolicy
        'net_arch': 'extra_large',  # Default: None
        'activation_fn': 'tanh',  # Default: tanh
        'ortho_init': True,  # Default: True
        'squash_output': False,  # Default: False
        'share_features_extractor': True,  # Default: True
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


RL_PARAMS['policy_kwargs']['net_arch'] = net_arch_dict[RL_PARAMS['policy_kwargs']['net_arch']]
RL_PARAMS['policy_kwargs']['activation_fn'] = activation_fn_dict[RL_PARAMS['policy_kwargs']['activation_fn']]

for run in range(EXP_PARAMS['n_runs']):
    train_rl_agent(
        agent='ppo',
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
