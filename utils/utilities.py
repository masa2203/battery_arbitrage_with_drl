import os
import time
import random
import numpy as np
import pandas as pd
import torch


# Get source directory
file_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(file_dir, os.pardir))


def get_env_log_data(env, mean_reward, start_time):
    """
    Gets the data to be logged for a given environment.

    :param env: A gym environment.
    :param mean_reward: A float that represents the mean reward.
    :param start_time: A float that represents the start time.
    :return: A dictionary that contains the data to be logged.
    """
    # Try-except to handle different env-wrappers
    try:
        episode_info = env.unwrapped.envs[0].unwrapped.return_episode_info()
    except AttributeError:
        episode_info = env.return_episode_info()

    stats = {
        'reward_sum': mean_reward,
        'compute_time': time.time() - start_time,
    }

    discharge_count = len(list(filter(lambda x: (x > 0), episode_info['bes_energy_flows'])))
    charge_count = len(list(filter(lambda x: (x < 0), episode_info['bes_energy_flows'])))

    bes_stats = {
        'degr_cost_sum': sum(episode_info['degr_costs']),
        'avg_soc': sum(episode_info['socs']) / len(episode_info['socs']),
        'num_charging': charge_count,
        'num_discharging': discharge_count,
    }
    stats = stats | bes_stats

    # Add tracked time-series
    log_data = stats | episode_info

    return log_data


def create_stats_file(path, exp_params):
    """
    Creates a CSV file that contains the training and evaluation statistics.

    :param path: A string that represents the path to the directory where the CSV file will be created.
    :param exp_params: A dictionary that contains the experiment parameters.
    """
    stats = pd.DataFrame(columns=[i for i in range(exp_params['n_episodes'] + 1)])
    if exp_params['eval_while_training']:
        eval_episodes = int(exp_params['n_episodes'] * 8760 / exp_params['eval_freq'])
        eval_stats = pd.DataFrame(columns=[i for i in range(eval_episodes)])
    count = 0
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('eval_monitor.csv'):
                df = pd.read_csv(os.path.join(subdir, file))
                eval_stats.loc[count] = [float(i) for i in df.index.values.tolist()[1:]]
            if file.endswith('train_monitor.csv') and 'eval' not in file:
                df = pd.read_csv(os.path.join(subdir, file))
                stats.loc[count] = [float(i) for i in df.index.values.tolist()[1:]]
                count += 1

    mean = stats.mean(axis=0)
    std = stats.std(axis=0)
    stats.loc['mean'] = mean
    stats.loc['std'] = std
    stats.to_csv(os.path.join(path, 'stats.csv'))
    print('Mean episodic rewards over all runs: ')
    print()
    print(mean)

    if exp_params['eval_while_training']:
        mean = eval_stats.mean(axis=0)
        std = eval_stats.std(axis=0)
        eval_stats.loc['mean'] = mean
        eval_stats.loc['std'] = std
        eval_stats.to_csv(os.path.join(path, 'eval_stats.csv'))


def set_seeds(seed):
    """
    Fixes the random seed for all relevant packages.

    :param seed: An integer that represents the seed to be set.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
