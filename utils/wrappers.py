import json
import os
import pickle
from collections import deque

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from forecasters.models import model_dict
from utils.net_design import activation_fn_dict

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Get variables from wrapper warnings


class DiscreteActions(gym.ActionWrapper):
    """
    A gymnasium action wrapper that converts discrete actions to continuous actions.

    :param env: A gymnasium environment.
    :param disc_to_cont: A list that represents the discrete actions.
    """
    def __init__(self, env, disc_to_cont):
        """
        Initializes the DiscreteActions class.

        :param env: A gymnasium environment.
        :param disc_to_cont: A list that represents the discrete actions.
        """
        super().__init__(env)
        self.disc_to_cont = disc_to_cont
        self.action_space = gym.spaces.Discrete(len(disc_to_cont))

    def action(self, act):
        """
        Converts the discrete action to a continuous action.

        :param act: An integer that represents the discrete action.
        :return: A numpy array that represents the continuous action.
        """
        return np.array(self.disc_to_cont[act]).astype(self.env.action_space.dtype)

    def reverse_action(self, action):
        """
        Raises a NotImplementedError.

        :param action: A numpy array that represents the action.
        :raises: NotImplementedError.
        """
        raise NotImplementedError


class PerfectPriceForecasts(gym.ObservationWrapper):
    """
    A gymnasium observation wrapper that adds perfect price forecasts to the observation.

    :param env: A gymnasium environment.
    :param forecasts: A list of integers that represent the number of hours ahead for each forecast.
    """
    def __init__(self, env, forecasts: list):
        """
        Initializes the PerfectPriceForecasts class.

        :param env: A gymnasium environment.
        :param forecasts: A list of integers that represent the number of hours ahead for each forecast.
        """
        super().__init__(env)
        self.forecasts = forecasts

        high = self.observation_space['pool_price'].high
        low = self.observation_space['pool_price'].low
        for f in self.forecasts:
            self.observation_space[f'pp_in_{f}h'] = spaces.Box(low=low, high=high, shape=(1,))

    def observation(self, obs):
        """
        Adds the perfect price forecasts to the observation.

        :param obs: A dictionary that represents the observation.
        :return: A dictionary that represents the modified observation.
        """
        for f in self.forecasts:
            try:
                obs[f'pp_in_{f}h'] = np.array([self.env.data.loc[self.env.count + f]['pool_price']])
            except KeyError:
                obs[f'pp_in_{f}h'] = obs['pool_price']
        return obs


class PriceForecasts(gym.ObservationWrapper):
    """
    A gymnasium observation wrapper that adds price forecasts from pretrained predictors to the observation.

    :param env: A gymnasium environment.
    :param log_folder_path: A string that represents the path to the log folder.
    :param path_datafile: A string that represents the path to the data file.
    """
    def __init__(self,
                 env,
                 log_folder_path: str,
                 path_datafile: str):
        """
        Initializes the PriceForecasts class.

        :param env: A gymnasium environment.
        :param log_folder_path: A string that represents the path to the log folder.
        :param path_datafile: A string that represents the path to the data file.
        """
        super().__init__(env)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get experiment inputs
        with open(os.path.join(log_folder_path, 'inputs.json'), 'r') as f:
            inputs = json.load(f)
        window_size = inputs['HYPERPARAMETERS']['WINDOW_SIZE']
        self.h = inputs['EXP_PARAMS']['HOURS_AHEAD']

        # Load dataframe
        df = pd.read_csv(path_datafile,
                         index_col=0,
                         parse_dates=['Date'])

        # Get relevant columns
        relevant_cols = inputs['HYPERPARAMETERS']['Features']
        in_dim = len(relevant_cols)
        if inputs['HYPERPARAMETERS']['Target_Column'] not in inputs['HYPERPARAMETERS']['Features']:
            relevant_cols.append(inputs['Target_Column'])

        # Remove all irrelevant columns and take only 2022
        df = df.drop(df.columns.difference(relevant_cols), axis=1)
        df = df[df.index.year == 2022]

        # Load scalers
        with open(os.path.join(log_folder_path, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(log_folder_path, 'target_scaler.pkl'), 'rb') as f:
            self.target_scaler = pickle.load(f)

        # Apply scaling on whole dataset and replace unscaled columns in dataframe by scaled ones
        scaled_array = pd.DataFrame(scaler.transform(df[inputs['HYPERPARAMETERS']['Columns_to_scale']].values),
                                    columns=inputs['HYPERPARAMETERS']['Columns_to_scale'])
        for column in inputs['HYPERPARAMETERS']['Columns_to_scale']:
            df[column] = scaled_array[column].values

        self.data = df.values

        activation = activation_fn_dict[inputs['HYPERPARAMETERS']['model_params']['activation']]()
        self.d = deque([self.data[0] for i in range(window_size)], maxlen=window_size)
        if inputs['MODEL'] == 'LSTM':
            self.model = model_dict[inputs['MODEL']](
                in_dim=in_dim,
                out_dim=1,
                # window_size=window_size,
                activation=activation,
                lstm_num_layer=inputs['HYPERPARAMETERS']['model_params']['lstm_num_layer'],
                lstm_layer_size=inputs['HYPERPARAMETERS']['model_params']['lstm_layer_size'],
                ann_net_shape=inputs['HYPERPARAMETERS']['model_params']['ann_net_shape']
            )

        if inputs['MODEL'] == 'CNN':
            self.model = model_dict[inputs['MODEL']](
                in_dim=in_dim,
                out_dim=1,
                window_size=window_size,
                activation=activation,
                cnn_net_shape=inputs['HYPERPARAMETERS']['model_params']['cnn_net_shape'],
                cnn_kernel_size=inputs['HYPERPARAMETERS']['model_params']['cnn_kernel_size'],
                cnn_stride=inputs['HYPERPARAMETERS']['model_params']['cnn_stride'],
                ann_net_shape=inputs['HYPERPARAMETERS']['model_params']['ann_net_shape']
            )

        if inputs['MODEL'] == 'Hybrid':
            self.model = model_dict[inputs['MODEL']](
                in_dim=in_dim,
                out_dim=1,
                # window_size=window_size,
                activation=activation,
                cnn_net_shape=inputs['HYPERPARAMETERS']['model_params']['cnn_net_shape'],
                cnn_kernel_size=inputs['HYPERPARAMETERS']['model_params']['cnn_kernel_size'],
                cnn_stride=inputs['HYPERPARAMETERS']['model_params']['cnn_stride'],
                lstm_num_layer=inputs['HYPERPARAMETERS']['model_params']['lstm_num_layer'],
                lstm_layer_size=inputs['HYPERPARAMETERS']['model_params']['lstm_layer_size'],
                ann_net_shape=inputs['HYPERPARAMETERS']['model_params']['ann_net_shape']
            )

        if inputs['MODEL'] == 'AttentionHybrid':
            self.model = model_dict[inputs['Model']](in_dim=in_dim,
                                                     out_dim=1,
                                                     activation=activation_fn_dict[inputs['ACTIVATION']](),
                                                     cnn_net_shape=inputs['CNN_NET_SHAPE'],
                                                     cnn_kernel_size=inputs['CNN_KERNEL_SIZE'],
                                                     cnn_stride=inputs['CNN_STRIDE'],
                                                     lstm_layer_size=inputs['LSTM_LAYER_SIZE'],
                                                     lstm_num_layer=inputs['LSTM_NUM_LAYER'],
                                                     mha_num_heads=inputs['MHA_HEADS'],
                                                     mha_dropout=inputs['MHA_DROPOUT'],
                                                     ann_net_shape=inputs['ANN_SHAPE'])

        # Load model weights
        self.model.load_state_dict(torch.load(os.path.join(log_folder_path, 'NN_params.pt'), map_location=self.device))

        high = self.observation_space['pool_price'].high
        low = self.observation_space['pool_price'].low
        self.observation_space[f'forecast_{self.h}h'] = spaces.Box(low=low, high=high, shape=(1,))

    def observation(self, obs):
        """
        Adds the price forecasts to the observation.

        :param obs: A dictionary that represents the observation.
        :return: A dictionary that represents the modified observation.
        """
        if self.env.count == 8760:
            return obs
        self.d.append(self.data[self.env.count])
        model_input = np.asarray(self.d)
        model_input = torch.tensor(model_input, dtype=torch.float).unsqueeze(0)  # Convert to tensor and add batch dim
        pred = self.model(model_input).detach().numpy()
        pred = self.target_scaler.inverse_transform(pred)
        obs[f'forecast_{self.h}h'] = pred[0]
        return obs
