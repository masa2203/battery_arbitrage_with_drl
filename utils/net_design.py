import torch.nn as nn

net_arch_dict = {
    "tiny": {"pi": [64], "vf": [64], "qf": [64]},
    'small': {"pi": [64, 64], "vf": [64, 64], "qf": [64, 64]},
    'medium': {"pi": [128, 128], "vf": [128, 128], "qf": [128, 128]},
    'large': {"pi": [128, 256, 128], "vf": [128, 256, 128], "qf": [128, 256, 128]},
    'extra_large': {"pi": [256, 512, 512, 256], "vf": [256, 512, 512, 256], "qf": [256, 512, 512, 256]},
    'ddpg': {"pi": [400, 300], "vf": [400, 300], "qf": [400, 300]},
    'sac': {"pi": [256, 256], "vf": [256, 256], "qf": [256, 256]},
}

activation_fn_dict = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
}

"""
Defaults:

PPO:
dict(pi=[64, 64], vf=[64, 64])
nn.Tanh

DDPG: 
[400, 300] for actor and critic(s)
nn.ReLU

DQN:
[64, 64]
nn.ReLU

SAC:
[256, 256]
nn.ReLU

A2C:
dict(pi=[64, 64], vf=[64, 64])
nn.Tanh
"""