from typing import List

import torch.nn as nn


class LSTM(nn.Module):
    """
    A PyTorch module that implements a Long Short-Term Memory (LSTM) neural network.
    """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 activation: nn.modules.activation = nn.ReLU(),
                 lstm_layer_size: int = 32,
                 lstm_num_layer: int = 2,
                 ann_net_shape: List[int] = [32, 32],  # not including output size!
                 ):
        """
        Initializes the LSTM class.

        :param in_dim: An integer that represents the input dimension.
        :param out_dim: An integer that represents the output dimension.
        :param activation: A PyTorch activation function.
        :param lstm_layer_size: An integer that represents the size of the LSTM layer.
        :param lstm_num_layer: An integer that represents the number of LSTM layers.
        :param ann_net_shape: A list of integers that represents the shape of the ANN.
        """
        super(LSTM, self).__init__()

        # LSTM with batch_first=True needs (batch, sequence, features)
        self.lstm = nn.LSTM(input_size=in_dim,
                            hidden_size=lstm_layer_size,
                            num_layers=lstm_num_layer,
                            batch_first=True)

        # ANN
        ann_layers = [nn.Linear(in_features=lstm_layer_size,
                                out_features=ann_net_shape[0]),
                      activation]
        for i in range(len(ann_net_shape) - 1):
            ann_layers.append(nn.Linear(ann_net_shape[i], ann_net_shape[i + 1]))
            ann_layers.append(activation)

        # Output layer
        ann_layers.append(nn.Linear(in_features=ann_net_shape[-1], out_features=out_dim))

        # Wrap with sequential module
        self.ann = nn.Sequential(*ann_layers)

    def forward(self, x):
        """
        Performs a forward pass through the LSTM neural network.

        :param x: A PyTorch tensor that represents the input.
        :return: A PyTorch tensor that represents the output.
        """
        x, _ = self.lstm(x)
        x = self.ann(x[:, -1, :])  # batch_first=True
        return x


class CNN(nn.Module):
    """
    A PyTorch module that implements a Convolutional Neural Network (CNN).
    """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 window_size: int,
                 activation: nn.modules.activation = nn.ReLU(),
                 cnn_net_shape: List[int] = [32],  # not including input dimension!
                 cnn_kernel_size: int = 3,
                 cnn_stride: int = 1,
                 ann_net_shape: List[int] = [32, 32],  # not including output size!
                 ):
        """
        Initializes the CNN class.

        :param in_dim: An integer that represents the input dimension.
        :param out_dim: An integer that represents the output dimension.
        :param window_size: An integer that represents the size of the window.
        :param activation: A PyTorch activation function.
        :param cnn_net_shape: A list of integers that represents the shape of the CNN.
        :param cnn_kernel_size: An integer that represents the size of the CNN kernel.
        :param cnn_stride: An integer that represents the stride of the CNN kernel.
        :param ann_net_shape: A list of integers that represents the shape of the ANN.
        """
        super(CNN, self).__init__()

        # 1D CNN needs (batch, features, sequence)
        # CNN input layer
        cnn_layers = [nn.Conv1d(in_channels=in_dim,
                                out_channels=cnn_net_shape[0],
                                kernel_size=cnn_kernel_size,
                                stride=cnn_stride),
                      activation]
        for i in range(len(cnn_net_shape) - 1):
            cnn_layers.append(nn.Conv1d(in_channels=cnn_net_shape[i],
                                        out_channels=cnn_net_shape[i + 1],
                                        kernel_size=cnn_kernel_size,
                                        stride=cnn_stride))
            cnn_layers.append(activation)

        # Wrap with sequential module
        self.cnn = nn.Sequential(*cnn_layers)

        # ANN
        # Compute size of CNN output after flattening
        in_features = ((window_size - cnn_kernel_size) // cnn_stride) + 1
        for i in range(len(cnn_net_shape) - 1):
            in_features = ((in_features - cnn_kernel_size) // cnn_stride) + 1
        in_features *= cnn_net_shape[-1]

        # Define ANN Layers
        ann_layers = [nn.Linear(in_features=in_features,
                                out_features=ann_net_shape[0]),
                      activation]
        for i in range(len(ann_net_shape) - 1):
            ann_layers.append(nn.Linear(ann_net_shape[i], ann_net_shape[i + 1]))
            ann_layers.append(activation)

        # Output layer
        ann_layers.append(nn.Linear(in_features=ann_net_shape[-1], out_features=out_dim))

        # Wrap with sequential module
        self.ann = nn.Sequential(*ann_layers)

    def forward(self, x):
        """
        Performs a forward pass through the CNN neural network.

        :param x: A PyTorch tensor that represents the input.
        :return: A PyTorch tensor that represents the output.
        """
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.flatten(1)
        x = self.ann(x)
        return x


class CnnLstmHybrid(nn.Module):
    """
    A PyTorch module that implements a hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) neural network.
    """
    def __init__(self,
                 in_dim: int = 2,
                 out_dim: int = 1,
                 activation: nn.modules.activation = nn.ReLU(),
                 cnn_net_shape: List[int] = [32, 32],  # not including input dimension!
                 cnn_kernel_size: int = 3,
                 cnn_stride: int = 1,
                 lstm_layer_size: int = 32,
                 lstm_num_layer: int = 2,
                 ann_net_shape: List[int] = [32, 32],  # not including output size!
                 ):
        """
        Initializes the CnnLstmHybrid class.

        :param in_dim: An integer that represents the input dimension.
        :param out_dim: An integer that represents the output dimension.
        :param activation: A PyTorch activation function.
        :param cnn_net_shape: A list of integers that represents the shape of the CNN.
        :param cnn_kernel_size: An integer that represents the size of the CNN kernel.
        :param cnn_stride: An integer that represents the stride of the CNN kernel.
        :param lstm_layer_size: An integer that represents the size of the LSTM layer.
        :param lstm_num_layer: An integer that represents the number of LSTM layers.
        :param ann_net_shape: A list of integers that represents the shape of the ANN.
        """
        super(CnnLstmHybrid, self).__init__()

        # 1D CNN needs (batch, features, sequence)
        # CNN input layer
        cnn_layers = [nn.Conv1d(in_channels=in_dim,
                                out_channels=cnn_net_shape[0],
                                kernel_size=cnn_kernel_size,
                                stride=cnn_stride),
                      activation]
        for i in range(len(cnn_net_shape) - 1):
            cnn_layers.append(nn.Conv1d(in_channels=cnn_net_shape[i],
                                        out_channels=cnn_net_shape[i + 1],
                                        kernel_size=cnn_kernel_size,
                                        stride=cnn_stride))
            cnn_layers.append(activation)

        # Wrap with sequential module
        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM with batch_first=True needs (batch, sequence, features)
        self.lstm = nn.LSTM(input_size=cnn_net_shape[-1],
                            hidden_size=lstm_layer_size,
                            num_layers=lstm_num_layer,
                            batch_first=True)

        # ANN
        ann_layers = [nn.Linear(in_features=lstm_layer_size,
                                out_features=ann_net_shape[0]),
                      activation]
        for i in range(len(ann_net_shape) - 1):
            ann_layers.append(nn.Linear(ann_net_shape[i], ann_net_shape[i + 1]))
            ann_layers.append(activation)

        # Output layer
        ann_layers.append(nn.Linear(in_features=ann_net_shape[-1], out_features=out_dim))

        # Wrap with sequential module
        self.ann = nn.Sequential(*ann_layers)

    def forward(self, x):
        """
        Performs a forward pass through the hybrid CNN-LSTM neural network.

        :param x: A PyTorch tensor that represents the input.
        :return: A PyTorch tensor that represents the output.
        """
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.ann(x[:, -1, :])
        return x


class CnnLstmAttentionHybrid(nn.Module):
    """
    A PyTorch module that implements a hybrid Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM),
    and Multi-head Attention (MHA) neural network.
    """
    def __init__(self,
                 in_dim: int = 2,
                 out_dim: int = 1,
                 activation: nn.modules.activation = nn.ReLU(),
                 cnn_net_shape: List[int] = [32, 32],  # not including input dimension!
                 cnn_kernel_size: int = 3,
                 cnn_stride: int = 1,
                 lstm_layer_size: int = 32,
                 lstm_num_layer: int = 2,
                 mha_num_heads: int = 4,
                 mha_dropout: float = 0.1,
                 ann_net_shape: List[int] = [32, 32],  # not including output size!
                 ):
        """
        Initializes the CnnLstmAttentionHybrid class.

        :param in_dim: An integer that represents the input dimension.
        :param out_dim: An integer that represents the output dimension.
        :param activation: A PyTorch activation function.
        :param cnn_net_shape: A list of integers that represents the shape of the CNN.
        :param cnn_kernel_size: An integer that represents the size of the CNN kernel.
        :param cnn_stride: An integer that represents the stride of the CNN kernel.
        :param lstm_layer_size: An integer that represents the size of the LSTM layer.
        :param lstm_num_layer: An integer that represents the number of LSTM layers.
        :param mha_num_heads: An integer that represents the number of heads in the MHA.
        :param mha_dropout: A float that represents the dropout rate for the MHA.
        :param ann_net_shape: A list of integers that represents the shape of the ANN.
        """
        super(CnnLstmAttentionHybrid, self).__init__()

        # 1D CNN needs (batch, features, sequence)
        # CNN input layer
        cnn_layers = [nn.Conv1d(in_channels=in_dim,
                                out_channels=cnn_net_shape[0],
                                kernel_size=cnn_kernel_size,
                                stride=cnn_stride),
                      activation]
        for i in range(len(cnn_net_shape) - 1):
            cnn_layers.append(nn.Conv1d(in_channels=cnn_net_shape[i],
                                        out_channels=cnn_net_shape[i + 1],
                                        kernel_size=cnn_kernel_size,
                                        stride=cnn_stride))
            cnn_layers.append(activation)

        # Wrap with sequential module
        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM with batch_first=True needs (batch, sequence, features)
        self.lstm = nn.LSTM(input_size=cnn_net_shape[-1],
                            hidden_size=lstm_layer_size,
                            num_layers=lstm_num_layer,
                            batch_first=True)

        # Multi-head Attention
        self.attention = nn.MultiheadAttention(embed_dim=lstm_layer_size,
                                               num_heads=mha_num_heads,
                                               dropout=mha_dropout,
                                               batch_first=True)

        # ANN
        ann_layers = [nn.Linear(in_features=lstm_layer_size,
                                out_features=ann_net_shape[0]),
                      activation]
        for i in range(len(ann_net_shape) - 1):
            ann_layers.append(nn.Linear(ann_net_shape[i], ann_net_shape[i + 1]))
            ann_layers.append(activation)

        # Output layer
        ann_layers.append(nn.Linear(in_features=ann_net_shape[-1], out_features=out_dim))

        # Wrap with sequential module
        self.ann = nn.Sequential(*ann_layers)

    def forward(self, x):
        """
        Performs a forward pass through the hybrid CNN-LSTM-MHA neural network.

        :param x: A PyTorch tensor that represents the input.
        :return: A PyTorch tensor that represents the output.
        """
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x, _ = self.attention(x, x, x)
        x = self.ann(x[:, -1, :])
        return x


model_dict = {
    'CNN': CNN,
    'LSTM': LSTM,
    'Hybrid': CnnLstmHybrid,
    'AttentionHybrid': CnnLstmAttentionHybrid,
}
