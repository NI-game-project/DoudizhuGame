import torch
import torch.nn as nn
from torch.nn import LSTM
import torch.nn.functional as F

from functools import reduce
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class cReLU(nn.Module):
    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), dim=1)


class DQN(nn.Module):
    def __init__(self, state_shape, num_actions, use_conv=False):
        super(DQN, self).__init__()

        self.flatten = Flatten()
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.use_conv = use_conv
        if self.use_conv:

            self.features = nn.Sequential(
                nn.Conv2d(self.state_shape[0], 32, kernel_size=5, stride=1),
                nn.BatchNorm2d(32),
                #cReLU(),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, kernel_size=1, stride=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
            )
            """
            conv1 = nn.Conv2d(self.state_shape[0], 32, kernel_size=5, stride=4, padding=2)
            torch.nn.init.xavier_uniform_(conv1.weight)
            torch.nn.init.constant_(conv1.bias, 0.1)
            conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2)
            torch.nn.init.xavier_uniform_(conv2.weight)
            torch.nn.init.constant_(conv2.bias, 0.1)
            conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2)
            torch.nn.init.xavier_uniform_(conv3.weight)
            torch.nn.init.constant_(conv3.bias, 0.1)
            conv4 = nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=2)
            torch.nn.init.xavier_uniform_(conv4.weight)
            torch.nn.init.constant_(conv4.bias, 0.1)
            self.features = nn.Sequential(
                conv1,
                nn.BatchNorm2d(32),
                nn.ReLU(),
                conv2,
                nn.BatchNorm2d(64),
                nn.ReLU(),
                #nn.MaxPool2d(kernel_size=2, stride=2),
                conv3,
                nn.BatchNorm2d(64),
                nn.ReLU(),
                #conv4,
                #nn.BatchNorm2d(64),
                #nn.ReLU(),
                #nn.MaxPool2d(kernel_size=2, stride=2)
            )
            """
            self.fc = nn.Sequential(
                nn.Linear(self._feature_size(), 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, self.num_actions),
            )
        else:
            flattened_state_shape = reduce(lambda x, y: x * y, self.state_shape)
            self.fc = nn.Sequential(
                nn.Linear(flattened_state_shape, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, self.num_actions),
                nn.Tanh()
            )

    def forward(self, state):
        if self.use_conv:
            x = self.features(state)
            x = self.flatten(x)
            q_values = self.fc(x)

        else:
            x = torch.flatten(state, start_dim=1)
            q_values = self.fc(x)

        return q_values

    def _feature_size(self):
        return self.features(torch.zeros(1, *self.state_shape)).view(1, -1).size(1)


class DuelingDQN(nn.Module):
    def __init__(self, state_shape, num_actions, use_conv):
        super(DuelingDQN, self).__init__()

        self.flatten = Flatten()
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.use_conv = use_conv

        if self.use_conv:

            self.features = nn.Sequential(
                nn.Conv2d(self.state_shape[0], 32, kernel_size=5, stride=1),
                cReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=1, stride=1),
                cReLU(),
                nn.BatchNorm2d(64),
                )

            self.fc = nn.Sequential(
                nn.Linear(self._feature_size(), 512),
                nn.ReLU(),
            )
        else:
            flattened_state_shape = reduce(lambda x, y: x * y, state_shape)
            self.fc = nn.Sequential(
                nn.Linear(flattened_state_shape, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
            )

        self.advantage = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions),
        )
        self.value = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        if self.use_conv:
            x = self.features(state)
            x = self.flatten(x)
        else:
            x = torch.flatten(state, start_dim=1)
        x = self.fc(x)
        advantage = self.advantage(x)
        value = self.value(x)
        adv_average = torch.mean(advantage, dim=1, keepdim=True)
        q_values = advantage + value - adv_average
        # q_values = torch.sigmoid(q_values)

        return q_values

    def _feature_size(self):
        return self.features(torch.zeros(1, *self.state_shape)).view(1, -1).size(1)


class NoisyLinear(nn.Module):
    def __init__(self, input_dim, output_dim, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(self.output_dim, self.input_dim))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(self.output_dim, self.input_dim))
        self.register_buffer('weight_epsilon', torch.FloatTensor(self.output_dim, self.input_dim))

        self.bias_mu = nn.Parameter(torch.FloatTensor(self.output_dim))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(self.output_dim))
        self.register_buffer('bias_epsilon', torch.FloatTensor(self.output_dim))

        self.reset_parameter()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def reset_parameter(self):
        mu_range = 1 / np.sqrt(self.input_dim)

        self.weight_mu.detach().uniform_(-mu_range, mu_range)
        self.bias_mu.detach().uniform_(-mu_range, mu_range)

        self.weight_sigma.detach().fill_(self.std_init / np.sqrt(self.input_dim))
        self.bias_sigma.detach().fill_(self.std_init / np.sqrt(self.input_dim))

    def _scale_noise(self, size):
        noise = torch.randn(size)
        noise = noise.sign().mul(noise.abs().sqrt())
        return noise

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.output_dim))


class DQNNoisy(nn.Module):
    def __init__(self, state_shape, num_actions):
        super(DQNNoisy, self).__init__()
        self.state_shape = state_shape
        self.num_actions = num_actions
        flattened_state_shape = reduce(lambda x, y: x * y, self.state_shape)
        self.noisy1 = NoisyLinear(512, 512)
        self.noisy2 = NoisyLinear(512, self.num_actions)
        self.fc = nn.Sequential(
            nn.Linear(flattened_state_shape, 512),
            nn.ReLU(),
            self.noisy1,
            nn.ReLU(),
            self.noisy2)

    def forward(self, state):
        state = torch.flatten(state, start_dim=1)
        x = self.fc(state)
        return x

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


class DRQN(nn.Module):
    """
    Recurrent q network in pytorch
    parameters:
        state_shape (list of int): shape of the state
        num_actions (int) : number of possible actions that an agent can take
        recurrent_layer_size (int) : size of hidden state of recurrent layer
        recurrent_layers_num (int) : number of recurrent layers to use
        mlp_layers (list): list of mlp hidden layer sizes
        describing the fully connected network from the hidden state to output
        activation (str): which activation func to use? 'tanh' or 'relu'
    """

    def __init__(self, state_shape, num_actions,
                 recurrent_layer_size, recurrent_layers_num,
                 mlp_layers, activation='relu'):
        super(DRQN, self).__init__()

        # initialize lstm layers
        self.flattened_state_size = reduce(lambda x, y: x * y, state_shape)
        self.recurrent_layer_size = recurrent_layer_size
        self.recurrent_layers_num = recurrent_layers_num
        self.lstm_layers = LSTM(input_size=self.flattened_state_size,
                                hidden_size=recurrent_layer_size,
                                num_layers=recurrent_layers_num,
                                )

        layer_dims = [recurrent_layer_size] + mlp_layers + [num_actions]
        fc_layers = []

        for i in range(len(layer_dims) - 2):
            fc_layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if activation == 'relu':
                fc_layers.append(nn.ReLU())
            else:
                fc_layers.append(nn.Tanh())

        fc_layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
        self.fc_layers = nn.Sequential(*fc_layers)
        self.init_hidden(1)

    def forward(self, state):
        state = torch.flatten(state, start_dim=1)
        x, (self.hidden, self.cell) = \
            self.lstm_layers(state.view(-1, 1, self.flattened_state_size), (self.hidden, self.cell))

        q_values = self.fc_layers(x)
        
        return q_values
    
    def init_hidden(self, size):
        self.hidden = torch.zeros(self.recurrent_layers_num, size, self.recurrent_layer_size)
        self.cell = torch.zeros(self.recurrent_layers_num, size, self.recurrent_layer_size)


class CategoricalDQNet(nn.Module):
    """
    actor(policy) model
    """

    def __init__(self, state_shape, num_actions, mlp_layers, num_atoms=51, Vmin=-10., Vmax=10., activation='relu'):
        """Initialize parameters and build model.
                Params
                    state_size (int): Dimension of each state
                    action_size (int): Dimension of each action
                    mlp_layers(list): list of fully connected layers
        """
        super(CategoricalDQNet, self).__init__()
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers

        flattened_state_size = reduce(lambda x, y: x * y, state_shape)
        layer_dims = [flattened_state_size] + self.mlp_layers
        fc = []
        for i in range(len(layer_dims) - 1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i + 1], bias=True))
            if activation == 'relu':
                fc.append(nn.ReLU())
            else:
                fc.append(nn.Tanh())
        fc.append(nn.Linear(layer_dims[-1], self.num_actions * self.num_atoms, bias=True))
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, state):
        """
        build a network that maps state to action values
        """
        state = torch.flatten(state, start_dim=1)
        x = self.fc_layers(state)
        # [batch_size, num_actions, num_atoms)
        output = F.softmax(x.view(-1, self.num_actions, self.num_atoms), dim=2)
        return output


class AveragePolicyNet(DQN):
    def __init__(self, state_shape, num_actions, use_conv=True):
        super(AveragePolicyNet, self).__init__(state_shape, num_actions)
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.use_conv = use_conv

        if self.use_conv:
            self.fc = nn.Sequential(
                nn.Linear(self._feature_size(), 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, self.num_actions),
                nn.Softmax(dim=-1)
            )
        else:
            flattened_state_shape = reduce(lambda x, y: x * y, self.state_shape)
            self.fc = nn.Sequential(
                nn.Linear(flattened_state_shape, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, self.num_actions),
                nn.Softmax(dim=-1)
            )
