import torch
import torch.nn as nn
from torch.nn import LSTM
import torch.nn.functional as F

from functools import reduce


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class cReLU(nn.Module):
    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), dim=1)


class DQNet(nn.Module):
    """
    Fully connected Q network in PyTorch
    Parameters:
        state_shape (list of int) : shape of state - (6, 5, 15)
        num_actions (int) : number of possible actions this agent can take - 309
        mlp_layers (list) : list of hidden layer sizes to use in fully connected network
        activation (str) : activation functions: 'tanh' or 'relu'
    """

    def __init__(self, state_shape, num_actions, mlp_layers, activation='relu'):
        super(DQNet, self).__init__()

        flattened_state_size = reduce(lambda x, y: x * y, state_shape)

        layer_dims = [flattened_state_size] + mlp_layers + [num_actions]

        # with batch normalization layer no need for flatten layer
        #fc = [nn.BatchNorm1d(layer_dims[0])]
        fc = []
        # experiment with dropout

        # activations on all layers but the last
        for i in range(len(layer_dims) - 2):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i + 1], bias=True))
            if activation == 'relu':
                fc.append(nn.ReLU())
            else:
                fc.append(nn.Tanh())

        fc.append(nn.Linear(layer_dims[-2], layer_dims[-1], bias=True))
        self.layers = nn.Sequential(*fc)

    def forward(self, state):
        state = torch.flatten(state, start_dim=1)
        q_values = self.layers(state)
        return q_values


class AveragePolicyNet(nn.Module):
    def __init__(self, state_shape, num_actions, mlp_layers, activation='relu'):
        super(AveragePolicyNet, self).__init__()
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.mlp_layers = mlp_layers

        flattened_state_size = reduce(lambda x, y: x * y, state_shape)

        layer_dims = [flattened_state_size] + self.mlp_layers + [self.num_actions]

        # with batch normalization layer no need for flatten layer
        fc = [nn.BatchNorm1d(layer_dims[0])]
        # fc = []
        # experiment with dropout

        # activations on all layers but the last
        for i in range(len(layer_dims) - 2):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if activation == 'relu':
                fc.append(nn.ReLU())
            else:
                fc.append(nn.Tanh())

        fc.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
        self.layers = nn.Sequential(*fc)

    def forward(self, state):
        state = torch.flatten(state, start_dim=1)
        logits = self.layers(state)
        # q_values = F.log_softmax(logits, dim=-1) # this one is used in rlcard
        q_values = F.softmax(logits, dim=-1)
        return q_values


class DRQNet(nn.Module):
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
        super(DRQNet, self).__init__()

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


class DuelingDQNet(nn.Module):
    # fully connected DuelingDQNet
    def __init__(self, state_shape, num_actions,):
        super(DuelingDQNet, self).__init__()
        self.state_shape = state_shape
        self.num_actions = num_actions
        flattened_state_size = reduce(lambda x, y: x * y, state_shape)
        self.fc1 = nn.Linear(flattened_state_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.adv_fc1 = nn.Linear(512, 512)
        self.adv_fc2 = nn.Linear(512, self.num_actions)

        self.value_fc1 = nn.Linear(512, 512)
        self.value_fc2 = nn.Linear(512, 1)

    def forward(self, state):
        state = torch.flatten(state, start_dim=1)
        # feature = self.fc1(state)
        feature = self.fc2(F.relu(self.fc1(state)))
        advantage = self.adv_fc2(F.relu(self.adv_fc1(F.relu(feature))))
        value = self.value_fc2(F.relu(self.value_fc1(F.relu(feature))))
        adv_average = torch.mean(advantage, dim=1, keepdim=True)
        q_values = advantage + value - adv_average

        return q_values


class DQNConv(nn.Module):
    def __init__(self, state_shape, num_actions, conv=True):
        super(DQNConv, self).__init__()

        self.flatten = Flatten()
        self.state_shape = state_shape
        self.num_actions = num_actions
        if conv:
            self.features = nn.Sequential(
                nn.Conv2d(self.state_shape[0], 10, kernel_size=5, stride=1),
                cReLU(),
            )

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
                nn.Linear(flattened_state_shape, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, self.num_actions),
            )

    def forward(self, state):
        x = self.features(state)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def _feature_size(self):
        return self.features(torch.zeros(1, *self.state_shape)).view(1, -1).size(1)


class AveragePolicyConv(DQNConv):
    def __init__(self, state_shape, num_actions):
        super(AveragePolicyConv, self).__init__(state_shape, num_actions)
        self.state_shape = state_shape
        self.num_actions = num_actions

        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(), 32),
            nn.ReLU(),
            nn.Linear(32, self.num_actions),
            nn.Softmax(dim=1)
        )

    def act(self, state):
        with torch.no_grad():
            state = state.unsqueeze(0)
            distribution = self.forward(state)
            action = distribution.multinomial(1).item()
        return distribution, action


class DuelingDQNConv(nn.Module):
    def __init__(self, state_shape, num_actions):
        super(DuelingDQNConv, self).__init__()

        self.flatten = Flatten()
        self.state_shape = state_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(self.state_shape[0], 10, kernel_size=5, stride=1),
            cReLU(),
            #nn.Conv2d(self.state_shape[0], 8, kernel_size=3, stride=1),
            #cReLU(),
            #nn.Conv2d(16, 8, kernel_size=3, stride=1),
            #cReLU(),
            )

        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(), 512),
            nn.ReLU(),
            #nn.Linear(512, 512),
            #nn.ReLU()
        )
        self.advantage = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, state):
        x = self.features(state)
        x = self.flatten(x)
        x = self.fc(x)
        advantage = self.advantage(x)
        value = self.value(x)
        adv_average = torch.mean(advantage, dim=1, keepdim=True)
        logits = advantage + value - adv_average
        q_values = F.softmax(logits, dim=-1)
        return q_values

    def _feature_size(self):
        return self.features(torch.zeros(1, *self.state_shape)).view(1, -1).size(1)
