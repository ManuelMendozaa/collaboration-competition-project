import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """ Actor (policy-based) model """
    def __init__(self, state_size, action_size, seed=14, layer_units=[128, 128]):
        super(Actor, self).__init__()
        self._state_size = state_size              #  environment state space
        self._action_size = action_size            #  environment discrete action space
        self._seed = torch.manual_seed(seed)       #  torch init seed

        # Arquitecture
        num_layers = len(layer_units)
        layers = [nn.Linear(self._state_size, layer_units[0])]

        for i in range(1, num_layers):
            layers.append(nn.Linear(layer_units[i-1], layer_units[i]))

        self._layers = nn.ModuleList(layers)
        self._output_layer = nn.Linear(layer_units[num_layers-1], action_size)

        # Initialize parameters
        self.init_parameters()

    def hidden_init(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)

    def init_parameters(self):
        for i in range(len(self._layers)):
            self._layers[i].weight.data.uniform_(*self.hidden_init(self._layers[i]))

        self._output_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = state
        for layer in self._layers:
            x = F.relu(layer(x))

        return F.tanh(self._output_layer(x))


class Critic(nn.Module):
    """ Critic (value-based) model """
    def __init__(self, state_size, action_size, seed=14, layer_units=[64, 64], dueling=True):
        super(Critic, self).__init__()
        self._state_size = state_size              #  environment state space
        self._action_size = action_size            #  environment discrete action space
        self._layer_units = layer_units            #  model arquitecture
        self._seed = torch.manual_seed(seed)       #  torch init seed
        self._dueling = dueling                    #  dueling boolean

        # Arquitecture
        num_layers = len(layer_units)
        final_units = layer_units[num_layers-1]
        layers = [nn.Linear(self._state_size, layer_units[0])]

        for i in range(1, num_layers):
            layers.append(nn.Linear(layer_units[i-1], layer_units[i]))

        self._layers = nn.ModuleList(layers)
        self._join_layer = nn.Linear(final_units + (self._action_size), final_units)
        self._V_layer = nn.Linear(final_units, 1)
        self._A_layer = nn.Linear(final_units, action_size)

        # Initialize parameters
        self.init_parameters()

    def hidden_init(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)

    def init_parameters(self):
        for i in range(len(self._layers)):
            self._layers[i].weight.data.uniform_(*self.hidden_init(self._layers[i]))

        self._join_layer.weight.data.uniform_(-3e-3, 3e-3)

        self._V_layer.weight.data.uniform_(-3e-3, 3e-3)
        self._A_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = state
        for layer in self._layers:
            x = F.relu(layer(x))

        # Join actions with state analysis
        x = torch.cat((x, action), dim=1)
        x = F.relu(self._join_layer(x))

        # Action values alone
        A_values = self._A_layer(x)
        if not self._dueling: return A_values

        # Dueling steps
        V_value = self._V_layer(x)
        Q_values = V_value + A_values - A_values.mean(1, keepdim=True)
        return Q_values
