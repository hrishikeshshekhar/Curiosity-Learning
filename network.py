import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, layers, softmax=True):
        super(NeuralNetwork, self).__init__()
        assert(len(layers) >= 2)

        last_dim = layers[0]
        self.softmax = softmax
        self.layers = nn.ModuleList()

        # Building all the neural network layers and saving the layers in self.layers
        for index in range(1, len(layers)):
            layer = layers[index]
            torch_layer = nn.Linear(last_dim, layer)

            # Initializing all the weights with the xavier initializer
            torch.nn.init.xavier_uniform_(torch_layer.weight)
            self.layers.append(torch_layer)
            last_dim = layer

    def forward(self, state):
        # If the input is an np array, converting the np array into a torch tensor
        if(isinstance(state, np.ndarray)):
            state = torch.tensor(state, dtype=torch.float32)

        # Adding the ReLu activation function for all layers except the last layer
        total_layers = len(self.layers)
        for index, layer in enumerate(self.layers):
            if(index != total_layers - 1):
                state = F.relu(layer(state))
            # Not adding ReLu to the last layer
            else:
                state = layer(state)

        # Applying a softmax on the final output if required
        if(self.softmax):
            layer = torch.nn.Softmax(dim=-1)
            state = layer(state)

        return state