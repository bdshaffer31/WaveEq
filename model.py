import torch
from torch import nn


class FCNN(nn.Module):
    """Fully Connected Neural Network (FCNN) / MLP, minimal implementation"""

    # fully connected neural network
    def __init__(
        self,
        input_dim,
        output_dim,
        n_hidden_layers=4,
        hidden_dim=128,
        activation_fn=nn.ReLU(),
    ):
        super(FCNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation_fn)
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn)
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class FCNNWithSkip(nn.Module):  # consider renaming to resnet
    """Fully Connected Neural Network (FCNN) with Skip Connections"""

    def __init__(
        self,
        input_dim,
        output_dim,
        n_hidden_layers=4,
        hidden_dim=128,
        activation_fn=nn.ReLU(),
    ):
        super(FCNNWithSkip, self).__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        self.activation_fn = activation_fn

        # Create hidden layers
        for _ in range(n_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def apply(self, x):
        # Apply input layer
        x = self.activation_fn(self.input_layer(x))

        # Pass through hidden layers with skip connections
        for layer in self.hidden_layers:
            x = x + self.activation_fn(layer(x))  # Add residual connection

        # Apply output layer
        return self.output_layer(x)

    def forward(self, x):
        return self.apply(x)