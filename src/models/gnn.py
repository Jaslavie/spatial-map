# graph neural network implementation for grid and place cell activity
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class GridCellModule(nn.Module):
    # implement grid cells to encode spatial position with hexagonal firing patterns
    # uses sinusoidal interference (combination of multiple sine wave patterns) to form a hexagonal pattern
    # import pre-existing nn architecture from pytorch modules
    def __init__(self, input_size, output_size, scale):
        super(GridCellModule, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.scale = scale # scale factor to normalize input

        # apply linear transformation to map input (cell encodings) -> output (grid cell activity patterns)
        # this places input features into the latent feature space 
        self.linear = nn.Linear(input_size, output_size, bias=True)

        # init frequency vectors (3 sine waves) based on triangular orientation
        self.k_vectors = nn.Parameter(torch.tensor([
            [1.0, 0.0],  # Direction 1 (0°)
            [-0.5, 0.866],  # Direction 2 (120°)
            [-0.5, -0.866]  # Direction 3 (240°)
        ]) * scale)

        # phase shifts
        # start all waves at 0
        # gradient descent will adjust these values iteratively during training (i.e. apply phase shifts)
        self.phases = nn.Parameter(torch.zeros(3))
        
    #TODO: potentially use RRN's here instead of current pos for more accurate contextual representation
    def forward(self, pos):
        # forward pass through grid cell modules (spatial encoding -> grid cell activation)
        # get raw activations
        latent = self.linear(pos)

        # compute periodic activation patterns
        # Compute activity at each point: dot product between latent space and sine wave vectors
        grid_outputs = torch.cos(torch.matmul(pos, self.k_vectors.T) + self.phases)

        # Compute pattern: Sum the contributions of all three vectors
        grid_encoding = torch.sum(grid_outputs, dim=-1, keepdim=True)

        # apply non-linear activation function
        return F.tanh(grid_encoding)

grid_cell_layer = GridCellModule(input_size=2, output_size=16, scale=5.0)
position = torch.tensor([[5.0, 5.0], [10.0, 15.0]])  # Example positions
grid_activation = grid_cell_layer(position)
print(grid_activation)