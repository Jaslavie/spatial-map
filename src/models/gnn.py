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
    # output: grid cell activity patterns as a 1D vector
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

class PlaceCellModule(nn.Module):
    # activate specific types of cells at specific landmark locations
    # num_place_cells: number of different locations of place cells (how finely tuned the env is mapped)
    # place fields are locations on the map where specific place cells are most active
    def __init__(self, input_size, num_place_cells, grid_map):
        super(PlaceCellModule, self).__init__()
        self.input_size = input_size
        self.num_place_cells = num_place_cells 
        self.grid_map = grid_map
        # map input to output space (all potential place cells)
        self.place_cells = nn.linear(input_size, num_place_cells)
        self._initialize_place_fields()
    
    def _initialize_place_fields(self):
        # init place fields with grid cell activations
        # rigid representation of the environment updated with rl agent
        nn.init.normal_(self.place_cells.weight)
        # generate grid map
        positions = torch.rand((self.num_place_cells, 2)) * 2 - 1 # (x, y)
        grid_activations = self.grid_map(positions)
        # initialize place fields based on grid activations
        # parameters are saved to train the rl agent
        self.place_fields = nn.Parameter(grid_activations.clone().detach())
    def forward(self, pos):
        # forward pass to get place cell activations based on position tensor
        # calculate the euclidean distance between the agent position and place cell
        # use a gaussian function to represent activation size (higher near the center, lower around edges)
        dist = torch.cdist(pos.unsqueeze(0), self.place_fields)
        activations = torch.exp(-dist ** 2 / 0.1)  # 0.1 controls place field width
        return activations
