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
        # self.linear = nn.Linear(input_size, output_size, bias=True)

        # init frequency vectors (3 sine waves) based on triangular orientation
        angles = torch.tensor([0, np.pi/3, 2*np.pi/3])
        self.freq_x = nn.Parameter(torch.cos(angles) * scale)
        self.freq_y = nn.Parameter(torch.sin(angles) * scale)
        # self.k_vectors = nn.Parameter(torch.tensor([
        #     [1.0, 0.0],  # Direction 1 (0°)
        #     [-0.5, 0.866],  # Direction 2 (120°)
        #     [-0.5, -0.866]  # Direction 3 (240°)
        # ]) * scale)

        # phase shifts
        # start all waves at 0
        # gradient descent will adjust these values iteratively during training (i.e. apply phase shifts)
        self.phases = nn.Parameter(torch.zeros(3))
        
    #TODO: potentially use RRN's here instead of current pos for more accurate contextual representation
    def forward(self, pos):
        # forward pass through grid cell modules (spatial encoding -> grid cell activation)
        # compute periodic activation patterns
        # Compute activity at each point: dot product between latent space and sine wave vectors
        batch_size = pos.shape[0]
        x = pos[:, 0].view(batch_size, 1)  
        y = pos[:, 1].view(batch_size, 1)

        grid_outputs = torch.cos(x * self.freq_x + y * self.freq_y + self.phases)
        return grid_outputs
        # # Compute pattern: Sum the contributions of all three vectors
        # grid_encoding = torch.tanh(grid_outputs)

        # # apply non-linear activation function
        # return F.tanh(grid_encoding)

class PlaceCellModule(nn.Module):
    # activate specific types of cells at specific landmark locations
    # num_place_cells: number of different locations of place cells (how finely tuned the env is mapped)
    # place fields are locations on the map where specific place cells are most active
    # output: place cell activations as a 1D vector
    def __init__(self, input_size, num_place_cells):
        super(PlaceCellModule, self).__init__()
        self.input_size = input_size
        self.num_place_cells = num_place_cells 
        self.place_fields = nn.Parameter(torch.rand((num_place_cells, 2)) * 2 - 1)
        self.sigma = nn.Parameter(torch.tensor(0.1)) # width of gaussian place fields
    
    # def _initialize_place_fields(self):
    #     # init place fields with grid cell activations
    #     # rigid representation of the environment updated with rl agent
    #     nn.init.normal_(self.place_cells.weight)
    #     # generate grid map
    #     positions = torch.rand((self.num_place_cells, 2)) * 2 - 1 # (x, y)
    #     grid_activations = self.grid_map(positions)
    #     # initialize place fields based on grid activations
    #     # parameters are saved to train the rl agent
    #     self.place_fields = nn.Parameter(grid_activations.clone().detach())
    def forward(self, pos):
        # forward pass to get place cell activations based on position tensor
        # calculate the euclidean distance between the agent position and place cell
        # use a gaussian function to represent activation size (higher near the center, lower around edges)
        squared_dist = torch.sum((pos.unsqueeze(1) - self.place_fields.unsqueeze(0))**2, dim=2)
        
        # Apply Gaussian function: A(x) = exp(-||x-x_c||²/2σ²)
        activations = torch.exp(-squared_dist / (2 * self.sigma**2))
        
        return activations
    
    def update_place_fields(self, new_place_fields):
        # update place fields with output from the rl agent
        with torch.no_grad():
            self.place_fields.copy_(new_place_fields)

class SpatialNetwork(nn.Module):
    # combines grid and place cells to form a grid representation of the environment
    # TODO: change input to 3D in the future
    def __init__(self, 
                input_size=2, # (x, y)
                hidden_size=128, 
                output_size=4, # (x, y, z, theta)
                num_grid_modules=6, # types of spatial representations (frequency)
                num_place_cells=100):
        super(SpatialNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # activate grid cell modules
        # grid cells scale geometrically to efficiently encode large environments
        self.grid_modules = nn.ModuleList([
            GridCellModule(
                input_size = input_size,
                output_size = hidden_size // num_grid_modules,
                scale=0.5 * (1.4**i) # geometric growth
            ) for i in range(num_grid_modules)
        ])

        # integrate grid and place cell layers
        # 1D vector of combined grid cell activation patterns
        self.place_cells = PlaceCellModule(input_size, num_place_cells)

        # integrate grid and place cell layers
        grid_output_dim = num_grid_modules * 3 # 3 sine waves per grid module
        self.integration_layer = nn.Linear(grid_output_dim + num_place_cells, output_size)
    def forward(self, pos):
        # forward pass through spatial network to get spatial representation
        # process grid cells and store position outputs
        grid_outputs = []
        for module in self.grid_modules:
            grid_outputs.append(module(pos))
        grid_concat = torch.cat(grid_outputs, dim=-1)
        # process place cells
        place_activations = self.place_cells(pos)

        # combine representation
        combined_representation = torch.cat([grid_concat, place_activations], dim=-1)
        return self.integration_layer(combined_representation)
    