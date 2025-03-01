# training loop to train model on env and agent interactions
import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, network, environment, device='cpu',
                 learning_rate=0.001, ## moderate learning rate
                 gamma=0.99 # favor long term rewards
                 ):
        self.network = network
        self.environment = environment
        self.device = device
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        self.gamma = gamma
        
    def train_episode(self, max_steps=1000):
        # train on a single series of agent interactions
        state = self.env.reset()
        total_reward = 0
        losses = []

        for step in range(max_steps):
            # convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # get action from network
            # apply action to env
            # calculate loss and update network
            # update grid and place cell activity
            # move to next state
            # log metrics
    def train(self, num_episodes=100):
        # train on multiple episodes
    def save_model(self, path):
        # save model to file
    def load_model(self, path):
        # load model from file
