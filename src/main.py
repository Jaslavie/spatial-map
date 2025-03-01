# process user input and update the env response. use response to update model and learning rate.
import os
import json
import numpy as np
import torch
from dotenv import load_dotenv

from envs.sandbox import Sandbox
from models.graph import GraphNetwork
from visualization.renderer import Renderer
from utils.logger import Logger
from utils.data_loader import load_data

load_dotenv()

logger = Logger()

def main():
    # init spatial data
    # init nn
    # init env
    # init renderer
    # run sim loop

if __name__ == "__main__":
    main()
