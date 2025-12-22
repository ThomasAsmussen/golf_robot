import os
from pathlib import Path
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import uuid
from reinforcement_learning.contextual_bandit import training
from vision.ball2hole_distance import compute_ball2hole_distance
from vision.ball_start_position import get_ball_start_position


def real_init_parameters():
    bx, by = get_ball_start_position(debug=False, use_cam=True)
    #hole_positions = 
    #disc_positions = 
    


def run_real()
    #compute_ball2hole_distance(ball_position)
    


#training(input_func=real_init_parameters, env_step=run_real)


