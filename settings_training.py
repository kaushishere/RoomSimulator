# Environment stuff
import pygame
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from gym.error import DependencyNotInstalled
from datetime import time as tim, datetime, timedelta

# Deep Q-Learning
import utils

# Deep Learning
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MSE
from tensorflow.keras.models import load_model

# PID
from scipy.integrate import odeint

# Helper functions 
import random
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import pickle
import os
from collections import namedtuple, deque
from typing import Optional
from os.path import join

# WEATHER STUFF
n = 96 # number of 15min periods in a day
t = np.linspace(0,95,n) # timestep array
MIN_TEMP_SUMMER = 12
MAX_TEMP_SUMMER = 25
SETPOINT_SUMMER = np.ones(n) * 21

MIN_TEMP_WINTER = 0
MAX_TEMP_WINTER = 14

# HYPERPARAMETERS 
TAU = 1e-3
GAMMA = 0.995
ALPHA = 1e-3

# epsilon greedy strategy
EPSILON = 1
E_MIN = 0.01
E_DECAY = 0.9975

# Experience Replay
MEMORY = 1000  
BATCH_SIZE = 20 
NUM_STEPS_UPD = 4

# episode
num_episodes = 3000
experiences = namedtuple('Experience', 'state, action, reward, new_state, done_val')
memory_buffer = deque(maxlen = MEMORY)
epsilon = EPSILON
score_hist = []
avg_frequency = 200
best_avg_score = 0
save_frequency = 500
