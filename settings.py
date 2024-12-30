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


