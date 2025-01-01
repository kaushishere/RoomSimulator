"""
Utilities Module containing helper functions for Agent Deep-Q learning 
"""

import random
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.losses import MSE

# SEED = 0
# random.seed(SEED)

def choose_action(env, epsilon, q_values):
    """
    Chooses an action to take based on the epsilon-greedy strategy
    """

    if random.uniform(0,1) >= epsilon:
        action = np.argmax(q_values)
    else:
        action = env.action_space.sample()

    return action

def get_experiences(memory_buffer, batch_size):
    """
    Returns a list, of size (batch_size,), of randomly selected experiences from the memory buffer.

    Args:
        memory_buffer (deque):
            A deque storing the most recent experiences
        batch_size (int):
            Number of experiences we want to take to form our mini-batch.

    Returns:
        sample (list):
            List of randomly selected experience tuples
    """

    # Python list of experiences
    experiences = random.sample(memory_buffer, batch_size)

    states = tf.convert_to_tensor(
        np.array([e.state for e in experiences]), dtype = tf.float32)
    actions = tf.convert_to_tensor(
        np.array([e.action for e in experiences]), dtype = tf.float32)
    rewards = tf.convert_to_tensor(
        np.array([e.reward for e in experiences]), dtype = tf.float32)
    new_states = tf.convert_to_tensor(
        np.array([e.new_state for e in experiences]), dtype = tf.float32)
    done_vals = tf.convert_to_tensor(
        np.array([e.done_val for e in experiences]), dtype = tf.uint8)

    # returns a tuple
    return (states, actions, rewards, new_states, done_vals)

def compute_loss_tf(policy_network, target_network, experiences, gamma):
    """
    Computes the MSE (Mean Squared Error) between Q-values in the policy network
    and the RHS of the Bellman equation.

    Args:
        experiences (deque):
            Randomly sampled mini-batch of experiences from the memory buffer.
        gamma (float):
            Hyperparameter for discounting future rewards

    Returns:
        loss ()
    """

    # unpack experiences tuples
    states, actions, rewards, new_states, done_vals = experiences

    ### COMPUTE Q-VALUES ###
    # returns q-values for all possible actions from that state
    q_values = policy_network(states)

    # we only want the q-value for the action that was taken from that experience
    # we will use tf.gather_nd to index q_values
    idx_one = tf.range(states.shape[0])
    idx_two = tf.cast(actions, dtype = tf.int32)
    idx_comb = tf.stack([idx_one, idx_two], axis = 1)
    q_values = tf.gather_nd(q_values, idx_comb)

    ### COMPUTE TARGETS ###
    q_msa = tf.reduce_max(target_network(new_states), axis = 1)
    done_vals_f = tf.cast(done_vals, dtype = tf.float32)
    targs = rewards + (gamma * (1 - done_vals_f) * q_msa)

    loss = MSE(targs, q_values)
    
    return loss

def check_update(t, num_steps_upd, memory_buffer, batch_size):
    """
    Returns a Boolean based on the following condition:
    
    Checks if the current timestep, t, is a multiple of the num_steps_upd and 
    that the memory buffer has enough experience tuples to draw a mini-batch from.
    
    Args:
        t (int): 
            Current training timestep 
        num_steps_upd (int): 
            For every {num_steps_upd} timesteps we update our networks. 
        memory_buffer (deque):
            A deque storing the most recent experience tuples. The max number
            of experiences it can store is specified by hyperparameter MEMORY.

    Returns:
        Boolean. If conditions are met, True is returned & False otherwise.
    """

    if t % num_steps_upd == 0 and len(memory_buffer) >= batch_size:
        return True
    else:
        return False

def update_target_network(target_network, policy_network, tau):
    """
    Softmax update of weights of the target network according to:

    w_ = (TAU * w) + ((1 - TAU) * w_)
    b_ = (TAU * b) + ((1 - TAU) * b_)

    where (w_, b_) are the weights for target network
          (w, b) are the weights for the policy network
    """

    for target_network_weights, policy_network_weights in zip(target_network.trainable_variables, policy_network.trainable_variables):
        target_network_weights.assign((policy_network_weights * tau) + ((1 - tau) * target_network_weights))

def update_epsilon(epsilon, e_min, e_decay):
    """
    Updates the value of epsilon, used for the epsiolon-greedy policy. The update is like so
    new_epsilon = MAX(epsilon * E_DECAY, E_MIN)
    """

    epsilon = max(e_min, epsilon * e_decay)

    return epsilon