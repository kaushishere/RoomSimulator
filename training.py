from settings_training import *
from env import RoomSimulator

# @tf.function allows computations to be carried out in graph-mode instead of eager execution
@tf.function
def agent_learn(policy_network, target_network, experiences, gamma, tau):
    """
    Agent performs a gradient descent step in the policy network &
    updates weights within the target network using a softmax update.

    Args:
        policy_network (Sequential):
            Policy network used to predict the best action in the env
        target_network (Sequential):
            Network outputting the targets (optimal sets of Q-values)
        experiences (tuple):
            tuple of experiences in the form (states, actions, reward, new_states, done_vals)
        gamma (float):
            Discount factor used in Bellman's equation
    """
    # tf needs to know what operations happened during the forward pass, and in what order, so 
    # that it can use back-propagation to compute gradients
    with tf.GradientTape() as tape:
        # forward pass
        loss = utils.compute_loss_tf(policy_network, target_network, experiences, gamma)

    # backward pass
    gradients = tape.gradient(loss, policy_network.trainable_variables)

    # perform a gradient descent step in policy network
    optimizer.apply_gradients(zip(gradients, policy_network.trainable_variables))

    # update target network
    utils.update_target_network(target_network, policy_network, tau)

# load environment
env = RoomSimulator(0.5,0.04,reward_mech='V3')
state_shape = np.shape(np.expand_dims(env.reset(), axis = 0))

# load networks
policy_network = Sequential(
    [
        Input(shape = state_shape),
        Dense(units = 64, activation = 'relu'),
        Dense(units = 64, activation = 'relu'),
        Dense(units = 3, activation = 'linear')
    ]
)

target_network = Sequential(
    [
        Input(shape = state_shape),
        Dense(units = 64, activation = 'relu'),
        Dense(units = 64, activation = 'relu'),
        Dense(units = 3, activation = 'linear')
    ]
)
optimizer = Adam(learning_rate = ALPHA)

save_dir = env.reward_mech + '_outputs'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

start_time = time.time()
for ep in range(num_episodes):
    # standard resets
    done = False
    score = 0
    state = env.reset()

    for i in range(1, len(env.num_timesteps)+1):

        # epsilon-greedy strategy
        state_net = np.expand_dims(state, axis = 0)
        q_values = policy_network(state_net)
        action = utils.choose_action(env, epsilon, q_values)

        # take action & store experience
        new_state, reward, done, info = env.step(action)
        experience = experiences(state, action, reward, new_state, done)
        memory_buffer.append(experience)

        # update networks?
        if utils.check_update(i, NUM_STEPS_UPD, memory_buffer, BATCH_SIZE):
            exps = utils.get_experiences(memory_buffer, BATCH_SIZE)
            agent_learn(policy_network, target_network, exps, GAMMA, TAU)

        # reset state and update score
        state = new_state
        score += reward

        # is the episode over?
        if done:
            break

    # append results
    score_hist.append(score)
    score_avg = np.mean(score_hist[-avg_frequency:])

    # update epsilon
    epsilon = utils.update_epsilon(epsilon, E_MIN, E_DECAY)

    # output metrics & save best models
    if (ep + 1) % avg_frequency == 0:
        print(f'Episode {ep + 1}: Average Reward over the last {avg_frequency} episodes: {score_avg}')
        if score_avg > best_avg_score:
            policy_network.save(join(f'{save_dir}','best_model.keras'))
            best_avg_score = score_avg
    
    if (ep+1) % save_frequency == 0:
        policy_network.save(join(f'{save_dir}',f'policy_network_{ep+1}.keras'))

# Save final policies
policy_network.save(join(f'{save_dir}','final_policy_network.keras'))
target_network.save(join(f'{save_dir}','final_target_network.keras'))
# Save scores 
with open(join(f'{save_dir}','score_hist.pkl'), 'wb') as file:
    pickle.dump(score_hist, file)
time_taken = time.time() - start_time
print(f"Training for {num_episodes} episodes took {time_taken/60:.0f} minutes")
