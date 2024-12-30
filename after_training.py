from settings import *
from env import RoomSimulator
from weather import OutdoorTemp

# load environment & pre-trained model
env = RoomSimulator(1,0.025,render_mode="human")
policy_network = load_model(join('V1_outputs','policy_network_180.keras'))

for ep in range(1):
    done = False
    state = env.reset()
    score = 0

    while not done:

        # policy network
        state = np.expand_dims(state, axis = 0)
        q_values = policy_network(state)
        
        # take action
        state, reward, done, info = env.step(np.argmax(q_values))
        env.render()
        
        # updates
        score += reward
        
        if done:
            break
    print(f'Episode {ep + 1}: Score: {score:.2f}')
env.close()

