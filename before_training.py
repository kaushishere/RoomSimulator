from settings import *
from env import RoomSimulator
from weather import OutdoorTemp

env = RoomSimulator(0.5,0.04,reward_mech='V1',render_mode="human")

for ep in range(1):
    done = False
    state = env.reset()
    score = 0

    while not done:
        state, reward, done, info = env.step(env.action_space.sample())
        env.render()
        score += reward
        
        if done:
            break
    print(f'Episode {ep + 1}: Score: {score:.1f}')
env.close()