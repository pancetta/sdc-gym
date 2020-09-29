import gym
import sdc_gym

import numpy as np

M = 3


env = gym.make('sdc-v0', M=M, dt=1.0, restol=1E-10)

env.reset()
print(env.lam)
action = np.zeros(M)
done = False
total_reward = 0
while not done:
    state, reward, done, info = env.step(action)
    total_reward += reward
    print(info['residual'], total_reward)


# print(env.step(action))
# print(env.step(action))
# print(env.step(action))
# print(env.step(action))
