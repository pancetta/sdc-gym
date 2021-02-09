import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2, ACKTR
import matplotlib.pyplot as plt
from operator import itemgetter

import sdc_gym

# Set parameters for SDC
M = 3  # in order to compare to the MIN-preconditioner, choose M=3 or M=5
restol = 1E-10  # defines when to stop iterating (lower means better solution, but more iterations)

# Set parameters for RL
fname = "sdc_model_acktr"
envname = 'sdc-v0'  # this is SDC with a full iteration per step (no intermediate observations)
# envname = 'sdc-v1'  # this is SDC with a single iteration per step (and intermediate observations)

# ---------------- TRAINING STARTS HERE ----------------

# Set up gym environment
env = gym.make(envname, M=M, dt=1.0, restol=restol)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=False, norm_reward=True)  # when training norm_reward = True, I hear..

# Set up model
policy_kwargs = {}
model = PPO2(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./sdc_tensorboard/")#, learning_rate=1E-05)
# model = ACKTR(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./sdc_tensorboard/")#, learning_rate=1E-03)

# Train the model (need to put at least 100k steps here to see something)
model.learn(total_timesteps=int(10000))

model.save(fname)
del model  # delete trained model to demonstrate loading, not really necessary

# ---------------- TESTING STARTS HERE ----------------

ntests = 5000

# Load the trained agent for testing
model = PPO2.load(fname)
# model = ACKTR.load(fname)

env = gym.make(envname, M=M, dt=1.0, restol=restol)
# Test the agent using ntests tests
mean_niter = 0
nsucc = 0
results_RL = []
for i in range(ntests):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
    if env.niter < 50 and info['residual'] < env.restol:
        nsucc += 1
        # Store each iteration count together with the respectivce lambda to make nice plots later on
        results_RL.append((env.lam.real, env.niter))
        mean_niter += env.niter

# Write out mean number of iterations (smaller is better) and the success rate (target: 100%)
if nsucc > 0:
    mean_niter /= nsucc
else:
    mean_niter = 666
print(f'RL  -- Mean number of iterations and success rate: {mean_niter:4.2f}, {nsucc / ntests * 100}%')

# Restart the whole thing, but now using the LU preconditioner (no RL here)
# LU is serial and the de-facto standard. Beat this (or at least be on par) and we win!
env = gym.make(envname, M=M, dt=1.0, restol=restol, prec="LU")
mean_niter = 0
nsucc = 0
results_LU = []
for i in range(ntests):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
    if env.niter < 50 and info['residual'] < env.restol:
        nsucc += 1
        results_LU.append((env.lam.real, env.niter))
        mean_niter += env.niter

if nsucc > 0:
    mean_niter /= nsucc
else:
    mean_niter = 666
print(f'LU  -- Mean number of iterations and success rate: {mean_niter:4.2f}, {nsucc / ntests * 100}%')

# Restart the whole thing, but now using the minization preconditioner (no RL here)
# This minimization approach are just magic numbers we found using indiesolver.com, parallel and proof-of-concept
env = gym.make(envname, M=M, dt=1.0, restol=1E-10, prec="min")
mean_niter = 0
nsucc = 0
results_min = []
for i in range(ntests):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
    if env.niter < 50 and info['residual'] < env.restol:
        nsucc += 1
        results_min.append((env.lam.real, env.niter))
        mean_niter += env.niter

if nsucc > 0:
    mean_niter /= nsucc
else:
    mean_niter = 666
print(f'MIN -- Mean number of iterations and success rate: {mean_niter:4.2f}, {nsucc / ntests * 100}%')

# Plot all three iteration counts over the lambda values
plt.plot([i[0] for i in sorted(results_RL, key=itemgetter(0))], [i[1] for i in sorted(results_RL, key=itemgetter(0))], color='b', label='RL')
plt.plot([i[0] for i in sorted(results_LU, key=itemgetter(0))], [i[1] for i in sorted(results_LU, key=itemgetter(0))], color='r', label='LU')
plt.plot([i[0] for i in sorted(results_min, key=itemgetter(0))], [i[1] for i in sorted(results_min, key=itemgetter(0))], color='g', label='MIN')
plt.legend()
plt.show()

