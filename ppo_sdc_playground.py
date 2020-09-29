import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2

import sdc_gym

M = 3
fname = "sdc_model_ppo2_normrew_1E05"

env = gym.make('sdc-v0', M=M, dt=1.0, restol=1E-10)
# env = gym.make('sdc-v1', M=M, dt=1.0, restol=1E-10)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=False, norm_reward=True)  # when training norm_reward = True

policy_kwargs = {}
# policy_kwargs = dict(net_arch=[32, 32])
model = PPO2(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./sdc_tensorboard/")
# model.set_env(env)
model.learn(total_timesteps=int(50000))

model.save(fname)
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = PPO2.load(fname)
env = gym.make('sdc-v0', M=M, dt=1.0, restol=1E-10)


ntests = 100
mean_niter = 0
nsucc = 0
for i in range(ntests):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        # print(rewards, info['residual'], done, action, env.lam, env.niter)
        # print(rewards, done, action, info[0]['residual'])
    if info['niter'] < 50 and info['residual'] < env.restol:
        nsucc += 1
        print(info['residual'], info['lam'], info['niter'])
        mean_niter += info['niter']

mean_niter /= nsucc
print(f"Success rate: {100 * nsucc / ntests:4.2f}% -- Mean number of iterations (of successes): {mean_niter:4.2f}")
