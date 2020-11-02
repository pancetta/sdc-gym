import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2, ACKTR

import sdc_gym

M = 3
fname = "sdc_model_acktr"

env = gym.make('sdc-v0', M=M, dt=1.0, restol=1E-10)
# env = gym.make('sdc-v1', M=M, dt=1.0, restol=1E-10)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=False, norm_reward=True)  # when training norm_reward = True

policy_kwargs = {}
# policy_kwargs = dict(net_arch=[128])
# policy_kwargs = dict(act_fun=tf.nn.tanh)
# policy_kwargs = dict(net_arch=[128, dict(vf=[128], pi=[128])])
# model = PPO2(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./sdc_tensorboard/")
model = ACKTR(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./sdc_tensorboard/")#, vf_fisher_coef=2.0)
# model = ACKTR(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./sdc_tensorboard/", lr_schedule='double_middle_drop')
model.learn(total_timesteps=int(1000000))

model.save(fname)
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = ACKTR.load(fname)
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
    if env.niter < 50 and info['residual'] < env.restol:
        nsucc += 1
        print(info['residual'], env.lam, env.niter, action)
        mean_niter += env.niter

mean_niter /= nsucc
print(mean_niter, nsucc / ntests)
