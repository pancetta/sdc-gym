import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

from stable_baselines import A2C
from stable_baselines import ACKTR

envname = "CartPole-v1"
env = gym.make(envname)

env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=False, norm_reward=True)  # when training norm_reward = True

policy_kwargs = {}
# policy_kwargs = dict(net_arch=[128])
model = ACKTR(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./cartpole_tensorboard/")
model.learn(total_timesteps=int(20000))

model.save(envname)
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = ACKTR.load(envname)

mean_nsteps = 0
for i in range(100):
    obs = env.reset()
    done = False
    nsteps = 0
    while not done:
        nsteps += 1
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
    mean_nsteps += nsteps
    print(i, nsteps, mean_nsteps / (i + 1))

