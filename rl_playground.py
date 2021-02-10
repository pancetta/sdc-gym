import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2, ACKTR
import matplotlib.pyplot as plt

import sdc_gym


def test_model(model, env, ntests, name):
    """Test the `model` in the Gym `env` `ntests` times.
    `name` is the name for the test run for logging purposes.
    """
    mean_niter = 0
    nsucc = 0
    results = []

    for i in range(ntests):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(
                obs,
                deterministic=True,
            )
            obs, rewards, done, info = env.step(action)

        if env.niter < 50 and info['residual'] < env.restol:
            nsucc += 1
            mean_niter += env.niter
            # Store each iteration count together with the respective
            # lambda to make nice plots later on
            results.append((env.lam.real, env.niter))

    # Write out mean number of iterations (smaller is better) and the
    # success rate (target: 100 %)
    if nsucc > 0:
        mean_niter /= nsucc
    else:
        mean_niter = 666
    print(f'{name}  -- Mean number of iterations and success rate: '
          f'{mean_niter:4.2f}, {nsucc / ntests * 100} %')
    return results


def plot_results(results, color, label):
    sorted_results = sorted(results, key=lambda x: x[0])
    plt.plot(
        [i[0] for i in sorted_results],
        [i[1] for i in sorted_results],
        color=color,
        label=label,
    )


def main():
    # Set parameters for SDC
    # In order to compare to the MIN-preconditioner,
    # choose M = 3 or M = 5.
    M = 3
    # When to stop iterating (lower means better solution, but
    # more iterations)
    restol = args.restol

    # Set parameters for RL
    # 'sdc-v0' –  SDC with a full iteration per step
    #             (no intermediate observations)
    # 'sdc-v1' –  SDC with a single iteration per step
    #             (and intermediate observations)
    envname = args.envname

    # ---------------- TRAINING STARTS HERE ----------------

    # Set up gym environment
    env = gym.make(envname, M=M, dt=1.0, restol=restol)
    env = DummyVecEnv([lambda: env])
    # When training, set `norm_reward = True`, I hear...
    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    # Set up model
    policy_kwargs = {}
    model = PPO2(
        MlpPolicy,
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log='./sdc_tensorboard/',
        # Learning rate for PPO2: 1E-05
        # Learning rate for ACKTR: 1E-03
        # learning_rate=1E-5,
    )

    # Train the model (need to put at least 100k steps to
    # see something)
    model.learn(total_timesteps=int(10000))

    fname = 'sdc_model_acktr'
    model.save(fname)
    # delete trained model to demonstrate loading, not really necessary
    # del model

    # ---------------- TESTING STARTS HERE ----------------

    ntests = int(args.tests)

    # Load the trained agent for testing
    # model = PPO2.load(fname)


    # Test the trained model.
    env = gym.make(envname, M=M, dt=1.0, restol=restol)
    results_RL = test_model(model, env, ntests, 'RL')

    # Restart the whole thing, but now using the LU preconditioner (no RL here)
    # LU is serial and the de-facto standard. Beat this (or at least be on par)
    # and we win!
    env = gym.make(envname, M=M, dt=1.0, restol=restol, prec='LU')
    results_LU = test_model(model, env, ntests, 'LU')

    # Restart the whole thing, but now using the minization preconditioner
    # (no RL here)
    # This minimization approach are just magic numbers we found using
    # indiesolver.com, parallel and proof-of-concept
    env = gym.make(envname, M=M, dt=1.0, restol=1E-10, prec='min')
    results_min = test_model(model, env, ntests, 'MIN')

    # Plot all three iteration counts over the lambda values
    plot_results(results_RL, color='b', label='RL')
    plot_results(results_LU, color='r', label='LU')
    plot_results(results_min, color='g', label='MIN')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
