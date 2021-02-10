import argparse
from pathlib import Path
import time

import gym
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
import stable_baselines
import matplotlib.pyplot as plt

import sdc_gym


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--steps',
        type=float,
        default=10000,
        help='Number of action steps to take in the environment.',
    )
    parser.add_argument(
        '--M',
        type=int,
        default=3,
        help=(
            '"Difficulty" of the problem '
            '(proportionally relates to nth-order differential equation)'
            '(choose M = 3 or M = 5 for comparing MIN-preconditioner)'
        ),
    )
    parser.add_argument(
        '--restol',
        type=float,
        default=1E-10,
        help=(
            'Residual tolerance; '
            'when residual is below this, stop iterating.'
            'Lower means better solution but more iterations.'
        ),
    )
    parser.add_argument(
        '--envname',
        type=str,
        default='sdc-v0',
        choices=['sdc-v0', 'sdc-v1'],
        help=(
            'Gym environment to use;\n    sdc-v0 – SDC with a full iteration '
            'per step (no intermediate observations),\n    sdc-v1 – SDC with '
            'a single iteration per step (with intermediate observations).'
        ),
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        # default for PPO2 in stable-baselines
        default=25E-05,
        help='Learning rate/step size of the model.',
    )
    parser.add_argument(
        '--model_class',
        type=str,
        default='PPO2',
        help='Class of model to instantiate.',
    )
    parser.add_argument(
        '--policy_class',
        type=str,
        default='MlpPolicy',
        help='Class of model policy.',
    )
    parser.add_argument(
        '--tests',
        type=float,
        default=5000,
        help='Number of test runs for each preconditioning method.',
    )
    return parser.parse_args()


def _get_model_class(model_class_str):
    """Return a model class according to `model_class_str`."""
    try:
        model_class = getattr(stable_baselines, model_class_str)
    except AttributeError:
        raise AttributeError(
            f"could not find model class '{model_class_str}' "
            f'in module `stable_baselines`'
        )
    assert issubclass(model_class, stable_baselines.common.BaseRLModel), \
        ('model class must be a subclass of '
         '`stable_baselines.common.BaseRLModel`')
    return model_class


def _get_policy_class(policy_class_str, model_class_str):
    if model_class_str.upper() == 'DDPG':
        policy_class_module = stable_baselines.ddpg.policies
    elif model_class_str.upper() == 'DQN':
        policy_class_module = stable_baselines.deepq.policies
    else:
        policy_class_module = stable_baselines.common.policies

    try:
        policy_class = getattr(
            policy_class_module,
            policy_class_str,
        )
    except AttributeError:
        try:
            policy_class = globals()[policy_class_str]
        except KeyError:
            raise AttributeError(
                f"could not find policy class '{policy_class_str}' "
                f'in module `stable_baselines.common.policies` '
                f'or in this module'
            )
    assert issubclass(
        policy_class,
        stable_baselines.common.policies.BasePolicy,
    ), ('policy class must be a subclass of '
        '`stable_baselines.common.policies.BasePolicy`')
    return policy_class


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


def _find_free_path(format_path):
    """Get a path following `format_path` into which a single incrementing
    number is interpolated until a non-existing path is found.
    """
    i = 0
    path = Path(format_path.format(i))
    while path.exists():
        i += 1
        path = Path(format_path.format(i))
    return path


def main():
    args = parse_args()

    # Set parameters for SDC
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
    model_class = _get_model_class(args.model_class)
    policy_class = _get_policy_class(args.policy_class, args.model_class)

    policy_kwargs = {}

    # Learning rate to try for PPO2: 1E-05
    # Learning rate to try for ACKTR: 1E-03
    learning_rate = args.learning_rate

    model = model_class(
        policy_class,
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(Path(
            f'./sdc_tensorboard/'
            f'{args.model_class.lower()}_{args.policy_class.lower()}/'
        )),
        learning_rate=learning_rate,
    )

    start_time = time.perf_counter()
    # Train the model (need to put at least 100k steps to
    # see something)
    model.learn(total_timesteps=int(args.steps))
    duration = time.perf_counter() - start_time
    print(f'Training took {duration} seconds.')

    fname = Path(f'sdc_model_{args.model_class.lower()}_'
                 f'{args.policy_class.lower()}_{learning_rate}.zip')
    model.save(str(fname))
    # delete trained model to demonstrate loading, not really necessary
    # del model

    # ---------------- TESTING STARTS HERE ----------------

    ntests = int(args.tests)

    # Load the trained agent for testing
    # model = model_class.load(fname)

    start_time = time.perf_counter()
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
    duration = time.perf_counter() - start_time
    print(f'Testing took {duration} seconds.')

    # Plot all three iteration counts over the lambda values
    plt.xlabel('re(λ)')
    plt.ylabel('iterations')

    plot_results(results_RL, color='b', label='RL')
    plot_results(results_LU, color='r', label='LU')
    plot_results(results_min, color='g', label='MIN')

    plt.legend()

    fig_path = _find_free_path('results_{:>04}.pdf')
    plt.savefig(fig_path, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    main()
