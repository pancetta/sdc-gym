import argparse
from pathlib import Path
import time

import gym
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
import stable_baselines
import matplotlib.pyplot as plt

import sdc_gym

PPO2_DEFAULT_NUM_MINIBATCHES = 4


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
        '--dt',
        type=float,
        default=1.0,
        help=(
            'Size of time step.'
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
        '--rescale_lr',
        type=bool,
        default=True,
        help=(
            'Whether to rescale the learning rate by the number'
            'of environments.'
        ),
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
    parser.add_argument(
        '--num_envs',
        type=int,
        default=4,
        help='How many environments to use in parallel.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help=(
            'Base seed for seeding the environments. For multiple '
            'environments, all will have different seeds based on this one.'
        ),
    )
    parser.add_argument(
        '--eval_freq',
        type=int,
        default=0,
        help=(
            'How often to evaluate the model during training, storing the '
            'best performing one. If this is 0, do not evaluate.'
        ),
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


def _maybe_fix_ntests(ntests_given, num_test_envs):
    """Return `ntests_given` approximately scaled to a vectorized environment
    with `num_test_envs` parallel environments.

    Print a warning if the rescaling does not result in the same number of
    test runs.
    """
    # Amount of test loops to run
    ntests = ntests_given // num_test_envs * num_test_envs
    if ntests != ntests_given:
        print(f'Warning: `ntests` set to {ntests} ({ntests_given} was given).')
    return ntests


def test_model(model, env, ntests, name):
    """Test the `model` in the Gym `env` `ntests` times.
    `name` is the name for the test run for logging purposes.
    """
    mean_niter = 0
    nsucc = 0
    results = []

    num_envs = env.num_envs
    # Amount of test that will be ran in total
    ntests_total = ntests * num_envs

    for i in range(ntests):
        state = None
        obs = env.reset()
        done = [False for _ in range(num_envs)]

        while not all(done):
            action, state = model.predict(
                obs,
                state=state,
                mask=done,
                deterministic=True,
            )
            obs, rewards, done, info = env.step(action)

        for (env_, info_) in zip(env.envs, info):
            # We work on the info here because its information is
            # not lost with the automatic env reset from a
            # vectorized environment.
            if info_['niter'] < 50 and info_['residual'] < env_.restol:
                nsucc += 1
                mean_niter += info_['niter']
                # Store each iteration count together with the respective
                # lambda to make nice plots later on
                results.append((info_['lam'].real, info_['niter']))

    # Write out mean number of iterations (smaller is better) and the
    # success rate (target: 100 %)
    if nsucc > 0:
        mean_niter /= nsucc
    else:
        mean_niter = 666
    print(f'{name}  -- Mean number of iterations and success rate: '
          f'{mean_niter:4.2f}, {nsucc / ntests_total * 100} %')
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


def make_env(
        args,
        num_envs=None,
        include_norm=False,
        norm_obs=False,
        norm_reward=True,
        **kwargs,
):
    if num_envs is None:
        num_envs = args.num_envs
    M = kwargs.pop('M', args.M)
    dt = kwargs.pop('dt', args.dt)
    restol = kwargs.pop('restol', args.restol)
    seed = kwargs.pop('seed', args.seed)

    env = DummyVecEnv([
        lambda: gym.make(
            args.envname,
            M=M,
            dt=dt,
            restol=restol,
            seed=seed + i if seed is not None else None,
            **kwargs,
        )
        for i in range(num_envs)
    ])
    if include_norm:
        # When training, set `norm_reward = True`, I hear...
        env = VecNormalize(
            env,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
        )
    return env


def _create_eval_callback(args, learning_rate):
    if args.eval_freq <= 0:
        return None

    eval_env = make_env(
        args,
        num_envs=1,
        include_norm=True,
        # In contrast to training, we don't give this for testing.
        norm_reward=False,
        seed=args.seed + args.num_envs,
    )

    best_dirname = Path(f'best_sdc_model_{args.model_class.lower()}_'
                        f'{args.policy_class.lower()}_{learning_rate}')
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dirname),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
    )
    return eval_callback


def _check_num_envs(args, policy_class):
    """Raise an error if the number of environments will cause issues with
    the model.
    """
    if not policy_class.recurrent:
        return
    if args.num_envs != 1 and args.eval_freq > 0:
        raise ValueError(
            'An `--eval_freq > 0` is given. Due to issues with recurrent '
            'models and vectorized environments, the number of environments '
            'must be set to 1 (so append ` --num_envs 1`)',
        )


def _maybe_fix_nminibatches(model_kwargs, args, policy_class):
    """Set the `nminibatches` parameter in `model_kwargs` to
    `args.num_envs` if the value would cause problems otherwise with the
    given `policy_class`.

    Print a warning as well if the value had to be changed.
    Only affects the `PPO2` model.
    """
    if args.model_class != 'PPO2' or not policy_class.recurrent:
        return

    nminibatches = model_kwargs.get(
        'nminibatches',
        PPO2_DEFAULT_NUM_MINIBATCHES,
    )
    if args.num_envs % nminibatches == 0:
        return

    print('Warning: policy is recurrent and the number of environments is '
          'not a multiple of `PPO2.nminibatches`. '
          'Setting `nminibatches = num_envs`...')
    model_kwargs['nminibatches'] = args.num_envs


def main():
    args = parse_args()
    seed = args.seed

    # ---------------- TRAINING STARTS HERE ----------------

    # Set up gym environment
    env = make_env(args, include_norm=True)
    # Set up model
    model_class = _get_model_class(args.model_class)
    policy_class = _get_policy_class(args.policy_class, args.model_class)

    policy_kwargs = {}

    # Learning rate to try for PPO2: 1E-05
    # Learning rate to try for ACKTR: 1E-03
    learning_rate = args.learning_rate
    if args.rescale_lr:
        learning_rate *= args.num_envs

    eval_callback = _create_eval_callback(args, learning_rate)

    model_kwargs = {
        'verbose': 1,
        'policy_kwargs': policy_kwargs,
        'tensorboard_log': str(Path(
            f'./sdc_tensorboard/'
            f'{args.model_class.lower()}_{args.policy_class.lower()}/'
        )),
        'learning_rate': learning_rate,
        'seed': seed,
    }

    _check_num_envs(args, policy_class)
    _maybe_fix_nminibatches(model_kwargs, args, policy_class)

    model = model_class(policy_class, env, **model_kwargs)

    start_time = time.perf_counter()
    # Train the model (need to put at least 100k steps to
    # see something)
    model.learn(total_timesteps=int(args.steps), callback=eval_callback)
    duration = time.perf_counter() - start_time
    print(f'Training took {duration} seconds.')

    fname = Path(f'sdc_model_{args.model_class.lower()}_'
                 f'{args.policy_class.lower()}_{learning_rate}.zip')
    model.save(str(fname))
    # delete trained model to demonstrate loading, not really necessary
    # del model

    # ---------------- TESTING STARTS HERE ----------------

    # Not vectorizing is faster for testing for some reason.
    num_test_envs = args.num_envs if policy_class.recurrent else 1

    ntests = int(args.tests)
    ntests = _maybe_fix_ntests(ntests, num_test_envs)

    # Load the trained agent for testing
    # model = model_class.load(fname)

    start_time = time.perf_counter()
    # Test the trained model.
    env = make_env(args, num_envs=num_test_envs, seed=seed + args.num_envs)
    results_RL = test_model(model, env, ntests, 'RL')

    # Restart the whole thing, but now using the LU preconditioner (no RL here)
    # LU is serial and the de-facto standard. Beat this (or at least be on par)
    # and we win!
    env = make_env(
        args,
        num_envs=num_test_envs,
        prec='LU',
        seed=seed + args.num_envs,
    )
    results_LU = test_model(model, env, ntests, 'LU')

    # Restart the whole thing, but now using the minization preconditioner
    # (no RL here)
    # This minimization approach are just magic numbers we found using
    # indiesolver.com, parallel and proof-of-concept
    env = make_env(
        args,
        num_envs=num_test_envs,
        prec='min',
        seed=seed + args.num_envs,
    )
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
