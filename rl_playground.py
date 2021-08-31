import datetime
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

import utils


def setup_model(args, env):
    "Return the model for the given `args` in the Gym `env`."
    model_class = utils.get_model_class(args.model_class)
    policy_class = utils.get_policy_class(args.policy_class, args.model_class)

    utils.check_num_envs(args, policy_class)

    policy_kwargs = args.policy_kwargs
    if args.activation_fn is not None:
        policy_kwargs['activation_fn'] = utils.get_activation_fn(
            args.activation_fn)

    # Learning rate to try for PPO2: 1E-05
    # Learning rate to try for ACKTR: 1E-03
    learning_rate = utils.compute_learning_rate(args)

    model_kwargs = {
        'verbose': 1,
        'tensorboard_log': str(Path(
            f'./sdc_tensorboard/'
            f'{args.model_class.lower()}_{args.policy_class.lower()}_'
            f'{args.script_start}/'
        )),
    }
    model_kwargs.update(args.model_kwargs)
    model_kwargs.update({
        'learning_rate': learning_rate,
        'policy_kwargs': policy_kwargs,
        'seed': args.seed,
    })

    utils.maybe_fix_nminibatches(model_kwargs, args, policy_class)

    if args.model_path is None:
        model = model_class(policy_class, env, **model_kwargs)
        return model

    if args.change_loaded_model:
        model_kwargs.pop('policy_kwargs', None)
        model = model_class.load(str(Path(args.model_path)), env,
                                 **model_kwargs)
    else:
        model = model_class.load(str(Path(args.model_path)), env)
    return model


def dry_run(model, env, nsteps):
    num_envs = env.num_envs
    # Amount of loops to run for vectorized environments
    nsteps = nsteps // num_envs * num_envs

    for i in range(nsteps):
        state = None
        obs = env.reset()
        done = [False for _ in range(num_envs)]
        if env.envs[0].prec is not None:
            action = [np.empty(env.action_space.shape,
                               dtype=env.action_space.dtype)
                      for _ in range(num_envs)]

        while not all(done):
            # Do not predict an action when we would discard it anyway
            if env.envs[0].prec is None:
                action, state = model.predict(
                    obs,
                    state=state if model.policy.recurrent else None,
                    mask=done if model.policy.recurrent else None,
                    deterministic=True,
                )

            obs, rewards, done, info = env.step(action)


def _store_test_stats(path, stats):
    np.savez_compressed(path, **stats)


def test_model(model, env, ntests, name, stats_path=None):
    """Test the `model` in the Gym `env` `ntests` times.
    `name` is the name for the test run for logging purposes.
    `stats_path` is an optional path where to save statistics about
    the test.
    """
    mean_niter = 0
    nsucc = 0
    results = []
    if stats_path is not None:
        stats = {
            key: []
            for key in [
                    'obs',
                    'action',
                    'reward',
                    'niter',
                    'lam',
                    'residual',
                    'terminal_observation',
            ]
        }

    num_envs = env.num_envs
    # Amount of test that will be ran in total
    ntests_total = ntests * num_envs

    for i in range(ntests):
        state = None
        obs = env.reset()
        done = [False for _ in range(num_envs)]
        if env.envs[0].prec is not None:
            action = [np.empty(env.action_space.shape,
                               dtype=env.action_space.dtype)
                      for _ in range(num_envs)]

        while not all(done):
            if stats_path is not None:
                stats['obs'].append(obs)

            # Do not predict an action when we would discard it anyway
            if env.envs[0].prec is None:
                action, state = model.predict(
                    obs,
                    state=state,
                    mask=done,
                    deterministic=True,
                )

            obs, rewards, done, info = env.step(action)

            if stats_path is not None:
                stats['action'].append(action)
                stats['reward'].append(rewards)
                for info_ in info:
                    for key in info_:
                        stats[key].append(info_[key])

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
    print(f'{name:<3} -- Mean number of iterations and success rate: '
          f'{mean_niter:4.2f}, {nsucc / ntests_total * 100} %')

    if stats_path is not None:
        _store_test_stats(stats_path, stats)

    return results


def plot_results(results, color, label):
    sorted_results = sorted(results, key=lambda x: x[0])
    plt.plot(
        [i[0] for i in sorted_results],
        [i[1] for i in sorted_results],
        color=color,
        label=label,
    )


def run_tests(model, args, seed=None, fig_path=None, stats_path=None):
    """Run tests for the given `model` and `args`, using `seed` as the
    random seed.

    `fig_path` is an optional path to store result plots at.
    `stats_path` is an optional path where to save statistics about the
    reinforcement learning test.
    """
    # Load the trained agent for testing
    if isinstance(model, (Path, str)):
        model_class = utils.get_model_class(args.model_class)
        model = model_class.load(str(model))

    # Not vectorizing is faster for testing for some reason.
    num_test_envs = args.num_envs \
        if not args.use_sb3 and model.policy.recurrent else 1

    ntests = int(args.tests)
    ntests = utils.maybe_fix_ntests(ntests, num_test_envs)

    start_time = time.perf_counter()
    # Test the trained model.
    env = utils.make_env(args, num_envs=num_test_envs, seed=seed,
                         lambda_real_interpolation_interval=None)
    results_RL = test_model(
        model, env, ntests, 'RL', stats_path=stats_path)

    # Restart the whole thing, but now using the LU preconditioner (no RL here)
    # LU is serial and the de-facto standard. Beat this (or at least be on par)
    # and we win!
    env = utils.make_env(
        args,
        num_envs=num_test_envs,
        prec='LU',
        seed=seed,
        lambda_real_interpolation_interval=None,
    )
    results_LU = test_model(model, env, ntests, 'LU')

    # Restart the whole thing, but now using the minization preconditioner
    # (no RL here)
    # This minimization approach are just magic numbers we found using
    # indiesolver.com, parallel and proof-of-concept
    env = utils.make_env(
        args,
        num_envs=num_test_envs,
        prec='min',
        seed=seed,
        lambda_real_interpolation_interval=None,
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

    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
    plt.show()


def main():
    script_start = str(datetime.datetime.now()
                       ).replace(':', '-').replace(' ', 'T')
    args = utils.parse_args()
    args.script_start = script_start
    args_path = Path(f'args_{script_start}.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    utils.setup(args.use_sb3, args.debug_nans)

    eval_seed = args.seed
    if eval_seed is not None:
        eval_seed += args.num_envs

    # ---------------- TRAINING STARTS HERE ----------------

    # Set up gym environment
    env = utils.make_env(args, include_norm=True)
    # Set up model
    model = setup_model(args, env)

    callbacks = []
    utils.append_callback(callbacks, utils.create_save_callback(args))
    utils.append_callback(callbacks, utils.create_eval_callback(args))

    dry_run(model, env, int(args.warmup_steps))
    env.seed(args.seed)
    for env_ in env.envs:
        env_.set_num_episodes(args.start_episodes)

    start_time = time.perf_counter()
    # Train the model (need to put at least 100k steps to
    # see something)
    model.learn(total_timesteps=int(args.steps), callback=callbacks)
    duration = time.perf_counter() - start_time
    print(f'Training took {duration} seconds.')
    # env.envs[0].plot_rewards()
    print('Number of episodes in each environment:',
          [env_.num_episodes for env_ in env.envs])

    model_fname = Path(f'sdc_model_{args.model_class.lower()}_'
                       f'{args.policy_class.lower()}_{script_start}.zip')
    model.save(str(model_fname))

    env_fname = Path(f'sdc_env_{script_start}.pkl')
    utils.save_env(env_fname, env)

    # delete trained model to demonstrate loading, not really necessary
    # del model

    # ---------------- TESTING STARTS HERE ----------------

    fig_path = Path(f'results_{script_start}.pdf')
    run_tests(model, args, seed=eval_seed, fig_path=fig_path)


if __name__ == '__main__':
    main()
