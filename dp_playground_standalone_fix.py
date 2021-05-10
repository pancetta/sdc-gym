import argparse
import datetime
from pathlib import Path
import time

import gym
import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental import optimizers
import matplotlib.pyplot as plt
import numpy as np

from rl_playground import _store_test_stats, plot_results
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--steps',
        type=float,
        default=10000,
        help='Number of learning steps to take.',
    )

    parser.add_argument(
        '--M',
        type=int,
        default=3,
        help=(
            '"Difficulty" of the problem '
            '(proportionally relates to nth-order differential equation)'
            '(choose M = 3 or M = 5 for comparing MIN-preconditioner).'
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
        '--lambda_real_interval',
        type=int,
        nargs=2,
        default=[-100, 0],
        help='Interval to sample the real part of lambda from.',
    )
    parser.add_argument(
        '--lambda_imag_interval',
        type=int,
        nargs=2,
        default=[0, 0],
        help='Interval to sample the imaginary part of lambda from.',
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=3E-4,
        help='Learning rate/step size of the model.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Number of samples for each training step',
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Model checkpoint to load (a .pt file).',
    )
    parser.add_argument(
        '--tests',
        type=float,
        default=5000,
        help='Number of test runs for each preconditioning method.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Base random number seed.',
    )

    args = parser.parse_args()
    args.envname = 'jax-sdc-v4'

    args.lambda_real_interval = sorted(args.lambda_real_interval)
    args.lambda_imag_interval = sorted(args.lambda_imag_interval)

    # Dummy values
    args.lambda_real_interpolation_interval = None
    args.norm_factor = 0.0
    args.residual_weight = 0.0
    args.step_penalty = 0.0
    args.reward_iteration_only = True
    args.reward_strategy = None
    args.collect_states = False
    args.model_class = 'PPG'
    return args


def build_model(M, train):
    scale = 1e-3
    glorot_normal = jax.nn.initializers.variance_scaling(
        scale, "fan_avg", "truncated_normal")
    normal = jax.nn.initializers.normal(scale)

    dropout_rate = 0
    mode = 'train' if train and dropout_rate > 0 else 'test'

    (model_init, model_apply) = stax.serial(
        stax.Flatten,
        stax.Dense(64, glorot_normal, normal),
        stax.Dropout(dropout_rate, mode),
        stax.Relu,
        # stax.Dense(256),
        # stax.Relu,
        stax.Dense(64, glorot_normal, normal),
        stax.Dropout(dropout_rate, mode),
        stax.Relu,
        stax.Dense(M, glorot_normal, normal),
    )
    return (model_init, model_apply)


def build_opt(lr, params):
    lr = optimizers.polynomial_decay(lr, 35000, lr * 1e-7, 2.0)

    (opt_init, opt_update, opt_get_params) = optimizers.adam(lr)
    opt_state = opt_init(params)
    return (opt_state, opt_update, opt_get_params)


def load_model(path):
    with open(path, 'rb') as f:
        weights = jnp.load(f, allow_pickle=True)
    with open(str(path) + '.steps', 'rb') as f:
        steps = jnp.load(f, allow_pickle=True)
    return list(weights), steps


def save_model(path, params, steps):
    with open(path, 'wb') as f:
        jnp.save(f, params)
    with open(str(path) + '.steps', 'wb') as f:
        jnp.save(f, steps)


def make_env(
        args,
        num_envs=None,
        include_norm=False,
        norm_reward=True,
        **kwargs,
):
    """Return a vectorized environment containing `num_envs` or `args.num_envs`
    environments (depending on whether `num_envs is None`).

    `args`, the command line arguments, specify several values. See `kwargs`
    for a more detailed explanation on their interaction.
    `include_norm` specifies whether the environment is wrapped in a
    normalizing environment.
    `norm_reward` indicates whether the rewards are normalized (only
    revelant if `include_norm is True`).
    `kwargs` are passed directly to the environment creation function. Any
    value given via `kwargs` has priority over the one given by `args`.
    """
    if num_envs is None:
        num_envs = args.num_envs

    # `kwargs` given via `args`
    args_kwargs = {}
    for arg in [
            'M',
            'dt',
            'restol',
            'seed',

            'lambda_real_interval',
            'lambda_imag_interval',
            'lambda_real_interpolation_interval',

            'norm_factor',
            'residual_weight',
            'step_penalty',
            'reward_iteration_only',
            'reward_strategy',
            'collect_states',
    ]:
        args_kwargs[arg] = kwargs.pop(arg, getattr(args, arg))
    all_kwargs = {**kwargs, **args_kwargs}

    # SAC does not support float64
    if args.model_class == 'SAC':
        all_kwargs['use_doubles'] = False

    return gym.make(
        args.envname,
        **all_kwargs,
    )


# def load_model(path):
#     with open(path, 'rb') as f:
#         cp = jnp.load(f)
#         return cp['opt_state'], cp.get('steps', 0)


# def save_model(path, opt_state, steps):
#     with open(path, 'wb') as f:
#         jnp.savez(f, opt_state=opt_state, steps=steps)


def test_model(model, params, rng_key, env, ntests, name,
               loss_func=None, stats_path=None):
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
                # print(obs.shape)
                # print(params.shape)
                rng_key, subkey = jax.random.split(rng_key)
                action = model(params, obs, rng=subkey)
                # loss = loss_func(action, obs)
                # print('test mean lam:', jnp.mean(obs).item(),
                #       'loss:', loss.item(), 'action:', action)
                action = np.array(action)

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


def run_tests(model, params, args,
              seed=None, fig_path=None, loss_func=None,
              stats_path=None):
    """Run tests for the given `model` and `args`, using `seed` as the
    random seed.

    `fig_path` is an optional path to store result plots at.
    `stats_path` is an optional path where to save statistics about the
    reinforcement learning test.
    """
    # Load the trained agent for testing
    if isinstance(model, (Path, str)):
        path = model
        model_init, model = build_model(args.M, train=False)
        params, _ = load_model(path)

    rng_key = jax.random.PRNGKey(seed)
    args.envname = 'sdc-v0'
    num_test_envs = args.batch_size
    ntests = int(args.tests)
    ntests = utils.maybe_fix_ntests(ntests, num_test_envs)

    start_time = time.perf_counter()
    # Test the trained model.
    env = utils.make_env(args, num_envs=num_test_envs, seed=seed,
                         lambda_real_interpolation_interval=None,
                         do_scale=False)
    results_RL = test_model(
        model, params, rng_key, env, ntests, 'RL', loss_func,
        stats_path=stats_path)

    # Restart the whole thing, but now using the LU preconditioner (no RL here)
    # LU is serial and the de-facto standard. Beat this (or at least be on par)
    # and we win!
    env = utils.make_env(
        args,
        num_envs=num_test_envs,
        prec='LU',
        seed=seed,
        lambda_real_interpolation_interval=None,
        do_scale=False,
    )
    results_LU = test_model(model, params, rng_key, env, ntests, 'LU')

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
        do_scale=False,
    )
    results_min = test_model(model, params, rng_key, env, ntests, 'MIN')
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
    args = parse_args()
    utils.setup(True)
    weight_decay_factor = 0.0
    max_grad_norm = 0.2

    eval_seed = args.seed
    if eval_seed is not None:
        eval_seed += 1

    rng_key = jax.random.PRNGKey(args.seed)

    env = make_env(args, num_envs=1, batch_size=args.batch_size,
                   do_scale=False)

    input_shape = env.observation_space.shape
    model_init, model = build_model(args.M, train=True)
    rng_key, subkey = jax.random.split(rng_key)
    _, params = model_init(subkey, input_shape)
    opt_state, opt_update, opt_get_params = build_opt(args.learning_rate,
                                                      params)

    if args.model_path is not None:
        params, old_steps = load_model(args.model_path)

    @jax.jit
    def loss(params, obs, rng_key):
        diags = model(params, obs, rng=rng_key)
        _, _, _, info = env.step(diags)
        norm_res = jnp.mean(info['residual'])

        weight_penalty = optimizers.l2_norm(params)
        return norm_res + weight_decay_factor * weight_penalty

    @jax.jit
    def update(i, opt_state, obs, rng_key):
        params = opt_get_params(opt_state)
        # print(params)
        rng_key, subkey = jax.random.split(rng_key)
        loss_, gradient = jax.value_and_grad(loss)(params, obs, subkey)
        # print(gradient)
        gradient = optimizers.clip_grads(gradient, max_grad_norm)
        opt_state = opt_update(i, gradient, opt_state)
        return loss_, opt_state, rng_key

    old_steps = 0
    steps = int(args.steps)
    steps_num_digits = len(str(steps))

    log_interval = 100
    best_loss = np.inf
    last_losses = np.zeros(log_interval)
    start_time = time.perf_counter()
    for step in range(steps):
        obs = env.reset()
        loss_, opt_state, rng_key = update(
            step + old_steps, opt_state, obs, rng_key)

        last_losses[step % len(last_losses)] = loss_.item()

        if step % log_interval == 0:
            mean_loss = jnp.mean(last_losses[:step + 1]).item()
            if mean_loss < best_loss and steps > 0:
                best_loss = mean_loss
                cp_path = Path(f'best_dp_model_{script_start}.npy')
                save_model(
                    cp_path, opt_get_params(opt_state), steps + old_steps)

            print(f'[{step:>{steps_num_digits}d}/{steps}] '
                  f'mean_loss: {mean_loss:.20f}')

        if step >= steps:
            break
    duration = time.perf_counter() - start_time
    print(f'Training took {duration} seconds.')

    _, model = build_model(args.M, train=False)
    params = opt_get_params(opt_state)
    if steps > 0:
        cp_path = Path(f'dp_model_{script_start}.npy')
        save_model(cp_path, params, steps + old_steps)
    elif args.model_path is not None:
        params, _ = load_model(args.model_path)
    fig_path = Path(f'dp_results_{script_start}.pdf')
    run_tests(model, params, args,
              seed=eval_seed, fig_path=fig_path, loss_func=loss)


if __name__ == '__main__':
    main()
