import argparse
import datetime
from pathlib import Path
import time

import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental import optimizers
import matplotlib.pyplot as plt
import numpy as np
from pySDC.implementations.collocation_classes.gauss_radau_right import \
    CollGaussRadau_Right

from rl_playground import _store_test_stats, plot_results
import utils


class DataGenerator:
    def __init__(
            self,
            M,
            lambda_real_interval,
            lambda_imag_interval,
            batch_size,
            rng_key,
    ):
        super().__init__()
        self.M = M
        self.coll = CollGaussRadau_Right(M, 0, 1)
        self.Q = self.coll.Qmat[1:, 1:]
        self.lambda_real_interval = lambda_real_interval
        self.lambda_imag_interval = lambda_imag_interval
        self.batch_size = batch_size
        self.rng_key = rng_key

        self.lam_real_low = self.lambda_real_interval[0]
        self.lam_real_high = self.lambda_real_interval[1]

        self.lam_imag_low = self.lambda_imag_interval[0]
        self.lam_imag_high = self.lambda_imag_interval[1]

    def _generate_lambdas(self):
        rng_key, subkey = jax.random.split(self.rng_key)
        self.rng_key, subkey2 = jax.random.split(rng_key)
        lams = (
            1 * jax.random.uniform(subkey, (self.batch_size, 1),
                                   minval=self.lam_real_low,
                                   maxval=self.lam_real_high)
            + 1j * jax.random.uniform(subkey2, (self.batch_size, 1),
                                      minval=self.lam_imag_low,
                                      maxval=self.lam_imag_high)
        )
        return lams

    def __iter__(self):
        while True:
            lams = self._generate_lambdas()
            yield lams


class NormLoss:
    def __init__(self, M, dt):
        coll = CollGaussRadau_Right(M, 0, 1)
        self.Q = jnp.array(coll.Qmat[1:, 1:])
        self.M = M
        self.dt = dt

    def _get_spectral_radius(self, lam, diag):
        Qdmat = jnp.diag(diag)

        # Precompute the inverse of P
        Pinv = jnp.linalg.inv(
            jnp.eye(self.M)
            - lam * self.dt * Qdmat
        )
        mPinv = jnp.dot(Pinv, self.Q - Qdmat)
        # print(mPinv)
        evals = jnp.linalg.eigvals(lam * self.dt * mPinv)
        # print(evals)
        # absed = jnp.abs(evals)
        # print(absed)
        spectral_radius = jnp.max(jnp.abs(evals))
        # print(spectral_radius)
        return spectral_radius

    def __call__(self, lams, diags):
        return jnp.mean(jax.vmap(self._get_spectral_radius)(lams, diags))


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
    args.envname = 'sdc-v0'

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


def build_model(M):
    scale = 1e-3
    glorot_normal = jax.nn.initializers.variance_scaling(
        scale, "fan_avg", "truncated_normal")
    normal = jax.nn.initializers.normal(scale)
    (model_init, model_apply) = stax.serial(
        stax.Dense(64, glorot_normal, normal),
        stax.Relu,
        # stax.Dense(256),
        # stax.Relu,
        stax.Dense(64, glorot_normal, normal),
        stax.Relu,
        stax.Dense(M, glorot_normal, normal),
    )
    return (model_init, model_apply)


def build_opt(lr, params):
    lr = optimizers.polynomial_decay(lr, 15000, lr * 1e-7, 2.0)

    (opt_init, opt_update, opt_get_params) = optimizers.adam(lr)
    opt_state = opt_init(params)
    return (opt_state, opt_update, opt_get_params)


def load_model(path):
    with open(path, 'rb') as f:
        weights = jnp.load(f, allow_pickle=True)
    with open(str(path) + '.steps', 'rb') as f:
        steps = jnp.load(f, allow_pickle=True)
    return weights, steps


def save_model(path, params, steps):
    with open(path, 'wb') as f:
        jnp.save(f, params)
    with open(str(path) + '.steps', 'wb') as f:
        jnp.save(f, steps)


# def load_model(path):
#     with open(path, 'rb') as f:
#         cp = jnp.load(f)
#         return cp['opt_state'], cp.get('steps', 0)


# def save_model(path, opt_state, steps):
#     with open(path, 'wb') as f:
#         jnp.savez(f, opt_state=opt_state, steps=steps)


def test_model(model, params, env, ntests, name,
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
        env.reset()
        obs = jnp.array([env_.lam for env_ in env.envs]).reshape(-1, 1)
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
                action = model(params, obs)
                # loss = loss_func(action, obs)
                # print('test mean lam:', jnp.mean(obs).item(),
                #       'loss:', loss.item(), 'action:', action)
                action = np.array(action)

            _, rewards, done, info = env.step(action)
            obs = jnp.array([env_.lam for env_ in env.envs]).reshape(-1, 1)

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
        model_init, model = build_model(args.M)
        params,  _ = load_model(path)

    num_test_envs = args.batch_size
    ntests = int(args.tests)
    ntests = utils.maybe_fix_ntests(ntests, num_test_envs)

    start_time = time.perf_counter()
    # Test the trained model.
    env = utils.make_env(args, num_envs=num_test_envs, seed=seed,
                         lambda_real_interpolation_interval=None,
                         do_scale=False)
    results_RL = test_model(
        model, params, env, ntests, 'RL', loss_func, stats_path=stats_path)

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
    results_LU = test_model(model, params, env, ntests, 'LU')

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
    results_min = test_model(model, params, env, ntests, 'MIN')
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

    eval_seed = args.seed
    if eval_seed is not None:
        eval_seed += 1

    rng_key = jax.random.PRNGKey(args.seed)

    dataloader = DataGenerator(
        args.M,
        args.lambda_real_interval,
        args.lambda_imag_interval,
        args.batch_size,
        rng_key,
    )

    input_shape = (args.batch_size, 1)
    model_init, model = build_model(args.M)
    rng_key, subkey = jax.random.split(rng_key)
    _, params = model_init(subkey, input_shape)
    opt_state, opt_update, opt_get_params = build_opt(args.learning_rate,
                                                      params)
    loss_func = NormLoss(args.M, args.dt)

    if args.model_path is not None:
        params, old_steps = load_model(args.model_path)

    @jax.jit
    def loss(params, lams):
        diags = model(params, lams)
        return loss_func(lams, diags)

    @jax.jit
    def update(i, opt_state, lams):
        params = opt_get_params(opt_state)
        loss_, gradient = jax.value_and_grad(loss)(params, lams)
        # print(gradient)
        opt_state = opt_update(i, gradient, opt_state)
        return loss_, opt_state

    old_steps = 0
    steps = int(args.steps)
    steps_num_digits = len(str(steps))

    best_loss = np.inf
    last_losses = np.zeros(100)
    start_time = time.perf_counter()
    for (step, lams) in enumerate(dataloader):
        loss_, opt_state = update(step + old_steps, opt_state, lams)

        last_losses[step % len(last_losses)] = loss_.item()

        if step % 100 == 0:
            mean_loss = jnp.mean(last_losses[:step + 1]).item()
            if mean_loss < best_loss and steps > 0:
                best_loss = mean_loss
                cp_path = Path(f'best_dp_model_{script_start}.npy')
                save_model(
                    cp_path, opt_get_params(opt_state), steps + old_steps)

            print(f'[{step:>{steps_num_digits}d}/{steps}] '
                  f'mean_loss: {mean_loss:.5f}')

        if step >= steps:
            break
    duration = time.perf_counter() - start_time
    print(f'Training took {duration} seconds.')

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
