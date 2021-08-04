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
import optax
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
        self.rng_key, subkey, subkey2 = jax.random.split(self.rng_key, 3)
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


def _from_model_arch(model_arch, train):
    # For smaller intervals, this should be higher (e.g. 1e-3);
    # for larger intervals lower is better (e.g. 1e-7).
    scale = 1e-7
    glorot_normal = jax.nn.initializers.variance_scaling(
        scale, "fan_avg", "truncated_normal")
    normal = jax.nn.initializers.normal(scale)

    dropout_rate = 0.0
    mode = 'train' if train else 'test'
    dropout_keep_rate = 1 - dropout_rate

    model_arch_real = []
    for tup in model_arch:
        if not isinstance(tup, tuple):
            tup = (tup,)
        name = tup[0]
        if len(tup) > 1:
            args = tup[1]
        if len(tup) > 2:
            kwargs = tup[2]

        layer = getattr(stax, name)
        if name == 'Dense':
            args = args + (glorot_normal, normal)
        elif name == 'Dropout':
            args = args + (dropout_keep_rate, mode)

        if len(tup) == 1:
            model_arch_real.append(layer)
        elif len(tup) == 2:
            model_arch_real.append(layer(*args))
        elif len(tup) == 3:
            model_arch_real.append(layer(*args, **kwargs))
        else:
            raise ValueError('error in model_arch syntax')
    (model_init, model_apply) = stax.serial(*model_arch_real)
    return (model_init, model_apply)


def build_model(M, train):
    # 12 (or more) hidden layers give good results sometimes.
    #
    # For very large intervals in both real and imaginary space, weird
    # architectures like 3 hidden layers with 2 neurons may work as well
    # as 3 hidden layers with 512 neurons. However, those results can
    # probably be improved.
    model_arch = [
        ('Dense', (128,)),
        ('Dropout', ()),
        ('Relu',),
        ('Dense', (128,)),
        ('Relu',),
        ('Dense', (128,)),
        ('Dropout', ()),
        ('Relu',),
        ('Dense', (M,)),
    ]

    (model_init, model_apply) = _from_model_arch(model_arch, train=train)

    return (model_init, model_apply, model_arch)


class RandLR:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, step):
        rng_key = jax.random.PRNGKey(step)
        return jax.random.uniform(
            rng_key, (), minval=self.low, maxval=self.high)


class SineLR:
    def __init__(self, lr, amplitude, steps_per_wave, phase=0.0):
        self.lr = lr
        self.amplitude = amplitude
        self.steps_per_wave = steps_per_wave
        self.phase = phase

    def __call__(self, step):
        return (
            self.lr
            + jnp.sin(
                step * jnp.pi * 2 / self.steps_per_wave
                + self.phase
            ) * self.amplitude
        )


class CosineLR:
    def __init__(self, lr, amplitude, steps_per_wave, phase=0.0):
        self.lr = lr
        self.amplitude = amplitude
        self.steps_per_wave = steps_per_wave
        self.phase = phase

    def __call__(self, step):
        return (
            self.lr
            + jnp.cos(
                step * jnp.pi * 2 / self.steps_per_wave
                + self.phase
            ) * self.amplitude
        )


def build_opt(lr, params, old_steps):
    # lr = optimizers.polynomial_decay(lr, 15000, lr * 1e-7, 2.0)
    # lr = optimizers.polynomial_decay(2 * lr, 50000, lr * 2e-9, 2.0)
    # lr = optax.cosine_onecycle_schedule(19000, 2e2 * lr, 0.3, 2e9)
    transition_steps = 30000
    max_lr = 2 * lr
    start_lr = 3e-13
    final_lr = 1e-9
    wave_base_lr = 1e-10
    wave_max_lr = 1e-7

    div_factor = max_lr / start_lr
    final_div_factor = start_lr / final_lr
    lr = optax.cosine_onecycle_schedule(transition_steps, max_lr, 0.1, div_factor, final_div_factor)
    lr2 = optax.cosine_onecycle_schedule(transition_steps, wave_max_lr, 0.1, wave_max_lr / final_lr, final_lr / 1e-10)
    lr3 = optax.cosine_onecycle_schedule(transition_steps, wave_max_lr * 1e-1, 0.1, wave_max_lr / final_lr, final_lr / 1e-11)
    lr4 = optax.cosine_onecycle_schedule(transition_steps, wave_max_lr * 1e-2, 0.1, wave_max_lr / final_lr, final_lr / 1e-12)
    lr = optax.join_schedules([lr, lr2, lr3, lr4], [transition_steps, transition_steps * 2, transition_steps * 3])
    # lr = RandLR(2e-9 * lr, 2 * lr)
    amplitude = wave_max_lr - wave_base_lr
    # lr2 = SineLR(wave_base_lr, amplitude, 50000)
    # lr = optax.join_schedules([lr, lr2], [transition_steps, 30000 + 50000 * 3])

    # These are good for generalization. However, we would like the
    # model to overfit so adaptive methods seem to be the better choice.
    # (opt_init, opt_update, opt_get_params) = optimizers.sgd(lr_)
    # (opt_init, opt_update, opt_get_params) = optimizers.nesterov(lr, 0.9)
    (opt_init, opt_update, opt_get_params) = optimizers.adam(lr)
    opt_state = opt_init(params)
    return (opt_state, opt_update, opt_get_params)


def load_model(path):
    with open(path, 'rb') as f:
        weights = jnp.load(f, allow_pickle=True)
    with open(str(path) + '.structure', 'rb') as f:
        model_arch = jnp.load(f, allow_pickle=True)
    with open(str(path) + '.steps', 'rb') as f:
        steps = jnp.load(f, allow_pickle=True)
    return weights, model_arch, steps


def save_model(path, params, model_arch, steps):
    with open(path, 'wb') as f:
        jnp.save(f, params)
    with open(str(path) + '.structure', 'wb') as f:
        jnp.save(f, model_arch)
    with open(str(path) + '.steps', 'wb') as f:
        jnp.save(f, steps)


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
                    'TimeLimit.truncated',
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
                stats['obs'].append(np.array(obs))

            # Do not predict an action when we would discard it anyway
            if env.envs[0].prec is None:
                rng_key, subkey = jax.random.split(rng_key)
                action = model(params, obs, rng=subkey)
                # loss = loss_func(action, obs)
                # print('test mean lam:', jnp.mean(obs).item(),
                #       'loss:', loss.item(), 'action:', action)
                action = np.array(action)

            _, rewards, done, info = env.step(action)
            obs = jnp.array([env_.lam for env_ in env.envs]).reshape(-1, 1)

            if stats_path is not None:
                stats['action'].append(action)
                stats['reward'].append(np.array(rewards))
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
        params, model_arch, _ = load_model(path)
        _, model = _from_model_arch(model_arch, train=False)

    model = jax.jit(model)

    rng_key = jax.random.PRNGKey(seed)
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


def get_cp_name(args, script_start):
    re_interval = '_'.join(map(str, sorted(args.lambda_real_interval)))
    im_interval = '_'.join(map(str, sorted(args.lambda_imag_interval)))
    return (
        f'dp_model_M_{args.M}_re_{re_interval}_im_{im_interval}_loss_{{}}_'
        f'{script_start}.npy'
    )


def main():
    script_start = str(datetime.datetime.now()
                       ).replace(':', '-').replace(' ', 'T')
    args = parse_args()
    utils.setup(True)

    eval_seed = args.seed
    if eval_seed is not None:
        eval_seed += 1

    rng_key = jax.random.PRNGKey(args.seed)
    rng_key, subkey = jax.random.split(rng_key)

    dataloader = DataGenerator(
        args.M,
        args.lambda_real_interval,
        args.lambda_imag_interval,
        args.batch_size,
        subkey,
    )

    input_shape = (args.batch_size, 1)
    model_init, model, model_arch = build_model(args.M, train=True)

    rng_key, subkey = jax.random.split(rng_key)
    _, params = model_init(subkey, input_shape)
    if args.model_path is not None:
        params, model_arch, old_steps = load_model(args.model_path)
        _, model = _from_model_arch(model_arch, train=True)
        params = list(params)
    else:
        old_steps = 0

    opt_state, opt_update, opt_get_params = build_opt(
        args.learning_rate, params, old_steps)
    loss_func = NormLoss(args.M, args.dt)

    max_grad_norm = 0.5
    # grad_clipping_schedule = optimizers.polynomial_decay(
    #     max_grad_norm, 15000, 1e-7, 1.0)

    # Better to avoid this; always worsened results.
    weight_decay_factor = 0.0
    weight_decay_schedule = optimizers.polynomial_decay(
        weight_decay_factor, 15000, 0.0, 2.0)

    # @jax.jit
    def loss(params, lams, i, rng_key):
        diags = model(params, lams, rng=rng_key)
        loss_ = loss_func(lams, diags)
        weight_penalty = optimizers.l2_norm(params)
        weight_decay_factor = weight_decay_schedule(i)
        return loss_ + weight_decay_factor * weight_penalty

    @jax.jit
    def update(i, opt_state, lams, rng_key):
        params = opt_get_params(opt_state)
        rng_key, subkey = jax.random.split(rng_key)
        loss_, gradient = jax.value_and_grad(loss)(
            params, lams, i.astype(float), subkey)
        # print(gradient)
        # max_grad_norm = grad_clipping_schedule(i)
        gradient = optimizers.clip_grads(gradient, max_grad_norm)
        opt_state = opt_update(i, gradient, opt_state)
        return loss_, opt_state, rng_key

    steps = int(args.steps)
    steps_num_digits = len(str(steps))

    cp_name = get_cp_name(args, script_start)
    best_cp_name = 'best_' + cp_name

    best_loss = np.inf
    last_losses = np.zeros(100)
    start_time = time.perf_counter()
    for (step, lams) in enumerate(dataloader):
        loss_, opt_state, rng_key = update(
            jnp.array(step + old_steps), opt_state, lams, rng_key)

        last_losses[step % len(last_losses)] = loss_.item()

        if step % 100 == 0:
            mean_loss = jnp.mean(last_losses[:step + 1]).item()
            if mean_loss < best_loss and steps > 0:
                best_loss = mean_loss
                best_cp_path = Path(best_cp_name.format(mean_loss))
                save_model(
                    best_cp_path,
                    opt_get_params(opt_state),
                    model_arch,
                    steps + old_steps,
                )

            print(f'[{step:>{steps_num_digits}d}/{steps}] '
                  f'mean_loss: {mean_loss:.5f}')

        if step >= steps:
            break
    duration = time.perf_counter() - start_time
    print(f'Training took {duration} seconds.')

    _, model = _from_model_arch(model_arch, train=False)
    params = opt_get_params(opt_state)
    if steps > 0:
        cp_path = Path(cp_name.format(mean_loss))
        save_model(
            cp_path,
            params,
            model_arch,
            steps + old_steps,
        )
    elif args.model_path is not None:
        params, model_arch, _ = load_model(args.model_path)
        params = list(params)
    fig_path = Path(f'dp_results_{script_start}.pdf')
    run_tests(model, params, args,
              seed=eval_seed, fig_path=fig_path, loss_func=loss)


if __name__ == '__main__':
    main()
