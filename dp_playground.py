import argparse
import datetime
import functools
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


Real = stax.elementwise(jnp.real)


def Params(out_dim, init):
    """Layer construction function for an identity layer of the parameters."""
    def init_fun(rng, input_shape):
        params = init(rng, (out_dim,))
        return (out_dim, (params,))

    def apply_fun(params, inputs, **kwargs):
        return jnp.tile(params[0], (len(inputs), 1))
    return init_fun, apply_fun


class UnknownPrecTypeError(NotImplementedError):
    def __init__(self, message=None, *args, **kwargs):
        if message is None:
            message = 'unknown `prec_type` (check your arguments)'
        super().__init__(message, *args, **kwargs)


class UnknownInputTypeError(NotImplementedError):
    def __init__(self, message=None, *args, **kwargs):
        if message is None:
            message = 'unknown `input_type` (check your arguments)'
        super().__init__(message, *args, **kwargs)


class UnknownLossTypeError(NotImplementedError):
    def __init__(self, message=None, *args, **kwargs):
        if message is None:
            message = 'unknown `loss_type` (check your arguments)'
        super().__init__(message, *args, **kwargs)


def _compute_residual(u0, u, C):
    return u0 - C @ u


class DataGenerator:
    def __init__(
            self,
            M,
            dt,
            lambda_real_interval,
            lambda_imag_interval,
            u_real_interval,
            u_imag_interval,
            batch_size,
            input_type,
            loss_type,
            rng_key,
    ):
        super().__init__()
        self.M = M
        self.dt = dt
        self.coll = CollGaussRadau_Right(M, 0, 1)
        self.Q = self.coll.Qmat[1:, 1:]
        self.lambda_real_interval = lambda_real_interval
        self.lambda_imag_interval = lambda_imag_interval
        self.u_real_interval = u_real_interval
        self.u_imag_interval = u_imag_interval
        self.batch_size = batch_size
        self.input_type = input_type
        self.loss_type = loss_type
        self.rng_key = rng_key

        self.lam_real_low = self.lambda_real_interval[0]
        self.lam_real_high = self.lambda_real_interval[1]
        self.lam_imag_low = self.lambda_imag_interval[0]
        self.lam_imag_high = self.lambda_imag_interval[1]

        self.u_real_low = self.u_real_interval[0]
        self.u_real_high = self.u_real_interval[1]
        self.u_imag_low = self.u_imag_interval[0]
        self.u_imag_high = self.u_imag_interval[1]

    def _generate_lambdas(self, rng_key):
        rng_keys = jax.random.split(rng_key, 3)
        lams = (
            1 * jax.random.uniform(rng_keys[1], (self.batch_size, 1),
                                   minval=self.lam_real_low,
                                   maxval=self.lam_real_high)
            + 1j * jax.random.uniform(rng_keys[2], (self.batch_size, 1),
                                      minval=self.lam_imag_low,
                                      maxval=self.lam_imag_high)
        )
        return lams, rng_keys[0]

    def _compute_system_matrix(self,lam):
        return jnp.eye(self.M) - lam * self.dt * self.Q

    def _generate_us(self, rng_key):
        rng_keys = jax.random.split(rng_key, 3)
        us = (
            1 * jax.random.uniform(rng_keys[1], (self.batch_size, self.M),
                                   minval=self.u_real_low,
                                   maxval=self.u_real_high)
            + 1j * jax.random.uniform(rng_keys[2], (self.batch_size, self.M),
                                      minval=self.u_imag_low,
                                      maxval=self.u_imag_high)
        )
        return us, rng_keys[0]

    @functools.partial(jax.jit, static_argnums=(0,))
    def _generate_inputs(self, rng_key):
        lams, rng_key = self._generate_lambdas(rng_key)
        # Stop early if we don't need other data.
        if self.input_type == 'lambda' and self.loss_type == 'spectral_radius':
            return lams, (), (), rng_key

        Cs = jax.vmap(self._compute_system_matrix)(lams)
        us, rng_key = self._generate_us(rng_key)
        residuals = jax.vmap(_compute_residual)(us, us, Cs)

        if self.loss_type == 'spectral_radius':
            loss_data = ()
        elif self.loss_type == 'residual':
            loss_data = (Cs, us, residuals)
        else:
            raise UnknownLossTypeError()

        if self.input_type == 'lambda':
            input_data = ()
        elif self.input_type == 'residual':
            input_data = (residuals,)
        elif self.input_type == 'lambda_u':
            input_data = (us,)
        else:
            raise UnknownInputTypeError()

        return lams, input_data, loss_data, rng_key

    def generate_inputs(self):
        lams, input_data, loss_data, self.rng_key = self._generate_inputs(self.rng_key)
        return lams, input_data, loss_data

    def __iter__(self):
        while True:
            inputs = self.generate_inputs()
            yield inputs


class SpectralRadiusLoss:
    def __init__(self, M, dt, prec_type):
        coll = CollGaussRadau_Right(M, 0, 1)
        self.Q = jnp.array(coll.Qmat[1:, 1:])
        self.M = M
        self.dt = dt
        self.prec_type = prec_type

    def get_qdmat(self, output):
        if self.prec_type == 'diag':
            Qdmat = jnp.diag(output)
        elif self.prec_type == 'lower_diag':
            Qdmat = jnp.diag(output, k=-1)
        elif self.prec_type == 'lower_tri':
            Qdmat = jnp.zeros((self.M, self.M))
            Qdmat = Qdmat.at[jnp.tril_indices(self.M)].set(output)
        elif self.prec_type == 'strictly_lower_tri':
            Qdmat = jnp.zeros((self.M, self.M))
            Qdmat = Qdmat.at[jnp.tril_indices(self.M, k=-1)].set(output)
        else:
            raise UnknownPrecTypeError()
        return Qdmat

    def compute_pinv(self, lam, Qdmat):
        # Compute the inverse of P
        Pinv = jnp.linalg.inv(
            jnp.eye(self.M) - lam * self.dt * Qdmat,
        )
        return Pinv

    def _get_spectral_radius(self, lam, output):
        Qdmat = self.get_qdmat(output)

        Pinv = self.compute_pinv(lam, Qdmat)
        mPinv = jnp.dot(Pinv, self.Q - Qdmat)
        # print(mPinv)
        evals = jnp.linalg.eigvals(lam * self.dt * mPinv)
        # print(evals)
        # absed = jnp.abs(evals)
        # print(absed)
        spectral_radius = jnp.max(jnp.abs(evals))
        # print(spectral_radius)
        return spectral_radius

    def __call__(self, lams, outputs):
        return jnp.mean(jax.vmap(self._get_spectral_radius)(lams, outputs))

NormLoss = SpectralRadiusLoss  # Backward compatibility


class ResidualLoss:
    def __init__(self, M, dt, prec_type):
        coll = CollGaussRadau_Right(M, 0, 1)
        self.Q = jnp.array(coll.Qmat[1:, 1:])
        self.M = M
        self.dt = dt
        self.u0 = jnp.ones(M, dtype=complex)
        self.prec_type = prec_type

    def _inf_norm(self, v):
        return jnp.linalg.norm(v, jnp.inf)

    def take_step(self, C, lam, output, u, old_residual):
        Qdmat = SpectralRadiusLoss.get_qdmat(self, output)
        Pinv = SpectralRadiusLoss.compute_pinv(self, lam, Qdmat)
        u = u + Pinv @ old_residual
        residual = _compute_residual(self.u0, u, C)
        return (u, residual)

    def __call__(self, lams, outputs, Cs, us, old_residuals):
        us, residuals = jax.vmap(self.take_step)(
            Cs, lams, outputs, us, old_residuals)
        residual_norms = jax.vmap(self._inf_norm)(residuals)
        return (jnp.mean(residual_norms), us, residuals)


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
        type=float,
        nargs=2,
        default=[-100.0, 0.0],
        help='Interval to sample the real part of lambda from.',
    )
    parser.add_argument(
        '--lambda_imag_interval',
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        help='Interval to sample the imaginary part of lambda from.',
    )
    parser.add_argument(
        '--u_real_interval',
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        help='Interval to sample the real part of u from.',
    )
    parser.add_argument(
        '--u_imag_interval',
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        help='Interval to sample the imaginary part of u from.',
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
        help='Model checkpoint to load (a .npy file).',
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
    parser.add_argument(
        '--envname',
        type=str,
        default=None,
        help=(
            'Gym environment to use;\n    sdc-v0 – SDC with a full iteration '
            'per step (no intermediate observations),\n    sdc-v1 – SDC with '
            'a single iteration per step (with intermediate observations).\n'
            'By default, choose based on other options.'
        ),
    )

    parser.add_argument(
        '--float64',
        type=utils.parse_bool,
        default=True,
        help='Whether to use double precision.',
    )
    parser.add_argument(
        '--prec_type',
        type=str,
        default='diag',
        help=(
            'How to shape the learned preconditioner. Valid values '
            'include "diag", "lower_diag", "lower_tri", and '
            '"strictly_lower_tri" (with zero-diagonal).'
        ),
    )
    parser.add_argument(
        '--input_type',
        type=str,
        default='lambda',
        help=(
            'What or how to train. Valid values '
            'include "lambda", "residual", and "lambda_u".'
        ),
    )
    parser.add_argument(
        '--loss_type',
        type=str,
        default='spectral_radius',
        help=(
            'Loss function to use for training. Valid values '
            'include "spectral_radius" (spectral radius of iteration matrix), '
            'and "residual" (residual after one iteration step).'
        ),
    )
    parser.add_argument(
        '--extensive_tests',
        type=utils.parse_bool,
        default=False,
        help=(
            'Whether to do testing on a zero-preconditioner and '
            'one using explicit Euler as well.'
        ),
    )

    args = parser.parse_args()
    if args.envname is None:
        if args.input_type != '' and args.loss_type == 'residual':
            args.envname = 'sdc-v1'
        else:
            args.envname = 'sdc-v0'

    args.lambda_real_interval = sorted(args.lambda_real_interval)
    args.lambda_imag_interval = sorted(args.lambda_imag_interval)
    args.u_real_interval = sorted(args.u_real_interval)
    args.u_imag_interval = sorted(args.u_imag_interval)
    args.prec_type = args.prec_type.lower()
    args.input_type = args.input_type.lower()
    args.loss_type = args.loss_type.lower()

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
        scale, "fan_avg", "truncated_normal", dtype=float)
    normal = jax.nn.initializers.normal(scale, dtype=float)
    dropout_rate = 0.0
    mode = 'train' if train else 'test'
    dropout_keep_rate = 1 - dropout_rate
    globals_ = globals()

    args_dict = {
        'Params': (normal,),
        'Dense': (glorot_normal, normal),
        'Dropout': (dropout_keep_rate, mode),
    }

    model_arch_real = []
    for tup in model_arch:
        if not isinstance(tup, tuple):
            try:
                tup = tuple(tup)
            except TypeError:
                tup = (tup,)

        if len(tup) == 0 or len(tup) > 3:
            raise ValueError('error in model_arch syntax')

        name = tup[0]
        if len(tup) > 1:
            args = tup[1]
        else:
            args = ()
        if len(tup) > 2:
            kwargs = tup[2]
        else:
            kwargs = {}

        if not hasattr(stax, name):
            if name not in globals_:
                raise ValueError(
                    f'unknown layer name "{name}". Names are case-sensitive.')
            layer = globals_[name]
        else:
            layer = getattr(stax, name)

        args = args + args_dict.get(name, ())

        if len(tup) > 1:
            model_arch_real.append(layer(*args, **kwargs))
        else:
            model_arch_real.append(layer)
    (model_init, model_apply) = stax.serial(*model_arch_real)
    return (model_init, model_apply)


def get_input_size(M, input_type):
    if input_type == 'lambda':
        input_size = 1
    elif input_type == 'residual':
        input_size = M
    elif input_type == 'lambda_u':
        input_size = 1 + M
    else:
        raise UnknownPrecTypeError()
    return input_size


def get_output_size(M, prec_type):
    if prec_type == 'diag':
        output_size = M
    elif prec_type == 'lower_diag':
        output_size = M - 1
    elif prec_type == 'lower_tri':
        output_size = (M * (M + 1)) // 2
    elif prec_type == 'strictly_lower_tri':
        output_size = ((M - 1) * M) // 2
    else:
        raise UnknownPrecTypeError()
    return output_size


def build_model(args, train):
    output_size = get_output_size(args.M, args.prec_type)

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
        ('Dense', (output_size,)),
        # ('Real',),
        # ('Params', (output_size,)),
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


def build_opt(args, params, old_steps):
    lr = args.learning_rate

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
    # lr2 = optax.cosine_onecycle_schedule(transition_steps, wave_max_lr * 1e-1, 0.1, wave_max_lr / final_lr, final_lr / 1e-11)
    lr3 = optax.cosine_onecycle_schedule(transition_steps, wave_max_lr * 1e-1, 0.1, wave_max_lr / final_lr, final_lr / 1e-11)
    # lr4 = optax.cosine_onecycle_schedule(transition_steps, wave_max_lr * 1e-1, 0.1, wave_max_lr / final_lr, final_lr / 1e-11)
    lr4 = optax.cosine_onecycle_schedule(transition_steps, wave_max_lr * 1e-2, 0.1, wave_max_lr / final_lr, final_lr / 1e-12)
    lr5 = optax.cosine_onecycle_schedule(transition_steps, wave_max_lr * 1e-3, 0.1, wave_max_lr / final_lr, final_lr / 1e-13)
    lr = optax.join_schedules([lr, lr2, lr3, lr4, lr5], [transition_steps, transition_steps * 2, transition_steps * 3, transition_steps * 4])
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


def delete_model(path):
    if path is None:
        return
    if not isinstance(path, Path):
        path = Path(path)
    path.unlink(missing_ok=True)
    path.with_name(path.name + '.structure').unlink(missing_ok=True)
    path.with_name(path.name + '.steps').unlink(missing_ok=True)


def get_loss_func(M, dt, prec_type, loss_type):
    if loss_type == 'spectral_radius':
        loss_func = SpectralRadiusLoss(M, dt, prec_type)
    elif loss_type == 'residual':
        loss_func = ResidualLoss(M, dt, prec_type)
    else:
        raise UnknownLossTypeError()
    return loss_func


def check_output_size(args, model, params):
    test_out = model(
        params,
        jnp.zeros((1, 1), dtype=float),
        rng=jax.random.PRNGKey(0),
    )
    expected_output_size = get_output_size(args.M, args.prec_type)
    assert test_out.size == expected_output_size, (
        f'Output size of the model on an example input does '
        f'not match expected size '
        f'({test_out.size} != {expected_output_size}). '
        f'Possibly the `prec_type` needs to be adjusted?'
    )


# def load_model(path):
#     with open(path, 'rb') as f:
#         cp = jnp.load(f)
#         return cp['opt_state'], cp.get('steps', 0)


# def save_model(path, opt_state, steps):
#     with open(path, 'wb') as f:
#         jnp.savez(f, opt_state=opt_state, steps=steps)


def get_obs(env, input_type):
    if input_type == 'lambda':
        obs = jnp.array([env_.lam for env_ in env.envs]).reshape(-1, 1)
    elif input_type == 'residual':
        obs = jnp.stack([env_.initial_residual for env_ in env.envs])
    elif input_type == 'lambda_u':
        obs = jnp.hstack((
            jnp.array([env_.lam for env_ in env.envs]).reshape(-1, 1),
            jnp.array([env_.state[0] for env_ in env.envs]).reshape(
                -1, env.envs[0].M),
        ))
    else:
        raise UnknownInputTypeError()
    return obs


def test_model(model, params, input_type, rng_key, env, ntests, name,
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
        obs = get_obs(env, input_type)
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
            obs = get_obs(env, input_type)

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
        params = list(params)
        check_output_size(args, model, params)

    model = jax.jit(model)

    input_type = args.input_type
    rng_key = jax.random.PRNGKey(seed)
    num_test_envs = 1
    ntests = int(args.tests)
    ntests = utils.maybe_fix_ntests(ntests, num_test_envs)

    old_envname = args.envname
    start_time = time.perf_counter()
    # Test the trained model.
    env = utils.make_env(args, num_envs=num_test_envs, seed=seed,
                         lambda_real_interpolation_interval=None,
                         do_scale=False)
    results_RL = test_model(
        model, params, input_type, rng_key, env, ntests, 'RL', loss_func,
        stats_path=stats_path)

    args.envname = 'sdc-v0'
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
    results_LU = test_model(
        model, params, input_type, rng_key, env, ntests, 'LU')

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
    results_min = test_model(
        model, params, input_type, rng_key, env, ntests, 'MIN')

    if args.extensive_tests:
        env = utils.make_env(
            args,
            num_envs=num_test_envs,
            prec='zeros',
            seed=seed,
            lambda_real_interpolation_interval=None,
            do_scale=False,
        )
        results_zeros = test_model(
            model, params, input_type, rng_key, env, ntests, 'zeros')

        env = utils.make_env(
            args,
            num_envs=num_test_envs,
            prec='EE',
            seed=seed,
            lambda_real_interpolation_interval=None,
            do_scale=False,
        )
        results_EE = test_model(
            model, params, input_type, rng_key, env, ntests, 'EE')

    duration = time.perf_counter() - start_time
    print(f'Testing took {duration} seconds.')
    args.envname = old_envname

    # Plot all three iteration counts over the lambda values
    plt.xlabel('re(λ)')
    plt.ylabel('iterations')

    plot_results(results_RL, color='b', label='RL')
    plot_results(results_LU, color='r', label='LU')
    plot_results(results_min, color='g', label='MIN')
    if args.extensive_tests:
        plot_results(results_zeros, color='orange', label='zeros')
        plot_results(results_EE, color='pink', label='EE')

    plt.legend()

    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
    plt.show()


def get_cp_name(args, script_start):
    lam_re_interval = '_'.join(map(str, args.lambda_real_interval))
    lam_im_interval = '_'.join(map(str, args.lambda_imag_interval))
    u_re_interval = '_'.join(map(str, args.u_real_interval))
    u_im_interval = '_'.join(map(str, args.u_imag_interval))
    return (
        f'dp_model_prec_{args.prec_type}_input_{args.input_type}_'
        f'lossf_{args.loss_type}_M_{args.M}_'
        f'lambda_re_{lam_re_interval}_im_{lam_im_interval}_'
        f'u_re_{u_re_interval}_im_{u_im_interval}_'
        f'loss_{{}}_{script_start}.npy'
    )


def main():
    script_start = str(datetime.datetime.now()
                       ).replace(':', '-').replace(' ', 'T')
    args = parse_args()
    if args.loss_type == 'spectral_radius':
        # Eigenvalue decomposition for our case currently not implemented
        # on GPU.
        # See https://github.com/google/jax/issues/1259
        jax.config.update('jax_platform_name', 'cpu')
    jax.config.update('jax_enable_x64', args.float64)
    utils.setup(True)

    eval_seed = args.seed
    if eval_seed is not None:
        eval_seed += 1

    rng_key = jax.random.PRNGKey(args.seed)
    rng_key, subkey = jax.random.split(rng_key)

    dataloader = DataGenerator(
        args.M,
        args.dt,
        args.lambda_real_interval,
        args.lambda_imag_interval,
        args.u_real_interval,
        args.u_imag_interval,
        args.batch_size,
        args.input_type,
        args.loss_type,
        subkey,
    )

    input_shape = (get_input_size(args.M, args.input_type),)
    model_init, model, model_arch = build_model(args, train=True)

    rng_key, subkey = jax.random.split(rng_key)
    _, params = model_init(subkey, input_shape)
    if args.model_path is not None:
        params, model_arch, old_steps = load_model(args.model_path)
        _, model = _from_model_arch(model_arch, train=True)
        params = list(params)
        check_output_size(args, model, params)
    else:
        old_steps = 0

    opt_state, opt_update, opt_get_params = build_opt(
        args, params, old_steps)
    loss_func = get_loss_func(args.M, args.dt, args.prec_type, args.loss_type)

    max_grad_norm = 0.5
    # grad_clipping_schedule = optimizers.polynomial_decay(
    #     max_grad_norm, 15000, 1e-7, 1.0)

    # Better to avoid this; always worsened results.
    weight_decay_factor = 0.0
    weight_decay_schedule = optimizers.polynomial_decay(
        weight_decay_factor, 15000, 0.0, 2.0)

    # @jax.jit
    def loss(params, lams, i, input_data, loss_data, rng_key):
        if args.input_type == 'lambda':
            outputs = model(params, lams, rng=rng_key)
        elif args.input_type == 'residual':
            (residuals,) = input_data
            outputs = model(params, residuals, rng=rng_key)
        elif args.input_type == 'lambda_u':
            (us,) = input_data
            outputs = model(params, jnp.hstack((lams, us)), rng=rng_key)
        else:
            raise UnknownInputTypeError()

        if args.loss_type == 'spectral_radius':
            loss_ = loss_func(lams, outputs)
            aux_data = ()
        elif args.loss_type == 'residual':
            Cs, us, residuals = loss_data
            loss_, us, residuals = loss_func(lams, outputs, Cs, us, residuals)
            aux_data = (us, residuals)
        else:
            raise UnknownLossTypeError()

        weight_penalty = optimizers.l2_norm(params)
        weight_decay_factor = weight_decay_schedule(i)
        loss_ = loss_ + weight_decay_factor * weight_penalty
        return loss_, aux_data

    @jax.jit
    def update(i, opt_state, lams, input_data, loss_data, rng_key):
        params = opt_get_params(opt_state)
        rng_key, subkey = jax.random.split(rng_key)
        (loss_, aux_data), gradient = jax.value_and_grad(
            loss,
            has_aux=True,
        )(params, lams, i.astype(float), input_data, loss_data, subkey)

        # print(gradient)
        # max_grad_norm = grad_clipping_schedule(i)
        gradient = optimizers.clip_grads(gradient, max_grad_norm)
        opt_state = opt_update(i, gradient, opt_state)
        return loss_, opt_state, aux_data, rng_key

    steps = int(args.steps)
    steps_num_digits = len(str(steps))

    cp_name = get_cp_name(args, script_start)
    best_cp_name = 'best_' + cp_name
    best_cp_path = None

    best_loss = np.inf
    last_losses = np.zeros(100)
    start_time = time.perf_counter()
    for (step, (lams, input_data, loss_data)) in enumerate(dataloader):
        loss_, opt_state, aux_data, rng_key = update(
            jnp.array(step + old_steps),
            opt_state,
            lams,
            input_data,
            loss_data,
            rng_key,
        )

        last_losses[step % len(last_losses)] = loss_.item()

        if step % 100 == 0:
            mean_loss = jnp.mean(last_losses[:step + 1]).item()
            if mean_loss < best_loss and steps > 0:
                best_loss = mean_loss
                prev_best_cp_path = best_cp_path
                best_cp_path = Path(best_cp_name.format(mean_loss))
                save_model(
                    best_cp_path,
                    opt_get_params(opt_state),
                    model_arch,
                    steps + old_steps,
                )
                delete_model(prev_best_cp_path)

            print(f'[{step:>{steps_num_digits}d}/{steps}] '
                  f'mean_loss: {mean_loss:.10f}')

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
        # Load best checkpoint for testing
        load_cp_path = best_cp_path
    elif args.model_path is not None:
        load_cp_path = args.model_path
    else:
        load_cp_path = None

    if load_cp_path is not None:
        params, _, _ = load_model(load_cp_path)
        print(f'Testing model at {load_cp_path}.')
        params = list(params)
        check_output_size(args, model, params)
    fig_path = Path(f'dp_results_{script_start}.pdf')
    run_tests(
        model,
        params,
        args,
        seed=eval_seed,
        fig_path=fig_path,
        loss_func=loss,
        # stats_path=args.model_path + '.stats.npz',
    )


if __name__ == '__main__':
    main()
