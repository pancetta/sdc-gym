import argparse
import datetime
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from pySDC.implementations.collocation_classes.gauss_radau_right import \
    CollGaussRadau_Right

from rl_playground import _store_test_stats, plot_results
import utils


class DataGenerator(th.utils.data.IterableDataset):
    def __init__(
            self,
            M,
            lambda_real_interval,
            lambda_imag_interval,
    ):
        super().__init__()
        self.M = M
        self.coll = CollGaussRadau_Right(M, 0, 1)
        self.Q = self.coll.Qmat[1:, 1:]
        self.lambda_real_interval = lambda_real_interval
        self.lambda_imag_interval = lambda_imag_interval

    def __iter__(self):
        lam_real_low = self.lambda_real_interval[0]
        lam_real_high = self.lambda_real_interval[1]
        lam_real_interval_size = lam_real_high - lam_real_low

        lam_imag_low = self.lambda_imag_interval[0]
        lam_imag_high = self.lambda_imag_interval[1]
        lam_imag_interval_size = lam_imag_high - lam_imag_low

        while True:
            lam = (
                1 * th.rand((1,)) * lam_real_interval_size + lam_real_low
                + 1j * th.rand((1,)) * lam_imag_interval_size + lam_imag_low
            )
            raise NotImplementedError(
                '`yield` (not `return`!) a tuple like '
                '`(lam, diag_vector_with_min_spectral_radius)` here'
            )
            # yield lam, th.rand(self.M)


class PreconditionerPredictor(th.nn.Module):
    def __init__(self, M):
        super().__init__()
        self.layers = th.nn.Sequential(
            th.nn.Linear(1, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, M),
        )

    def forward(self, inputs):
        return self.layers(inputs)


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
    args.collect_states = False
    args.model_class = 'PPG'
    return args


def load_model(path, model, opt, device):
    cp = th.load(path, map_location=device)
    model.load_state_dict(cp['model_state_dict'])
    model.to(device)
    if opt is not None:
        opt.load_state_dict(cp['opt_state_dict'])
    return model, opt, cp.get('steps', 0)


def save_model(path, model, opt, steps):
    th.save({
        'steps': steps,
        'model_state_dict': model.state_dict(),
        'opt_state_dict': opt.state_dict(),
    }, path)


def test_model(model, device, env, ntests, name, stats_path=None):
    """Test the `model` on `device` in the Gym `env` `ntests` times.
    `name` is the name for the test run for logging purposes.
    `stats_path` is an optional path where to save statistics about
    the test.
    """
    model.eval()
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

    with th.no_grad():
        for i in range(ntests):
            env.reset()
            obs = th.tensor([env_.lam for env_ in env.envs]
                            ).float().unsqueeze(0).to(device)
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
                    action = model(obs)
                    action = [action.cpu().numpy()]

                _, rewards, done, info = env.step(action)
                obs = th.tensor([env_.lam for env_ in env.envs]
                                ).float().unsqueeze(0).to(device)

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

    model.train()
    return results


def run_tests(model, device, args, seed=None, fig_path=None, stats_path=None):
    """Run tests for the given `model` and `args`, using `seed` as the
    random seed.

    `fig_path` is an optional path to store result plots at.
    `stats_path` is an optional path where to save statistics about the
    reinforcement learning test.
    """
    # Load the trained agent for testing
    if isinstance(model, (Path, str)):
        path = model
        model = PreconditionerPredictor(args.M).to(device)
        model, _, _ = load_model(path, model, None, device)

    num_test_envs = 1
    ntests = int(args.tests)
    ntests = utils.maybe_fix_ntests(ntests, num_test_envs)

    start_time = time.perf_counter()
    # Test the trained model.
    env = utils.make_env(args, num_envs=num_test_envs, seed=seed,
                         lambda_real_interpolation_interval=None)
    results_RL = test_model(
        model, device, env, ntests, 'RL', stats_path=stats_path)

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
    results_LU = test_model(model, device, env, ntests, 'LU')

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
    results_min = test_model(model, device, env, ntests, 'MIN')
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

    th.manual_seed(args.seed)
    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

    dataloader = th.utils.data.DataLoader(
        DataGenerator(
            args.M,
            args.lambda_real_interval,
            args.lambda_imag_interval,
        ),
        batch_size=args.batch_size,
    )

    model = PreconditionerPredictor(args.M).to(device)
    loss_func = th.nn.MSELoss()
    opt = th.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.model_path is not None:
        model, opt, old_steps = load_model(args.model_path, model, opt, device)
    model.train()
    steps = int(args.steps)
    steps_num_digits = len(str(steps))

    for (step, (lams, min_diags)) in enumerate(dataloader):
        lams = lams.float().to(device)
        min_diags = min_diags.to(device)

        diags = model(lams)
        loss = loss_func(diags, min_diags)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f'[{step:>{steps_num_digits}d}/{steps}] '
                  f'loss: {loss.item():.5f}')

        if step >= steps:
            break

    if steps > 0:
        cp_path = Path(f'sl_model_{script_start}.pt')
        save_model(cp_path, model, opt, steps + old_steps)
    fig_path = Path(f'sl_results_{script_start}.pdf')
    run_tests(model, device, args, seed=eval_seed, fig_path=fig_path)


if __name__ == '__main__':
    main()
