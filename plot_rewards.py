import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from pySDC.implementations.collocation_classes.gauss_radau_right import \
    CollGaussRadau_Right


def _norm(x):
    return np.linalg.norm(x, np.inf)


def calc_default_initial_residual(M, lam, dt):
    coll = CollGaussRadau_Right(M, 0, 1)
    Q = coll.Qmat[1:, 1:]

    return ((np.ones(coll.num_nodes, dtype=np.complex128))
            - (np.eye(coll.num_nodes)
               - lam * dt * Q)
            @ (np.ones(coll.num_nodes, dtype=np.complex128)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=3)
    parser.add_argument('--lam', type=float, default=-1.0)
    parser.add_argument('--restol', type=float, default=1E-10)
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--norm_initial_residual', type=float, default=None)
    parser.add_argument('--norm_old_residual', type=float, default=1E-10)
    parser.add_argument('--norm_factor', type=float, default=1.0)
    parser.add_argument('--residual_weight', type=float, default=1.0)
    parser.add_argument('--step_penalty', type=float, default=0.0)
    return parser.parse_args()


def reward_func(
        norm_residual,
        norm_old_residual,
        norm_initial_residual,
        restol,
        norm_factor,
        residual_weight,
        step_penalty,
):
    # summe ueber alle Iterationen liegt auf [0,1] (hoffentlich)
    reward = abs((math.log(norm_old_residual * norm_factor)
                  - math.log(norm_residual * norm_factor))
                 / (math.log(norm_initial_residual * norm_factor)
                    - math.log(restol * norm_factor)))
    reward *= residual_weight
    reward -= step_penalty
    return reward


def main():
    args = parse_args()
    norm_initial_residual = args.norm_initial_residual
    if norm_initial_residual is None:
        initial_residual = calc_default_initial_residual(
            args.M,
            args.lam,
            args.dt,
        )
        norm_initial_residual = _norm(initial_residual)

    # Create a vectorized of the reward function for efficiency
    def curr_reward_func(norm_residual):
        return np.vectorize(lambda norm_residual: reward_func(
            norm_residual,
            args.norm_old_residual,
            norm_initial_residual,
            args.restol,
            args.norm_factor,
            args.residual_weight,
            args.step_penalty,
        ))(norm_residual)

    norm_residuals = np.linspace(np.nextafter(0.0, 1), 1.5, 10000)
    rewards = curr_reward_func(norm_residuals)

    plt.title(
        f'norm(old_residual) = {args.norm_old_residual},\n'
        f'norm(initial_residual) = {norm_initial_residual},\n'
        f'λ = {args.lam}, M = {args.M}, dt = {args.dt},\n'
        f'norm_factor = {args.norm_factor},\n'
        f'residual_weight = {args.residual_weight},\n'
        f'step_penalty = {args.step_penalty},\n'
        f'restol={args.restol}',
    )
    # Make title visible
    plt.subplots_adjust(top=0.65)
    plt.xlabel('norm(residual)')
    plt.ylabel('reward')

    plt.plot(norm_residuals, rewards, label='rewards')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
