import numpy as np


class logistics_equation():

    def __init__(self, problem_params):
        self.params = problem_params

    def eval_f(self, u):
        return self.params['lam'] * u * (1 - u)

    def solve_system(self, rhs, dt, u_guess):
        d = (1 - dt * self.params['lam']) ** 2 + 4 * dt * self.params['lam'] * rhs
        u = (- (1 - dt * self.params['lam']) + np.sqrt(d)) / (2 * dt * self.params['lam'])
        return u

    def u_exact(self, t):
        return self.params['u0'] * np.exp(self.params['lam'] * t) / \
               (1 - self.params['u0'] + self.params['u0'] * np.exp(self.params['lam'] * t))

