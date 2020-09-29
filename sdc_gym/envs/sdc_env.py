import gym
from gym import spaces
from gym.utils import seeding

from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right

import numpy as np


class SDC_Full_Env(gym.Env):

    action_space = None
    observation_space = None
    num_envs = 1

    def __init__(self, M=None, dt=None, restol = None):

        self.np_random = None
        self.niter = None
        self.restol = restol
        self.dt = dt
        self.coll = CollGaussRadau_Right(M, 0, 1)
        self.Q = self.coll.Qmat[1:, 1:]
        self.C = None
        self.lam = None
        self.u0 = np.ones(self.coll.num_nodes)
        self.old_res = None

        self.observation_space = spaces.Box(low=-np.finfo(np.float64).max, high=np.finfo(np.float64).max, shape=(2, M), dtype=np.float64)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(M,), dtype=np.float64)

        self.seed()
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        u, old_res = self.state

        scaled_action = np.interp(action, (-1, 1), (0, 1))

        Qdmat = np.zeros_like(self.Q)
        np.fill_diagonal(Qdmat, scaled_action)
        Pinv = np.linalg.inv(np.eye(self.coll.num_nodes) - self.lam * self.dt * Qdmat)
        residual = self.u0 - self.C @ u
        norm_res_old = np.linalg.norm(residual, np.inf)
        done = False
        err = False
        reward = 0
        self.niter = 0
        while not done and not self.niter >= 50 and not err:
            self.niter += 1
            u += Pinv @ (self.u0 - self.C @ u)

            residual = self.u0 - self.C @ u
            norm_res = np.linalg.norm(residual, np.inf)
            err = norm_res > norm_res_old * 10
            if err:
                reward = -50
                break
            done = norm_res < self.restol
            reward -= 1
            # norm_res_old = norm_res
        done = True

        self.state = (u, residual)

        return self.state, reward, done, {'residual': norm_res, 'niter': self.niter, 'lam': self.lam}

    def reset(self):
        self.lam = 1 * np.random.uniform(low=-100.0, high=0.0) + 0j * np.random.uniform(low=0.0, high=1.0)
        # self.lam = -3.9
        self.C = np.eye(self.coll.num_nodes) - self.lam * self.dt * self.Q
        u = np.ones(self.coll.num_nodes, dtype=np.complex128)
        residual = self.u0 - self.C @ u
        self.niter = 0

        self.state = (u, residual)
        return self.state


class SDC_Step_Env(SDC_Full_Env):

    def step(self, action):

        u, residual = self.state

        Qdmat = np.zeros_like(self.Q)
        np.fill_diagonal(Qdmat, action)
        Pinv = np.linalg.inv(np.eye(self.coll.num_nodes) - self.lam * self.dt * Qdmat)

        self.niter += 1
        u += Pinv @ residual
        residual = self.u0 - self.C @ u
        norm_res = np.linalg.norm(residual, np.inf)

        err = norm_res > 1E03
        done = norm_res < self.restol or self.niter >= 50 or err

        if not err:
            reward = -1
        else:
            reward = -50 + self.niter

        self.state = (u, residual)

        return self.state, reward, done, {'residual': norm_res, 'niter': self.niter, 'lam': self.lam}
