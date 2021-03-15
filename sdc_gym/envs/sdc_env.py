import itertools
import math

import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
import numpy as np
import scipy

from pySDC.implementations.collocation_classes.gauss_radau_right import \
    CollGaussRadau_Right


class SDC_Full_Env(gym.Env):
    """This environment implements a full iteration of SDC, i.e. for
    each step we iterate until
        (a) convergence is reached (residual norm is below restol),
        (b) more than 50 iterations are done (not converged),
        (c) diverged.
    """
    action_space = None
    observation_space = None
    num_envs = 1

    def __init__(
            self,
            M=None,
            dt=None,
            restol=None,
            prec=None,
            seed=None,
            lambda_real_interval=[-100, 0],
            lambda_imag_interval=[0, 0],
            lambda_real_interpolation_interval=None,
            norm_factor=1,
            residual_weight=0.5,
            step_penalty=0.1,
            reward_iteration_only=True,
            collect_states=False,
    ):

        self.np_random = None
        self.niter = None
        self.restol = restol
        self.dt = dt
        self.coll = CollGaussRadau_Right(M, 0, 1)
        self.Q = self.coll.Qmat[1:, 1:]
        self.C = None
        self.lam = None
        self.u0 = np.ones(self.coll.num_nodes, dtype=np.complex128)
        self.old_res = None
        self.prec = prec
        self.initial_residual = None

        self.lambda_real_interval = lambda_real_interval
        self.lambda_real_interval_reversed = list(
            reversed(lambda_real_interval))
        self.lambda_imag_interval = lambda_imag_interval
        self.lambda_real_interpolation_interval = \
            lambda_real_interpolation_interval

        self.norm_factor = norm_factor
        self.residual_weight = residual_weight
        self.step_penalty = step_penalty
        self.reward_iteration_only = reward_iteration_only
        self.collect_states = collect_states

        self.num_episodes = 0
        # self.rewards = []
        # self.episode_rewards = []
        # self.norm_resids = []
        # Setting the spaces: both are continuous, observation box
        # artificially bounded by some large numbers
        # note that because lambda can be complex, U can be complex,
        # i.e. the observation space should be complex
        self.observation_space = spaces.Box(
            low=-1E10,
            high=+1E10,
            shape=(M * 2, 50) if collect_states else (2, M),
            dtype=np.complex128,
        )
        # I read somewhere that the actions should be scaled to [-1,1],
        # values will be real.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(M,),
            dtype=np.float64,
        )

        self.seed(seed)
        self.state = None
        if collect_states:
            self.old_states = np.zeros((M * 2, 50), dtype=np.complex128)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def _get_prec(self, scaled_action, M):
        """Return a preconditioner based on the `scaled_action`.
        `M` is the problem size.
        """
        # Decide which preconditioner to use
        # (depending on self.prec string)... not very elegant
        if self.prec is None:
            Qdmat = np.zeros_like(self.Q)
            np.fill_diagonal(Qdmat, scaled_action)
        elif self.prec.upper() == 'LU':
            QT = self.Q.T
            [_, _, U] = scipy.linalg.lu(QT, overwrite_a=True)
            Qdmat = U.T
        elif self.prec.lower() == 'min':
            Qdmat = np.zeros_like(self.Q)
            if M == 7:
                x = [
                    0.15223871397682717,
                    0.12625448001038536,
                    0.08210714764924298,
                    0.03994434742760019,
                    0.1052662547386142,
                    0.14075805578834127,
                    0.15636085758812895
                ]
            elif M == 5:
                x = [
                    0.2818591930905709,
                    0.2011358490453793,
                    0.06274536689514164,
                    0.11790265267514095,
                    0.1571629578515223,
                ]
            elif M == 4:
                x = [
                    0.3198786751412953,
                    0.08887606314792469,
                    0.1812366328324738,
                    0.23273925017954,
                ]
            elif M == 3:
                x = [
                    0.3203856825077055,
                    0.1399680686269595,
                    0.3716708461097372,
                ]
            else:
                # if M is some other number, take zeros. This won't work
                # well, but does not raise an error
                x = np.zeros(M)
            np.fill_diagonal(Qdmat, x)
        else:
            raise NotImplementedError()
        return Qdmat

    def step(self, action):

        u, old_residual = self.state

        # I read somewhere that the actions should be scaled to [-1,1],
        # scale it back to [0,1] here...
        scaled_action = np.interp(action, (-1, 1), (0, 1))

        # Get Q_delta, based on self.prec (and/or scaled_action)
        Qdmat = self._get_prec(scaled_action=scaled_action, M=u.size)

        # Precompute the inverse of P
        Pinv = np.linalg.inv(
            np.eye(self.coll.num_nodes) - self.lam * self.dt * Qdmat,
        )
        norm_res_old = np.linalg.norm(old_residual, np.inf)

        done = False
        err = False
        self.niter = 0
        # Start the loop
        while not done and not self.niter >= 50:
            self.niter += 1

            # This is the iteration (yes, there is a typo in the slides,
            # this one is correct!)
            u += Pinv @ (self.u0 - self.C @ u)
            # Compute the residual and its norm
            residual = self.u0 - self.C @ u
            norm_res = np.linalg.norm(residual, np.inf)
            # stop if something goes wrong
            err = np.isnan(norm_res) or np.isinf(norm_res)
            # so far this seems to be the best setup:
            #   - stop if residual gets larger than the initial one
            #     (not needed, but faster)
            #   - reward = -50, if this happens (crucial!)
            if self.collect_states and self.niter < 50:
                self.old_states[:, self.niter] = np.concatenate((u, residual))
            err = err or norm_res > norm_res_old * 100
            if err:
                reward = -self.step_penalty * 51
                # reward = -51
                break
            # check for convergence
            done = norm_res < self.restol

        if not err:
            reward = self.reward_func(
                self.initial_residual,
                residual,
                self.niter,
            )

        done = True
        # self.episode_rewards.append(reward)
        # self.norm_resids.append(norm_res)

        self.state = (u, residual)

        info = {
            'residual': norm_res,
            'niter': self.niter,
            'lam': self.lam,
        }
        if self.collect_states:
            return (self.old_states, reward, done, info)
        else:
            return (self.state, reward, done, info)

    def reset(self):
        self.num_episodes += 1
        # Draw a lambda (here: negative real for starters)
        # The number of episodes is always smaller than the number of
        # time steps, keep that in mind for the interpolation
        # hyperparameters.
        if self.lambda_real_interpolation_interval is not None:
            lam_low = np.interp(self.num_episodes,
                                self.lambda_real_interpolation_interval,
                                self.lambda_real_interval_reversed)
        else:
            lam_low = self.lambda_real_interval[0]
        self.lam = (
            1 * self.np_random.uniform(
                low=lam_low,
                high=self.lambda_real_interval[1])
            + 1j * self.np_random.uniform(
                low=self.lambda_imag_interval[0],
                high=self.lambda_imag_interval[1])
        )

        # Compute the system matrix
        self.C = np.eye(self.coll.num_nodes) - self.lam * self.dt * self.Q

        # Initial guess u^0 and initial residual for the state
        u = np.ones(self.coll.num_nodes, dtype=np.complex128)
        residual = self.u0 - self.C @ u
        self.initial_residual = residual

        self.state = (u, residual)
        if self.collect_states:
            # Try if this works instead of the line below it.
            # I didn't use it for safety, but it's a bit faster.
            # self.old_states[:] = 0
            self.old_states = np.zeros((u.size * 2, 50), dtype=np.complex128)
            self.old_states[:, 0] = np.concatenate((u, residual))

        # self.rewards.append(self.episode_rewards)
        # self.episode_rewards = []
        self.niter = 0

        if self.collect_states:
            return self.old_states
        else:
            return self.state

    def reward_func(self, old_residual, residual, steps=1):
        """Return the reward obtained with the `old_residual` with the
        new `residual`.
        `steps` indicates how many time steps to penalize.
        """
        if self.reward_iteration_only:
            return -steps * self.step_penalty

        # reward = -self.initial_residual / 100
        # reward = -np.linalg.norm(residual, np.inf)
        reward = abs(
            (math.log(np.linalg.norm(old_residual * self.norm_factor, np.inf))
             - math.log(np.linalg.norm(residual * self.norm_factor, np.inf)))
            / (math.log(np.linalg.norm(self.initial_residual
                                       * self.norm_factor, np.inf))
               - math.log(self.restol * self.norm_factor)),
        )
        reward *= self.residual_weight
        # jede der 50 Iterationen wird bestraft
        reward -= steps * self.step_penalty
        return reward

    def plot_rewards(self):
        plt.xlabel('time')
        plt.ylabel('reward/residual norm')

        all_rewards = [reward for ep in self.rewards for reward in ep]
        plt.plot(
            np.arange(len(all_rewards)),
            all_rewards,
            label='individual rewards',
        )

        episode_lengths = (len(ep) for ep in self.rewards)
        episode_ends = list(itertools.accumulate(episode_lengths))
        episode_rewards = [sum(ep) for ep in self.rewards]
        plt.plot(
            episode_ends,
            episode_rewards,
            label='episodic rewards',
            marker='.',
        )

        max_reward = max(map(abs, all_rewards))
        max_norm_resid = max(self.norm_resids)
        plt.plot(
            np.arange(len(self.norm_resids)),
            [r / max_norm_resid * max_reward for r in self.norm_resids],
            label='residual norm (rescaled)',
        )

        plt.legend()
        plt.savefig('rewards.pdf', bbox_inches='tight')
        plt.show()


class SDC_Step_Env(SDC_Full_Env):
    """This environment implements a single iteration of SDC, i.e.
    for each step we just do one iteration and stop if
        (a) convergence is reached (residual norm is below restol),
        (b) more than 50 iterations are done (not converged),
        (c) diverged.
    """

    def step(self, action):

        u, old_residual = self.state

        # I read somewhere that the actions should be scaled to [-1,1],
        # scale it back to [0,1] here...
        scaled_action = np.interp(action, (-1, 1), (0, 1))

        # Get Q_delta, based on self.prec (and/or scaled_action)
        Qdmat = self._get_prec(scaled_action=scaled_action, M=u.size)

        # Compute the inverse of P
        Pinv = np.linalg.inv(
            np.eye(self.coll.num_nodes) - self.lam * self.dt * Qdmat,
        )

        # Do the iteration (note that we already have the residual)
        u += Pinv @ old_residual

        # The new residual and its norm
        residual = self.u0 - self.C @ u

        norm_res = np.linalg.norm(residual, np.inf)
        norm_res_old = np.linalg.norm(old_residual, np.inf)

        self.niter += 1

        # Check if something went wrong
        err = np.isnan(norm_res) or np.isinf(norm_res)
        # so far this seems to be the best setup:
        #   - stop if residual gets larger than the initial one
        #     (not needed, but faster)
        #   - reward = -50, if this happens (crucial!)
        err = err or norm_res > norm_res_old * 100
        # Stop iterating when converged, when iteration count is
        # too high or when something bad happened
        done = norm_res < self.restol or self.niter >= 50 or err

        if not err:
            reward = self.reward_func(old_residual, residual, self.niter)
            # print(reward)
        else:
            # return overall reward of -51
            # (slightly worse than -50 in the "not converged" scenario)
            # reward = -self.step_penalty * (52 - self.niter)
            reward = -self.step_penalty * 51
            # reward = -51 + self.niter
            # reward = -50 + self.niter

        # self.episode_rewards.append(reward)
        # self.norm_resids.append(norm_res)

        self.state = (u, residual)
        if self.collect_states and self.niter < 50:
            self.old_states[:, self.niter] = np.concatenate((u, residual))

        info = {
            'residual': norm_res,
            'niter': self.niter,
            'lam': self.lam,
        }
        if self.collect_states:
            return (self.old_states, reward, done, info)
        else:
            return (self.state, reward, done, info)
