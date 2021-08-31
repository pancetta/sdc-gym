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
        (b) more than `self.max_iters` iterations are done (not converged),
        (c) diverged.
    """
    action_space = None
    observation_space = None
    num_envs = 1
    max_iters = 50

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
            reward_iteration_only=None,
            reward_strategy='iteration_only',
            collect_states=False,
            use_doubles=True,
            do_scale=True,
            free_action_space=False,
    ):

        self.np_random = None
        self.niter = None
        self.restol = restol
        self.dt = dt
        self.M = M
        self.coll = CollGaussRadau_Right(M, 0, 1)
        self.Q = self.coll.Qmat[1:, 1:]
        self.C = None
        self.lam = None
        self.u0 = np.ones(M, dtype=np.complex128)
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
        if reward_iteration_only is None:
            self.reward_strategy = reward_strategy.lower()
        elif reward_iteration_only:
            self.reward_strategy = 'iteration_only'
        else:
            self.reward_strategy = 'residual_change'
        self.collect_states = collect_states
        self.do_scale = do_scale

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
            shape=(M * 2, self.max_iters) if collect_states else (2, M),
            dtype=np.complex128,
        )
        if free_action_space:
            self.action_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(M,),
                dtype=np.complex128 if use_doubles else np.complex64,
            )
        else:
            # I read somewhere that the actions should be scaled to [-1,1],
            # values will be real.
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(M,),
                dtype=np.float64 if use_doubles else np.float32,
            )

        self.seed(seed)
        self.state = None
        if collect_states:
            self.old_states = np.zeros((M * 2, self.max_iters),
                                       dtype=np.complex128)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def set_num_episodes(self, num_episodes):
        self.num_episodes = num_episodes

    def _scale_action(self, action):
        # I read somewhere that the actions should be scaled to [-1,1],
        # scale it back to [0,1] here...
        if self.do_scale:
            scaled_action = np.interp(action, (-1, 1), (0, 1))
        else:
            scaled_action = action
        return scaled_action

    def _get_prec(self, scaled_action):
        """Return a preconditioner based on the `scaled_action`."""
        # Decide which preconditioner to use
        # (depending on self.prec string)... not very elegant
        if self.prec is None:
            Qdmat = np.zeros_like(self.Q, dtype=self.action_space.dtype)
            np.fill_diagonal(Qdmat, scaled_action)
        elif self.prec.upper() == 'LU':
            QT = self.Q.T
            [_, _, U] = scipy.linalg.lu(QT, overwrite_a=True)
            Qdmat = U.T
        elif self.prec.lower() == 'min':
            Qdmat = np.zeros_like(self.Q)
            if self.M == 7:
                x = [
                    0.15223871397682717,
                    0.12625448001038536,
                    0.08210714764924298,
                    0.03994434742760019,
                    0.1052662547386142,
                    0.14075805578834127,
                    0.15636085758812895
                ]
            elif self.M == 5:
                x = [
                    0.2818591930905709,
                    0.2011358490453793,
                    0.06274536689514164,
                    0.11790265267514095,
                    0.1571629578515223,
                ]
            elif self.M == 4:
                x = [
                    0.3198786751412953,
                    0.08887606314792469,
                    0.1812366328324738,
                    0.23273925017954,
                ]
            elif self.M == 3:
                x = [
                    0.3203856825077055,
                    0.1399680686269595,
                    0.3716708461097372,
                ]
            else:
                # if M is some other number, take zeros. This won't work
                # well, but does not raise an error
                x = np.zeros(self.M)
            np.fill_diagonal(Qdmat, x)
        elif self.prec.upper() == 'EE':
            Qdmat = np.zeros_like(self.Q)
            for m in range(self.M):
                Qdmat[m, 0:m] = self.coll.delta_m[1:m + 1]
        elif self.prec.lower() == 'zeros':
            Qdmat = np.zeros_like(self.Q)
        else:
            raise NotImplementedError()
        return Qdmat

    def _compute_pinv(self, scaled_action):
        # Get Q_delta, based on self.prec (and/or scaled_action)
        Qdmat = self._get_prec(scaled_action=scaled_action)

        # Compute the inverse of P
        Pinv = np.linalg.inv(
            np.eye(self.M) - self.lam * self.dt * Qdmat,
        )
        return Pinv

    def _compute_residual(self, u):
        return self.u0 - self.C @ u

    def _inf_norm(self, v):
        return np.linalg.norm(v, np.inf)

    def step(self, action):
        u, old_residual = self.state

        scaled_action = self._scale_action(action)

        Pinv = self._compute_pinv(scaled_action)
        norm_res_old = self._inf_norm(old_residual)

        # Re-use what we already have
        residual = old_residual

        done = False
        err = False
        self.niter = 0
        # Start the loop
        while not done and not self.niter >= self.max_iters:
            self.niter += 1

            # This is the iteration (yes, there is a typo in the slides,
            # this one is correct!)
            u += Pinv @ residual
            # Compute the residual and its norm
            residual = self._compute_residual(u)
            norm_res = self._inf_norm(residual)
            # stop if something goes wrong
            err = np.isnan(norm_res) or np.isinf(norm_res)
            # so far this seems to be the best setup:
            #   - stop if residual gets larger than the initial one
            #     (not needed, but faster)
            #   - reward = -self.max_iters, if this happens (crucial!)
            if self.collect_states and self.niter < self.max_iters:
                self.old_states[:, self.niter] = np.concatenate((u, residual))
            err = err or norm_res > norm_res_old * 100
            if err:
                reward = -self.step_penalty * (self.max_iters + 1)
                # reward = -(self.max_iters + 1)
                break
            # check for convergence
            done = norm_res < self.restol

        if not err:
            reward = self.reward_func(
                self.initial_residual,
                residual,
                done,
                self.niter,
                scaled_action,
                Pinv,
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

    def _reset_vars(self):
        self.num_episodes += 1
        self.niter = 0

        # self.rewards.append(self.episode_rewards)
        # self.episode_rewards = []

    def _generate_lambda(self):
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

    def _compute_system_matrix(self):
        # Compute the system matrix
        self.C = np.eye(self.M) - self.lam * self.dt * self.Q

    def _compute_initial_state(self):
        self._generate_lambda()
        self._compute_system_matrix()

        # Initial guess u^0 and initial residual for the state
        u = np.ones(self.M, dtype=np.complex128)
        residual = self._compute_residual(u)
        self.initial_residual = residual
        return (u, residual)

    def reset(self):
        self._reset_vars()
        (u, residual) = self._compute_initial_state()
        self.state = (u, residual)

        if self.collect_states:
            # Try if this works instead of the line below it.
            # I didn't use it for safety, but it's a bit faster.
            self.old_states[:, 0] = np.concatenate(self.state)
            self.old_states[:, 1:] = 0
            # self.old_states = np.zeros((self.M * 2, self.max_iters),
            #                            dtype=np.complex128)

        if self.collect_states:
            return self.old_states
        else:
            return self.state

    def _reward_iteration_only(self, steps):
        return -steps * self.step_penalty

    def _reward_residual_change(self, old_residual, residual, steps):
        # reward = -self.initial_residual / 100
        # reward = -self._inf_norm(residual)
        reward = abs(
            (math.log(self._inf_norm(old_residual * self.norm_factor))
             - math.log(self._inf_norm(residual * self.norm_factor)))
            / (math.log(self._inf_norm(self.initial_residual
                                       * self.norm_factor))
               - math.log(self.restol * self.norm_factor)),
        )
        reward *= self.residual_weight
        # jede der `self.max_iters` Iterationen wird bestraft
        reward -= steps * self.step_penalty
        return reward

    def _reward_gauss_kernel(self, residual, reached_convergence, steps):
        self.gauss_facts = [1]
        self.gauss_invs = [1 / self.restol]
        norm_res = self._inf_norm(residual)
        gauss_dist = sum(
            (gauss_fact
             * np.exp(-(norm_res * gauss_inv)**2 / 2))
            for (gauss_fact, gauss_inv) in zip(self.gauss_facts,
                                               self.gauss_invs)
        )
        if reached_convergence:
            extra_fact = (self.max_iters + 1 - steps)**2 * 10
        else:
            extra_fact = 1

        reward = gauss_dist * extra_fact
        return reward

    def _reward_fast_convergence(self, residual, reached_convergence, steps):
        norm_res = self._inf_norm(residual)
        if reached_convergence:
            extra_fact = (self.max_iters + 1 - steps)**2 * 10
        else:
            extra_fact = 1

        if norm_res == 0:
            # Smallest double exponent 1e-323's -log is about 744
            reward = 1000
        else:
            reward = -math.log(norm_res)
        reward *= extra_fact
        return reward

    def _reward_smooth_fast_convergence(
            self, residual, reached_convergence, steps):
        norm_res = self._inf_norm(residual)
        if reached_convergence:
            extra_fact = (self.max_iters + 1 - steps)**2 * 10
        else:
            extra_fact = 1

        if norm_res == 0:
            # Smallest double exponent 1e-323's -log is about 744
            reward = 1000
        else:
            reward = -math.log(norm_res)
        if reward > 1:
            reward = 1 + math.log(reward)
        reward *= extra_fact
        return reward

    def _reward_smoother_fast_convergence(
            self, residual, reached_convergence, steps):
        norm_res = self._inf_norm(residual)
        if reached_convergence:
            extra_fact = (self.max_iters + 1 - steps)**2 * 10
        else:
            extra_fact = 1

        if norm_res == 0:
            # Smallest double exponent 1e-323's -log is about 744
            reward = 1000
        else:
            reward = -math.log(norm_res)
        reward *= extra_fact
        if reward > 1:
            reward = 1 + math.log(reward)
        return reward

    def _reward_spectral_radius(self, scaled_action, Pinv):
        Qdmat = self._get_prec(scaled_action)
        mulpinv = Pinv.dot(self.Q - Qdmat)
        eigvals = np.linalg.eigvals(self.lam * self.dt * mulpinv)
        return max(abs(eigvals))

    def reward_func(
            self,
            old_residual,
            residual,
            reached_convergence,
            steps,
            scaled_action,
            Pinv,
    ):
        """Return the reward obtained with the `old_residual` with the
        new `residual`.
        `reached_convergence` indicates whether convergence was reached.
        `steps` indicates how many time steps to penalize.
        `scaled_action` is the action taken.
        `Pinv` is the iteration matrix.
        """
        if self.reward_strategy == 'iteration_only':
            return self._reward_iteration_only(steps)
        elif self.reward_strategy == 'residual_change':
            return self._reward_residual_change(old_residual, residual, steps)
        elif self.reward_strategy == 'gauss_kernel':
            return self._reward_gauss_kernel(
                residual, reached_convergence, steps)
        elif self.reward_strategy == 'fast_convergence':
            return self._reward_fast_convergence(
                residual, reached_convergence, steps)
        elif self.reward_strategy == 'smooth_fast_convergence':
            return self._reward_smooth_fast_convergence(
                residual, reached_convergence, steps)
        elif self.reward_strategy == 'smoother_fast_convergence':
            return self._reward_smoother_fast_convergence(
                residual, reached_convergence, steps)
        elif self.reward_strategy == 'spectral_radius':
            return self._reward_spectral_radius(scaled_action, Pinv)

        raise NotImplementedError(
            f'unknown reward strategy {self.reward_strategy}')

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
        (b) more than `self.max_iters` iterations are done (not converged),
        (c) diverged.
    """

    def step(self, action):

        u, old_residual = self.state

        scaled_action = self._scale_action(action)

        Pinv = self._compute_pinv(scaled_action)

        # Do the iteration (note that we already have the residual)
        u += Pinv @ old_residual

        # The new residual and its norm
        residual = self._compute_residual(u)

        norm_res = self._inf_norm(residual)
        norm_res_old = self._inf_norm(old_residual)

        self.niter += 1

        # Check if something went wrong
        err = np.isnan(norm_res) or np.isinf(norm_res)
        # so far this seems to be the best setup:
        #   - stop if residual gets larger than the initial one
        #     (not needed, but faster)
        #   - reward = -self.max_iters, if this happens (crucial!)
        err = err or norm_res > norm_res_old * 100
        # Stop iterating when converged
        done = norm_res < self.restol

        if not err:
            reward = self.reward_func(
                old_residual,
                residual,
                done,
                self.niter,
                scaled_action,
                Pinv,
            )
            # print(reward)
        else:
            # return overall reward of -(self.max_iters + 1)
            # (slightly worse than -self.max_iters in the
            # "not converged" scenario)
            # reward = -self.step_penalty * ((self.max_iters + 2) - self.niter)
            reward = -self.step_penalty * (self.max_iters + 1)
            # reward = -(self.max_iters + 1) + self.niter
            # reward = -self.max_iters + self.niter
        # Stop iterating when iteration count is too high or when
        # something bad happened
        done = done or self.niter >= self.max_iters or err
        # self.episode_rewards.append(reward)
        # self.norm_resids.append(norm_res)

        self.state = (u, residual)
        if self.collect_states and self.niter < self.max_iters:
            self.old_states[:, self.niter] = np.concatenate(self.state)

        info = {
            'residual': norm_res,
            'niter': self.niter,
            'lam': self.lam,
        }
        if self.collect_states:
            return (self.old_states, reward, done, info)
        else:
            return (self.state, reward, done, info)
