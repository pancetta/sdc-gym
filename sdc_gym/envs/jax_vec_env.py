import itertools
import math

import gym
from gym import spaces
from gym.utils import seeding
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from pySDC.implementations.collocation_classes.gauss_radau_right import \
    CollGaussRadau_Right

jax.config.update('jax_enable_x64', True)


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
            batch_size=None,
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
    ):

        self.rng_key = None
        self.niter = None
        self.restol = restol
        self.dt = dt
        self.M = M
        self.coll = CollGaussRadau_Right(M, 0, 1)
        self.Q = self.coll.Qmat[1:, 1:]
        self.C = None
        self.lam = None
        self.u0 = jnp.ones(M, dtype=jnp.complex128)
        self.old_res = None
        self.prec = prec
        self.initial_residual = None
        self.batch_size = batch_size

        self.lambda_real_interval = jnp.array(lambda_real_interval)
        self.lambda_real_interval_reversed = list(
            reversed(lambda_real_interval))
        self.lambda_imag_interval = jnp.array(lambda_imag_interval)
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
            shape=((batch_size, M * 2, self.max_iters)
                   if collect_states else (batch_size, 2 * M)),
            # shape=(M * 2, self.max_iters) if collect_states else (2 * M + 1),
            dtype=jnp.complex128,
        )
        # I read somewhere that the actions should be scaled to [-1,1],
        # values will be real.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(batch_size, M),
            dtype=jnp.float64 if use_doubles else jnp.float32,
        )

        self.seed(seed)
        self.state = None
        if collect_states:
            self.old_states = jnp.zeros((batch_size, M * 2, self.max_iters),
                                        dtype=jnp.complex128)
        self._setup_jit()

    def _setup_jit(self):
        # for func in [
        #         # '_scale_action',
        #         # '_get_prec',
        #         '_compute_residual',
        #         '_inf_norm',
        #         '_update_u',
        #         # '_generate_lambda',
        #         # '_compute_system_matrix',
        #         # '_compute_initial_state',
        #         # 'reset',
        #         # 'reward_func',
        # ]:
        #     setattr(self, func, jax.jit(getattr(self, func)))
        # self._scale_action = jax.jit(self._scale_action, static_argnums=(0,))
        # self._get_prec = jax.jit(self._get_prec, static_argnums=(0, 2))

        self._map_scale_action = jax.jit(jax.vmap(self._scale_action,
                                                  (None, 0)),
                                         static_argnums=(0,))
        self._map_compute_pinv = jax.jit(jax.vmap(self._compute_pinv))
        self._map_inf_norm = jax.jit(jax.vmap(self._inf_norm))
        self._map_update_u = jax.jit(jax.vmap(self._update_u))
        self._map_compute_residual = jax.jit(jax.vmap(self._compute_residual,
                                                      (None, 0, 0)))
        self._map_isnan = jax.jit(jax.vmap(jnp.isnan))
        self._map_isinf = jax.jit(jax.vmap(jnp.isinf))
        self._map_reward_func = jax.jit(jax.vmap(self.reward_func))
        self._map_compute_system_matrix = jax.jit(jax.vmap(
            self._compute_system_matrix))

    def seed(self, seed=None):
        seed = seeding.create_seed(seed)
        self.rng_key = jax.random.PRNGKey(seed)
        return seed

    def set_num_episodes(self, num_episodes):
        self.num_episodes = num_episodes

    @staticmethod
    # @partial(jax.jit, static_argnums=(0,))
    def _scale_action(do_scale, action):
        # I read somewhere that the actions should be scaled to [-1,1],
        # scale it back to [0,1] here...
        if do_scale:
            scaled_action = jnp.interp(
                action, jnp.array([-1, 1]), jnp.array([0, 1]))
        else:
            scaled_action = action
        return scaled_action

    @staticmethod
    # @partial(jax.jit, static_argnums=(0, 2))
    def _get_prec(prec, Q, M, scaled_action):
        """Return a preconditioner based on the `scaled_action`."""
        # Decide which preconditioner to use
        # (depending on self.prec string)... not very elegant
        if prec is None:
            Qdmat = jnp.diag(scaled_action)
        elif prec.upper() == 'LU':
            QT = Q.T
            [_, _, U] = jax.scipy.linalg.lu(QT, overwrite_a=True)
            Qdmat = U.T
        elif prec.lower() == 'min':
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
                x = jnp.zeros(M)
            Qdmat = jnp.diag(x)
        else:
            raise NotImplementedError()
        return Qdmat

    def _compute_pinv(self, lam, scaled_action):
        # Get Q_delta, based on self.prec (and/or scaled_action)
        Qdmat = self._get_prec(self.prec, self.Q, self.M,
                               scaled_action=scaled_action)

        # Compute the inverse of P
        Pinv = jnp.linalg.inv(
            jnp.eye(self.M) - lam * self.dt * Qdmat,
        )
        return Pinv

    @staticmethod
    # @jax.jit
    def _compute_residual(u0, C, u):
        return u0 - C @ u

    @staticmethod
    # @jax.jit
    def _inf_norm(v):
        return jnp.linalg.norm(v, jnp.inf)

    @staticmethod
    # @jax.jit
    def _update_u(u, Pinv, residual):
        return u + Pinv @ residual

    def step(self, action):
        u, old_residual = self.state
        # u, old_residual, _ = self.state

        scaled_action = self._map_scale_action(self.do_scale, action)

        Pinv = self._map_compute_pinv(self.lam, scaled_action)
        norm_res_old = self._map_inf_norm(old_residual)

        # Re-use what we already have
        residual = old_residual

        done = jnp.zeros((self.batch_size,), dtype=bool)
        err = jnp.zeros((self.batch_size,), dtype=bool)
        self.niter = jnp.zeros((self.batch_size,), dtype=jnp.uint8)
        # Start the loop
        # print('new ep!')
        while not jnp.all(done) and not jnp.all(self.niter >= self.max_iters):
            self.niter += ~done

            # This is the iteration (yes, there is a typo in the slides,
            # this one is correct!)
            u = self._map_update_u(u, Pinv, residual)
            # Compute the residual and its norm
            residual = self._map_compute_residual(self.u0, self.C, u)
            norm_res = self._map_inf_norm(residual)
            # print(f'{self.niter:>2}: {norm_res}')
            # stop if something goes wrong
            err = self._map_isnan(norm_res) ^ self._map_isinf(norm_res)
            # so far this seems to be the best setup:
            #   - stop if residual gets larger than the initial one
            #     (not needed, but faster)
            #   - reward = -self.max_iters, if this happens (crucial!)
            if self.collect_states and self.niter < self.max_iters:
                self.old_states[:, self.niter] = jnp.concatenate((u, residual))
            err = err ^ (norm_res > norm_res_old * 100)
            # if norm_res > norm_res_old * 100:
            #     raise ValueError('HEHEHEHEH')
            if jnp.any(err):
                reward = -self.step_penalty * (self.max_iters + 1)
                # reward = -(self.max_iters + 1)
                break
            # check for convergence
            done = norm_res < self.restol

        if not jnp.any(err):
            reward = jnp.mean(self._map_reward_func(
                self.initial_residual,
                residual,
                done,
                self.niter,
                scaled_action,
                Pinv,
            )).item()

        done = True
        # self.episode_rewards.append(reward)
        # self.norm_resids.append(norm_res)

        self.state = (u, residual)
        # self.state = (u, residual, self.niter)

        info = {
            'residual': norm_res,
            'niter': self.niter,
            'lam': self.lam,
        }
        # for i, norm_res_ in enumerate(norm_res):
        #     info['residual' + str(i)] = norm_res_
        if self.collect_states:
            return (self.old_states, reward, done, info)
        else:
            return (jnp.hstack(self.state), reward, done, info)

    def _reset_vars(self):
        self.num_episodes += 1
        self.niter = jnp.zeros((self.batch_size,), dtype=jnp.uint8)

        # self.rewards.append(self.episode_rewards)
        # self.episode_rewards = []

    def _generate_lambda(self):
        # Draw a lambda (here: negative real for starters)
        # The number of episodes is always smaller than the number of
        # time steps, keep that in mind for the interpolation
        # hyperparameters.
        if self.lambda_real_interpolation_interval is not None:
            lam_low = jnp.interp(self.num_episodes,
                                 self.lambda_real_interpolation_interval,
                                 self.lambda_real_interval_reversed)
        else:
            lam_low = self.lambda_real_interval[0]
        rng_key, subkey = jax.random.split(self.rng_key)
        self.rng_key, subkey2 = jax.random.split(rng_key)
        self.lam = (
            1 * jax.random.uniform(
                subkey,
                (self.batch_size,),
                minval=lam_low,
                maxval=self.lambda_real_interval[1])
            + 1j * jax.random.uniform(
                subkey2,
                (self.batch_size,),
                minval=self.lambda_imag_interval[0],
                maxval=self.lambda_imag_interval[1])
        )

    def _compute_system_matrix(self, lam):
        # Compute the system matrix
        C = jnp.eye(self.M) - lam * self.dt * self.Q
        return C

    def _compute_initial_state(self):
        self._generate_lambda()
        self.C = self._map_compute_system_matrix(self.lam)

        # Initial guess u^0 and initial residual for the state
        u = jnp.ones((self.batch_size, self.M), dtype=jnp.complex128)
        residual = self._map_compute_residual(self.u0, self.C, u)
        self.initial_residual = residual
        return (u, residual)

    def reset(self):
        self._reset_vars()
        (u, residual) = self._compute_initial_state()
        self.state = (u, residual)

        if self.collect_states:
            # Try if this works instead of the line below it.
            # I didn't use it for safety, but it's a bit faster.
            self.old_states[:, 0] = jnp.concatenate(self.state)
            self.old_states[:, 1:] = 0
            # self.old_states = jnp.zeros((self.M * 2, self.max_iters),
            #                             dtype=jnp.complex128)

        if self.collect_states:
            return self.old_states
        else:
            return jnp.hstack(self.state)

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
             * jnp.exp(-(norm_res * gauss_inv)**2 / 2))
            for (gauss_fact, gauss_inv) in zip(self.gauss_facts,
                                               self.gauss_invs)
        )
        if reached_convergence:
            extra_fact = (self.max_iters + 1 - steps)**2 * 10
        else:
            extra_fact = 1
        # reward = extra_fact * gauss_dist
        # reward = -norm_res + extra_fact * gauss_dist
        # reward = jnp.clip(reward, -math.log(/rewa))

        reward = gauss_dist * extra_fact
        # reward = 0
        # if norm_res > 1:
        #     reward -= math.log(norm_res)
        # if gauss_dist > 0:
        #     reward += math.log(extra_fact * gauss_dist)
        # min_reward = -200
        # max_reward = (self.max_iters + 1)**2 * 10
        # reward = jnp.clip(reward, min_reward, max_reward)
        # reward -= min_reward
        # reward /= max_reward - min_reward

        # reward = 3/20 * -math.log(norm_res) + extra_fact * gauss_dist
        # reward = -math.log(norm_res)
        # print(f'norm_res: {norm_res}\nreward: {reward}')
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
        Qdmat = self._get_prec(self.prec, self.Q, self.M, scaled_action)
        mulpinv = Pinv.dot(self.Q - Qdmat)
        eigvals = jnp.linalg.eigvals(self.lam * self.dt * mulpinv)
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
            jnp.arange(len(all_rewards)),
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
            jnp.arange(len(self.norm_resids)),
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
        # u, old_residual, _ = self.state

        scaled_action = self._scale_action(self.do_scale, action)

        Pinv = self._compute_pinv(scaled_action)

        # Do the iteration (note that we already have the residual)
        u += Pinv @ old_residual

        # The new residual and its norm
        residual = self._compute_residual(self.u0, self.C, u)

        norm_res = self._inf_norm(residual)
        # norm_res_old = self._inf_norm(old_residual)

        self.niter += 1

        # Check if something went wrong
        err = jnp.isnan(norm_res) or jnp.isinf(norm_res)
        # so far this seems to be the best setup:
        #   - stop if residual gets larger than the initial one
        #     (not needed, but faster)
        #   - reward = -self.max_iters, if this happens (crucial!)
        # err = err or norm_res > norm_res_old * 100
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
        # self.state = (u, residual, self.niter)
        if self.collect_states and self.niter < self.max_iters:
            self.old_states[:, self.niter] = jnp.concatenate(self.state)

        info = {
            'residual': norm_res,
            'niter': self.niter,
            'lam': self.lam,
        }
        if self.collect_states:
            return (self.old_states, reward, done, info)
        else:
            return (self.state, reward, done, info)


class SDC_Fix_Env(SDC_Full_Env):
    max_iters = 5

    def _setup_jit(self):
        self._map_scale_action = jax.vmap(self._scale_action, (None, 0))
        self._map_compute_pinv = jax.vmap(self._compute_pinv)
        self._map_inf_norm = jax.vmap(self._inf_norm)
        self._map_update_u = jax.vmap(self._update_u)
        self._map_compute_residual = jax.vmap(
            self._compute_residual, (None, 0, 0))
        self._map_isnan = jax.vmap(jnp.isnan)
        self._map_isinf = jax.vmap(jnp.isinf)
        self._map_reward_func = jax.vmap(self.reward_func)
        self._map_compute_system_matrix = jax.vmap(self._compute_system_matrix)

    def step(self, action):
        u, old_residual = self.state

        scaled_action = self._map_scale_action(self.do_scale, action)

        Pinv = self._map_compute_pinv(self.lam, scaled_action)

        # Re-use what we already have
        residual = old_residual

        # Start the loop
        # print('new ep!')
        for i in range(self.max_iters):
            # This is the iteration (yes, there is a typo in the slides,
            # this one is correct!)
            u = self._map_update_u(u, Pinv, residual)
            # Compute the residual and its norm
            residual = self._map_compute_residual(self.u0, self.C, u)
            # print(f'{self.niter:>2}: {norm_res}')
        norm_res = self._map_inf_norm(residual)

        self.niter = jnp.full(
            (self.batch_size,), self.max_iters, dtype=jnp.uint8)

        reward = 0
        done = True
        # self.episode_rewards.append(reward)
        # self.norm_resids.append(norm_res)

        self.state = (u, residual)
        # self.state = (u, residual, self.niter)

        info = {
            'residual': norm_res,
            'niter': self.niter,
            'lam': self.lam,
        }
        if self.collect_states:
            return (self.old_states, reward, done, info)
        else:
            return (jnp.hstack(self.state), reward, done, info)
