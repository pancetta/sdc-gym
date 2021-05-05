from gym import spaces
import numpy as np

from .sdc_env import SDC_Full_Env


class SDC_Full_Force_Env(SDC_Full_Env):
    """This environment implements doing full iterations of SDC, taking
    actions until we reach convergence for each lambda.
    """
    max_tries = 50

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        M = self.coll.num_nodes
        self.observation_space = spaces.Box(
            low=-1E10,
            high=+1E10,
            shape=(
                (M * 2, self.max_tries)
                if self.collect_states
                else (2, M)
            ),
            dtype=np.complex128,
        )
        if self.collect_states:
            self.old_states = np.zeros((M * 2, self.max_tries),
                                       dtype=np.complex128)

    def step(self, action):

        old_residual, old_diag = self.state
        u = self.u0.copy()
        # u, old_residual, _ = self.state

        scaled_action = self._scale_action(action)
        if self.prec is None:
            scaled_action = scaled_action + old_diag

        Pinv = self._compute_pinv(scaled_action)
        norm_res_old = self._inf_norm(old_residual)

        done = False
        err = False
        self.niter = 0
        # Start the loop
        # print('new ep!')
        while not done and not self.niter >= self.max_iters:
            self.niter += 1

            # This is the iteration (yes, there is a typo in the slides,
            # this one is correct!)
            u += Pinv @ (self.u0 - self.C @ u)
            # Compute the residual and its norm
            residual = self.u0 - self.C @ u
            norm_res = self._inf_norm(residual)
            # print(f'{self.niter:>2}: {norm_res}')
            # stop if something goes wrong
            err = np.isnan(norm_res) or np.isinf(norm_res)
            # so far this seems to be the best setup:
            #   - stop if residual gets larger than the initial one
            #     (not needed, but faster)
            #   - reward = -self.max_iters, if this happens (crucial!)
            err = err or norm_res > norm_res_old * 100
            if err:
                reward = -self.step_penalty * (self.max_tries + 1)
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
            )
            if done:
                reward *= (self.max_tries + 1 - self.ntries)**2 * 10

        self.state = (residual, scaled_action)

        self.ntries += 1
        if self.collect_states and self.ntries < self.max_tries:
            self.old_states[:, self.ntries] = np.concatenate(
                (residual, scaled_action))
        done = done or self.ntries >= self.max_tries

        info = {
            'residual': norm_res,
            'niter': self.niter,
            'ntries': self.ntries,
            'lam': self.lam,
        }
        if self.collect_states:
            return (self.old_states, reward, done, info)
        else:
            return (self.state, reward, done, info)

    def reset(self):
        self.ntries = 0
        self._reset_vars()

        (u, residual) = self._compute_initial_state()
        self.state = (residual, np.zeros_like(u))

        if self.collect_states:
            # Try if this works instead of the line below it.
            # I didn't use it for safety, but it's a bit faster.
            self.old_states[:, 1:] = 0
            # self.old_states = np.zeros((u.size * 2, self.max_iters),
            #                            dtype=np.complex128)
            self.old_states[:, 0] = np.concatenate(
                (residual, np.zeros_like(u)))

        if self.collect_states:
            return self.old_states
        else:
            return self.state
