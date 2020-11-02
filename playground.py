import gym
import sdc_gym
from stable_baselines import PPO2, ACKTR
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right

import numpy as np

M = 3
fname = "sdc_model_acktr"

model = ACKTR.load(fname)

lam = 1 * np.random.uniform(low=-100.0, high=0.0) + 0j * np.random.uniform(low=0.0, high=1.0)
dt = 1.0
restol  = 1E-10

coll = CollGaussRadau_Right(M, 0, 1)
Q = coll.Qmat[1:, 1:]
u0 = np.ones(coll.num_nodes, dtype=np.complex128)
u = np.ones(coll.num_nodes, dtype=np.complex128)
C = np.eye(coll.num_nodes) - lam * dt * Q
residual = u0 - C @ u
norm_res_old = np.linalg.norm(residual, np.inf)
done = False
err = False
niter = 0
obs = (u, residual)
action, _states = model.predict(obs, deterministic=True)
scaled_action = np.interp(action, (-1, 1), (0, 1))
while not done and not niter >= 50 and not err:
    niter += 1



    Qdmat = np.zeros_like(Q)
    np.fill_diagonal(Qdmat, scaled_action)
    Pinv = np.linalg.inv(np.eye(coll.num_nodes) - lam * dt * Qdmat)

    u += Pinv @ (u0 - C @ u)

    residual = u0 - C @ u
    norm_res = np.linalg.norm(residual, np.inf)
    print(niter, norm_res, action)

    done = norm_res < restol


