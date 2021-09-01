from sdc_gym.envs.sdc_env_nonlinear import SDC_Full_Env
from sdc_gym.problems.logistics_equation import logistics_equation
import numpy as np

def main():

    M = 3
    dt = 0.1
    restol = 1E-10
    lambda_real_interval=[-1, -1]
    prec = 'LU'
    u0 = 0.5
    prob = logistics_equation(problem_params={'N': 1})

    sdc = SDC_Full_Env(M=M, prob=prob, u0=u0, dt=dt, restol=restol, prec=prec, lambda_real_interval=lambda_real_interval)
    sdc.reset()
    _, rewards, done, info = sdc.step(action=None)
    print(info)


if __name__ == '__main__':
    main()