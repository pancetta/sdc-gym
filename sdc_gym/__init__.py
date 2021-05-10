from gym.envs.registration import register

register(
    id='sdc-v0',
    entry_point='sdc_gym.envs:SDC_Full_Env',
    max_episode_steps=1,
)

register(
    id='sdc-v1',
    entry_point='sdc_gym.envs:SDC_Step_Env',
    max_episode_steps=50,
)

register(
    id='sdc-v4',
    entry_point='sdc_gym.envs:SDC_Full_Force_Env',
    max_episode_steps=50,
)

register(
    id='jax-sdc-v2',
    entry_point='sdc_gym.envs:JAX_SDC_Full_Vec_Env',
    max_episode_steps=1,
)

register(
    id='jax-sdc-v3',
    entry_point='sdc_gym.envs:JAX_SDC_Step_Vec_Env',
    max_episode_steps=50,
)
