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