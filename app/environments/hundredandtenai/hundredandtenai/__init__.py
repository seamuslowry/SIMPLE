from gym.envs.registration import register

register(
    id='HundredAndTen-v0',
    entry_point='hundredandtenai.envs:HundredAndTenEnv',
)

