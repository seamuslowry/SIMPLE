from gym.envs.registration import register

register(
    id='HundredAndTen-v0',
    entry_point='hundredandten.envs:HundredAndTenEnv',
)

