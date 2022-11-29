from gym.envs.registration import register


register(
    id='gcl-target-v0',
    entry_point='gym_gcl.envs:Target',
    max_episode_steps=50,
)
