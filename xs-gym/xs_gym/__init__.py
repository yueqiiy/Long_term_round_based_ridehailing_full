from gym.envs.registration import register

register(
		id='myEnv-v0',
		entry_point='xs_gym.envs:MyEnv',
)
