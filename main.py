import gym
from ppo import PPO

# Creating the environment
env = gym.make('Acrobot-v1')

# Setting hyper parameters
max_episode_length = 500
timesteps_per_batch = 1000
actor_learning_rate = 1e-2
critic_learning_rate = 1e-3
clip = 0.1
entropy_coeff = 0.1
actor_layers = [12, 12, 12]
critic_layers = [12, 12, 12]
entropy_coefficients = [0, 0.1, 1e-2, 1e-3, 1e-4]
rollouts = 50

#  Creating the model
model = PPO(env, timesteps_per_batch=timesteps_per_batch, max_episode_length = max_episode_length, verbose=True, entropy_coeff=entropy_coeff, critic_layers=critic_layers, actor_layers=actor_layers, actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, clip=clip)
model.learn(rollouts=rollouts)

# Saving the weights
model.save()
