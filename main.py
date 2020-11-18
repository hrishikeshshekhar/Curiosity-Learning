import gym
from ppo import PPO

# Creating the environment
env = gym.make('CartPole-v1')

# Setting hyper parameters
learning_steps = 25000
actor_learning_rate = 1e-2
critic_learning_rate = 1e-3
clip = 0.2

#  Creating the model
model = PPO(env, verbose=True, actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, clip=clip)

# Training the model and saving the weights
model.learn(learning_steps, plot=True)
model.save()

# Testing the model
model.test(episodes=5, display=True)
