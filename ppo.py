from network import NeuralNetwork
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

class PPO:
    def __init__(self, env, critic_layers=[24, 24, 24], actor_layers = [24, 24, 24], verbose=False, max_episode_length=1000, timesteps_per_batch=2000, seed=0, clip=0.1, gamma=0.99, updates_per_rollout = 5, actor_learning_rate=1e-3, critic_learning_rate=1e-3, entropy_coeff=1e-4):
        # Assigning hyper parameters
        self.timesteps_per_batch = timesteps_per_batch
        self.max_timesteps_per_episode = max_episode_length
        self.updates_per_rollout = updates_per_rollout
        self.gamma = gamma
        self.clip = clip
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.critic_layers = critic_layers
        self.actor_layers = actor_layers
        self.entropy_coeff = entropy_coeff
        self.verbose = verbose

        # Setting that seed for the model to produce reproducible results
        self.set_seed(seed)

        # Environment variables
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        # Initializing actor and critic models
        self.initialize_models()

    def initialize_models(self):
        # Creating the actor and critic networks
        actor_layers = [self.state_dim, *self.actor_layers, self.action_dim]
        critic_layers = [self.state_dim, *self.critic_layers, 1]
        self.actor = NeuralNetwork(actor_layers)
        self.critic = NeuralNetwork(critic_layers, softmax=False)

        # Creating the optimizers for the actor and critic networks
        self.actor_optim  = torch.optim.Adam(self.actor.parameters(),  lr=self.actor_learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

    def set_seed(self, seed):
        # Setting seeds for numpy and torch
        torch.manual_seed(seed)
        np.random.seed(seed)

    def save(self, model_name = None):
        # Getting the env name if no model_name is given
        if not model_name:
            model_name = self.env.unwrapped.spec.id

        # Checking for the weights folder and navigating to the weights folder
        if(not os.path.exists("./weights")):
            os.mkdir("./weights")
        os.chdir("./weights")

        # Checking for the model name folder and navigating to that folder
        if(not os.path.exists("./" + model_name)):
            os.mkdir("./" + model_name)
        os.chdir("./" + model_name)

        # Saving the actor and the critic weights
        torch.save(self.actor.state_dict(), "./actor_weights")
        torch.save(self.critic.state_dict(), "./critic_weights")
        print("Successfully saved weights for the environment : {}".format(model_name))

        # Changing back to the root directory
        os.chdir("./../../")

    def load(self, model_name = None):
        # Getting the env name if no model_name is given
        if not model_name:
            model_name = self.env.unwrapped.spec.id

        # Checking for the weights folder and navigating to the weights folder
        if(not os.path.exists("./weights")):
            print("No weights folder exists")
            return
        os.chdir("./weights")

        # Checking for the model name folder and navigating to that folder
        if(not os.path.exists("./" + model_name)):
            print("No saved weights exists for the env : {}".format(model_name))
            return
        os.chdir("./" + model_name)

        # Loading the actor and critic models
        try:
            self.actor.load_state_dict(torch.load("./actor_weights"))
            self.critic.load_state_dict(torch.load("./critic_weights"))
        except:
            print("Error loading weights of actor or critic network for the environment : {} ".format(model_name))

        # Changing back to the root directory
        os.chdir("./../../")

        print("Successfully loaded weights : {}".format(model_name))

    def test(self, episodes=10, display=False):
        # Storing rewards from each episode
        rewards = []
        model_name = self.env.unwrapped.spec.id

        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = []

            for timestep in range(self.max_timesteps_per_episode):
                # Choose the action with the highest probability rather than sampling the action
                action, _, _ = self.get_action(state, deterministic=True)
                state, reward, done, _ = self.env.step(action)
                episode_reward.append(reward)
                if display:
                    self.env.render()
                if done:
                    break

            rewards.append(np.sum(episode_reward))

        print("Average reward : {}".format(np.mean(rewards)))
        plt.plot(rewards)
        plt.title("PPO Performance on environment : {}".format(model_name))
        plt.xlabel("Episode Number")
        plt.ylabel("Total episode reward")
        plt.show()

    def get_entropy_loss(self, batch_states):
        # Getting all output probabilities
        output_probs = self.actor.forward(batch_states)

        # Getting log probs
        log_probs = torch.log(output_probs)

        # Entropy
        entropy = self.entropy_coeff * torch.sum(output_probs * log_probs, dim=1).mean()
        
        return entropy

    def learn(self, rollouts = 10, plot=False):
        timestep = 0

        # Storing the average rollout reward to track training progress
        average_rollout_rewards = []
        
        for rollout_no in range(rollouts):
            # Performing one rollout and getting rollout states, actions and rewards
            batch_states, batch_actions, batch_log_probs, batch_rewards, average_rollout_reward = self.rollout()
            average_rollout_rewards.append(average_rollout_reward)

            if (self.verbose):
                print("Average rollout reward for rollout number {} is : {}".format(rollout_no, average_rollout_reward))

            # Calculating the values for each state
            V, _ = self.get_values(batch_states, batch_actions)

            # Calculating advantages
            A_k = batch_rewards - V.detach()

            for _ in range(self.updates_per_rollout):
                # Calculating current log probabilities and values
                V, curr_log_probs = self.get_values(batch_states, batch_actions)

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculating the surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculating the losses for the actor and critic
                actor_loss  = -1 * (torch.min(surr1, surr2)).mean() 
                critic_loss = torch.nn.MSELoss()(V, batch_rewards)

                # Updating the actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Updating the critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            timestep += len(batch_states)

        # Converting average rollout rewards into a numpy array
        average_rollout_rewards = np.array(average_rollout_rewards)
        
        if(plot):
            plt.plot(average_rollout_rewards)
            plt.title("Average rollout reward vs rollout number")
            plt.xlabel("Rollout Number")
            plt.ylabel("Average rollout reward")
            plt.show()

        return average_rollout_rewards

    def investigate_entropy(self, total_seeds = 3, rollouts=10, entropy_coefficients=[1e-1, 1e-2, 1e-3, 1e-4, 0]):
        # Saving the initial value of the entropy coeff 
        initial_entropy = self.entropy_coeff

        # Creating a set of seeds
        seeds = [random.randint(0, 1e6) for _ in range(total_seeds)]

        # Iterating over differnt entropy coefficient values
        for entropy_coeff in entropy_coefficients:
            # Storing the rewards for each entropy coeff
            entropy_rewards = []

            # Iterating over many random seeds for each entropy
            for seed in seeds:
                # Setting the seed
                self.set_seed(seed)

                # Setting the entropy coeff
                self.entropy_coeff = entropy_coeff

                # Reinitializing the models
                self.initialize_models()

                # Training the model
                rewards = self.learn(rollouts=rollouts)

                # Saving the entropy rewards
                entropy_rewards.append(rewards)

            # Taking an average across all the seeds
            all_rewards = np.array(entropy_rewards)
            average_rewards = np.mean(all_rewards, axis=0)

            # Plotting the average rewards
            plt.plot(average_rewards, label="E = " + str(entropy_coeff))
            print("Completed training for entropy coefficient : {}".format(entropy_coeff))
                        
        # Plotting the whole graph
        plt.legend()
        plt.show()

        # Resetting the entropy coefficient
        self.entropy_coeff = initial_entropy

    def evaluate(self, no_seeds=5, rollouts=50):
        # Iterate over random seeds and record performance
        for seed_no in range(no_seeds):
            seed = random.randint(0, 1e5)

            # Setting the seed
            self.set_seed(seed)

            # Initializing new parameters for the actor and critic
            self.initialize_models()

            # Training the model
            agent_rewards = self.learn(rollouts=rollouts)
            plt.plot(agent_rewards, label="Seed no : {}".format(seed))

        # Plotting rewards vs rollout number
        plt.title("Reward vs rollout number")
        plt.xlabel("Rollout Number")
        plt.ylabel("Reward")
        plt.legend()
        plt.show()

    def get_action(self, state, deterministic=False):
        # Getting the action output probabilities
        action_output_probs = self.actor.forward(state).detach().numpy()
        action_output_log_probs = np.log(action_output_probs)
        entropy = -1 * np.sum(action_output_probs * action_output_log_probs)
        
        # Selecting the most probable action
        if deterministic:
            action = np.argmax(action_output_probs)
        else:
            # Using the action output probabilities to sample an action
            action = random.choices([i for i in range(self.action_dim)], weights=action_output_probs)[0]

        # Extracting the probability and log probability of the action chosen
        probability = action_output_probs[action]
        log_prob = np.log(probability)

        return action, log_prob, entropy

    def calculate_rewards(self, batch_rewards, batch_entropies):
        # Calculating rewards per episode
        gamma_discounted_rewards = []
        total_episodes = len(batch_rewards)

        # Finding the total number of episodes
        for episode_num in range(total_episodes):
            episode_rewards = batch_rewards[episode_num]
            episode_entropies = batch_entropies[episode_num]

            episode_entropy = 0
            episode_reward = 0
            episode_length = len(episode_rewards)
            episode_gamma_rewards = []
            episode_gamma_entropies = []

            # Reversing the rewards and multiplying by gamma at each stage to get gamma discounted rewards
            for timestep in reversed(range(episode_length)):
                episode_reward *= self.gamma
                episode_entropy *= self.gamma
                episode_reward += episode_rewards[timestep]
                episode_entropy += episode_entropies[timestep]
                episode_gamma_rewards.append(episode_reward)
                episode_gamma_entropies.append(episode_entropy)

            # Reversing again to get the rewards in the right order
            episode_gamma_rewards = np.array(episode_gamma_rewards[::-1])
            episode_gamma_entropies = self.entropy_coeff * np.array(episode_gamma_entropies[::-1])
            
            # Modifying the episode rewards by adding the entropies
            modified_episode_rewards = episode_gamma_rewards + episode_gamma_entropies

            # Subtracting mean and dividing by standard deviation to reduce variance and improve stability
            modified_episode_rewards = (modified_episode_rewards - modified_episode_rewards.mean()) / (modified_episode_rewards.std() + 1e-8)

            # Appending all the rewards to gamma discounted rewards
            for reward in modified_episode_rewards:
                gamma_discounted_rewards.append(reward)

        # Creating an output tensor from the rewards
        gamma_discounted_tensor = torch.tensor(gamma_discounted_rewards, dtype=torch.float32)

        return gamma_discounted_tensor

    def get_values(self, batch_states, batch_actions):
        # Calculating the value of each state using the current version of the critic network
        V = self.critic.forward(batch_states).squeeze()

        # Calculating the probability of each action using the current version of the policy network
        output_probs = self.actor.forward(batch_states)

        # Batch actions is made into a one hot vector. Hence, only the chosen action's probability will be multiplied by 1 and the other actions will be multiplied by 0
        probs = output_probs * batch_actions

        # Taking a summation across each timestep, we will get the action probability we need
        probs = probs.sum(dim=1)

        # Calculating log probabilities
        log_probs = torch.log(probs)

        return V, log_probs

    def onehot(self, action, size):
        # Generating a one hot vector of length = size
        output = np.zeros(size)
        output[action] = 1
        return output

    def rollout(self):
        # Collecting data of the rollout
        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_entropies = []
        all_episode_rewards = []
        
        # Timestep counter
        t = 0

        while t < self.timesteps_per_batch:
            # Rewards this episode
            state = self.env.reset()
            episode_rewards = []
            episode_entropies = []

            for episode_steps in range(self.max_timesteps_per_episode):
                # Incrementing the total timesteps for this batch
                t += 1

                # Collecting the observations
                batch_states.append(state)

                # Choosing an action using the current actor network
                action, log_prob, entropy = self.get_action(state)
                state, reward, done, _ = self.env.step(action)

                # Collecting the reward, action and log_prob
                episode_rewards.append(reward)

                # Changing the action from an integer to a one hot vector
                batch_actions.append(self.onehot(action, self.action_dim))
                
                # Storing the entropy
                episode_entropies.append(entropy)

                # Saving the log probabilities of each action
                batch_log_probs.append(log_prob)

                if done:
                    break

            # Collecting the episode rewards and lengths
            batch_rewards.append(episode_rewards)
            all_episode_rewards.append(np.sum(episode_rewards))
            batch_entropies.append(episode_entropies)

        # Convert data into tensors
        batch_states = torch.tensor(batch_states, dtype=torch.float32)
        batch_actions = torch.tensor(batch_actions, dtype=torch.long)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32)

        # Calculating the batch rewards
        batch_rewards = self.calculate_rewards(batch_rewards, batch_entropies)
        average_rollout_reward = np.mean(all_episode_rewards)

        return batch_states, batch_actions, batch_log_probs, batch_rewards, average_rollout_reward

