# import gym
# from ppo import PPO

# # Creating the environment
# env = gym.make('Acrobot-v1')

# # Setting hyper parameters
# max_episode_length = 500
# timesteps_per_batch = 1000
# actor_learning_rate = 1e-2
# critic_learning_rate = 1e-3
# clip = 0.1
# entropy_coeff = 0
# actor_layers = [12, 12, 12]
# critic_layers = [12, 12, 12]
# entropy_coefficients = [0, 0.1, 1e-2, 1e-3, 1e-4]

# #  Creating the model
# model = PPO(env, timesteps_per_batch=timesteps_per_batch, max_episode_length = max_episode_length, verbose=False, entropy_coeff=entropy_coeff, critic_layers=critic_layers, actor_layers=actor_layers, actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, clip=clip)

# # Testing how entropy coefficient influences learning
# model.investigate_entropy(total_seeds=3, rollouts=50, entropy_coefficients=entropy_coefficients)




import os, time
import gym
from ppo import PPO
from procgen import ProcgenEnv
import random
import torch


from assets.logger import Logger
from assets.storage import Storage1
from network import ImpalaModel
from assets.policy import CategoricalPolicy
from assets.utils_misc import set_global_seeds, set_global_log_levels
from utils import *



if __name__ == '__main__':
    exp_name = 'test'
    env_name = 'fruitbot'
    start_level = int(0)
    num_levels = int(50)
    distribution_mode = 'easy'
    gamma = 0.999
    laamdba = 0.95
    num_epochs = 3
    alpha = 0.0005
    grad_eps = .5
    value_coef = .5
    entropy_coef = [0, 0.1, 1e-2, 1e-3, 1e-4]
    mini_batch_per_epoch = 8
    batch_size = 2048
    num_timesteps = int(20e6)
    seed = random.randint(0, 9999)
    log_level = int(40)
    num_checkpoints = int(1)
    eps = .2
    
    set_global_seeds(seed)
    set_global_log_levels(log_level)

    ### DEVICE ###
    device = torch.device('cuda')
    #print("dev", device)
   
    ### ENVIRONMENT ###
    print('INITIALIZAING THE ENVIRONMENTS...............')
    n_steps = 256
    n_envs = 64
    torch.set_num_threads(1)    #increasing the number of threads will usually leads to faster execution on CPU
    env = ProcgenEnv(num_envs=n_envs,env_name=env_name,start_level=start_level,
                    num_levels=num_levels,distribution_mode=distribution_mode,
                    use_backgrounds=False,restrict_themes=True)
    
    normalize_reward = True
    env = VecExtractDictObs(env, "rgb")
    if normalize_reward:
        env = VecNormalize(env, ob=False)  # normalizing returns, but not the img frames
    env = TransposeFrame(env)
    env = ScaledFloatFrame(env)

    v_env = ProcgenEnv(num_envs=n_envs,env_name=env_name,start_level=start_level,
                    num_levels=num_levels,distribution_mode=distribution_mode,
                    use_backgrounds=False,restrict_themes=True)
    v_env = VecExtractDictObs(v_env, "rgb")
    if normalize_reward:
        v_env = VecNormalize(v_env, ob=False)
    v_env = TransposeFrame(v_env)
    v_env = ScaledFloatFrame(v_env)

    
    ### LOGGER ###
    print('INITIALIZAING LOGGER..................')
    logdir = 'procgen/' + env_name + '/' + exp_name + '/' + 'seed' + '_' + \
            str(seed) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('logs', logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    logger = Logger(n_envs, logdir)

    
    ### MODEL ###
    print('INTIALIZING MODEL...................')
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    in_channels = observation_shape[0]
    action_space = env.action_space
    # Model's architecture
    model = ImpalaModel(in_channels=in_channels)  #Importance Weighted Actor-Learner Architectures(paper: https://arxiv.org/abs/1802.01561)
    recurrent = True
    action_size = action_space.n
    policy = CategoricalPolicy(model, recurrent, action_size)
    policy.to(device)
    ### STORAGE ###
    print('INITIALIZAING STORAGE...............')
    hidden_state_dim = model.output_dim
    storage = Storage1(obs_shape=observation_shape, hidden_state_size=hidden_state_dim, num_steps=n_steps, num_envs=n_envs, device=device)
    #print("HEYYYYY dimm", storage.num_steps,storage.obs_shape,storage.hidden_state_size,storage.num_envs,storage.device)
    
    
    # TESTING HOW ENTROPY COEFFICIENT INFLUENCES LEARNING
    
    for entropy in entropy_coef:
        ### AGENT ###
        print('INTIALIZING AGENT...................')
        agent = PPO(env, v_env, policy, logger, storage, device, num_checkpoints, n_steps,
                n_envs, num_epochs, mini_batch_per_epoch=mini_batch_per_epoch,
                mini_batch_size=batch_size,gamma=gamma, laambda=laamdba, learning_rate=alpha,
                grad_clip_norm=grad_eps,eps_clip=eps, value_coef=value_coef,
                entropy_coef=entropy,normalize_adv=True, use_gae=True)
        agent.train(num_timesteps)
