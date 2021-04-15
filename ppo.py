import torch
from torch.distributions import MultivariateNormal

from network import FeedForwardNN


class ppo:
    def __init__(self, env):
        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

    def learn(self,total_timesteps):
        t_so_far = 0 # Timesteps simulated so far

        while t_so_far < total_timesteps:
            # Increment t_so_far somewhere below
            
    
    def _init_hyperparameters(self):
        # Default values for hyperparameters, will need to change later.
        self.timesteps_per_batch = 4800      # timesteps per batch
        self.max_timesteps_per_episode = 1600      # timesteps per episode

    def rollout(self):
        # Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch
