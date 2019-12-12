import torch

env_name = 'CartPole-v1'
goal_score = 200

max_episodes = 1000
initial_exploration = 1000

experience_size = 1000
batch_size = 32

gamma = 0.99

alpha = 0.5
e_adj = 0.0001

min_beta = 0.1
max_beta = 1.0
beta_anneal = 0.005

sigma_zero = 0.5

update_target = 100

log_interval = 10
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
