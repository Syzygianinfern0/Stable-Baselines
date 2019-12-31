import torch

env_name = 'CartPole-v1'
goal_score = 200

max_episodes = 1000

gamma = 0.99
lmbda = 0.95
delta = 0.01

cg_iters = 10
eps = 1e-8
damping_coeff = 0.1
delta = 0.01
epsilon = 1e-8

log_interval = 10
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
