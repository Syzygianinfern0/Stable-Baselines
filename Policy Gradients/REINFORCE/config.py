import torch

env_name = 'CartPole-v1'
goal_score = 200

max_episodes = 1000

batch_size = 32

log_interval = 10
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
