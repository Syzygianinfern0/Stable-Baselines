import torch

env_name = 'CartPole-v1'
goal_score = 200

max_episodes = 1000
initial_exploration = 1000

experience_size = 1000
batch_size = 32

gamma = 0.99
max_epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.005

log_interval = 10
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
