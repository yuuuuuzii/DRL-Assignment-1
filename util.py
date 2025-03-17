import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque 
import random
import numpy as np
import argparse
class DQNnetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNnetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64,128)
        self.fc4 = nn.Linear(128, action_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen = capacity)
        self.device = device
    def push(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=float)
        # state[:10]/=(np.max(state[2:6])+1)
        new_array = state[2:].copy()
        new_array[[0, 2, 4, 6]] -= state[0]
        new_array[[1, 3, 5, 7]] -= state[1]
        new_array = torch.tensor(new_array, dtype=torch.float32).to(self.device)

        next_state = np.array(next_state, dtype=float)
        # next_state[:10]/=(np.max(next_state[2:6])+1)
        next_array = next_state[2:].copy()
        next_array[[0, 2, 4, 6]] -= next_state[0]
        next_array[[1, 3, 5, 7]] -= next_state[1]
        next_array = torch.tensor(next_array, dtype=torch.float32).to(self.device)

        self.buffer.append((new_array, action, reward, next_array, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)    
        
        return(
            torch.stack(states).to(self.device),
            torch.tensor(actions, dtype=torch.int64).to(self.device),
            torch.tensor(rewards, dtype=torch.float32).to(self.device),
            torch.stack(next_states).to(self.device),
            torch.tensor(dones, dtype=torch.float32).to(self.device)
        )

def get_config():
    parser = argparse.ArgumentParser(description="DQN Agent Configuration")
    
    parser.add_argument("--episodes", type=int, default=15000, help="Number of episodes")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate for optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batchsize", type=int, default=256, help="Batch size")
    parser.add_argument("--buffersize", type=int, default=10000, help="Replay buffer size")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial epsilon for exploration")
    parser.add_argument("--epsilon_end", type=float, default=0.05, help="Final epsilon for exploration")
    parser.add_argument("--epsilon_decay", type=float, default=0.99985, help="Epsilon decay rate")
    parser.add_argument("--update_time", type=int, default=500, help="Target network update frequency")
    parser.add_argument("--token", type=str, help="Authentication token")
    args = parser.parse_args()
    return vars(args)


