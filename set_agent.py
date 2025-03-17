import numpy as np
import random
from util import DQNnetwork, ReplayBuffer
import torch
import torch.nn as nn
import torch.optim as optim


class DQN_agent:
    def __init__(self, state_dim, action_dim,  config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = config["gamma"]  # 折扣因子
        self.batchsize = config["batchsize"]
        self.buffer_size = config["buffersize"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # 初始化 Q-Network 和 Target Network
        self.q_net = DQNnetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.SGD(self.q_net.parameters(), lr=config["learning_rate"])
        self.target_net = DQNnetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())  # 初始化為相同的權重
        self.target_net.eval() 
        
        self.criterion = nn.MSELoss()
        # 記憶體 Replay Buffer
        self.buffer = ReplayBuffer(self.buffer_size, self.device)

    def select_action(self, state, epsilon):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device) #把state 轉為 [1, dimension]的維度
        with torch.no_grad():
            action_values = self.q_net(state_tensor)  # 計算 Q_net預測的action 分佈

        if epsilon > np.random.rand(): ## 亂選
            return random.choice(range(self.action_dim))
        return torch.argmax(action_values, dim=1).item()
    
    def train(self): ##其實這步是update

        self.optimizer.zero_grad()
        ## 先sample 
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batchsize)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # states : (batchsize, states_dim) 的tensor 丟進去Q net後變成 [batchsize, action_space_size]

        # actions = [batchsize] 的tensor, unsqueeze 後變成[[1],[2], ...] 變成[batchsize, 1]
        # gather input [batchsize, action_space_size] 匹配 [batchsize, 1] 
        # output對應的  [batchsize, 1], 再squeeze(1) 成 [batchsize]的tensor
        with torch.no_grad(): #用這些sample出的state來計算現在的Q值
            max_next_q_values = self.target_net(next_states).max(1)[0] 
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
            
        loss = self.criterion(q_values, target_q_values)
        # 反向傳播更新 Q-Network
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save_model(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            torch.save(self.q_net.state_dict(), f)
        
    def load_model(self, filename="q_table.pkl"):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ✅ 檢查是否有 GPU
        self.q_net.to(device)
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location=device, weights_only=True)  
        self.q_net.load_state_dict(state_dict)
        self.q_net.eval()
