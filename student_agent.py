from set_agent import DQN_agent
import torch

from util import get_config

def get_action(obs):
    return agent.select_action(obs, 0)

config = get_config()
agent = DQN_agent(14, 6, config)
agent.load_model()