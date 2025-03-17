from tqdm import tqdm
from set_agent import DQN_agent
from simple_custom_taxi_env import SimpleTaxiEnv
from util import get_config

def train_agent(config):
    # The default parameters should allow learning, but you can still adjust them to achieve better training performance.
    env = SimpleTaxiEnv()
    state_size = 16
    action_size = 6
    agent = DQN_agent(state_size, action_size, config)
    episodes = config["episodes"]
    epsilon = config["epsilon_start"] # 初始探索率
    epsilon_min = config["epsilon_end"] # 最低探索率
    epsilon_decay = config["epsilon_decay"]
    update_time = config["update_time"]

    for episode in tqdm(range(episodes)):
        
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
          
          action = agent.select_action(obs, epsilon)
          next_obs, reward, done, _ = env.step(action)

          agent.buffer.push(obs, action, reward, next_obs, done)
          if len(agent.buffer.buffer) > agent.batchsize:
            agent.train()
  
          obs = next_obs
          total_reward += reward

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % update_time == 0:
          agent.update_target()

        # Print progress every 100 episodes.
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    agent.save_model()

if __name__ == '__main__':
    config = get_config()
    train_agent(config)
