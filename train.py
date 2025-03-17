from tqdm import tqdm
from set_agent import DQN_agent
from simple_custom_taxi_env import SimpleTaxiEnv
from util import get_config
import random
def train_agent(config):
    # The default parameters should allow learning, but you can still adjust them to achieve better training performance.
    
    state_size = 14
    action_size = 6
    agent = DQN_agent(state_size, action_size, config)
    episodes = config["episodes"]
    epsilon = config["epsilon_start"] # 初始探索率
    epsilon_min = config["epsilon_end"] # 最低探索率
    epsilon_decay = config["epsilon_decay"]
    update_time = config["update_time"]
    pickup = 0
    for episode in tqdm(range(episodes)):
        env = SimpleTaxiEnv(grid_size = random.randint(5, 10))
        obs, _ = env.reset()
        done = False
        total_reward = 0
        visited_station = {1:0, 2:0, 3:0, 4:0}
        actions = [0]* 6
        
        while not done:
          
          action = agent.select_action(obs, epsilon)
          next_obs, reward, done, _ = env.step(action)
          actions[action] += 1
          for i in range(1, 5):  # 站點索引從 1 到 4
            if (next_obs[0], next_obs[1]) == (next_obs[2 * i], next_obs[2 * i + 1]):
                visited_station[i] += 1
                if visited_station[i] == min(visited_station.values()):
                    reward += 10


          if next_obs[14] != 1:
            reward -= 2

          agent.buffer.push(obs, action, reward, next_obs, done)
          if len(agent.buffer.buffer) > agent.batchsize:
            agent.train()

          obs = next_obs
          total_reward += reward

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % update_time == 0:
          agent.update_target()
        if env.passenger_picked_up:
           pickup += 1
        # Print progress every 100 episodes.
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")
            print(f"actions: {actions}, pickup: {pickup}/{100}")
            actions = [0] * 6
            pickup = 0
    agent.save_model()
visited_states = set()


if __name__ == '__main__':
    config = get_config()
    train_agent(config)
