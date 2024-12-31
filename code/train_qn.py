
# import pandas as pd
import matplotlib.pyplot as plt
from sumo_env import SumoRampEnv  #
from qn_agent import QLearningAgent  #
import traci


# Training Loop
if __name__ == "__main__":
    env = SumoRampEnv()
    agent = QLearningAgent(env.state_size, env.action_size)

    rewards_per_episode = []
    avg_speed_per_episode = []  # Average speed per episode
    queue_length_per_episode = []  # Queue length per episode
    episodes = 150
    batch_size = 64
    max_steps_per_episode = 1000

    for episode in range(episodes):
        # Reset environment and get initial state
        state = env.reset()
        total_reward = 0  # Track total reward for this episode

        for step in range(max_steps_per_episode):
            # Agent chooses an action
            action = agent.act(state)

            # Environment returns next_state, reward, and done flag
            next_state, reward, done = env.step(action)

            # Agent updates Q-value
            agent.update_q_value(state, action, reward, next_state)

            # Update state
            state = next_state

            # Accumulate reward
            total_reward += reward
            # End episode if done
            if done:
                break
        
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")
        # Decay epsilon (exploration rate)
        agent.decay_epsilon()

        # Store total reward for analysis
        rewards_per_episode.append(total_reward)
        avg_speed_per_episode.append(traci.edge.getLastStepMeanSpeed("highway_entry"))
        queue_length_per_episode.append(traci.edge.getLastStepHaltingNumber("ramp_entry"))

    # smoothed_rewards = pd.Series(rewards_per_episode).rolling(window=10).mean()
    plt.plot(rewards_per_episode)
    plt.title('Total Reward Across episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    # Close the SUMO environment
    env.close()
