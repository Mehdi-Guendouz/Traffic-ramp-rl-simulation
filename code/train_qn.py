
# import pandas as pd
import matplotlib.pyplot as plt
from sumo_env import SumoRampEnv  #
from qn_agent import QLearningAgent  #

def plot_results(episodes, rewards, avg_speeds, queue_lengths):
    # Plot Total Rewards
    plt.figure(figsize=(12, 6))
    plt.plot(range(episodes), rewards, label="Total Rewards", color='b')
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Total Reward Across Episodes")
    plt.legend()
    plt.show()
    
    # Plot Average Speeds
    plt.figure(figsize=(12, 6))
    plt.plot(range(episodes), avg_speeds, label="Average Speed (m/s)", color='g')
    plt.xlabel("Episodes")
    plt.ylabel("Average Speed (m/s)")
    plt.title("Average Speed Across Episodes")
    plt.legend()
    plt.show()
    
    # Plot Queue Lengths
    plt.figure(figsize=(12, 6))
    plt.plot(range(episodes), queue_lengths, label="Average Queue Length", color='r')
    plt.xlabel("Episodes")
    plt.ylabel("Queue Length (vehicles)")
    plt.title("Queue Length Across Episodes")
    plt.legend()
    plt.show()



# Training Loop
if __name__ == "__main__":
    env = SumoRampEnv()
    agent = QLearningAgent(env.state_size, env.action_size)

    rewards_per_episode = []
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

        # Logging for every 50 episodes
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

    # smoothed_rewards = pd.Series(rewards_per_episode).rolling(window=10).mean()
    plt.plot(rewards_per_episode)
    plt.title('Total Reward Across episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
    # plt.plot(smoothed_rewards)
    # plt.title('Total Reward Across episodes')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.show()


    # Plot results after training
    plot_results(episodes, rewards_per_episode)

    # Save the trained model
    agent.model.save("qn_model.h5")

    # Close the SUMO environment
    env.close()
