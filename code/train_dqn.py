
from sumo_env import SumoRampEnv  #
from dqn_agent import DQNAgent  #
import traci
import numpy as np
from plot import plot_results




# Training Loop
if __name__ == "__main__":
    env = SumoRampEnv()
    agent = DQNAgent(env.state_size, env.action_size)
    episodes = 100
    batch_size = 64
    max_steps_per_episode = 1000

    rewards_per_episode = []  # Track total reward per episode
    avg_speed_per_episode = []  # Average speed per episode
    queue_length_per_episode = []  # Queue length per episode

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0 

        while not done and step_count < max_steps_per_episode:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1

        rewards_per_episode.append(total_reward)
        avg_speed_per_episode.append(traci.edge.getLastStepMeanSpeed("highway_entry"))
        queue_length_per_episode.append(traci.edge.getLastStepHaltingNumber("ramp_entry"))
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e % agent.target_update_freq == 0:
            agent.update_target_model()

        if e  > 30:
            agent.exploration_strategy = "adaptive_reset"
        agent.update_exploration(avg_reward= total_reward / (e +1),episode=e, max_episodes=episodes)
        print(f"Episode: {e+1}, Reward: {total_reward:.2f}, Avg Speed: {np.mean(avg_speed_per_episode):.2f}")
        print(f"Queue: {np.mean(queue_length_per_episode):.2f}, Epsilon: {agent.epsilon:.2f}") 

    # Plot results after training
    plot_results(episodes, rewards_per_episode, avg_speed_per_episode, queue_length_per_episode)

    # Save the trained model
    agent.model.save("dqn_model.h5")

    # Close the SUMO environment
    env.close()