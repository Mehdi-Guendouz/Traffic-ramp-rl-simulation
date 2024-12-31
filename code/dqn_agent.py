import numpy as np
import tensorflow as tf
from collections import deque
import random



class DQNAgent:
    def __init__(self, state_size, action_size, exploration_strategy="exponential_decay", reset_period=10):
        self.state_size = state_size   # Size of the state vector
        self.action_size = action_size  # Number of possible actions
        self.memory = deque(maxlen=30000)  # Experience replay memory
        
        # Hyperparameters
        self.gamma = 0.95              # Discount factor
        self.epsilon = 1.5             # Exploration rate
        self.epsilon_decay = 0.999      # Decay rate for epsilon
        self.epsilon_min = 0.01         # Minimum exploration rate
        self.learning_rate = 0.0001   # Learning rate for the neural network
        
        # Build models
        self.model = self.build_model()           # Primary Q-network
        self.target_model = self.build_model()    # Target Q-network
        self.target_update_freq = 10          # Update target model every 10 episodes
        self.episode_count = 0
        self.update_target_model()                # Initialize target weights

        self.exploration_strategy = exploration_strategy  # Exploration strategy
        self.reset_period = reset_period

    # The neural network will approximate the Q-values for each action given a state.
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='huber')
        return model

    
    # The agent will store experiences (state, action, reward, next_state, done) in a deque.
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Use an ε-greedy policy for exploration vs exploitation.
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1) # random action
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)  # Exploit: choose best action
        return np.argmax(q_values[0])


    # Sample a minibatch from memory and train the Q-network.
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return  # Not enough experiences to sample
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.target_model.predict(next_state[np.newaxis, :], verbose=0))
            
            # Update the Q-value for the taken action
            target_f = self.model.predict(state[np.newaxis, :], verbose=0)
            target_f[0][action] = target
            
            # Train the model
            self.model.fit(state[np.newaxis, :], target_f, epochs=1, verbose=0)



    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Update the exploration rate based on the chosen strategy.
    def update_exploration(self, avg_reward=None, episode=None, max_episodes=None):
        self.episode_count += 1
      # Minimum epsilon value (stop exploring eventually)
        epsilon_min = 0.01  

        # Exploration decay rate
        decay_rate = 0.999  

        if self.exploration_strategy == "exponential_decay":
            # Exponential decay of epsilon
            self.epsilon = max(
                epsilon_min, 
                self.epsilon * decay_rate * (1 - (episode / max_episodes))
            )

        elif self.exploration_strategy == "adaptive_reset":
            # Periodic reset of epsilon based on reward performance
            if self.episode_count % self.reset_period == 0 and avg_reward is not None:
                if avg_reward < 0:  # Poor performance, increase exploration
                    self.epsilon = 0.7
                else:  # Good performance, reduce exploration
                    self.epsilon = max(
                        epsilon_min, 
                        self.epsilon * decay_rate * (1 - (episode / max_episodes))
                    )

        