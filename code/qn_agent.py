

import os
import traci
import random
import numpy as np
from collections import deque

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.replay_buffer = deque(maxlen=1000)  # Add to class initialization

    def discretize_state(self, state):
        state = np.array(state)  # Convert the tuple to a numpy array
        normalized_state = (state - state.mean()) / state.std()  # Normalize
        return tuple(np.round(normalized_state, decimals=2))  # Discretize


    def get_q_value(self, state, action):
        state = self.discretize_state(state)
        return self.q_table.get((state, action), 0.0)

    def ensure_q_entry(self, state, action):
        state = self.discretize_state(state)
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0.0

    def update_q_value(self, state, action, reward, next_state):
        state = self.discretize_state(state)
        next_state = self.discretize_state(next_state)

        max_next_q = max([self.get_q_value(next_state, a) for a in range(self.action_size)])
        current_q = self.get_q_value(state, action)
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_next_q)
        self.q_table[(state, action)] = new_q

    def act(self, state):
        state = self.discretize_state(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)  # Explore
        return np.argmax([self.get_q_value(state, a) for a in range(self.action_size)])  # Exploit

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


    def reset_agent(self):
        self.q_table = {}
        self.epsilon = 1.0

    def remember(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))

    def replay(self):
        for state, action, reward, next_state in random.sample(self.replay_buffer, min(len(self.replay_buffer), batch_size)):
            self.update_q_value(state, action, reward, next_state)
