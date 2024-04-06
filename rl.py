# rl_agent.py

import numpy as np

class RLAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_rate=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.q_table = np.zeros((state_size, action_size))
    
    def decide(self, state):
        """Make a decision based on the current state."""
        action = np.argmax(self.q_table[state])
        return action
    
    def learn(self, state, action, reward, next_state):
        """Update Q-table based on the action taken and the reward received."""
        q_update = reward + self.discount_rate * np.max(self.q_table[next_state]) - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * q_update
