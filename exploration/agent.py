import math
import numpy as np

class Agent:
    def __init__(self, x, y, theta):
        self.pos = np.array([x, y])
        self.prev = np.zeros_like(self.pos)
        self.theta = theta

    def random_update(self, grid_size):
        self.prev = np.array([self.pos[0], self.pos[1]])

        # Choose randomly between -1 or +1 to move in both axes
        self.pos[0] = (self.pos[0] + np.random.choice([-1, 1])) 
        self.pos[1] = (self.pos[1] + np.random.choice([-1, 1])) 

        # Clamp min to 0 and max to grid_size
        self.pos[0] = np.clip(self.pos[0], 0, grid_size)
        self.pos[1] = np.clip(self.pos[1], 0, grid_size)

    def apply_action(self, action):
        
        self.prev = np.array([self.pos[0], self.pos[1]])

        self.pos += action

        return self.pos
