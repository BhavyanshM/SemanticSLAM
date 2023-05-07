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


    def check_action_obstacles(self, action, obstacles):

        position = self.pos + action

        collision = False
        for obstacle in obstacles:

            obstacle_size = obstacle[2]
            obstacle_min_x = obstacle[0] - obstacle_size
            obstacle_max_x = obstacle[0] + obstacle_size
            obstacle_min_y = obstacle[1] - obstacle_size
            obstacle_max_y = obstacle[1] + obstacle_size

            if position[0] > obstacle_min_x and position[0] < obstacle_max_x:
                if position[1] > obstacle_min_y and position[1] < obstacle_max_y:
                    collision = True
                    break

        return not(collision)
                
    def check_action_boundaries(self, action, grid_size):

        position = self.pos + action

        if position[0] < 0 or position[0] > grid_size:
            return False
        if position[1] < 0 or position[1] > grid_size:
            return False

        return True
