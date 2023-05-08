import numpy as np

class World:
    def __init__(self, obstacles, goal, margin, height, width):
        self.obstacles = obstacles
        self.grid = grid  = np.zeros((height, width))
        self.grid_height = height
        self.grid_width = width
        self.goal = goal
        self.goal_margin = margin
        