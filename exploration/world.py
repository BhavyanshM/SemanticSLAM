import numpy as np

class World:
    def __init__(self, obstacles, height, width):
        self.obstacles = obstacles
        self.grid = grid  = np.zeros((height, width))
        self.grid_height = height
        self.grid_width = width
        self.goal = np.array([width - 1, height - 1])