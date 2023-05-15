import numpy as np

class World:
    def __init__(self, obstacles, goal, margin, height, width):
        self.obstacles = obstacles
        self.grid = grid  = np.zeros((height+1, width+1))
        self.grid_height = height
        self.grid_width = width
        self.goal = goal
        self.goal_margin = margin
        self.grid = grid

    def update_grid(self, agent_state, radius):
        
        # set a circle of pixels around the agent to be 50
        agent_min_x = agent_state[0] - radius
        agent_max_x = agent_state[0] + radius
        agent_min_y = agent_state[1] - radius
        agent_max_y = agent_state[1] + radius

        for x in range(agent_min_x, agent_max_x):
            for y in range(agent_min_y, agent_max_y):
                # if point is within 5 pixels circular radius
                if np.linalg.norm(np.array([x, y]) - agent_state) < radius:
                    # check if inside the world boundaries
                    if x >= 0 and x <= self.grid_width and y >= 0 and y <= self.grid_height:
                        self.grid[x][y] = 50
        