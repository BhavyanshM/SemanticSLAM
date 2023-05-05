import numpy as np
from plot_utils import *

from matplotlib.patches import Rectangle
import matplotlib as mpl

import os
os.environ['MPLBACKEND'] = 'Agg'

import sys
import select
import keyboard
import time
import math
import cv2

class Agent:
    def __init__(self, x, y, theta):
        self.pos = np.array([x, y])
        self.prev = np.zeros_like(self.pos)
        self.theta = theta

    def update_agent_pos(self, i, grid_size):
        self.prev = np.array([self.pos[0], self.pos[1]])

        
        self.pos[0] = (self.pos[0] + 3 * math.sin(i * 0.1))
        self.pos[1] = (self.pos[1] + 3 * math.cos(i * 0.1))

        # Clamp min to 0 and max to grid_size
        self.pos[0] = np.clip(self.pos[0], 0, grid_size)
        self.pos[1] = np.clip(self.pos[1], 0, grid_size)

    def random_update(self, i, grid_size):
        self.prev = np.array([self.pos[0], self.pos[1]])

        # Choose randomly between -1 or +1 to move in both axes
        self.pos[0] = (self.pos[0] + np.random.choice([-1, 1])) 
        self.pos[1] = (self.pos[1] + np.random.choice([-1, 1])) 

        # Clamp min to 0 and max to grid_size
        self.pos[0] = np.clip(self.pos[0], 0, grid_size)
        self.pos[1] = np.clip(self.pos[1], 0, grid_size)


# Plot the grid with the point and the obstacles
def plot_world(fig, grid, obstacles, agent, world_height_in_meters, world_width_in_meters):
    
        obstacle_size = 5

        # Set an area of 20 x 20 pixels around each obstacle as 100
        for obstacle in obstacles:
            grid[obstacle[0] - obstacle_size:obstacle[0] + obstacle_size, obstacle[1] - obstacle_size:obstacle[1] + obstacle_size] = 100

        # Set the agent's position as 50
        grid[int(agent.prev[0]), int(agent.prev[1])] = 0
        grid[int(agent.pos[0]), int(agent.pos[1])] = 50
    
        cmap = mpl.colors.ListedColormap(['blue','black','red'])
        bounds=[-6,-2,2,6]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # tell imshow about color map so that only set colors are used
        img = plt.imshow(grid,interpolation='nearest',
                            cmap = cmap,norm=norm)
    
# Use OpenCV cv2 to plot the same grid
def plot_world_cv(grid, obstacles, agent, world_height_in_meters, world_width_in_meters):
    
    obstacle_size = 5

    # Set an area of 20 x 20 pixels around each obstacle as 100
    for obstacle in obstacles:
        grid[obstacle[0] - obstacle_size:obstacle[0] + obstacle_size, obstacle[1] - obstacle_size:obstacle[1] + obstacle_size] = 100

    # Set the agent's position as 50
    grid[int(agent.prev[0]), int(agent.prev[1])] = 0
    grid[int(agent.pos[0]), int(agent.pos[1])] = 50

    # Plot the grid

    # Convert the floating point grid to 8-bit grayscale then convert it to RGB image
    grid = cv2.cvtColor(np.uint8(grid), cv2.COLOR_GRAY2RGB)

    cv2.imshow('Grid', grid)
    cv2.resizeWindow('Grid', 1600, 1600)
    code = cv2.waitKeyEx(30)

    return code


# Create an even loop that runs for 1000 iterations and plots the grid
def plot_world_loop(grid, obstacles, world_height_in_meters, world_width_in_meters):
    
    agent = Agent(20, 60, 0)

    # Define the physical size of the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    running = True
    i = 0
    while running:
        # ax.clear()

        code = plot_world_cv(grid, obstacles, agent, world_height_in_meters, world_width_in_meters)

        agent.random_update(i, world_height)

        print(agent.pos)

        # plt.pause(1 / 100)

        i += 1


        # print(code) 1048689
        if code != -1:
            running = False

        # # Check for user input
        # if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        #     input_char = sys.stdin.read(1)
        #     if input_char == 'q':
        #         running = False
    


if __name__ == "__main__":
    plotting_style = 'mplot'  # 'mplot' or 'sns'

    scale = 2
    world_height = 100
    world_width = 100

    world_height_in_meters = 4.0
    world_width_in_meters = 6.0

    grid  = np.zeros((world_height, world_width))

    obstacles = [(10, 20), (45, 61), (70, 82), (90, 43), (20, 34)]

    point = (2, 2)

    agent = (20,60)

    cv2.namedWindow('Grid', cv2.WINDOW_NORMAL)

    plot_world_loop(grid, obstacles, world_height_in_meters, world_width_in_meters)