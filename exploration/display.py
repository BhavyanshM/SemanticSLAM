import cv2
import numpy as np
from plot_utils import *
from agent import *

# Plot the grid with the point and the obstacles
def plot_world(fig, grid, obstacles, agent):
    
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
def plot_world_cv(grid, obstacles, agent, scale):
    
    for obstacle in obstacles:

        obstacle_size = obstacle[2]
        obstacle_min_x = obstacle[0] - obstacle_size
        obstacle_max_x = obstacle[0] + obstacle_size
        obstacle_min_y = obstacle[1] - obstacle_size
        obstacle_max_y = obstacle[1] + obstacle_size

        grid[ obstacle_min_x:obstacle_max_x, obstacle_min_y:obstacle_max_y] = 100

    # Set the agent's position as 50
    grid[int(agent.prev[0]), int(agent.prev[1])] = 0
    grid[int(agent.pos[0]), int(agent.pos[1])] = 50

    # Plot the grid

    # Convert the floating point grid to 8-bit grayscale then convert it to RGB image
    grid = cv2.cvtColor(np.uint8(grid), cv2.COLOR_GRAY2RGB)

    cv2.imshow('Grid', grid)
    cv2.resizeWindow('Grid', scale, scale)
    code = cv2.waitKeyEx(30)

    return code
