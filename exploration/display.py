import cv2
import numpy as np
from plot_utils import *
from agent import *
from world import *

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
def plot_world_cv(world : World, agent : Agent, scale):
    
    for obstacle in world.obstacles:

        obstacle_size_x = obstacle[2]
        obstacle_size_y = obstacle[3]
        obstacle_min_x = obstacle[0] - obstacle_size_x
        obstacle_max_x = obstacle[0] + obstacle_size_x
        obstacle_min_y = obstacle[1] - obstacle_size_y
        obstacle_max_y = obstacle[1] + obstacle_size_y

        world.grid[ obstacle_min_x:obstacle_max_x, obstacle_min_y:obstacle_max_y] = 100

    # Display a yellow square on goal of total width goal_margin
    goal_margin = world.goal_margin
    goal_min_x = world.goal[0] - goal_margin
    goal_max_x = world.goal[0] + goal_margin
    goal_min_y = world.goal[1] - goal_margin
    goal_max_y = world.goal[1] + goal_margin
    world.grid[goal_min_x:goal_max_x, goal_min_y:goal_max_y] = 200

    # Convert the floating point grid to 8-bit grayscale then convert it to RGB image
    grid_color = cv2.cvtColor(np.uint8(world.grid), cv2.COLOR_GRAY2RGB)
    
    # Set the agent's position as 50
    grid_color[int(agent.prev[0]), int(agent.prev[1])] = np.array([0, 0, 0])
    grid_color[int(agent.pos[0]), int(agent.pos[1])] = np.array([0, 255, 255])

    grid_color[int(agent.average_state[0]), int(agent.average_state[1])] = np.array([100, 100, 255])

    # Plot lidar scan points as filled red cells
    for point in agent.scan_points:
        if int(point[0]) < grid_color.shape[0] and int(point[1]) < grid_color.shape[1] and int(point[0]) >= 0 and int(point[1]) >= 0:
            grid_color[int(point[0]), int(point[1])] = [0, 0, 255]
         


    cv2.imshow('Grid', grid_color)
    cv2.resizeWindow('Grid', scale, scale)
    code = cv2.waitKeyEx(1)

    return code
