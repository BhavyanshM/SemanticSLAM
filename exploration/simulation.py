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
    cv2.resizeWindow('Grid', scale, scale)
    code = cv2.waitKeyEx(30)

    return code


# Create an even loop that runs for 1000 iterations and plots the grid
def run_simulation(grid, obstacles, agent : Agent, height, width):
    
    scale = 1400

    # Define the physical size of the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    running = True
    i = 0
    while running:
        # ax.clear()

        code = plot_world_cv(grid, obstacles, agent, scale)

        agent.random_update(i, height)

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