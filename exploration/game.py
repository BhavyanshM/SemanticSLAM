import numpy as np

from matplotlib.patches import Rectangle
import matplotlib as mpl

import os
os.environ['MPLBACKEND'] = 'Agg'

import cv2

from agent import *
from display import *    
from monte_carlo_planner import *
from world import *

# Create an even loop that runs for 1000 iterations and plots the grid
def run_simulation(mcp : MonteCarloPlanner, world : World, agent : Agent, iterations):
    
    scale = 1400

    # Define the physical size of the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    running = True
    i = 0
    while running:
        # ax.clear()
        agent.measure(world.obstacles)

        code = plot_world_cv(world, agent, scale)

        # print(agent.scan_points)

        new_state, action_values = mcp.plan()

        print("New State: {}, Action Values: {}".format(new_state, action_values))

        agent.update_state(new_state)

        world.update_grid(new_state, agent.max_range)

        i += 1

        # print(code) 1048689
        if code != -1:
            running = False

if __name__ == "__main__":
    plotting_style = 'mplot'  # 'mplot' or 'sns'

    scale = 2
    world_height = 100
    world_width = 100

    iterations = 10
    simulation_count = 10

    max_range = 20

    world_height_in_meters = 4.0
    world_width_in_meters = 6.0

    # obstacles = [(10, 20, 5, 5), (45, 61, 8, 8), (70, 82, 10, 8), (90, 43, 3, 3), (20, 34, 6, 6)]

    obstacles = [(30, 10, 1, 10), (30, 28, 1, 2), (15, 30, 15, 1), 
                 (60, 30, 1, 30), (10, 60, 10, 1), (58, 60, 2, 1),
                 (80, 40, 1, 40), (30, 80, 30, 1), (75, 80, 5, 1)]

    point = (2, 2)

    agent = (20,60)

    cv2.namedWindow('Grid', cv2.WINDOW_NORMAL)

    goal = np.array([10, world_height - 10])
    goal_margin = 5

    agent = Agent(0, 0, 0, 20)
    world = World(obstacles, goal, goal_margin, world_height, world_width)
    planner = MonteCarloPlanner(world, agent, iterations, simulation_count)

    run_simulation(planner, world, agent, iterations)