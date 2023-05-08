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
def run_simulation(mcp : MonteCarloPlanner, world, agent : Agent, iterations):
    
    scale = 1400

    # Define the physical size of the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    running = True
    i = 0
    while running:
        # ax.clear()

        code = plot_world_cv(world, agent, scale)

        new_state, action_values = mcp.plan(iterations)

        print("New State: {}, Action Values: {}".format(new_state, action_values))

        agent.update_state(new_state)

        i += 1

        # print(code) 1048689
        if code != -1:
            running = False

if __name__ == "__main__":
    plotting_style = 'mplot'  # 'mplot' or 'sns'

    scale = 2
    world_height = 100
    world_width = 100

    simulation_iterations = 100

    world_height_in_meters = 4.0
    world_width_in_meters = 6.0

    obstacles = [(10, 20, 5), (45, 61, 8), (70, 82, 10), (90, 43, 3), (20, 34, 6)]

    point = (2, 2)

    agent = (20,60)

    cv2.namedWindow('Grid', cv2.WINDOW_NORMAL)

    goal = np.array([world_width - 1, world_height - 1])
    goal_margin = 10

    agent = Agent(0, 0, 0)
    world = World(obstacles, goal, goal_margin, world_height, world_width)
    planner = MonteCarloPlanner(world, agent)

    run_simulation(planner, world, agent, simulation_iterations)