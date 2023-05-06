import numpy as np

from matplotlib.patches import Rectangle
import matplotlib as mpl

import os
os.environ['MPLBACKEND'] = 'Agg'

import cv2

from agent import *
from simulation import *    


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

    agent = Agent(20, 60, 0)

    run_simulation(grid, obstacles, agent, world_height, world_width)