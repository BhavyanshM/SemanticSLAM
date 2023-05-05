import pygame
import time

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Set the width and height of the screen [width, height]
WIDTH = 500
HEIGHT = 500
SCREEN_SIZE = (WIDTH, HEIGHT)

# Set the refresh rate (in Hz)
REFRESH_RATE = 100

# Define the size of the grid world
GRID_SIZE = 100
CELL_SIZE = 20

# Initialize Pygame
pygame.init()

# Set the screen size
screen = pygame.display.set_mode(SCREEN_SIZE)

# Set the title of the window
pygame.display.set_caption("2D Grid World")

# Set the font for displaying the agent's position
font = pygame.font.SysFont(None, 25)

# Set the initial position of the agent
agent_pos = [0, 0]

# Define a function to draw the grid world
def draw_grid_world():
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            # Calculate the position of the cell
            x = i * CELL_SIZE
            y = j * CELL_SIZE

            # Draw the cell
            pygame.draw.rect(screen, WHITE, [x, y, CELL_SIZE, CELL_SIZE], 1)

            # Draw the agent if it is in this cell
            if agent_pos == [i, j]:
                pygame.draw.circle(screen, BLACK, [x + CELL_SIZE // 2, y + CELL_SIZE // 2], CELL_SIZE // 4)

# Define a function to update the agent's position
def update_agent_pos():
    # Update the agent's position randomly
    agent_pos[0] = (agent_pos[0] + 1) % GRID_SIZE
    agent_pos[1] = (agent_pos[1] + 1) % GRID_SIZE

# Set the clock for controlling the refresh rate
clock = pygame.time.Clock()

# Set the loop for updating the agent's position and refreshing the screen
running = True
while running:
    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the agent's position
    update_agent_pos()

    # Clear the screen
    screen.fill(BLACK)

    # Draw the grid world
    draw_grid_world()

    # Draw the agent's position as a circle
    pygame.draw.circle(screen, BLACK, [agent_pos[0] * CELL_SIZE + CELL_SIZE // 2, agent_pos[1] * CELL_SIZE + CELL_SIZE // 2], CELL_SIZE // 4)

    
    

    # Refresh the screen
    pygame.display.flip()

    # Control the refresh rate
    clock.tick(REFRESH_RATE)

# Quit Pygame
pygame.quit()