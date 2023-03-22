import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plotting_style = 'mplot'  # 'mplot' or 'sns'

scale = 2
world_height = 1600
world_width = 2048

world_height_in_meters = 4.0
world_width_in_meters = 6.0

x = 2.1
y = 3.2

xx, yy = np.meshgrid(np.arange(0, world_height_in_meters, world_height_in_meters / world_height),
                     np.arange(0, world_width_in_meters, world_width_in_meters / world_width))



distances = np.sqrt((xx - x)**2 + (yy - y)**2)

print(distances)

# normalized_distances = distances / np.max(distances)


if plotting_style == 'mplot':

    plt.figure(figsize=(20, 10))
    plt.imshow(distances, cmap='YlOrRd', extent=[0, world_width_in_meters, 0, world_height_in_meters])

    plt.colorbar()
    plt.show()

else:

    # Define the physical size of the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create the heatmap plot
    sns.heatmap(distances, cmap='YlOrRd')

    plt.show()