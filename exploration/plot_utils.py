import seaborn as sns
import matplotlib.pyplot as plt

def plot(plotting_style, distances, world_height_in_meters, world_width_in_meters):
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