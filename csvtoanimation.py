import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import os

# Load the saved positions from the CSV file
def load_positions_from_csv(positions_filename):
    positions_data = []
    with open(positions_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            if row:  # skip empty lines
                positions_data.append(np.array(row, dtype=float).reshape(-1, 2))
    return np.array(positions_data)

# Create the animation
def create_matplotlib_animation(directory, positions_filename, predator_filename):
    positions_data = load_positions_from_csv(f"{directory}/{positions_filename}")
    
    # Try to load predator data (if not empty)
    predator_data = []
    if os.path.getsize(f"{directory}/{predator_filename}") > 0:
        predator_data = load_positions_from_csv(f"{directory}/{predator_filename}")
    
    has_predators = len(predator_data) > 0

    # Create the plot
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-100, 200)
    ax.set_ylim(-100, 200)
    ax.scatter(17.09402801, -19.89900665, marker='+', c='black', s=100, label='Target')
    #ax.scatter(30.5171265, 29.4989665, marker='+', c='black', s=100, label='Target2')

    # Create the scatter plots
    scatter_animals = ax.scatter([], [], c='blue', label='Animals')
    scatter_predators = ax.scatter([], [], c='red', label='Predators') if has_predators else None

    ax.legend()
    def init():
        scatter_animals.set_offsets(np.empty((0, 2)))
        if scatter_predators:
            scatter_predators.set_offsets(np.empty((0, 2)))
        return (scatter_animals, scatter_predators) if scatter_predators else (scatter_animals,)

    def update(frame):
        scatter_animals.set_offsets(positions_data[frame])
        if scatter_predators:
            scatter_predators.set_offsets(predator_data[frame])
            return scatter_animals, scatter_predators
        return scatter_animals,

    ani = animation.FuncAnimation(
        fig, update, frames=positions_data.shape[0], init_func=init,
        blit=False, interval=50
    )
    ani.save(f'{directory}/animation.mp4', writer='ffmpeg', fps=30)
    #plt.show()

# Call the function to run the animation
subfolders = [ f.path for f in os.scandir("./simulations") if f.is_dir() ]
for subfolder in subfolders:
    predator_file = next((f for f in os.listdir(subfolder) if f.startswith("Pred")))
    prey_file = next((f for f in os.listdir(subfolder) if f.startswith("Prey")))
    print(subfolder)
    create_matplotlib_animation(f"{subfolder}", prey_file, predator_file)
