"""
Visualization tools for simulation results
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import os
from typing import List, Optional


def load_positions_csv(filepath: str) -> np.ndarray:
    """Load position trajectories from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Array of shape (num_ticks, num_agents, 2)
    """
    positions = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            if row:
                positions.append(np.array(row, dtype=float).reshape(-1, 2))
    
    return np.array(positions)


def load_targets(filepath: str) -> List[np.ndarray]:
    """Load target positions from text file.
    
    Args:
        filepath: Path to targets file
        
    Returns:
        List of target positions
    """
    targets = []
    with open(filepath, 'r') as f:
        for line in f:
            if 'Target' in line and ':' in line:
                # Extract array from line like "Target 1: [x y]"
                array_str = line.split(':')[1].strip()
                coords = array_str.strip('[]').split()
                targets.append(np.array([float(coords[0]), float(coords[1])]))
    return targets


def animate_simulation(
    simulation_dir: str,
    prey_filename: str = "prey_positions.csv",
    predator_filename: str = "predator_positions.csv",
    targets_filename: str = "targets.txt",
    output_filename: str = "animation.mp4",
    show_targets: bool = True,
    xlim: tuple = (-100, 200),
    ylim: tuple = (-100, 200),
    fps: int = 30,
    interval: int = 50
):
    """Create an animation from simulation results.
    
    Args:
        simulation_dir: Directory containing simulation results
        prey_filename: Name of prey positions CSV
        predator_filename: Name of predator positions CSV
        targets_filename: Name of targets file
        output_filename: Output video filename
        show_targets: Whether to show targets
        xlim: X-axis limits
        ylim: Y-axis limits
        fps: Frames per second for output video
        interval: Milliseconds between frames
    """
    # Load data
    prey_positions = load_positions_csv(os.path.join(simulation_dir, prey_filename))
    
    predator_positions = None
    pred_path = os.path.join(simulation_dir, predator_filename)
    if os.path.exists(pred_path) and os.path.getsize(pred_path) > 0:
        predator_positions = load_positions_csv(pred_path)
    
    targets = []
    target_path = os.path.join(simulation_dir, targets_filename)
    if show_targets and os.path.exists(target_path):
        targets = load_targets(target_path)
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Agent Simulation')
    ax.grid(True, alpha=0.3)
    
    # Plot targets
    if targets:
        for i, target in enumerate(targets):
            ax.scatter(
                target[0], target[1],
                marker='*',
                c='gold',
                s=300,
                edgecolors='black',
                linewidths=2,
                label=f'Target {i+1}' if i == 0 else None,
                zorder=3
            )
    
    # Initialize scatter plots
    prey_scatter = ax.scatter([], [], c='blue', s=50, label='Prey', zorder=2)
    predator_scatter = None
    if predator_positions is not None:
        predator_scatter = ax.scatter([], [], c='red', s=100, marker='^', 
                                     label='Predators', zorder=2)
    
    ax.legend(loc='upper right')
    
    def init():
        """Initialize animation."""
        prey_scatter.set_offsets(np.empty((0, 2)))
        if predator_scatter:
            predator_scatter.set_offsets(np.empty((0, 2)))
        return (prey_scatter, predator_scatter) if predator_scatter else (prey_scatter,)
    
    def update(frame):
        """Update animation frame."""
        prey_scatter.set_offsets(prey_positions[frame])
        
        if predator_scatter and predator_positions is not None:
            predator_scatter.set_offsets(predator_positions[frame])
            return prey_scatter, predator_scatter
        
        return prey_scatter,
    
    # Create animation
    num_frames = prey_positions.shape[0]
    ani = animation.FuncAnimation(
        fig, update,
        frames=num_frames,
        init_func=init,
        blit=True,
        interval=interval
    )
    
    # Save
    output_path = os.path.join(simulation_dir, output_filename)
    ani.save(output_path, writer='ffmpeg', fps=fps)
    plt.close()
    
    print(f"Animation saved to: {output_path}")


def plot_ring_attractor(state: np.ndarray, title: str = "Ring Attractor State"):
    """Plot the current state of a ring attractor network.
    
    Args:
        state: Array of neuron states (+1 or -1)
        title: Plot title
    """
    n = len(state)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_facecolor('lightgray')
    
    # Plot neurons
    for xi, yi, neuron_state in zip(x, y, state):
        color = 'white' if neuron_state == 1 else 'black'
        circle = plt.Circle((xi, yi), 0.05, color=color, ec='gray', linewidth=1)
        ax.add_patch(circle)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def animate_ring_attractor(states: List[np.ndarray], interval: int = 200):
    """Animate the evolution of a ring attractor network.
    
    Args:
        states: List of network states over time
        interval: Milliseconds between frames
    """
    n = len(states[0])
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_facecolor('lightgray')
    ax.axis('off')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Create circles for each neuron
    circles = []
    for xi, yi in zip(x, y):
        circle = plt.Circle((xi, yi), 0.05, color='gray', ec='black', linewidth=1)
        ax.add_patch(circle)
        circles.append(circle)
    
    title_text = ax.text(0, 1.1, '', ha='center', va='center', fontsize=12)
    
    def update(frame):
        """Update animation frame."""
        state = states[frame]
        for i, neuron_state in enumerate(state):
            color = 'white' if neuron_state == 1 else 'black'
            circles[i].set_facecolor(color)
        title_text.set_text(f"Tick {frame}/{len(states)}")
        return circles + [title_text]
    
    ani = animation.FuncAnimation(
        fig, update,
        frames=len(states),
        interval=interval,
        blit=True
    )
    
    plt.show()


def batch_animate_simulations(simulations_dir: str = "simulations"):
    """Create animations for all simulation subdirectories.
    
    Args:
        simulations_dir: Parent directory containing simulation folders
    """
    if not os.path.exists(simulations_dir):
        print(f"Directory not found: {simulations_dir}")
        return
    
    subdirs = [f.path for f in os.scandir(simulations_dir) if f.is_dir()]
    
    if not subdirs:
        print(f"No simulation subdirectories found in {simulations_dir}")
        return
    
    print(f"Found {len(subdirs)} simulation(s) to animate\n")
    
    for subdir in subdirs:
        print(f"Processing: {subdir}")
        try:
            animate_simulation(subdir)
        except Exception as e:
            print(f"  Error: {e}")
        print()


if __name__ == "__main__":
    # Animate all simulations in the simulations directory
    batch_animate_simulations()