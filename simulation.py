"""
Agent-based simulation with Ring Attractor Networks
Implements allocentric and egocentric navigation with predator-prey dynamics
"""
import numpy as np
import csv
import json
import os
import time
import logging
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm
from ring_attractor import RingAttractor, AttractorParams


# Global configuration
SPEED_MULTIPLIER = 8


class Agent:
    """An autonomous agent controlled by a Ring Attractor Network."""
    
    def __init__(
        self,
        agent_id: int,
        position: np.ndarray,
        network: RingAttractor,
        is_allocentric: bool,
        steps_per_tick: int = 50,
        heading: float = 0.0,
        speed: float = 0.0
    ):
        """Initialize an agent.
        
        Args:
            agent_id: Unique identifier
            position: Initial 2D position [x, y]
            network: Ring attractor network controller
            is_allocentric: If True, uses world coordinates; if False, egocentric
            steps_per_tick: Number of network updates per simulation tick
            heading: Initial heading angle (radians)
            speed: Initial speed
        """
        self.id = agent_id
        self.position = np.array(position, dtype=np.float32)
        self.heading = heading
        self.speed = speed
        self.network = network
        self.is_allocentric = is_allocentric
        self.steps_per_tick = steps_per_tick
    
    def compute_target_angle(self, target_pos: np.ndarray) -> Tuple[float, float]:
        """Convert target position to angle relative to agent.
        
        Args:
            target_pos: Target position [x, y]
            
        Returns:
            (angle, distance) tuple
        """
        relative_pos = target_pos - self.position
        distance = np.linalg.norm(relative_pos)
        angle = np.arctan2(relative_pos[1], relative_pos[0])
        angle = (angle + 2 * np.pi) % (2 * np.pi)
        
        if not self.is_allocentric:
            # Convert to egocentric (relative to current heading)
            angle = (angle - self.heading) % (2 * np.pi)
        
        logging.debug(f"Agent {self.id}: target angle={angle:.2f}, distance={distance:.2f}")
        return angle, distance
    
    def add_target_influence(
        self,
        target_pos: np.ndarray,
        weight: float = 1.0,
        repulsion_distance: float = 0.0
    ):
        """Add a target's influence to the agent's network.
        
        Args:
            target_pos: Target position
            weight: Influence strength (negative for repulsion)
            repulsion_distance: Distance below which influence becomes repulsive
        """
        angle, distance = self.compute_target_angle(target_pos)
        
        # Reverse influence if too close
        if distance < repulsion_distance:
            weight = -abs(weight)
        
        self.network.set_target(angle, strength=weight)
    
    def update_movement(self):
        """Update heading and speed based on network output."""
        direction = self.network.get_direction()
        magnitude = np.linalg.norm(direction)
        
        if magnitude > 0:
            angle = np.arctan2(direction[1], direction[0])
            
            # Convert to absolute heading
            if self.is_allocentric:
                self.heading = angle
            else:
                self.heading = (self.heading + angle) % (2 * np.pi)
            
            self.speed = magnitude * SPEED_MULTIPLIER
        
        logging.debug(f"Agent {self.id}: heading={self.heading:.2f}, speed={self.speed:.2f}")
    
    def move(self):
        """Update position based on current heading and speed."""
        dx = self.speed * np.cos(self.heading)
        dy = self.speed * np.sin(self.heading)
        self.position += np.array([dx, dy], dtype=np.float32)
    
    def tick(self):
        """Execute one simulation tick: run network, update movement, move."""
        self.network.run(self.steps_per_tick)
        self.update_movement()
        self.move()


class AgentGroup:
    """Manages a group of interacting agents."""
    
    def __init__(
        self,
        visibility_radius: float,
        is_allocentric: bool,
        friendly_weight: float = 1.0,
        unfriendly_weight: float = -1.0,
        repulsion_distance: float = 0.5,
        target_weight: float = 1.0
    ):
        """Initialize agent group.
        
        Args:
            visibility_radius: Maximum interaction distance
            is_allocentric: Coordinate system for all agents
            friendly_weight: Attraction weight for same-group agents
            unfriendly_weight: Weight for predator interactions
            repulsion_distance: Distance for repulsion behavior
            target_weight: Weight for target attraction
        """
        self.agents: Dict[int, Agent] = {}
        self.positions: Dict[int, np.ndarray] = {}
        self.visibility_radius = visibility_radius
        self.is_allocentric = is_allocentric
        self.friendly_weight = friendly_weight
        self.unfriendly_weight = unfriendly_weight
        self.repulsion_distance = repulsion_distance
        self.target_weight = target_weight
    
    def add_agent(self, agent: Agent):
        """Add an agent to the group."""
        self.agents[agent.id] = agent
        self.positions[agent.id] = agent.position.copy()
    
    def get_nearby_positions(self, position: np.ndarray) -> List[np.ndarray]:
        """Find all agent positions within visibility radius.
        
        Args:
            position: Reference position
            
        Returns:
            List of nearby positions
        """
        nearby = []
        for pos in self.positions.values():
            distance = np.linalg.norm(pos - position)
            if 0 < distance <= self.visibility_radius:
                nearby.append(pos.copy())
        return nearby
    
    def tick_all(
        self,
        targets: List[np.ndarray],
        predator_positions: Optional[Dict[int, np.ndarray]] = None,
        target_weight: Optional[float] = None
    ):
        """Update all agents for one simulation tick.
        
        Args:
            targets: List of target positions
            predator_positions: Dict of predator positions (if any)
            target_weight: Override default target weight
        """
        if target_weight is None:
            target_weight = self.target_weight
        
        if predator_positions is None:
            predator_positions = {}
        
        # Update each agent
        for agent_id, agent in self.agents.items():
            agent.network.reset_target()
            
            # Add influences from nearby agents
            nearby = self.get_nearby_positions(self.positions[agent_id])
            for pos in nearby:
                agent.add_target_influence(
                    pos,
                    weight=self.friendly_weight,
                    repulsion_distance=self.repulsion_distance
                )
            
            # Add predator influences (always repulsive)
            for pred_pos in predator_positions.values():
                agent.add_target_influence(
                    pred_pos,
                    weight=self.unfriendly_weight,
                    repulsion_distance=0.0
                )
            
            # Add target influences
            for target_pos in targets:
                agent.add_target_influence(
                    target_pos,
                    weight=target_weight,
                    repulsion_distance=0.0
                )
            
            # Execute agent tick
            agent.tick()
            self.positions[agent_id] = agent.position.copy()
    
    def get_all_positions(self) -> Dict[int, np.ndarray]:
        """Get current positions of all agents."""
        return self.positions.copy()
    
    def __len__(self):
        return len(self.agents)
    
    def __repr__(self):
        return f"AgentGroup({len(self)} agents, allocentric={self.is_allocentric})"


def create_agent_group(
    num_agents: int,
    steps_per_tick: int,
    is_allocentric: bool,
    visibility_radius: float,
    beta: float,
    friendly_weight: float = 1.0,
    unfriendly_weight: float = -1.0,
    repulsion_distance: float = 0.5,
    target_weight: float = 1.0,
    spawn_area: Tuple[float, float] = (-10, 10)
) -> AgentGroup:
    """Factory function to create a group of agents with identical parameters.
    
    Args:
        num_agents: Number of agents to create
        steps_per_tick: Network steps per simulation tick
        is_allocentric: Use allocentric coordinates
        visibility_radius: Interaction radius
        beta: Network inverse temperature
        friendly_weight: Same-group attraction
        unfriendly_weight: Predator repulsion
        repulsion_distance: Distance for repulsion
        target_weight: Target attraction
        spawn_area: (min, max) for random spawn positions
        
    Returns:
        AgentGroup with initialized agents
    """
    group = AgentGroup(
        visibility_radius=visibility_radius,
        is_allocentric=is_allocentric,
        friendly_weight=friendly_weight,
        unfriendly_weight=unfriendly_weight,
        repulsion_distance=repulsion_distance,
        target_weight=target_weight
    )
    
    num_neurons = 100
    for i in range(num_agents):
        # Create network
        params = AttractorParams(
            N=num_neurons,
            beta=beta,
            nu_exp=1.0,
            sigma=2 * np.pi / num_neurons * 5,
            h0=0.25
        )
        network = RingAttractor(params)
        
        # Random initial conditions
        position = np.random.uniform(spawn_area[0], spawn_area[1], size=2)
        heading = np.random.uniform(0, 2 * np.pi)
        
        # Create agent
        agent = Agent(
            agent_id=i,
            position=position,
            network=network,
            is_allocentric=is_allocentric,
            steps_per_tick=steps_per_tick,
            heading=heading
        )
        group.add_agent(agent)
    
    return group


def generate_filename(
    agent_type: str,
    info_type: str,
    steps: int,
    beta: float,
    is_allocentric: bool,
    extension: str
) -> str:
    """Generate a structured filename for simulation data.
    
    Args:
        agent_type: 'Prey' or 'Pred'
        info_type: 'P' (position) or 'H&B' (heading & speed)
        steps: Steps per tick
        beta: Network beta parameter
        is_allocentric: Coordinate system
        extension: File extension (csv, json)
        
    Returns:
        Formatted filename
    """
    return f"{agent_type}_{info_type}_b={beta}_steps={steps}_allo={is_allocentric}.{extension}"


def run_simulation(
    num_ticks: int,
    targets: List[np.ndarray],
    prey_group: AgentGroup,
    predator_group: Optional[AgentGroup] = None,
    output_dir: Optional[str] = None,
    show_progress: bool = True
) -> Tuple[List, List, List]:
    """Run the full simulation and save results.
    
    Args:
        num_ticks: Number of simulation ticks
        targets: List of target positions
        prey_group: Group of prey agents
        predator_group: Optional group of predator agents
        output_dir: Directory to save results (auto-generated if None)
        show_progress: Show progress bar
        
    Returns:
        (prey_positions, predator_positions, brain_states)
    """
    # Create output directory
    if output_dir is None:
        timestamp = str(int(time.time() * 1000))
        output_dir = os.path.join("simulations", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Storage for trajectories
    prey_trajectories = []
    predator_trajectories = []
    brain_states_log = []
    
    # Main simulation loop
    iterator = tqdm(range(num_ticks)) if show_progress else range(num_ticks)
    for tick in iterator:
        # Get predator positions if they exist
        pred_positions = predator_group.get_all_positions() if predator_group else {}
        
        # Update prey
        prey_group.tick_all(targets, pred_positions)
        prey_pos_array = np.array(list(prey_group.get_all_positions().values()))
        prey_trajectories.append(prey_pos_array)
        
        # Update predators (target = prey positions)
        if predator_group:
            prey_positions_list = list(prey_group.get_all_positions().values())
            predator_group.tick_all(prey_positions_list)
            pred_pos_array = np.array(list(predator_group.get_all_positions().values()))
            predator_trajectories.append(pred_pos_array)
        
        # Log brain states
        for agent_id, agent in prey_group.agents.items():
            state_row = [tick, agent_id] + agent.network.state.tolist()
            brain_states_log.append(state_row)
        
        if show_progress:
            iterator.set_description(f"Tick {tick}/{num_ticks}")
    
    # Save results
    _save_positions(prey_trajectories, os.path.join(output_dir, "prey_positions.csv"))
    if predator_trajectories:
        _save_positions(predator_trajectories, os.path.join(output_dir, "predator_positions.csv"))
    _save_brain_states(brain_states_log, os.path.join(output_dir, "brain_states.csv"), prey_group)
    _save_targets(targets, os.path.join(output_dir, "targets.txt"))
    
    print(f"Simulation complete. Results saved to: {output_dir}")
    return prey_trajectories, predator_trajectories, brain_states_log


def _save_positions(trajectories: List[np.ndarray], filepath: str):
    """Save position trajectories to CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        if trajectories:
            num_agents = trajectories[0].shape[0]
            header = [f"Agent_{i+1}_{axis}" for i in range(num_agents) for axis in ['x', 'y']]
            writer.writerow(header)
        
        # Write data
        for tick_positions in trajectories:
            writer.writerow(tick_positions.flatten())


def _save_brain_states(states: List, filepath: str, group: AgentGroup):
    """Save brain state time series to CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        num_neurons = len(list(group.agents.values())[0].network.state)
        header = ['tick', 'agent_id'] + [f'neuron_{i}' for i in range(num_neurons)]
        writer.writerow(header)
        
        # Data
        writer.writerows(states)


def _save_targets(targets: List[np.ndarray], filepath: str):
    """Save target positions to text file."""
    with open(filepath, 'w') as f:
        f.write("Target positions:\n")
        for i, target in enumerate(targets):
            f.write(f"  Target {i+1}: {target}\n")