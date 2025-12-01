"""
Ring Attractor Network Implementation
Based on Salahshour & Couzin (2025) - Allocentric flocking
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class AttractorParams:
    """Parameters for the Ring Attractor Network.
    
    Attributes:
        N: Number of neurons in the ring
        beta: Inverse temperature (controls stochasticity)
        nu_exp: Exponent for connection function
        h0: External target influence strength (eq. 3)
        sigma: Width of the Gaussian field
        hb: Baseline external field
    """
    N: int
    beta: float
    nu_exp: float
    h0: float = 0.0025
    sigma: float = 1.0
    hb: float = 0.0


class RingAttractor:
    """Ring Attractor Network for directional decision-making."""
    
    def __init__(self, params: AttractorParams):
        """Initialize the network with given parameters."""
        self.params = params
        
        # Initialize random spin states (+1 or -1)
        self.state = np.random.randint(0, 2, params.N) * 2 - 1
        
        # Create direction vectors for each neuron
        self.angles = np.linspace(0, 2 * np.pi, params.N, endpoint=False)
        self.directions = np.array([np.cos(self.angles), np.sin(self.angles)])
        
        # Initialize connection matrix and external field
        self.J = self._compute_connections(self.angles, params.nu_exp)
        self.h = np.zeros(params.N)
        self.hb = params.hb
        
    def _compute_connections(self, angles: np.ndarray, nu: float) -> np.ndarray:
        """Compute connection strengths between neurons (eq. 1).
        
        Args:
            angles: Array of neuron angles
            nu: Exponent parameter
            
        Returns:
            N×N connection matrix
        """
        angle_diff = np.abs(angles[:, np.newaxis] - angles)
        return np.cos(np.pi * (angle_diff / np.pi) ** nu)
    
    def glauber_step(self):
        """Update a single random neuron using Glauber dynamics (eq. 4)."""
        i = np.random.randint(0, self.params.N)
        
        # Calculate energy change from flipping neuron i
        interaction_term = 4 * self.state[i] * (
            (self.J[i, :] @ self.state) - self.state[i]
        ) / self.params.N
        field_term = 2 * (self.h[i] - self.hb) * self.state[i]
        dE = field_term + interaction_term
        
        # Flip with temperature-dependent probability
        flip_prob = 1 / (1 + np.exp(self.params.beta * dE))
        if np.random.rand() < flip_prob:
            self.state[i] *= -1
    
    def step(self):
        """Perform one full network update (N neuron updates)."""
        for _ in range(self.params.N):
            self.glauber_step()
    
    def run(self, steps: int):
        """Run the network for multiple steps.
        
        Args:
            steps: Number of full updates to perform
        """
        for _ in range(steps):
            self.step()
    
    def reset_target(self):
        """Clear all external field influences."""
        self.h = np.zeros(self.params.N)
    
    def set_target(self, theta: float, strength: float = 1.0):
        """Add a directional target influence to the network (eq. 2-3).
        
        Args:
            theta: Target angle in radians
            strength: Influence strength multiplier
        """
        # Compute periodic angle difference
        angle_diff = (self.angles - theta + 3 * np.pi) % (2 * np.pi) - np.pi
        
        # Add Gaussian field centered at target angle
        gaussian_field = np.exp(-0.5 * angle_diff**2 / self.params.sigma**2)
        normalization = np.sqrt(2 * np.pi * self.params.sigma**2)
        self.h += self.params.h0 * strength * normalization * gaussian_field
    
    def get_direction(self) -> np.ndarray:
        """Compute the network's output direction vector.
        
        Returns:
            2D unit direction vector [x, y]
        """
        return self.directions @ self.state / self.params.N


# Example usage
if __name__ == "__main__":
    # Create network
    params = AttractorParams(N=100, beta=400, nu_exp=1.0, sigma=2*np.pi/100)
    network = RingAttractor(params)
    
    # Set target and run
    network.set_target(0)  # Target at 0 radians (pointing right)
    network.run(200)
    
    # Get output
    direction = network.get_direction()
    print(f"Output direction: {direction}")
    print(f"Final state (first 10): {network.state[:10]}")