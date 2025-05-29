import numpy as np # useful library for fast math operations and for arrays (vectors and matrices) and operations on them
import matplotlib.pyplot as plt # nice simple plotting library
from dataclasses import dataclass # just helper to make code cleaner


@dataclass
class AttractorParams:
    '''
    Attributes:
        N: int - number of neurons
        beta: float - inverse temperature
        nu_exp: float - exponent to_calculate the connection functions
        h0: float - how strongly the external target influences the network (eq 3 from paper)
        hb: float - external field (idk what it really does)
    '''
    N: int
    beta: float
    nu_exp: float
    h0: float = 0.0025
    sigma: float = 1.0
    hb: float = 0.0


class RingAttractor:
    def __init__(self, params: AttractorParams): # function that sets up all the necessary variables when you create a new object (a network for a specific animal for example)
        '''
        1) We initialize the state of the network to some random values of +1 and -1
        2) We initialize direction vectors - each of them points from the center of a circle to one neuron. They are used to calculate the direction of movement.
        3) We initialize the connections between neurons - the J matrix (N by N).
        4) We initialize the field h as zero - we don't have any targets to reach yet.
        '''
        self.params = params
        self.state = np.random.randint(0, 1, params.N) * 2 - 1
        self.angles = np.linspace(0, 2 * np.pi, params.N, endpoint=False)
        self.directions = np.array([np.cos(self.angles), np.sin(self.angles)])
        self.J = get_connections(self.angles, params.nu_exp)
        self.h = np.zeros(params.N)
        self.hb = params.hb
        
    def glauber_step(self):
        '''Update a single random spin of the network
        1) we chose "i" - a random neuron
        2) we calculate dE - the change in energy resulting from hypothetically flipping this one neuron (equation 4 from the paper, optimized to avoid unnecessary calculations)
        3) we flip the neuron with probability 1/(1 + exp(beta * dE))
        '''
        i = np.random.randint(0, self.params.N)
        dE = 2 * (self.h[i] - self.hb) * self.state[i] + 4*self.state[i] * ((self.J[i, :] @ self.state) - self.state[i]) / self.params.N
        if np.random.rand() < 1/(1 + np.exp(self.params.beta * dE)): 
            self.state[i] *= -1

    def step(self):
        '''Update the whole network once (N spin updates)
        '''
        for _ in range(self.params.N):
            self.glauber_step()

    def run(self, steps: int):
        '''Run the network for a given number of steps
        '''
        for _ in range(steps):
            self.step()

    def reset_target(self):
        self.h = np.zeros(self.params.N) # reset the target to zero

    def set_target(self, theta: float, strength: float = 1.0):
        '''Set h field to point towards a given target (with angle theta)
        (equation 2, 3 from the paper)
        The angle difference is modified to work periodically for any angle.
        '''
        angles_diff = (self.angles - theta + 3*np.pi) % (2*np.pi) - np.pi
        self.h += self.params.h0 * strength * np.sqrt(2 * np.pi * self.params.sigma**2) * np.exp(-0.5 * angles_diff**2 / self.params.sigma**2)

    def get_direction(self) -> np.ndarray:
        '''Return the average resulting direction of the network (used for movement later)
        '''
        return self.directions @ self.state / self.params.N


def get_connections(angles: np.ndarray, nu: float) -> np.ndarray:
    '''Calculate the connections between neurons (equation 1 from the paper)
    '''
    return np.cos(np.pi * (np.abs(angles[:, np.newaxis] - angles) / np.pi)**nu)


if __name__ == "__main__": # EXAMPLE OF USAGE
    basic_params = AttractorParams(N=100, beta=400, nu_exp=1.0, sigma=2*np.pi/100) # we make parameters
    attractor = RingAttractor(basic_params) # we make the attractor
    #attractor.run(50)
    attractor.set_target(0) # we set the target to be at angle 0 (0 degrees, up)
    attractor.run(200)
    print(attractor.get_direction()) # direction turns out to be around [0.25, 0.00] so the animal would probably move in a good direction. Success!
    print((attractor.state))
