import numpy as np
from ring_attractor import RingAttractor, AttractorParams
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import csv
import time
import json
import os
from tqdm import tqdm
import logging

speedAdjustment = 8

class Animal:
    def __init__(self, animal_id: int, attractor: RingAttractor, position: np.ndarray, allocentric: bool, stepsToRun: int = 50, heading: float = 0., speed: float = 0.):
        '''
        Represents an agent
        
        Args:
            animal_id (int): Unique identifier.
            position (np.ndarray): Initial position (2D vector).
            heading (float): Initial heading angle in radians.
            speed (float): Initial speed.
            attractor (RingAttractor): The neural network controller.
            allocentric (bool): If True, uses allocentric coordinates.
            stepsToRun (int): Number of steps in each update.
        '''
        self.id = animal_id
        self.position = np.array(position, dtype=np.float32)
        self.heading = heading
        self.speed = speed
        self.network = attractor
        self.allocentric = allocentric
        self.stepsToRun = stepsToRun


    #this function takes the position of a target an translates it to a heading (depending on if its allocentric or egocentric)
    def targetPosToAngle(self, target: np.ndarray):
        head2Targetpos = target - self.position
        head2Target = (np.arctan2(head2Targetpos[1], head2Targetpos[0]) + 2*np.pi) % (2*np.pi)
        distanceToTarget = np.linalg.norm(head2Targetpos)
        
        if(self.allocentric):
            logging.debug(f"target angle allo:\t{(head2Target + 2*np.pi)%(2*np.pi)}distance:\t{distanceToTarget}")
            return [head2Target, distanceToTarget]
        else:    
            logging.debug(f"target angle ego:\t{(head2Target - self.heading)%(2*np.pi)}distance:\t{distanceToTarget}")
            return [(head2Target-self.heading), distanceToTarget]
    
    #this function takes a heading and translates it to an absolute one
    def headingToAbsolute(self, heading: float):
        if(self.allocentric):
            return(heading)
        else:
            return (self.heading+heading)

    def update_brain(self, target_pos: np.ndarray = None, cutoff: float = 0.0, weight: int = 1): 
        '''Run the network and optionally update target'''
        targetHeading = self.targetPosToAngle(target_pos)
        if target_pos is not None:
            if np.linalg.norm(target_pos - self.position) < cutoff:
                self.network.set_target(targetHeading[0], strength=-1*weight) # negative weight on very short distances
            else:
                self.network.set_target(targetHeading[0], strength=weight)
 
    def update_movement(self):
        '''Adjust heading and speed based on attractor output. This also gets adjusted according to if it is egocentric or allocentric'''
        network_direction = self.network.get_direction() # cartesian coordinates
        norm = np.linalg.norm(network_direction)
        angle = np.arctan2(network_direction[1], network_direction[0])
        logging.debug("Animal " + str(self.id) + " now has the direction of " + str(angle) + " and speed of " + str(norm))
        if norm > 0:
            self.heading = self.headingToAbsolute(angle)  # convert to absolute heading
            self.speed = norm * speedAdjustment  # or clamp if needed (e.g., min/max speed)

    def step(self):
        '''this makes the network run its steps, then update the heading, speed and position of the agent'''
        self.network.run(self.stepsToRun)
        self.update_movement()
        dx = self.speed * np.cos(self.heading)
        dy = self.speed * np.sin(self.heading)
        self.position += np.array([dx, dy], dtype=np.float32)

    #just the functions to access variables
    def get_position(self):
        return self.position
    
    def get_heading(self):
        return self.heading
    
    def get_speed(self):
        return self.speed

class GlobalTracker:
    def __init__(self, visibility: float, allocentric: bool, friendly_weight: float = 1.0, unfriendly_weight: float = -1.0, cutoff: float = 0.5, target_weight = 1):
        self.animals: dict[int, Animal] = {}
        self.positions: dict[int, np.ndarray] = {}
        #this serves to save the future positions of the animals, but be able to calculate each animals step
        self.newPos: dict[int, np.ndarray] = {}
        #the radius of visibility that the animals have
        self.visibility = visibility
        self.cutoff = cutoff
        self.allocentric = allocentric
        self.friendly_weight = friendly_weight
        self.unfriendly_weight = unfriendly_weight # FOR PREY -1 in all our simulations
        self.target_weight = target_weight

    def add_animal(self, animal: Animal):
        self.animals[animal.id] = animal
        self.positions[animal.id] = animal.get_position().copy()
        #when initializing the newPos and positions are the same
        self.newPos[animal.id] = animal.get_position().copy()

    #updates all positions
    def update_position(self, animal_id: int, new_position: np.ndarray):
        if animal_id in self.animals:
            self.animals[animal_id].position = np.array(new_position, dtype=np.float32)
            self.newPos[animal_id] = self.animals[animal_id].position.copy()

    def get_nearby_positions(self, position: np.ndarray):
        '''Returns list of positions within radius of a given agents position'''
        nearby_positions = []
        for pos in self.positions.values():
            if (np.linalg.norm(pos - position) <= self.visibility and np.linalg.norm(pos - position) > 1e-3):
                nearby_positions.append(pos.copy())
        return nearby_positions

    #a global step for this class
    def step_all(self, targetAnimals: list, target_weight: float = None, predatoranimals: dict = {}):
        '''Updates all animals: run brains, save next positinos then replaces old positions with new ones. setting the target weight here does as expected, but can also be set at other places
        '''
        if target_weight is None:
            target_weight = self.target_weight
        for aid, animal in self.animals.items():
            animal.network.reset_target() # start the network with no "external field" from previous steps
            logging.debug("now updating animal" + str(aid))
            nearbyTargetPositions = self.get_nearby_positions(self.positions[aid])
            for pos in nearbyTargetPositions:
                #first we enter all the surrounding animals
                animal.update_brain(pos, weight=self.friendly_weight, cutoff=self.cutoff)
            for id, pos in predatoranimals.items():
                #then we enter all the surrounding predators
                animal.update_brain(pos, weight=self.unfriendly_weight, cutoff=0.0) # CUTOFF IS ZERO HERE SO THAT THE PREDATORS CAN "EAT" THE PREY
            for posTarget in targetAnimals:
                animal.update_brain(posTarget, weight=target_weight, cutoff=0.0) # the weight to target is the same as to other animals actually
            #then we make the brain process the new information inputted, and let the animal move to a new position
            animal.step()
            logging.debug(animal.get_position())
            #this now also updates the position of the animal in the GlobalTracker
            self.update_position(aid, animal.get_position())

    def updateGlobalPositions(self):
        self.positions = self.newPos.copy()
    
    def get_positions(self):
        return self.positions


    def __repr__(self):
        return f"AnimalTracker with {len(self.animals)} animals."

#I think the name of the function is self explaining, but this spawns animals with given parameters
def spawnAnimals(stepsToRun: int, animalsToSpawn: int, isallocentric: bool, visibility: float, beta: float, friendly_weight: float = 1.0, unfriendly_weight: float = -1.0, cutoff: float = 0.5, target_weight = 1):
    animals = GlobalTracker(visibility=visibility, allocentric=isallocentric, friendly_weight=friendly_weight, unfriendly_weight=unfriendly_weight, cutoff=cutoff, target_weight=target_weight)
    neurons = 100
    for i in range(animalsToSpawn):
        basic_params = AttractorParams(N=neurons, beta=beta, nu_exp=1.0, sigma=2*np.pi/neurons * 5, h0=0.25)
        attractor = RingAttractor(basic_params)
        position = np.random.uniform(-10, 10, size=(2))
        heading = np.random.uniform(0,2*np.pi)
        newAnimal = Animal(animal_id=i, position=position, heading=heading, allocentric=isallocentric, attractor=attractor, stepsToRun=stepsToRun)
        animals.add_animal(newAnimal)
    for i in range(animalsToSpawn):
        logging.debug(animals.positions[i])
    return animals

#this just spits out structured filenames for the data to be saved
def fileNameSystem(animalType: str, infotype: str, steps: float, beta: float, allocentric: bool, filetype: str):
    return f'{animalType}_{infotype}_b={str(beta)}_gStp={str(steps)}_allo={str(allocentric)}.{filetype}'

#this function is where the animation is run with given preyanimals, predators and trackers
def run_and_save_simulation_data(n_ticks, targets: list, preyanimals: GlobalTracker, predators: GlobalTracker, filenames:list, subfolder: str = None, progress_bar: bool = False):

    #creates a folder for all the simulation data
    if not subfolder:
        uid = str(int(time.time() * 1000))
    else:
        uid = subfolder
    SimulationsFolder = "simulations"
    os.makedirs(SimulationsFolder, exist_ok=True)
    thisSimulationFolder = os.path.join(SimulationsFolder, uid)
    os.makedirs(thisSimulationFolder, exist_ok=True)
    
    # Create a list to store positions at each tick
    Prey_positions = []
    brain_states = []
    Prey_heading_and_speed = []
    Pred_positions = []
    Pred_heading_and_speed = []
    brain_states_all_animals = [] #Added by Ryo


    for i in range(len(preyanimals.animals)):
        Prey_heading_and_speed.append([])
    
    if(predators != None):
        for i in range(len(predators.animals)):
            Pred_heading_and_speed.append([])
    
    for t in (tqdm(range(n_ticks)) if progress_bar else range(n_ticks)): # progress bar
        if(predators != None):
            preyanimals.step_all(targetAnimals=targets, predatoranimals=predators.get_positions())
            preyanimalsList = list(preyanimals.positions.values())
            predators.step_all(targetAnimals=preyanimalsList)
            preyanimals.updateGlobalPositions()
            predators.updateGlobalPositions()
            # Extract positions as a 2D array (num_animals, 2)
            CurrentPreyPositions = np.array([pos for pos in preyanimals.positions.values()])
            Prey_positions.append(CurrentPreyPositions)
            CurrentPredPositions = np.array([pos for pos in predators.positions.values()])
            Pred_positions.append(CurrentPredPositions)

            for i in range(len(preyanimals.animals)):
                Prey_heading_and_speed[i].append([preyanimals.animals[i].get_heading(), preyanimals.animals[i].get_speed()])
            for i in range(len(predators.animals)):
                Pred_heading_and_speed[i].append([predators.animals[i].get_heading(), predators.animals[i].get_speed()])
            print("now doing tick number " + str(t) + "")
            
        else:
            preyanimals.step_all(targetAnimals=targets)
            preyanimals.updateGlobalPositions()
            # Extract positions as a 2D array (num_animals, 2)
            positions = np.array([pos for pos in preyanimals.positions.values()])
            Prey_positions.append(positions)
            for i in range(len(preyanimals.animals)):
                Prey_heading_and_speed[i].append([preyanimals.animals[i].get_heading(), preyanimals.animals[i].get_speed()])
            print("now doing tick number " + str(t) + "")


        for aid, animal in preyanimals.animals.items():
            state_list = animal.network.state.tolist()
            brain_states_all_animals.append([t, aid] + state_list)


    #saving the prey positions to a csv file
    PreyPosFile = os.path.join(thisSimulationFolder, filenames[0])
    with open(PreyPosFile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header (optional)
        num_animals = len(preyanimals.positions)
        header = [f"Animal_{i+1}_{axis}" for i in range(num_animals) for axis in ['x', 'y']]
        writer.writerow(header)
        # Write positions for each tick
        for tick in Prey_positions:
            row = tick.flatten()  # Flatten positions from (num_animals, 2) to a 1D array
            writer.writerow(row)
        
    #saving pred positions to csv 
    #saving the prey positions to a csv file
    if(predators != None):
        PredPosFile = os.path.join(thisSimulationFolder, filenames[2])
        with open(PredPosFile, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
        
            # Write the header (optional)
            pred_num_animals = len(predators.positions)
            header2 = [f"Animal_{i+1}_{axis}" for i in range(pred_num_animals) for axis in ['x', 'y']]
            writer.writerow(header2)
            # Write positions for each tick
            for tick in Pred_positions:
                row = tick.flatten()  # Flatten positions from (num_animals, 2) to a 1D array
                writer.writerow(row)
        
    #saving the target in this animation
    notesfile = os.path.join(thisSimulationFolder, "notes.txt")
    with open(notesfile, 'w') as file:
        for element in targets:
            file.write(f"\n target position: ")
            file.write(str(element))
            

    # Save the heading and speed data to json
    PreyHandBFile = os.path.join(thisSimulationFolder, filenames[1])
    with open(PreyHandBFile, 'w') as f:
        json.dump(Prey_heading_and_speed, f)
       
    serializable_state = [
    [state.tolist() for state in brain_states]
    #for tick in brain_states
    ]
    statesFilepath = 'states.json'
    with open(statesFilepath, 'w') as f2:
        json.dump(serializable_state, f2)


    # Save all animals' brain states to CSV
    brain_states_file = os.path.join(thisSimulationFolder, 'all_brain_states.csv')
    with open(brain_states_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['tick', 'animal_id'] + [f'neuron_{i}' for i in range(len(preyanimals.animals[0].network.state))]
        writer.writerow(header)
        writer.writerows(brain_states_all_animals)
    return brain_states_all_animals

#it starts with the amounts of animals to spawn, which get a random position, random heading and random speed
if __name__ == "__main__":
    #the global parameters
    animationSteps = 400

    #the parameters for the prey
    stepsToRun = 75
    animalsToSpawn = 10
    preyIsAlloC = False
    PreyVis = 500
    Preybeta = 0.05
    PreyUnfriendlyWeight = -19

    #Target parameters
    targetExists = True
    targetPosition = []
    target_weight = animalsToSpawn+3
    targets_to_spawn = 1
    if(targetExists):
        for i in range(targets_to_spawn):
            targetPosition.append(np.random.uniform(20, 40, size=(2)))
        

    #parameters for predator animals
    predsExist = True
    predSteps = 100
    predToSpawn = 1
    predAllocentric = False
    predVis = 500 #in our simulations we did not play around with this
    predBeta = 0.05
    hungry_weight = 3
    
    #this spawns the preyanimals
    preyanimals = spawnAnimals(stepsToRun=stepsToRun, animalsToSpawn=animalsToSpawn, isallocentric=preyIsAlloC, visibility=PreyVis, beta=Preybeta, target_weight=target_weight, unfriendly_weight=PreyUnfriendlyWeight)
    
    #this spawns the predators if desired
    if(predsExist):
        predators = spawnAnimals(stepsToRun=predSteps, visibility=predVis, isallocentric=predAllocentric, animalsToSpawn=predToSpawn, beta=predBeta, target_weight=hungry_weight)
    else:
        predators = None


    #the filenames 
    PreyPosFileName = fileNameSystem(animalType="Prey", infotype="P", steps=stepsToRun, beta=Preybeta, allocentric=preyIsAlloC, filetype="csv")
    #PreyPosFileNameJson = fileNameSystem(animalType="Prey", infotype="P", steps=stepsToRun, beta=Preybeta, allocentric=preyIsAlloC, filetype="json")
    PreyHandB = fileNameSystem(animalType="Prey", infotype="H&B", steps=stepsToRun, beta=Preybeta, allocentric=preyIsAlloC, filetype="json")
    PredPosFile = fileNameSystem(animalType="Pred", infotype="P", steps=predSteps, beta=predBeta, allocentric=predAllocentric, filetype="csv")
    PredHandB = fileNameSystem(animalType="Pred", infotype="H&B", steps=predSteps, beta=predBeta, allocentric=predAllocentric, filetype="csv")
    filenames = [PreyPosFileName, PreyHandB, PredPosFile, PredHandB]

    #this is where the simulation is executed
    brain_states = run_and_save_simulation_data(animationSteps, targets=targetPosition, preyanimals=preyanimals, predators=predators, filenames=filenames)
    logging.debug("Prey did " + str(animationSteps)+ " ticks with beta of " + str(Preybeta))
    print(targetPosition)
    winsound.Beep(2000, 1500) 

    def plot_ring_attractor(state, title="Ring Attractor"):
        """
        state: list or numpy array of -1 or 1 values
        """
        n = len(state)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        
        x = np.cos(angles)
        y = np.sin(angles)
        
        colors = ['white' if s == 1 else 'black' for s in state]
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_aspect('equal')
        ax.set_facecolor('lightgray')  # 背景色

        # show nneurons
        for xi, yi, ci in zip(x, y, colors):
            circle = plt.Circle((xi, yi), 0.05, color=ci, ec='gray')
            ax.add_patch(circle)

        # no axis
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        ax.set_title(title)
        plt.show()

    #plot_ring_attractor(preyanimals.animals[0].network.state, title="Tick 0")


    def animate_ring_attractor(states, interval=200):
        """
        states: List of brain states (each state is a list or np.array of -1/1)
        interval: Milliseconds between frames
        """

        n = len(states[0])  # the number of neurons
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_aspect('equal')
        ax.set_facecolor('lightgray')
        ax.axis('off')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_title("Ring Attractor Over Time")

        circles = [plt.Circle((xi, yi), 0.05, color='gray', ec='black') for xi, yi in zip(x, y)]
        for c in circles:
            ax.add_patch(c)

        def update(frame):
            state = states[frame]
            for i, neuron in enumerate(state):
                color = 'white' if neuron == 1 else 'black'
                circles[i].set_facecolor(color)
            ax.set_title(f"Tick {frame}")
            return circles

        ani = animation.FuncAnimation(fig, update, frames=len(states), interval=interval, blit=True)
        plt.show()

    #animate_ring_attractor(brain_states)

