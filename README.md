# Allocentric flocking simulation
Based on the paper [Allocentric flocking (Salahshour & Couzin, 2025)](https://doi.org/10.1101/2025.01.27.634610) with additional feature of avoiding a predator.


Our goal was to reproduce the results in the paper "allocentric flocking" (linked above). We created our own neural network and dynamics between agents and targets, then added the functionality of a predator (or multiple predators) to test our model in further scenarios. 
The code we used is in the respective files in this repository, along with some simulations that we ran ourselves.  

To run and animate the simulation: 
1. Copy the files and resolve all dependencies
2. If desired, change the variables (like beta, stepstorun (the amount of steps the neural network takes for one tick), animation ticks, wether to have a target or not, wether to have predators or not) as desired in the code of "runningmultipleanimations.py"
3. Run the file "runningmultipleanimations.py". This should create 4 new subfolders in the new folder "simulations". Each simulation has its own folder, where the positions of the prey, predator and the Heading and speed of the prey are stored in csv and json files.
4. Create two new files "positions.csv" and "predators.csv" in your main folder. Copy the respective data into these files. then run "csvtoanimation.py" to animate the simulation. Only the data in "positions.csv" and "predators.csv" will be run. use the commented out code to add the target in the animation (in the "csvtoanimation.py" file)
