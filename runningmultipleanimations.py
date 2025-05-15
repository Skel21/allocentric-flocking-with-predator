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
#import winsound
from AnimalsAndGlobalCSV import *

def runninganimation(param1tochange, param2tochange):
        #the global parameters
    animationSteps = 200

    #the parameters for the prey
    stepsToRun = 100
    animalsToSpawn = 10
    preyIsAlloC = param1tochange
    PreyVis = 500
    Preybeta = 10
    PreyUnfriendlyWeight = -animalsToSpawn-2

    #Target parameters
    targetExists = True
    targetPosition = []
    target_weight = animalsToSpawn+1
    targets_to_spawn = 1
    if(targetExists):
        for i in range(targets_to_spawn):
            targetPosition.append(np.random.uniform(-20, 40, size=(2)))
        

    #parameters for predator animals
    predsExist = True
    predSteps = 100
    predToSpawn = 1
    predAllocentric = param2tochange
    predVis = 500 #at the moment setting it this high doesn't make it have any effect
    predBeta = 10
    hungry_weight = 3
    
    #this spawns the preyanimals
    preyanimals = spawnAnimals(stepsToRun=stepsToRun, animalsToSpawn=animalsToSpawn, isallocentric=preyIsAlloC, visibility=PreyVis, beta=Preybeta, target_weight=target_weight, unfriendly_weight=PreyUnfriendlyWeight)
    
    #this spawns the predators if desired
    if(predsExist):
        predators = spawnAnimals(stepsToRun=predSteps, visibility=predVis, isallocentric=predAllocentric, animalsToSpawn=predToSpawn, beta=predBeta, target_weight=hungry_weight)
    else:
        predators = None


    #the filenames (positions being saved in json will be implemented later probably)
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



if __name__ == "__main__":
    #this time I am running it to test the differences in allo and egocentric performance
    runninganimation(True, True)
    print("running second animation now")
    runninganimation(True, False)
    print("running third animation now")
    runninganimation(False, True)
    print("runninig fourth animation now")
    runninganimation(False, False)
