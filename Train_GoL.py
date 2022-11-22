# -*- coding: utf-8 -*-

from random import Random
import time

import sys
import math
import shutil
import os.path
import inspyred
import subprocess
import matplotlib
import numpy as np
from firebase import Firebase

#from pylab import *

import plot_utils

from inspyred import ec
from inspyred_utils import NumpyRandomWrapper

"""
-------------------------------------------------------------------------
Edit this part to do the exercises
"""

display = True

# possible options: config2d.txt, config2d_no_obstacles.txt
config_file = "config.txt"

popSize = 50                # population size
numGen = 30                 # used with generation_termination
numEval = 2500              # used with evaluation_termination
tournamentSize = 4#2          # tournament size (default 2)
mutationRate = 0.3          # mutation rate, per gene (default 0.1)
gaussianMean = 0            # mean of the Gaussian distribution used for mutation
gaussianStdev = 0.1         # std. dev. of the Gaussian distribution used for mutation
crossoverRate = 0.8         # rate at which crossover is performed (default 1.0)
numCrossoverPoints = 1      # number of crossover points used (default 1)
selectionSize = popSize     # selection size (i.e. how many individuals are selected for reproduction)
numElites = 2#1               # no. of elites (i.e. best individuals that are kept in the population)


config = {
  "apiKey": "your-apiKey",
  "authDomain": "yourProject.firebaseapp.com",
  "databaseURL": "https://yourProject.firebaseio.com/",
  "storageBucket": "yourProject.appspot.com"
}

# firebase = Firebase(config) # Uncomment to configure upload of best solution to firebase

"""
-------------------------------------------------------------------------
"""

def readConfigFile(file):
    myvars = {}
    with open(file) as f:
        lines = f.read().splitlines()
        for line in lines:
            if line.startswith("#"):
                pass
            else:
                if "=" in line:
                    name, var = line.partition("=")[::2]
                    myvars[name.strip()] = var.strip()
    return myvars

def writeCandidatesToFile(file,candidate):
    with open(file, "w") as f:
        for i in np.arange(candidate.shape[0]-1):
            f.write(str(candidate[i]) + " ")

def readFileAsMatrix(file):
    with open(file) as f:
        lines = f.read().splitlines()
        matrix = []
        for line in lines:
            row = []
            for value in line.split():
                row.append(float(value.replace(",",".")))
            matrix.append(row)

        return matrix

# this object is used for single-thread evaluations (only pickleable objects can be used in multi-thread)
class RobotEvaluator():

    def __init__(self,config_file,seed):
        self.config_file = config_file
        self.numRobots = popSize

        parameters = readConfigFile(self.config_file)

        nrInputNodes = int(parameters["nrInputs"])
        nrHiddenNodes = int(parameters["nrHiddenNodes"])
        nrOutputNodes = 1

        # calculate the no. of weights
        networkType = parameters["networkType"]
        if nrHiddenNodes > 0:
            if networkType == "FFNN":
                nrWeights = (nrInputNodes+1)*nrHiddenNodes + (nrHiddenNodes+1)*nrOutputNodes
            elif networkType == "ELMAN":
                nrWeights = (nrInputNodes+1)*nrHiddenNodes + (nrHiddenNodes+1)*nrOutputNodes + nrHiddenNodes*nrHiddenNodes
                pass
            elif networkType == "JORDAN":
                nrWeights = (nrInputNodes+1)*nrHiddenNodes + (nrHiddenNodes+1)*nrOutputNodes + nrOutputNodes*nrOutputNodes
                pass
        else:
            nrWeights = (nrInputNodes+1)*nrOutputNodes

        print("N_weights: ", nrWeights)

        self.geneMin = float(parameters["geneMin"])
        self.geneMax = float(parameters["geneMax"])
        self.nrTimeStepsGen = int(parameters["nrTimeStepsGen"])
        
        self.nrWeights = nrWeights
        self.seed = seed
        self.bounder = ec.Bounder([self.geneMin]*self.nrWeights, [self.geneMax]*self.nrWeights)
        self.maximize = False

        self.genCount = 0
    
    def generator(self, random, args):
        return [random.uniform(self.geneMin,self.geneMax) for _ in range(self.nrWeights)]
    
    def evaluator(self, candidates, args):
        communication_dir = "."
        request_file = communication_dir + "/weights.txt"
        results_file = communication_dir + "/iterations.txt"

        fitness = []

        print("matrix shape: ", np.asarray(candidates).shape)
        candidates = np.asarray(candidates)


        for candidate_idx in range(self.numRobots):
            candidate = candidates[candidate_idx]

            results = writeCandidatesToFile(request_file, candidate)


            #command = "./a.out 64 2000"
            #result = os.system(command)

            proc = subprocess.Popen("../run.sh")
            #subprocess.call([sys.executable, 'run.sh'])
            proc.wait()

            results = readFileAsMatrix(results_file)
            print("iter.: ", results[0][0])     #NUMBER OF ITERATIONS (0-5000)
            n_iterations = results[0][0]
            print("n.part.: ", results[0][1])   #AVERAGE NUMBER OF PARTICLES (0-144)
            n_particles = results[0][1]
            print("n.diff.: ", results[0][2])   #VARIANZA (0-144)
            n_var = results[0][2]
            print("n.agg.: ", results[0][3])    #AGGREGAZIONE (0-12)
            n_agg = results[0][3]

            fit_iterations = (5000-int(n_iterations))/50    # 0 - 100
            fit_particles = abs((n_particles - 20))         # 0 - 124
            fit_var = abs(n_var - ((n_particles*2)/4))      # 0 - 139 (n_var = 2*n_particles if all particles change)
            fit_agg = abs(n_agg-8)*10                       # 0 ~ 100

            fit = fit_iterations + fit_particles + fit_var + fit_agg
            print("fit: ", fit)

            fitness.append(fit)

        
        if not os.path.exists(str(seed)):
            os.makedirs(str(seed))
        shutil.copy(request_file, str(seed) + "/candidates_" + str(self.genCount) + ".txt")

        self.genCount += 1
        
        return fitness

def main(rng, seed, display=False):
    
    # the robot maze navigation problem
    problem = RobotEvaluator(config_file,seed)
    
    # --------------------------------------------------------------------------- #
    # EA configuration
    
    # the evolutionary algorithm (EvolutionaryComputation is a fully configurable evolutionary algorithm)
    # standard GA, ES, SA, DE, EDA, PAES, NSGA2, PSO and ACO are also available
    ea = inspyred.ec.EvolutionaryComputation(rng)
    
    # observers: provide various logging features
    if display:
        ea.observer = [inspyred.ec.observers.stats_observer,
                       plot_utils.plot_observer]
                       #inspyred.ec.observers.file_observer,
                       #inspyred.ec.observers.best_observer,
                       #inspyred.ec.observers.population_observer,

    # selection operator
    #ea.selector = inspyred.ec.selectors.truncation_selection
    #ea.selector = inspyred.ec.selectors.uniform_selection
    #ea.selector = inspyred.ec.selectors.fitness_proportionate_selection
    #ea.selector = inspyred.ec.selectors.rank_selection
    ea.selector = inspyred.ec.selectors.tournament_selection

    # variation operators (mutation/crossover)
    ea.variator = [inspyred.ec.variators.gaussian_mutation,
                   inspyred.ec.variators.n_point_crossover]
                    #inspyred.ec.variators.random_reset_mutation,
                    #inspyred.ec.variators.inversion_mutation,
                    #inspyred.ec.variators.uniform_crossover,
                    #inspyred.ec.variators.partially_matched_crossover,

    # replacement operator
    #ea.replacer = inspyred.ec.replacers.truncation_replacement
    #ea.replacer = inspyred.ec.replacers.steady_state_replacement
    #ea.replacer = inspyred.ec.replacers.random_replacement
    #ea.replacer = inspyred.ec.replacers.plus_replacement
    #ea.replacer = inspyred.ec.replacers.comma_replacement
    #ea.replacer = inspyred.ec.replacers.crowding_replacement
    #ea.replacer = inspyred.ec.replacers.simulated_annealing_replacement
    #ea.replacer = inspyred.ec.replacers.nsga_replacement
    #ea.replacer = inspyred.ec.replacers.paes_replacement
    ea.replacer = inspyred.ec.replacers.generational_replacement

    # termination condition
    #ea.terminator = inspyred.ec.terminators.evaluation_termination
    #ea.terminator = inspyred.ec.terminators.no_improvement_termination
    #ea.terminator = inspyred.ec.terminators.diversity_termination
    #ea.terminator = inspyred.ec.terminators.time_termination
    ea.terminator = inspyred.ec.terminators.generation_termination

    # --------------------------------------------------------------------------- #

    args = {}
    args["fig_title"] = "EA"

    # run the EA
    final_pop = ea.evolve(generator=problem.generator,
                  evaluator=problem.evaluator,
                  bounder=problem.bounder,
                  maximize=problem.maximize,
                  pop_size=popSize,
                  max_generations=numGen,
                  #max_evaluations=numEval,
                  tournament_size=tournamentSize,
                  mutation_rate=mutationRate,
                  gaussian_mean=gaussianMean,
                  gaussian_stdev=gaussianStdev,
                  crossover_rate=crossoverRate,
                  num_crossover_points=numCrossoverPoints,
                  num_selected=selectionSize,
                  num_elites=numElites, **args)
                              
    # --------------------------------------------------------------------------- #

    if display:
        # print best solution and fitness
        best_candidate = final_pop[0].candidate
        best_fitness = final_pop[0].fitness
        print("Best Solution: {0}: {1}".format(str(best_candidate), best_fitness))
        writeCandidatesToFile("best.txt", np.asarray(best_candidate))

        # db = firebase.database()

        data = {"weights": best_candidate}
        # db.child("Lampada/GoL").set(data)

        #os.chdir("Java2dRobotSim")
        
        #if not os.path.exists(str(seed)):
        #    os.makedirs(str(seed))
        #shutil.copy("best.txt", str(seed) + "/best_candidate.txt")
        #shutil.copy("results.txt", str(seed) + "/best_results.txt")
        if os.path.exists("results.txt"):
            os.remove("results.txt")

if __name__ == "__main__":
    
    if len(sys.argv) > 1 :
        seed = int(sys.argv[1])
    else :
        seed = int(time.time())
    rng = NumpyRandomWrapper(seed)
    
    main(rng,seed,display)



    if display:
        ioff()
        show()
