

"""
TSP Solver using Genetic Algorithm
Implementation of various GA operators for solving the Traveling Salesman Problem
Original implementation: 2019
"""

import random
from individual import Individual
import sys
import math

# Set random seed for reproducibility
random.seed(42)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# # # CONFIGURATION RUN SELECTIONS
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

run_direct = True # <--------------------- set True if run direct
run_configuration = False # <------------- set True if run specific config
config_number = 8 # <--------------------- select config version (1-8)

# ALTERNATIVE: RUN THE FULL EXPERIMENT (see bottom of the file)
run_experiment = False # <------------- set True to run experiment
# other options MUST be set to False!

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - = = GLOBAL SETTINGS = = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

set_population = 300 # 300 / 100
set_mutation = 0.1 # 0.1
set_iterations = 500 # 500 / 300

enable_elitist = False # <---------------- ELITIST FLAG

enable_PMX = False
enable_uniform_crossover = False
enable_inversion_mutation = False
enable_reciprocal_exchange_mutation = False

init_sol_rand = True # True is Random / False=nearest city neighbour insertion
rand_selection = False


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - = = Debug Flags = = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Enabling some of the below with large data samples and
# high amount of iterations will flood console and slow down processing

# init infos
debug_show_euclid_distance_of_first = False
debug_show_init_population_data = False

# method debug - only enable those with very small iteration & sample size !
debug_method_stochasticUniversalSampling_show = False
debug_method_uniform_crossover_show = False
debug_method_pmx_show = False
debug_method_inversion_mutation_show = False
debug_method_order_1_crossover_show = False
debug_elitist = False

# full debug - only enable those with very small iteration & sample size !
debug_iteration_count = False
debug_show_mating_pool_size_every_iteration = False
debug_show_population_size_every_iteration = False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 



class BasicTSP:
    def __init__(self, _fName, _popSize, _mutationRate, _maxIterations):
        """
        Parameters and general variables
        """

        self.population     = [] # Chromosomes / Individuals ( all "cities" inside each Individual )
        self.matingPool     = [] # use to create new population
        self.best           = None # best Chromosome / Individual - here: from overall distance - the smaller the better
        self.popSize        = _popSize # "allowed" population size (= amount of Inviduals)
        self.geneSize        = None # number of "cities" (assigned in read)
        self.mutationRate   = _mutationRate
        self.maxIterations  = _maxIterations
        self.iteration      = 0 # counter
        self.fName          = _fName
        self.data           = {} # dict with data from file

        self.readInstance()
        self.initPopulation()

    # READ FILE --------------------------------------------------------------
    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.fName, 'r')
        self.geneSize = int(file.readline()) # first line = number of genes in the file
        self.data = {}
        for line in file:
            (id, x, y) = line.split()
            self.data[int(id)] = (int(x), int(y)) # data --> dict {id: (coord 1, coord 2)}
        file.close()
        
        if debug_show_init_population_data: # DEBUG ONLY
                print("Data Dict:", self.data)
                

    # INIT POPULATION --------------------------------------------------------
    def initPopulation(self):
        """
        Creating random individuals in the population
        """
        for i in range(0, self.popSize):
                
            individual = Individual(self.geneSize, self.data, init_sol_rand) # pass No. of cities & Dict - returns individual (Chromosome)            
            individual.computeFitness() # assigns Chromosome fitness value
            self.population.append(individual) # add Chromosome to population

        self.best = self.population[0].copy() # use first Chromosome initially as best to compare
        if debug_show_euclid_distance_of_first: # DEBUG ONLY
            print("First Chromosome Fitness (Distance):", self.best.getFitness())
        
        # Assign Best (Initial)
        for individual_i in self.population:
            if self.best.getFitness() > individual_i.getFitness(): # Looking for smallest value from whole population
                self.best = individual_i.copy()
        print ("Best initial solution: ", self.best.getFitness()) # output float (calculated euclidean distance)


    # ------------------------------------------------------------------------

    def updateBest(self, candidate):
        if self.best == None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()
            print ("iteration: ",self.iteration, "best: ",self.best.getFitness())            

    
    # ------------------------------------------------------------------------
    
    def returnTwoIndividuals_withChecks_fromIndicies(self, index_list_1, index_list_2):
        """
        * creates two new Individual objects from 2 given indicies 
        * computes fitness 
        * updates best 
        * returns
        """
        
        temp_dictA = {k: self.data[k] for k in index_list_1}
        new_individual_A = Individual(self.geneSize, temp_dictA, init_sol_rand)
        
        temp_dictB = {k: self.data[k] for k in index_list_2}
        new_individual_B = Individual(self.geneSize, temp_dictB, init_sol_rand)
        
        new_individual_A.computeFitness()
        new_individual_B.computeFitness()
        
        self.updateBest(new_individual_A)
        self.updateBest(new_individual_B)
        
        return new_individual_A ,new_individual_B
    
    def returnRandomIntWithOptions(self):
        """
        Setup to protect the top 5 from mutation so they can survive
        for the next generation
        """
        
        if(enable_elitist):
            return random.randint(5, self.geneSize-1) # 5 starting point
        else:
            return random.randint(0, self.geneSize-1) # regular
    
        
    
    # ------------------------------------------------------------------------

    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[ random.randint(0, self.popSize-1) ]
        indB = self.matingPool[ random.randint(0, self.popSize-1) ]
        return [indA, indB]


    # SUS - - - - - - - - 
    def stochasticUniversalSampling(self):
        """
        Your stochastic universal sampling Selection Implementation
        
        Using current population to select a "fitter" version for new mating pool
        """        
        
        fitness_sum_all = 0        
        mark_distance = 0
        mark_start = 0
        ruler = []
        
        # sum of fitness values - all chromosomes in population
        for individual in self.population:
            fitness_sum_all += individual.fitness
        
        # distance P (between successive points)
        mark_distance = fitness_sum_all/self.popSize
        #mark_distance = 6245478
        
        # starting point for first marker
        mark_start = random.randint(0, int(mark_distance))      
        
        # setup ruler
        for i in range(self.popSize): # using int here to get rid of floating point in keys (should make no difference)            
            ruler.append((int(mark_start)) + (int(mark_distance)) * i)
                
        tempPopulation = []        
        
        if debug_method_stochasticUniversalSampling_show: # DEBUG ONLY
            print("Ruler:", ruler)
            print("Population:", self.popSize)
            
        cumulative_fitness = 0
        individual_list = []
        individual_index = 0
        
        # go through each Individual in Population        
        for individual in self.population:
            cumulative_fitness = cumulative_fitness + individual.fitness
            for ruler_pointer in ruler: # check ruler and items in range
                if ruler_pointer < cumulative_fitness and ruler_pointer > cumulative_fitness - individual.fitness:
                    individual_list.append(individual_index)
                    tempPopulation.append(individual.copy())
            individual_index+=1
        
        if debug_method_stochasticUniversalSampling_show: # DEBUG ONLY
            print("Individual List:", individual_list)
        
        return tempPopulation
    



    # Uniform Order Based Crossover  (Permutation Encoding) - - - - - - - - 
    def uniformCrossover(self, individual_A, individual_B):
        """
        Your Uniform Crossover Implementation        
        """        
        
        # SETUP - - - - - ->
        static_genes_A = [] # using A to pick genes at random
        static_genes_B = [] # will be matched (by position) later
        static_indicies = [] # static indicies for both Chromosomes A & B (matching positions)
        childA = [] 
        childB = []
        
        # get random choices from Chromosome A (50% chance of getting picked)
        static_genes_A = random.sample(individual_A.genes,int(self.geneSize/2))
        
        # now get a list of indicies of those picked
        for item in static_genes_A:
                static_indicies.append(individual_A.genes.index(item)) # indicies
                #static_genes_B.append(individual_B.genes[individual_A.genes.index(item)]) 
                # would be nice but unsorted - won't work :/
        
        static_indicies.sort() # get them in order to match up A & B properly
        static_genes_A = [] # reset A
        
        # A & B Static - - - - - ->
        
        # now get static parts A & B in correct indexed order
        for itemA, itemB in zip(individual_A.genes, individual_B.genes):
            if individual_A.genes.index(itemA) in static_indicies:
                static_genes_A.append(itemA)
            if individual_B.genes.index(itemB) in static_indicies:
                static_genes_B.append(itemB)
        
        # CHILDREN - - - - - ->        
        
        counter_static = 0
        for i in range(0, self.geneSize):
            # static part
            if i in static_indicies: 
                childA.append(static_genes_A[counter_static])
                childB.append(static_genes_B[counter_static])
                counter_static += 1
            else: # use parts of other parent as they occur filling in slots but NOT duplicates!
                for item in individual_B.genes:
                    if item not in childA and item not in static_genes_A:
                        childA.append(item)
                        break
                for item in individual_A.genes:
                    if item not in childB and item not in static_genes_B:
                        childB.append(item)
                        break
                
        
        if debug_method_uniform_crossover_show: # DEBUG ONLY            
            
            print("Chromosome (A):", individual_A.genes)
            print("Chromosome (B):", individual_B.genes)            
            print("Indicies Of Those Originally Picked:", static_indicies)
            print("Genes Picked 50% at Random (A):", static_genes_A)
            print("Genes Picked -> corresponding (B):", static_genes_B)
            print("Child A Result:", childA)
            print("Child B Result:", childB)
            
        # run checks (update best) and return 2 new Children
        return self.returnTwoIndividuals_withChecks_fromIndicies(childA,childB)





    # PMX - - - - - - - - - - - - - - - -
    def pmxCrossover(self, individual_A, individual_B):
        """
        Your PMX Crossover Implementation
        
        # WOW! working this out was a cool challenge!
        
        """
        
        tmpA = {} # maps True/False indicies
        tmpB = {}
        staticA = [] # holds static part indicies
        staticB = []
        childA = [] # holds the final crossover indicies
        childB = []

        # 2 random indices (range)
        indexA = random.randint(0, self.geneSize-1)
        indexB = random.randint(0, self.geneSize-1)
        
        # tmp dic {key = assign gene number} FROM PARENT A: if in between the 2 indices set to False otherwise True
        for i in range(0, self.geneSize):
            if i >= min(indexA, indexB) and i <= max(indexA, indexB):
                tmpA[individual_A.genes[i]] = False
                tmpB[individual_B.genes[i]] = False
            else:
                tmpA[individual_A.genes[i]] = True
                tmpB[individual_B.genes[i]] = True
                
        # returns 2 lists with static parts for A & B 
        staticA = [index for index, value in tmpA.items() if value == False]
        staticB = [index for index, value in tmpB.items() if value == False]
        
        static_part_index = 0;
        # merge the 3 parts (start, static mid and end) ->  cases with 2 parts (where static is at start or end) will also work
        for i in range(0, self.geneSize):
            # non static part
            if i < min(indexA, indexB) or i > max(indexA, indexB):
                
                # check for duplicates between crossed over statics and parents (AxB & BxA) to "change" those
                if individual_A.genes[i] not in staticB:
                    childA.append(individual_A.genes[i])                    
                
                else: # crossreference that index with static parts to "map" it
                    matching_value = individual_A.genes[i] # value to map and test if it fits (if not update and keep going)                    
                    desired_value = None                    
                    for j in range(0, len(staticB)): # run loop static part length times                        
                        desired_value = staticA[staticB.index(matching_value)]                        
                        if desired_value in childA or desired_value in staticB: # if already in child or static part
                            matching_value = desired_value # reassign                            
                        else:                            
                            childA.append(desired_value) # add                            
                            break # break inner loop and go to next i
                        
                
                # same for other child
                if individual_B.genes[i] not in staticA:
                    childB.append(individual_B.genes[i])
                else:
                    matching_value = individual_B.genes[i]
                    desired_value = None
                    for p in range(0, len(staticA)):
                        desired_value = staticB[staticA.index(matching_value)] 
                        if desired_value in childB or desired_value in staticA:
                            matching_value = desired_value
                        else:
                            childB.append(desired_value)
                            break                
            
            #static part implementation
            else:
                childA.append(staticB[static_part_index])
                childB.append(staticA[static_part_index])
                static_part_index+=1     
        
        if debug_method_pmx_show: # DEBUG ONLY
            print("Parent A:", individual_A.genes)
            print("Parent B:", individual_B.genes)
            print("TEMPA:", tmpA)
            print("TEMPB:", tmpB)
            print("STATIC A:", staticA)
            print("STATIC B:", staticB)
            print("CHILD A:", childA)
            print("CHILD B:", childB)            
          
        # run checks (update best) and return 2 new Children
        return self.returnTwoIndividuals_withChecks_fromIndicies(childA,childB)                  
        
        
    # Reciprocal Exchange - - - - - - - - - - - -
    def reciprocalExchangeMutation(self, individual):
        """
        Your Reciprocal Exchange Mutation implementation
        """        
        # as implemented already by Dr. Diarmuid Grimes (def mutation)
        pass


    # Inversion Mutation - - - - - - - - - - - -
    def inversionMutation(self, individual):
        """
        Your Inversion Mutation implementation
        """
        
        if random.random() > self.mutationRate:
            return       
        # -------------------------------------
        
        # 2 random indicies
        indexA = self.returnRandomIntWithOptions()
        indexB = self.returnRandomIntWithOptions()
        
        # case: 0 - do nothing
        if (indexA == indexB): # must be different!
            if debug_method_inversion_mutation_show: # DEBUG ONLY
                print("Indicies The Same - Aborted!")
            return
        
        # -------------------------------------
        
        # if B < A invert them first (B must be > A)
        if indexB < indexA: 
            temp_index = indexA            
            indexA = indexB
            indexB = temp_index
            
        if debug_method_inversion_mutation_show: # DEBUG ONLY
            print("Individual Before:", individual.genes)
            print("Index A:", indexA)
            print("Index B:", indexB)

        # case: 1            
        # = Form of Reciprocal Exchange Mutator
        if indexB-indexA == 1:                
            temp_index = individual.genes[indexA]
            individual.genes[indexA] = individual.genes[indexB]
            individual.genes[indexB] = temp_index
        
        # -------------------------------------
        
        # case: all other
        else:
        
            inversion_steps = ((indexB-indexA)/2) # half of the range                
            
            for i in range(0, math.ceil(inversion_steps)):
                if i < math.ceil(inversion_steps):
                    temp_index = individual.genes[indexA]            
                    individual.genes[indexA] = individual.genes[indexB]
                    individual.genes[indexB] = temp_index
                    
                    #inversion_steps-=1
                    indexA += 1
                    indexB -= 1
                else:
                    break        
    
        individual.computeFitness()
        self.updateBest(individual)
            
        if debug_method_inversion_mutation_show: # DEBUG ONLY
            print("Individual After:", individual.genes)

        

    # ------------------------------------------------------------------------
    # two function below were given initially in the original file
    # ------------------------------------------------------------------------

    # Order 1 Cross Over - - - - - - - - - - - -
    def crossover(self, individual_A, individual_B):
        """
        Executes a 1 order crossover and returns a new individual object
        Original Method by Dr. Diarmuid Grimes
        """
        
        # ---------------------------------------------------------
        # BELOW COMMENTS for initial step by step understanding only
        # ---------------------------------------------------------
        
        child = []
        tmp = {}
        
        indexA = random.randint(0, self.geneSize-1)
        indexB = random.randint(0, self.geneSize-1)
        
        for i in range(0, self.geneSize):
            if i >= min(indexA, indexB) and i <= max(indexA, indexB):
                tmp[individual_A.genes[i]] = False
            else:
                tmp[individual_A.genes[i]] = True
        
        aux = []

        for i in range(0, self.geneSize):
            if not tmp[individual_B.genes[i]]:
                child.append(individual_B.genes[i])
            else:
                aux.append(individual_B.genes[i])
                
        child += aux
        
        if debug_method_order_1_crossover_show: # DEBUG ONLY
            print("Parent A:", individual_A.genes)
            print("Parent B:", individual_B.genes)
            print("TEMP:", tmp)
            print("AUX:", aux)
            print("Child:",child)
            
        
        temp_dict = {k: self.data[k] for k in child}
        
        individual = Individual(self.geneSize, temp_dict, init_sol_rand)
        individual.computeFitness()
        self.updateBest(individual)
        return individual
    

    # Mutation - Reciprocal Exchange - (as original file version) - - - - - - 
    def mutation(self, individual):
        """
        Mutate an individual by swaping two cities with certain probability (i.e., mutation rate)
        Original Method by Dr. Diarmuid Grimes
        """
        if random.random() > self.mutationRate:
            return
        indexA = self.returnRandomIntWithOptions()
        indexB = self.returnRandomIntWithOptions()

        tmp = individual.genes[indexA]
        individual.genes[indexA] = individual.genes[indexB]
        individual.genes[indexB] = tmp

        individual.computeFitness()
        self.updateBest(individual)
    

    # ------------------------------------------------------------------------    
    # SUB STEPS ---------- mating pool -> new generation ---------------------
    # ------------------------------------------------------------------------
    
    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation
        
        Either Random or SUS options
        """
        
        self.matingPool = [] # reset mating pool
        
        # using SUS to create a new mating pool
        if not rand_selection:
            self.matingPool = self.stochasticUniversalSampling()
        
        # basic implementation to copy population
        # results in random selection (Crossover implementations)
        if rand_selection:
            for individual_i in self.population:
                self.matingPool.append( individual_i.copy() )      
        
        if(debug_show_mating_pool_size_every_iteration):
            print("Mating Pool:", len(self.matingPool)) # DEBUG ONLY

    def newGeneration(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        
        # required later if elitist enabled
        poolSize = len(self.matingPool)
        if enable_elitist: 
            poolSize = len(self.matingPool) - 20        
        
        # elitist option (20 preserved)
        if enable_elitist:
            tempList = self.elitist()
            self.population = []
            self.population.extend(tempList) # add 20 elite to population  
            
            ### noticed here that sometimes #1 top position does not stay
            ### changed mutation to take into account but to no effect
            ### where is this coming from?? -> investigate!            
            #print(self.population[0].getFitness())
        else:
            self.population = [] # resetting population to zero before appending new below        
        
        for i in range(0, poolSize):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """

            # SET GLOBAL FLAGS TO ENABLE ANY OF THE BELOW
            
            # Uniform Crossover
            if enable_uniform_crossover:
                if i % 2 == 0: # run x times to match pool size (as 2 return)
                    rand_ind1, rand_ind2 = self.randomSelection()
                    new_ind1, new_ind2 = self.uniformCrossover(rand_ind1,rand_ind2)
                    self.population.append(new_ind1)
                    self.population.append(new_ind2)
            
            # Cross Over - PMX
            if enable_PMX:                
                if i % 2 == 0: # run x times to match pool size (as 2 return)
                    rand_ind1, rand_ind2 = self.randomSelection()
                    new_ind1, new_ind2 = self.pmxCrossover(rand_ind1,rand_ind2)
                    self.population.append(new_ind1)
                    self.population.append(new_ind2)
                
            # Mutation - Reciprocal Exchange
            if enable_reciprocal_exchange_mutation:                
                self.mutation(self.population[i])
            
            
            # Mutation - Inversion
            if enable_inversion_mutation:
                self.inversionMutation(self.population[i])
            

    def elitist(self):
        """
        preserves top 20 for each run and feeds them back into the pool if enabled
        """
        
        # I wanted to experiment and trying to push this as I kept getting
        # more and more duplicates into the list in higher numbers of iterations
        # instead I was hoping to avoid those and only carry
        # forward unique values each time as with the option below
        
        # while the below is the shortest version I could think of I 
        # got better (but a lot slower) results with other options as well
        
        tpl = []
        for i in range(0,20):
            tpl.append(self.population[i].copy())
        
        top_fitness_list = sorted(tpl, key=lambda ind : ind.getFitness())
        
        if debug_elitist:
            print("Top 20:")
            #for i in range(0,len(top_fitness_list)):
            for i in range(0,4):
                print(top_fitness_list[i].getFitness())
        
        
        return top_fitness_list
        
        
    # STEP -------------------------------------------------------------------
    def GAStep(self):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """
        
        if debug_show_population_size_every_iteration: 
            print("Population:", len(self.population)) # DEBUG ONLY
        
        
        self.updateMatingPool()
        self.newGeneration()
        

    def search(self):
        """
        General search template.
        Iterates for a given number of steps
        """
        
        self.iteration = 0
        while self.iteration < self.maxIterations:
            if debug_iteration_count:
                print("Iteration Number:", self.iteration)
            self.GAStep()
            self.iteration += 1

        print ("Total iterations: ",self.iteration)
        print ("Best Fitness: ", self.best.getFitness())

# checking file input on console
if len(sys.argv) < 2:
    print ("Error - Incorrect input")
    print ("Expecting python BasicTSP.py [instance] ")
    sys.exit(0)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# runs directly from global variables
def run_directly():
    #problem_file = sys.argv[1]
    ga = BasicTSP(sys.argv[1], set_population, set_mutation, set_iterations) # file, population (300), mutation (0.1), iterations (500)
    ga.search()
    
# ENABLE RUN DIRECTLY
if run_direct:
    run_directly()
    
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - = = RUN VARIATIONS = = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def run_config(config): # random / uniform XO / ivnersion mut
    
    global enable_PMX
    global enable_uniform_crossover
    global enable_inversion_mutation
    global enable_reciprocal_exchange_mutation   
    global init_sol_rand
    global rand_selection
    global set_population
    global set_mutation
    global set_iterations
    
    set_population = 300 # 300
    set_mutation = 0.1
    set_iterations = 500 # 500
    
    enable_PMX = False
    enable_uniform_crossover = False
    enable_inversion_mutation = False
    enable_reciprocal_exchange_mutation = False
    init_sol_rand = True
    rand_selection = False
    
    
    if config == 1:        
        init_sol_rand = True
        enable_uniform_crossover = True
        enable_inversion_mutation = True        
        rand_selection = True
        set_population = 100
        set_mutation = 0.1
        print("RUNNING CONFIG 1")
        print("Iterations:", set_iterations, "Mutation: ", set_mutation, "Population:", set_population)
        

    if config == 2:        
        init_sol_rand = True
        enable_PMX = True
        enable_reciprocal_exchange_mutation = True
        rand_selection = True
        set_population = 100
        set_mutation = 0.1
        print("RUNNING CONFIG 2")
        print("Iterations:", set_iterations, "Mutation: ", set_mutation, "Population:", set_population)
        
    if config == 3:        
        init_sol_rand = True
        enable_uniform_crossover = True
        enable_reciprocal_exchange_mutation = True
        rand_selection = False
        print("RUNNING CONFIG 3")
        print("Iterations:", set_iterations, "Mutation: ", set_mutation, "Population:", set_population)
        
    if config == 4:        
        init_sol_rand = True
        enable_PMX = True
        enable_reciprocal_exchange_mutation = True
        rand_selection = False
        print("RUNNING CONFIG 4")
        print("Iterations:", set_iterations, "Mutation: ", set_mutation, "Population:", set_population)
        
    if config == 5:        
        init_sol_rand = True
        enable_PMX = True
        enable_inversion_mutation = True
        rand_selection = False
        print("RUNNING CONFIG 5")
        print("Iterations:", set_iterations, "Mutation: ", set_mutation, "Population:", set_population)
        
    if config == 6:
        init_sol_rand = True
        enable_uniform_crossover = True
        enable_inversion_mutation = True
        rand_selection = False
        print("RUNNING CONFIG 6")
        print("Iterations:", set_iterations, "Mutation: ", set_mutation, "Population:", set_population)
        
    if config == 7:
        init_sol_rand = False
        enable_PMX = True
        enable_reciprocal_exchange_mutation = True
        rand_selection = False
        print("RUNNING CONFIG 7")
        print("Iterations:", set_iterations, "Mutation: ", set_mutation, "Population:", set_population)
        
    if config == 8:
        init_sol_rand = False
        enable_uniform_crossover = True
        enable_inversion_mutation = True
        rand_selection = False
        print("RUNNING CONFIG 8")
        print("Iterations:", set_iterations, "Mutation: ", set_mutation, "Population:", set_population)
    
    ga = BasicTSP(sys.argv[1], set_population, set_mutation, set_iterations)
    ga.search()

if run_configuration:
    run_config(config_number)
    
def experiment(numer_of_runs):
        
    for i in range(0,numer_of_runs):
        print("Run Number:", i+1)
        run_config(1)
        
    for i in range(0,numer_of_runs):
        print("Run Number:", i+1)
        run_config(2)
        
    for i in range(0,numer_of_runs):
        print("Run Number:", i+1)
        run_config(3)
        
    for i in range(0,numer_of_runs):
        print("Run Number:", i+1)
        run_config(4)
        
    for i in range(0,numer_of_runs):
        print("Run Number:", i+1)
        run_config(5)
        
    for i in range(0,numer_of_runs):
        print("Run Number:", i+1)
        run_config(6)
        
    for i in range(0,numer_of_runs):
        print("Run Number:", i+1)
        run_config(7)
        
    for i in range(0,numer_of_runs):
        print("Run Number:", i+1)
        run_config(8)
     
if run_experiment:
    experiment(5)