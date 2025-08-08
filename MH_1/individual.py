"""
Basic TSP Example
file: Individual.py
"""

import random
import math


class Individual:
    def __init__(self, _size, _data, _init_random):
        
        """
        Parameters and general variables
        """
        self.fitness    = 0 # fitness
        self.genes      = [] # genes (city) list
        self.genSize    = _size # length of gene
        self.data       = _data # passed in dictionary
        self.genes = list(self.data.keys()) # genes = dict keys (list)
        self.initial_population = _init_random

        if self.initial_population: # random selection (and shuffle around keys)
            for i in range(0, self.genSize): 
                n1 = random.randint(0, self.genSize-1)
                n2 = random.randint(0, self.genSize-1)
                
                tmp = self.genes[n2]
                self.genes[n2] = self.genes[n1]
                self.genes[n1] = tmp
        
        else: # heuristic (nearest neighbour) - adapted from original Lab 1 solution
            
            cities = list(self.data.keys())
            cIndex = random.randint(0, self.genSize-1)            
            
            solution = [cities[cIndex]]         
            del cities[cIndex]
        
            current_city = solution[0]            
            while len(cities) > 0:
                bCity = cities[0]                
                bCost = self.euclideanDistance(current_city, bCity)
                bIndex = 0
        
                for city_index in range(1, len(cities)):
                    city = cities[city_index]                    
                    cost = self.euclideanDistance(current_city, city)
        
                    if bCost > cost:
                        bCost = cost
                        bCity = city
                        bIndex = city_index
            
                current_city = bCity
                solution.append(current_city)
                del cities[bIndex]
                
            self.genes = solution
                
        
            
    
    def setGene(self, _genes):
        """
        Updating current chromosome
        """
        self.genes = []
        for gene_i in _genes:
            self.genes.append(gene_i)

    def copy(self):
        """
        Creating a new individual
        """
        ind = Individual(self.genSize, self.data,self.initial_population)
        for i in range(0, self.genSize):
            ind.genes[i] = self.genes[i]
        ind.fitness = self.getFitness()
        return ind

    def euclideanDistance(self, c1, c2):
        """
        Distance between two cities
        """
        d1 = self.data[c1]
        d2 = self.data[c2]
        return math.sqrt( (d1[0]-d2[0])**2 + (d1[1]-d2[1])**2 )

    def getFitness(self):
        return self.fitness

    def computeFitness(self):
        """
        Computing the cost or fitness of the individual
        """
        self.fitness    = self.euclideanDistance(self.genes[0], self.genes[len(self.genes)-1]) # last city back to first        
        for i in range(0, self.genSize-1):
            self.fitness += self.euclideanDistance(self.genes[i], self.genes[i+1]) # all other cities distance

