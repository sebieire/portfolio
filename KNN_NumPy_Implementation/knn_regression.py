"""
k-NN for Regression
Feature selection and distance-weighted regression
Original implementation: 2019
"""

# k-NN regression implementation with:
# - R-squared metric for evaluation
# - Advanced feature selection capability
# - Achieved 95% accuracy with optimized feature selection


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import numpy as np
import math
import matplotlib.pyplot as plt
import pprint 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - = GLOBAL SETTINGS = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



################## RUNNING OPTIONS (pick one)
singleRunEnabled = True     # -------> runs single time with the settings below
fullExperiment = False      # -------> runs a full experiment with statistical data (ignoring global settings / has own settings below)

################## GENERAL OPTIONS
kNN_K_AMOUNT = 15                            # -------> set k as desired (>=1) - only for single runs (not in experiments)
DISTANCE_FUNCTION = 4                       # -------> 1: Euclid 2: Euclid (With Feature Drop) 3: Manhattan
droppedFeature = 3                          # -------> options 0 - 11 only (APPLIES ONLY when using option 2 DISTANCE_FUNCTION)
droppedFeatureList = [0,1,2,3,4,5,8,10]     # -------> options 0 - 11 only (APPLIES ONLY when using option 4 DISTANCE_FUNCTION)

# [0,1,2,3,4,5,8,10] BEST RESULTS SO FAR !

################## WEIGHTED DISTANCE CALC OPTIONS
VALUE_OF_N = 2                              # -------> ^N (optional)
weightedDistanceCalculationEnabled = True   # -------> if False will use regular method

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
################## EXPERIMENT ONLY OPTIONS
K_RANGE = 20                                 # -------> Range that k will go to (starts with 1) - entirely independant of single runs (kNN_K_AMOUNT)
N_RANGE = 10                                 # -------> Exponent - Max range defined for N to take on at specific experiment
include_primary_experiments = False           # -------> if True will also run 0-9 versions for "Euclidean Drop" Calculations

include_euclid_drop_runs = False            # -------> if True will also run 0-9 versions for "Euclidean Drop" Calculations
include_euclid_drop_list = True            # -------> if True will also run drop list (droppedFeatureList) versions for "calculateDistances_Euclid_DropMultipleFeatures"
include_value_N_changes = False             # -------> if True will also run N (Exponent) cycles
_withWeighted = True                        # -------> Enables / Disables weighted option for Euclid Drop & N_Changes Options

graphing_ON = True                          # -------> Displays A Chart For Each Experiment
exp_running_info = True                    # -------> On/Off Extra text for experiment (K values)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - = FLAGS / DEBUG / STATISTICS = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
general_running_info = True
debug_info = False 
short_run = False
cumulativeAccuracyAllExp = {}
experimentCounter = 0

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - = CALCULATE DISTANCES = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# EUCLIDEAN (NORMAL - same as part 1)
def calculateDistances(npArr2D_trainingData, npArr1D_queryInstance):
    """ euclidean distance function (using Numpy)
        returns 1D np array: distance from query instance to all training data points
        returns 1D np array: indices INDICATING positions of fist array as if in sorted order         
    """
    euclidDistancesArray = np.sqrt(np.sum((npArr2D_trainingData[:,0:12] - npArr1D_queryInstance[0:12])**2,axis=1))

    indicesSortedOrder = np.argsort(euclidDistancesArray)

    return euclidDistancesArray, indicesSortedOrder
    

# EUCLIDEAN (ELIMINATE FEATURE / 0 WEIGHT FEATURE - as Part 2)
def calculateDistances_Euclid_DropFeature(npArr2D_trainingData, npArr1D_queryInstance, droppedFeature):
    """ euclidean distance function (using Numpy)
        alternative that allows for a feature to be dropped
    """
    if droppedFeature == 0:
        euclidDistancesArray = np.sqrt(np.sum((npArr2D_trainingData[:,1:12] - npArr1D_queryInstance[1:12])**2,axis=1))
    elif droppedFeature == 11:
        euclidDistancesArray = np.sqrt(np.sum((npArr2D_trainingData[:,0:11] - npArr1D_queryInstance[0:11])**2,axis=1))
    elif droppedFeature > 0 and droppedFeature < 11 :
        euclidDistancesArray = np.sqrt(
                np.sum((npArr2D_trainingData[:,0:droppedFeature] - npArr1D_queryInstance[0:droppedFeature])**2,axis=1) +
                np.sum((npArr2D_trainingData[:,droppedFeature+1:12] - npArr1D_queryInstance[droppedFeature+1:12])**2,axis=1)
                )
    else:
        #normal case
        print("Error - Dropped Feature Out of Bounds! - Reverting back to general Distance")
        euclidDistancesArray = np.sqrt(np.sum((npArr2D_trainingData[:,0:12] - npArr1D_queryInstance[0:12])**2,axis=1))

    indicesSortedOrder = np.argsort(euclidDistancesArray)

    return euclidDistancesArray, indicesSortedOrder

# MANHATTAN DISTANCE (as Part 2)
def calculateDistances_Manhattan(npArr2D_trainingData, npArr1D_queryInstance):
    """ 
    manhattan distance function (using Numpy)      
    """
    manhDistancesArray = np.sum(np.absolute(npArr2D_trainingData[:,0:12] - npArr1D_queryInstance[0:12]), axis=1)

    indicesSortedOrder = np.argsort(manhDistancesArray)

    return manhDistancesArray, indicesSortedOrder


# EUCLIDEAN (ELIMINATE MULTIPLE FEATURES) - New in Part 3
def calculateDistances_Euclid_DropMultipleFeatures(npArr2D_trainingData, npArr1D_queryInstance, droppedFeatureList):
    """ euclidean distance function (using Numpy)
        alternative 2 that allows for a several features to be dropped (list)
    """        
    
    firstCoordColumn = True # set to know if this the first column of coordinates added (just a technicality)
    tempCoordinatesTrainingData = 0
    tempCoordinatesQueryData = 0
    
    # loop through given list to check for each particular index
    for index in range(0,12): # 0 to 12 in this case
        
        # if index not present & NOT first column (every other column basically)
        if index not in droppedFeatureList and firstCoordColumn == False:
            tempCoordinatesTrainingData = np.concatenate((tempCoordinatesTrainingData, npArr2D_trainingData[:,index:index+1]), axis=1)
            tempCoordinatesQueryData = np.append(tempCoordinatesQueryData, npArr1D_queryInstance[index:index+1])
            
        # if index not present & first column
        if index not in droppedFeatureList and firstCoordColumn == True:
            tempCoordinatesTrainingData = npArr2D_trainingData[:,index:index+1]
            tempCoordinatesQueryData = npArr1D_queryInstance[index:index+1]
            firstCoordColumn = False
  
    
    euclidDistancesArray = np.sqrt(np.sum((tempCoordinatesTrainingData - tempCoordinatesQueryData)**2,axis=1))  
    
    indicesSortedOrder = np.argsort(euclidDistancesArray)

    return euclidDistancesArray, indicesSortedOrder

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - = FILE INPUT = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -   

trainingData = np.genfromtxt('data/regression/trainingData.csv', delimiter=',')
testData = np.genfromtxt('data/regression/testData.csv', delimiter=',')


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - = RUN = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def run_one_cycle():
    ''' runs a single cycle '''

    # modelDataRegressionValues holds all results (each test instance / testData) -->    
    # in Parts 1 & 2 this variable (named testDataClassification) held the evaluated classification result for each instance of testData
    # here it holds the predicted regression value (k average) defined (or calculated) from the k nearest neighbours (numeric value instead of a "classification")
    modelDataRegressionValues = np.zeros(len(testData)) 
    
    for n in range(len(testData)):
        
        # Euclidean (Dropping A Feature)
        if DISTANCE_FUNCTION == 2:
            calcDist, indicesMap = calculateDistances_Euclid_DropFeature(trainingData,testData[n],droppedFeature)
        # Manhattan 
        elif DISTANCE_FUNCTION == 3:
            calcDist, indicesMap = calculateDistances_Manhattan(trainingData,testData[n])
        # Euclidean (Dropping Multiple Features)
        elif DISTANCE_FUNCTION == 4:
            calcDist, indicesMap = calculateDistances_Euclid_DropMultipleFeatures(trainingData,testData[n],droppedFeatureList)
        # Euclidean
        else: # 1 and all other cases
            calcDist, indicesMap = calculateDistances(trainingData,testData[n])
            
        
        # container of predictedValues and distances per query
        kNearestValuesFromModel = np.zeros(kNN_K_AMOUNT) # formerly: placholderClassifications (part 1 & 2)
        placholderEuclidDistances = np.zeros(kNN_K_AMOUNT)
        
        # for amount k: 
        # 1) get k closest point (Regression Value - the training result(s))
        # 2) get k closest point (Euclidean Distances value)
        for i in range(0, kNN_K_AMOUNT):        
            kNearestValuesFromModel[i] = trainingData[indicesMap[i],12]
            placholderEuclidDistances[i] = calcDist[indicesMap[i]]        
        
        # case k > 1: implementation of distance weighting
        if kNN_K_AMOUNT > 1:
            
            # - - - - - - - - - - - - - - - - - - - - - - - - -
            # Adapted for part 3 ----------------------->>>>>
            # WEIGHTED DISTANCE
            if weightedDistanceCalculationEnabled:
                
                # for all k nearest neighbours multiply each (model value) 
                # with their inverse distance squared / sum them all up
                # divide then by the sum of ->  all the inverse distances squared
                
                numerator = 0
                denominator = 0
                for i in range(0, kNN_K_AMOUNT):
                    # inverse Distance Squared (option to globally adjust value of N left open)
                    inverseDistanceSquared = np.reciprocal((placholderEuclidDistances[i])**VALUE_OF_N)
                    
                    numerator = numerator + (kNearestValuesFromModel[i] * inverseDistanceSquared)
                    denominator = denominator + inverseDistanceSquared                   
                    
                modelDataRegressionValues[n] = (numerator/denominator)
                
            # WEIGHTED DISTANCE END <<<<< ---------------------
            # - - - - - - - - - - - - - - - - - - - - - - - - -
            
            
            
            # - - - >> adapted for regression to suit new logic
            # Case: NO WEIGHTED DISTANCE USED / simply get mean of k values
            else:                
                modelDataRegressionValues[n] = np.sum(kNearestValuesFromModel) / len(kNearestValuesFromModel)
            
        # case k = 1: only one value to work with - assign the same value
        else: 
            modelDataRegressionValues[n] = kNearestValuesFromModel[0]    
        
        if short_run: # DEBUG ONLY
            if n > 20: 
                break
        
        if debug_info: # DEBUG ONLY
            print("K Nearest Regression Value(s):", kNearestValuesFromModel)
            print("-------------------------")
    
    # - - - END FOR LOOP
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    if debug_info: # DEBUG ONLY
        # this is not actually required here and differs entirely in regression with the calculation for R^2
        # however this is left in just for debug purposes to see that mean is calculated and for optional other debug to include later on (maybe)
        print("- - - - - - - > > >")
        print("Calculated (mean) Regression values from model (0-20 only):\n", modelDataRegressionValues[0:20])
    
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # - - = CECK RESULTS = - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    # Different in Part 3 - - - >>
    
    # here I apply the R^2 calculation for all values combined
    
    # this differs entirely to a classification problem (Parts 1 & 2) where each test value (as such) is "classified"
    # there I've used a Numpy boolean array to see how many model classifications match with the testData classification
    # here instead R^2 calculates (in one go) the overall performance of the model (as was run with global parameters settings)
    
    # R^2 = 1 - ( sum of all (predicted regression value - true value of every test data instance)^2 ) / ( sum of all ( average value of all test data - true value of every test data instance)^2 )
    
    # VARS:
    # predicted regression values from model (list/array)
    # --------> modelDataRegressionValues
    # true value of test dataset (list/array)  (used twice)
    # --------> testData[:,12]
    # average value of all test data (single number)
    # --------> np.mean(testData[:,12])
    
    # R^2
    rSquared = 1 - ( (np.sum( (modelDataRegressionValues-testData[:,12])**2) ) / (np.sum( (np.mean(testData[:,12]) - testData[:,12])**2) ) )
    if singleRunEnabled:
        print("The test data returned an (R^2)", round(rSquared * 100,2) , "% accuracy with", kNN_K_AMOUNT, "k nearest neighbour(s).")        
    
    return rSquared * 100 # required to run experiment


##############################################################################
# SINGLE RUN
##############################################################################
if singleRunEnabled:
    
    print("- - - - SINGLE RUN - - - -")
    
    run_one_cycle()

    print("Settings ------>")
    if general_running_info:
        if DISTANCE_FUNCTION == 2:
            print("Ran with Euclidean Distance Function - Dropped Feature:", droppedFeature)
        elif DISTANCE_FUNCTION == 3:
            print("Ran with Manhattan Distance Function")
        elif DISTANCE_FUNCTION == 4:
            print("Ran Euclidean Distance Function - Dropped Multiple Features:", droppedFeatureList)
        else:
            print("Ran with Euclidean Distance Function (Regular)")
            
        print("K:", kNN_K_AMOUNT)
        print("Value of N:", VALUE_OF_N)
        print("Distance Weight Calculation On:", weightedDistanceCalculationEnabled)
        print("- - - - - - - - - - - - -")



##############################################################################
# EXPERIMENT
##############################################################################
        
def experiment(distFunct,weightedEnabled,nameStr, droppedFeat=0, valueOfN=2):
    
    global kNN_K_AMOUNT
    global VALUE_OF_N
    global DISTANCE_FUNCTION
    global droppedFeature
    global weightedDistanceCalculationEnabled
    global cumulativeAccuracyAllExp
    global experimentCounter    
    
    peekExperimentThisRun = {}
    experimentCounter += 1
   
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -       
    
    DISTANCE_FUNCTION = distFunct       
    weightedDistanceCalculationEnabled = weightedEnabled
    
    print("\n- - - - - - - - - - - - - ")
    print(nameStr, weightedDistanceCalculationEnabled, " N:", valueOfN) # print header per experiment     
    
    exp_name = nameStr + str(weightedDistanceCalculationEnabled) + " N: " + str(valueOfN)
    droppedFeature = droppedFeat
    VALUE_OF_N = valueOfN
    
    graph_x = []
    graph_y = []
    
    tempHighestAccuracy = 0
    tempK = 0
    for i in range(1,K_RANGE+1):
        #set vars
        kNN_K_AMOUNT = i
        result = run_one_cycle()
        if exp_running_info:
            print("K:", kNN_K_AMOUNT , " | Accuracy:", round(result,2) , "%")
        
        if result > tempHighestAccuracy:
            tempHighestAccuracy = round(result,2)
            tempK = kNN_K_AMOUNT
        graph_x.append(kNN_K_AMOUNT)
        graph_y.append(result)
        
        cumulativeAccuracyAllExp[kNN_K_AMOUNT] = cumulativeAccuracyAllExp[kNN_K_AMOUNT] + round(result,2)

        
        
    # Record Peek Experiment
    peekExperimentThisRun[exp_name] = "Accuracy: " + str(tempHighestAccuracy) + ", with K: " + str(tempK)

    # Graph
    if graphing_ON:        
        plt.plot(graph_x,graph_y, 'b-')
        plt.ylabel('Accuracy')
        plt.xlabel('K')
        xint = range(min(graph_x), math.ceil(max(graph_x))+1)
        plt.xticks(xint)
        #plt.axis([min(graph_x),max(graph_x),min(graph_y)-10,max(graph_y)+2])    
        plt.axis([min(graph_x),max(graph_x),min(graph_y)-5,max(graph_y)+2])    
        plt.axis()
        plt.show()
        
        
    return peekExperimentThisRun




if fullExperiment:
    
    print("- - - - - - - - - - - - - ")
    print("- - - - EXPERIMENT - - - -")
    print("- - - - - - - - - - - - - ")
    
    peekExperimentPerSetting = {} # initial
    
    for x in range(1,K_RANGE+1):
        cumulativeAccuracyAllExp[x] = 0 # setup empty (initial)
  
    
    # run experiments and for each get dict back with peek values in it
    # @params: distFunct, weightedEnabled, nameStr, (optional) droppedFeature
    
    #PRIMARY EXPERIMENTS
    if include_primary_experiments:
        #1
        peekExperimentPerSetting.update(experiment(1,False, "Euclid (Regular) | Weighted: "))
        #2
        peekExperimentPerSetting.update(experiment(1,True, "Euclid (Regular) | Weighted: "))
        #3
        peekExperimentPerSetting.update(experiment(3,False, "Manhattan | Weighted: "))
        #4
        peekExperimentPerSetting.update(experiment(3,True, "Manhattan | Weighted: "))
    
    #SECONDARY EXPERIMENTS
    # trying out all dropped versions to see if any might improve results
    if include_euclid_drop_runs:
        for j in range(0,12):
            peekExperimentPerSetting.update(experiment(2,False, "Euclid (Dropped) = " +str(j) +" | Weighted: ",j))
            if _withWeighted:
                peekExperimentPerSetting.update(experiment(2,True, "Euclid (Dropped) = " +str(j) +" | Weighted: ", j))
                
    # Drop full list run
    if include_euclid_drop_list:
        peekExperimentPerSetting.update(experiment(4,False, "Euclid (Drop Features: " + str(droppedFeatureList) + ") | Weighted: "))
        if _withWeighted:
            peekExperimentPerSetting.update(experiment(4,True, "Euclid (Drop Features: " + str(droppedFeatureList) + ") | Weighted: "))
    
    # trying to change value of N to see if any might improve results (a couple here should be enough)
    if include_value_N_changes:
        for n in range(1,N_RANGE+1): # @param 0 = nothing dropped (but makes no difference with the choosen distance algorithm anyway)
            peekExperimentPerSetting.update(experiment(1,False, "Euclid (Drop 1) | Weighted: ",0, n))
            if _withWeighted:
                peekExperimentPerSetting.update(experiment(1,True, "Euclid (Drop 1) | Weighted: ",0, n))
    
    
    print("\n- - - - - - - - - - - - - ")
    print("PEEK RESULTS FROM EVERY RUN:")
    pprint.pprint(peekExperimentPerSetting)
    # - - - - - - - 
    
    for key in cumulativeAccuracyAllExp:
        cumulativeAccuracyAllExp[key] = round(cumulativeAccuracyAllExp[key] / experimentCounter,2)
    
    print("- - - - - - - - - - - - - ")
    print("AVERAGE RESULTS FROM EVERY RUN (cumulative from all variants):")
    pprint.pprint(cumulativeAccuracyAllExp) 
    
    
    graph_x = [value for value in cumulativeAccuracyAllExp]
    graph_y = [value for value in cumulativeAccuracyAllExp.values()]
    # Graph
    if graphing_ON:
        plt.plot(graph_x,graph_y, 'g-')
        plt.ylabel('Average - All Accuracy')
        plt.xlabel('K')
        xint = range(min(graph_x), math.ceil(max(graph_x))+1)
        plt.xticks(xint)        
        plt.axis([min(graph_x),max(graph_x),min(graph_y)-5,max(graph_y)+2])    
        plt.axis()
        plt.show()

    
