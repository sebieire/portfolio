"""
k-NN Enhanced Implementation
Distance-weighted k-NN with multiple distance metrics
Original implementation: 2019
"""

# Enhanced k-NN implementation with:
# - Distance-weighted voting
# - Multiple distance metrics (Euclidean, Manhattan)
# - Feature dropping capability
# - Comprehensive experiment mode for hyperparameter tuning

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import numpy as np
import math
import matplotlib.pyplot as plt
import pprint # pretty print for outputting statistic dictionaries a bit better


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - = GLOBAL SETTINGS = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



################## RUNNING OPTIONS (pick one)
singleRunEnabled = True     # -------> runs single time with the settings below
fullExperiment = False      # -------> runs a full experiment with statistical data (ignoring global settings / has own settings)

################## GENERAL OPTIONS
kNN_K_AMOUNT = 9           # -------> set k as desired (>=1) - only for single runs (not in experiments)
DISTANCE_FUNCTION = 1       # -------> 1: Euclid 2: Euclid (With Feature Drop) 3: Manhattan
droppedFeature = 3          # -------> options 0 - 9 only (APPLIES ONLY when using option 2 above)

################## WEIGHTED DISTANCE CALC OPTIONS
VALUE_OF_N = 1                              # -------> ^N (optional)
weightedDistanceCalculationEnabled = True   # -------> if False will use regular method
inverseDistance = True                      # -------> True if distance weight is to be inverted for calculation

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
################## EXPERIMENT ONLY OPTIONS
K_RANGE = 30                                # -------> Range that k will go to (starts with 1) - entirely independant of single runs (kNN_K_AMOUNT)
include_euclid_drop_runs = False         # -------> if True will also run 0-9 versions for "Euclidean Drop" Calculations
graphing_ON = True                          # -------> Displays A Chart For Each Experiment
exp_running_info = True                     # -------> On/Off Extra text for experiment (K values)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - = FLAGS / DEBUG / STATISTICS = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


general_running_info = True

debug_info = False # just some info when running (includes 50/50 cases)
short_run = False # helps to debug and stops flooding console (just run 20 cases)

cumulativeAccuracyAllExp = {}
experimentCounter = 0


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - = CALCULATE DISTANCES = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# EUCLIDEAN (NORMAL - as part 1)
def calculateDistances(npArr2D_trainingData, npArr1D_queryInstance):
    """ euclidean distance function (using Numpy)
        returns 1D np array: distance from query instance to all training data points
        returns 1D np array: indices INDICATING positions of fist array as if in sorted order         
    """
    euclidDistancesArray = np.sqrt(np.sum((npArr2D_trainingData[:,0:10] - npArr1D_queryInstance[0:10])**2,axis=1))

    indicesSortedOrder = np.argsort(euclidDistancesArray)

    return euclidDistancesArray, indicesSortedOrder
    

# EUCLIDEAN (ELIMINATE FEATURE / 0 WEIGHT FEATURE) ---> NEW IN PART 2
def calculateDistances_Euclid_DropFeature(npArr2D_trainingData, npArr1D_queryInstance, droppedFeature):
    """ euclidean distance function (using Numpy)
        alternative that allows for a feature to be dropped
    """
    if droppedFeature == 0:
        euclidDistancesArray = np.sqrt(np.sum((npArr2D_trainingData[:,1:10] - npArr1D_queryInstance[1:10])**2,axis=1))
    elif droppedFeature == 9:
        euclidDistancesArray = np.sqrt(np.sum((npArr2D_trainingData[:,0:9] - npArr1D_queryInstance[0:9])**2,axis=1))
    elif droppedFeature > 0 and droppedFeature < 9 :
        euclidDistancesArray = np.sqrt(
                np.sum((npArr2D_trainingData[:,0:droppedFeature] - npArr1D_queryInstance[0:droppedFeature])**2,axis=1) +
                np.sum((npArr2D_trainingData[:,droppedFeature+1:10] - npArr1D_queryInstance[droppedFeature+1:10])**2,axis=1)
                )
    else:
        #normal case
        print("Error - Dropped Feature Out of Bounds! - Reverting back to general Distance")
        euclidDistancesArray = np.sqrt(np.sum((npArr2D_trainingData[:,0:10] - npArr1D_queryInstance[0:10])**2,axis=1))

    indicesSortedOrder = np.argsort(euclidDistancesArray)

    return euclidDistancesArray, indicesSortedOrder

# MANHATTAN DISTANCE ---> NEW IN PART 2
def calculateDistances_Manhattan(npArr2D_trainingData, npArr1D_queryInstance):
    """ 
    manhattan distance function (using Numpy)      
    """
    manhDistancesArray = np.sum(np.absolute(npArr2D_trainingData[:,0:10] - npArr1D_queryInstance[0:10]), axis=1)

    indicesSortedOrder = np.argsort(manhDistancesArray)

    return manhDistancesArray, indicesSortedOrder

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - = FILE INPUT = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -   

trainingData = np.genfromtxt('data/classification/trainingData.csv', delimiter=',')
testData = np.genfromtxt('data/classification/testData.csv', delimiter=',')


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - = RUN = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# now a function in Part 2
def run_one_cycle():
    ''' runs a single cycle '''

    # temporary placeholder
    testDataClassification = np.zeros(len(testData)) 
    
    for n in range(len(testData)):    
        
        # Euclidean (Dropping A Feature)
        if DISTANCE_FUNCTION == 2:
            calcDist, indicesMap = calculateDistances_Euclid_DropFeature(trainingData,testData[n],droppedFeature)
        # Manhattan 
        elif DISTANCE_FUNCTION == 3:
            calcDist, indicesMap = calculateDistances_Manhattan(trainingData,testData[n])        
        # Euclidean
        else: # 1 and all other cases
            calcDist, indicesMap = calculateDistances(trainingData,testData[n])
            
        
        # initial placholders per query
        placholderClassifications = np.zeros(kNN_K_AMOUNT)
        placholderEuclidDistances = np.zeros(kNN_K_AMOUNT) 
        
        # for amount k: 
        # 1) get k closest point (Classification)
        # 2) get k closest point (Euclidean Distances value)
        for i in range(0, kNN_K_AMOUNT):        
            placholderClassifications[i] = trainingData[indicesMap[i],10]
            placholderEuclidDistances[i] = calcDist[indicesMap[i]]
        
        # case k > 1: implementation of distance weighting
        if kNN_K_AMOUNT > 1:       
            
            
            
            # NEW IN PART 2 ----------------------->>>>>
            # WEIGHTED DISTANCE
            if weightedDistanceCalculationEnabled:
                
                # for same indices -> combine  // then use on placholderEuclidDistances to calculate // highest value wins        
                
                # required temp values
                unique_vals = np.unique(placholderClassifications)        
                tempMaxValue = 0
                tempClassification = 0
                
                # for each unique value (mostly Numpy used here)
                for i in range(len(unique_vals)):
                    # map a boolean array (match current unique value with classification array)
                    boolIteratorMatchIndices = unique_vals[i] == placholderClassifications
                    # with that, get matching values (indices) from the distance array
                    tempDistanceList = placholderEuclidDistances[boolIteratorMatchIndices]
                    
                    # calculate value (reciprocal per value)
                    if inverseDistance:
                        tempDistanceList = np.reciprocal(tempDistanceList)                
                    
                    # using N
                    calculated = np.power(np.sum(tempDistanceList),VALUE_OF_N)
                    # if highest value -> correct classification
                    if calculated > tempMaxValue:
                        tempMaxValue = calculated
                        tempClassification = unique_vals[i]
                        
                # assign correct classification
                testDataClassification[n] = tempClassification
                
                
                if debug_info: # DEBUG ONLY                
                    print(tempMaxValue)
            
            
            
            # SAME AS PART 1 ----------------------->>>>>
            else: 
                
                # BELOW IS SAME AS PART 1 (used for comparison testing)
                unique_values, indices = np.unique(placholderClassifications, return_inverse=True)        
                tempCounter = 1
                while np.bincount(indices).tolist().count(np.bincount(indices)[np.argmax(np.bincount(indices))]) > 1:
                    if len(indicesMap) > i + tempCounter:
                        placholderClassifications = np.append(placholderClassifications,trainingData[indicesMap[i+tempCounter],10])
                        tempCounter+=1
                        unique_values, indices = np.unique(placholderClassifications, return_inverse=True)
                testDataClassification[n] = unique_values[np.argmax(np.bincount(indices))]
            
    
            
        # case k = 1: only one value to work with - assign classification
        else: 
            testDataClassification[n] = placholderClassifications[0]    
        
        if short_run: # DEBUG ONLY
            if n > 20: 
                break
        
        if debug_info: # DEBUG ONLY
            print("Classifications:", placholderClassifications)
            print("-------------------------")
    
    # - - - END FOR LOOP
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    if debug_info: # DEBUG ONLY
        print("- - - - - - - > > >")
        print("Classification List (0-20 only):", testDataClassification[0:20])
    
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # - - = CECK RESULTS = - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    # check original test data classification with the result classification (numpy boolean)
    kNN_results = testData[:,10] == testDataClassification
    if singleRunEnabled:
        print("The test data returned an", np.mean(kNN_results) * 100 , "% accuracy with", kNN_K_AMOUNT, "k nearest neighbour.")
        
    return np.mean(kNN_results) * 100 # required to run experiment
    


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
        else:
            print("Ran with Euclidean Distance Function (Regular)")
            
        print("K:", kNN_K_AMOUNT)
        print("Distance Weight Calculation On:", weightedDistanceCalculationEnabled)
        print("- - - - - - - - - - - - -")



##############################################################################
# EXPERIMENT
##############################################################################            
        
def experiment(distFunct,weightedEnabled,nameStr, droppedFeat=0):
    
    global kNN_K_AMOUNT
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
    print(nameStr, weightedDistanceCalculationEnabled)         
    exp_name = nameStr + str(weightedDistanceCalculationEnabled)
    droppedFeature = droppedFeat
    
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
        plt.axis([min(graph_x),max(graph_x),75,95])    
        plt.axis()
        plt.show()
        
        
    return peekExperimentThisRun




if fullExperiment:
    
    print("- - - - - - - - - - - - - ")
    print("- - - - EXPERIMENT - - - -")
    print("- - - - - - - - - - - - - ")
    
    peekExperimentPerSetting = {} # initial
    
    for x in range(1,K_RANGE+1):
        cumulativeAccuracyAllExp[x] = 0 # setup 
  
    
    # run experiments and for each get dict back with peek values in it
    # @params: distFunct, weightedEnabled, nameStr, (optional) droppedFeature
    
    #1
    peekExperimentPerSetting.update(experiment(1,False, "Euclid (Regular) | Weighted: "))
    #2
    peekExperimentPerSetting.update(experiment(1,True, "Euclid (Regular) | Weighted: "))
    #3
    peekExperimentPerSetting.update(experiment(3,False, "Manhattan | Weighted: "))
    #4
    peekExperimentPerSetting.update(experiment(3,True, "Manhattan | Weighted: "))    
    
    # trying out all dropped versions to see if any might improve results
    if include_euclid_drop_runs:
        for j in range(0,10):
            peekExperimentPerSetting.update(experiment(2,False, "Euclid (Dropped) = " +str(j) +" | Weighted: ",j))    
            peekExperimentPerSetting.update(experiment(2,True, "Euclid (Dropped) = " +str(j) +" | Weighted: ", j))
    
    
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
        plt.axis([min(graph_x),max(graph_x),75,95])    
        plt.axis()
        plt.show()

    
