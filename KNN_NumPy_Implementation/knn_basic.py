"""
k-NN Basic Implementation
Pure NumPy implementation with no loops in distance calculation
Original implementation: 2019
"""

# k-NN classifier implementation for multi-class classification
# Features: Euclidean distance, tie-breaking for equal votes 

import numpy as np

kNN_K_AMOUNT = 1            # ----------> set k as desired (>=1)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - = FLAGS / DEBUG = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

debug_info = False # just some info when running (includes 50/50 cases)
short_run = False # helps to debug and stops flooding console (just run 20 cases)

# NOTE: this is only for debugging/info
# short run will affect result percentage (most values won't be included)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - = CALCULATE EUCLIDEAN DISTANCES = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def calculateDistances(npArr2D_trainingData, npArr1D_queryInstance):
    """ euclidean distance function (using Numpy)
        returns 1D np array: distance from query instance to all training data points
        returns 1D np array: indices INDICATING positions of fist array as if in sorted order 
        (note: array itself is NOT sorted! - see below)
    """
    
    # THE BELOW LINE OF CODE calculates the euclidean distance step by step as follows (from inside to outside)
    # 1. take trainingData array and subtract query instance from each row (exclude last column of data)
    # each feature in query is subtracted from each (matching position) feature in training data
    # 2. apply power of 2 (to each result) and them sum along every sinlge row
    # 3. square root of each result 
    # --> result is 1D Numpy Array that holds all euclidean distances from query to each point in the 2D array
    euclidDistancesArray = np.sqrt(np.sum((npArr2D_trainingData[:,0:10] - npArr1D_queryInstance[0:10])**2,axis=1))
    
    # THE BELOW LINE OF CODE creates an array with sorted indices in relation to euclidDistancesArray    
    # the first index [0] will return the index of the lowest euclidean distance for example and so on
    # this is basically to be understood as a map of the above euclidDistancesArray
    indicesSortedOrder = np.argsort(euclidDistancesArray)
    
    # Return both 1D arrays
    return euclidDistancesArray, indicesSortedOrder
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - = FILE INPUT = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
# reading csv data sources (Numpy)
trainingData = np.genfromtxt('data/classification/trainingData.csv', delimiter=',')
testData = np.genfromtxt('data/classification/testData.csv', delimiter=',')


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - = RUN = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

#create a match list to be populated
testDataClassification = np.zeros(len(testData)) # temporary placeholder

# go through each row in test data
for n in range(len(testData)):
    
    # calculate Euclidean Distances for query to test data
    euclidDist, indicesMap = calculateDistances(trainingData,testData[n]) # n times    
    
    # setup empty list --> assuming any number of k here so need to find "most" value
    placholderClassifications = np.zeros(kNN_K_AMOUNT) # k values to be appended    
    
    # for amount k - get matches closest distance and append classification result
    for i in range(0,kNN_K_AMOUNT):
        # use only col 10 = classification
        placholderClassifications[i] = trainingData[indicesMap[i],10]
    
    if kNN_K_AMOUNT > 1:
        # Numpy implementation to check for most common value (includes basic failsafe if a "tie" 50/50)
        # (usually this can be done with for example collections/Counter)
        unique_values, indices = np.unique(placholderClassifications, return_inverse=True)
        
        # failsafe (if we get 2 classifications in equal "strength" e.g. 2 vs 2)
        
        # the line below is a bit convoluted but works really well - it checks if there is more than 1 argmax present
        # note: the indicies here are really an "inverse map" of the placholderClassifications array (from unique_values)
        # the bincount really returns which INDEX is most prominent        
        
        # one way to do this with numpy instead of using a generic for loop
        tempCounter = 1
        while np.bincount(indices).tolist().count(np.bincount(indices)[np.argmax(np.bincount(indices))]) > 1:
            
            if debug_info: # DEBUG ONLY
                print("50/50 occured")
                
            # add another k as exception until one wins over the other
            # later should use weighting instead - but not in Part 1 yet
            if len(indicesMap) > i + tempCounter: # make sure we don't go out of bounds for the very last one (unlikely but just in case)
                placholderClassifications = np.append(placholderClassifications,trainingData[indicesMap[i+tempCounter],10])
                tempCounter+=1
                # update (as we are having new values now obviously)
                unique_values, indices = np.unique(placholderClassifications, return_inverse=True)
        
        if debug_info: # DEBUG ONLY
            print("Class is:", unique_values[np.argmax(np.bincount(indices))])
        
        # assign value to array
        testDataClassification[n] = unique_values[np.argmax(np.bincount(indices))]
        
        
    else: # k = 1 (simple) - assign value to array
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
print("The test data returned an", np.mean(kNN_results) * 100 , "% accuracy with", kNN_K_AMOUNT, "k nearest neighbour.")
    


