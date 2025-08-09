import random
from . import settings # import module
from . thing_objects import Rabbit, Deer, Hut # import things
from . huntsman import Huntsman # import 

# ---------------------------------------------------------------------------
# - - = AGENTS  = - -
# ---------------------------------------------------------------------------


# ------------------------------------------
# Simple Reflex Agent
# ------------------------------------------
def simple_reflex_agent(percepts):
    ''' simple reflex agent - returns action solely based on percepts '''
    
    things, the_agent = percepts
    
    if settings.full_debug:
        print("PERCEPTS:", things)
    
    for thing in things: # going through the list of Things
        # Rabbit
        if isinstance(thing, Rabbit):
            return 'hunt rabbit'
        # Deer
        if isinstance(thing, Deer):
            return 'hunt deer'
        # Hut
        if isinstance(thing, Hut):
            # check if agent has either full energy OR rested already (max) 2 times
            if the_agent.energy < 100 and the_agent.rest_amount < 3:
                return 'rest'
            elif settings.basic_debug:
                print("Resting not allowed (full energy or resting amount > 1)")
    
    # Otherwise move either randomly in either direction
    return random.choice(['travel north', 'travel east','travel south', 'travel west'])



# ------------------------------------------
# Model Based Agent    
# ------------------------------------------
def model_based_agent(percepts):
    ''' keeps track of perceived environment (model of the world) returns an appropriate action based on percepts and model '''
    
    things, the_agent = percepts
    
    if settings.full_debug:
        print("PERCEPTS:", percepts)
    
    # model - keeping track of the world
    # set current agent location in world model to "visited"
    currentLocation = the_agent.location # list    
    the_agent.world_model[tuple(currentLocation)] = "visited"
    
    for thing in things: # going through the list of Things
        if isinstance(thing, Rabbit):
            return 'hunt rabbit'
        if isinstance(thing, Deer):
            return 'hunt deer'
        if isinstance(thing, Hut):            
            # if hut set model to "hut" instead of "visited" (so it can be visited again)
            the_agent.world_model[tuple(currentLocation)] = "hut"
            # check if agent has either full energy OR rested already (max) 2 times
            if the_agent.energy < 100 and the_agent.rest_amount < 3:
                return 'rest'
            elif settings.basic_debug:
                print("Resting not allowed (full energy or resting amount > 1)")
            
    
    # Check Directions / Compare To Existing Model / Get Appropriate Directions:
    def directionNotVisited(x,y):
        ''' checks if direction was not yet visited '''
        
        coord_X = currentLocation[0]
        coord_Y = currentLocation[1]
        
        #check if edges of ROUND world
        if x == -1 and coord_X == 0: # going north and currently most northern
            coord_X = the_agent.world_size_x #set to max x for below calc to end up on correct point
        if x == 1 and coord_X == the_agent.world_size_x-1: # going south and currently most southern
            coord_X = -1 #set to min x - 1
        if y == -1 and coord_Y == 0: # going west and currently most western
            coord_Y = the_agent.world_size_y #set to max y for below calc to end up on correct point
        if y == 1 and coord_Y == the_agent.world_size_y-1: # going east and currently most eastern
            coord_Y = -1 #set to min y - 1 
            
        # exists?
        if tuple( [coord_X+x , coord_Y+y] ) in the_agent.world_model.keys():            
            if the_agent.world_model[tuple( [coord_X+x , coord_Y+y] )] == "visited":
                # return False (== visited)
                return False
        else:
            return True
        
    
    tempListTravelOptions = []
    # check where agent has visited before (excludes hut locations)
    
    #north
    if directionNotVisited(-1,0):
        tempListTravelOptions.append('travel north')
    #south
    if directionNotVisited(1,0):
        tempListTravelOptions.append('travel south')
    #east
    if directionNotVisited(0,1):
        tempListTravelOptions.append('travel east')
    #west
    if directionNotVisited(0,-1):
        tempListTravelOptions.append('travel west')
    
    if tempListTravelOptions: # not empty
        return random.choice(tempListTravelOptions)
    
    else: #if there is no direction available at all - must pick one!
        return random.choice(['travel north', 'travel east','travel south', 'travel west'])
    
    #if none of above
    return 'Do Nothing At All' # if this is returned something is wrong.
    
    
# ------------------------------------------
# Utility Based Agent
# ------------------------------------------
def utility_based_agent(percepts):
    ''' 
        similar to model based agent keeps track of perceived environment (model of the world)
        however this agent will aim to maximise performance (secondary goals) but also try to survive the simulation (primary goal)
        if it feels it is close to zero energy it will ignore secondar goals and try to "find a hut" instead
        this agent takes the environment into consideration and can observe a radius of +1 from the current tile it is on
        (that includes diagonally as well -> up to 8 fields around it)
    '''
    
    things, the_agent = percepts
    
    if settings.full_debug:
        print("PERCEPTS:", percepts)
    
    # update agent world model
    for thing in things:
        #add huts and critter to world model
        if not isinstance(thing, Huntsman):
            the_agent.world_model[tuple(thing.location)] = thing
        else:
            the_agent.world_model[tuple(the_agent.location)] = 'empty'
    
    
    def returnDistanceBetweenTwoPoints(pointA,pointB):
        ''' returns distance between 2 points - takes world edges - round - into consideration'''
        
        dX = 0
        dY = 0
                
        # regular case
        dX = abs(pointA[0] - pointB[0])
        dY = abs(pointA[1] - pointB[1])        
        
        # world edge cases (overwrites regular cases)
        if pointA[0] == 0 and pointB[0] == the_agent.world_size_x-1:
            dX = 1
        elif pointB[0] == 0 and pointA[0] == the_agent.world_size_x-1:
            dX = 1
            
        if pointA[1] == 0 and pointB[1] == the_agent.world_size_y-1:
            dY = 1
        elif pointB[1] == 0 and pointA[1] == the_agent.world_size_y-1:
            dY = 1
            
        distance = dX + dY            
        return distance
        
    
    def getClosest(desired_thing):
        ''' returns closest to agent '''
        foundResultCoordinates = []
        # go through the current world model and find all instances of desired_thing
        for element in the_agent.world_model.values():
            if isinstance(element, desired_thing):
                # collect all coordinates
                foundResultCoordinates.append(element.location)
        
        # get coordinates of closest one
        if foundResultCoordinates: # not empty            
            # assign first one randomly
            theClosestOne = foundResultCoordinates[0]           
            for coordinate in foundResultCoordinates:                
                # measure distance for each found                 
                closestDistance = returnDistanceBetweenTwoPoints(the_agent.location,theClosestOne)
                # if better then previous assign current one
                if returnDistanceBetweenTwoPoints(the_agent.location,coordinate) < closestDistance:
                    theClosestOne = coordinate
            return theClosestOne
        else:
            return None
        
    
    def getClosestUnknownTile():
        ''' returns closest unexplored tile to the agent / not including current tile standing on'''
        
        foundResultCoordinates = []
        # go through the current world model and find tiles NOT EXPLORED (adjacent to explored tiles)                
        
        
        
        # below is NOT taking into account world edge (simply filtered out here - maybe implement later - still works fine though)
        for coordinate in the_agent.world_model:
            
            if coordinate[0]+1 < the_agent.world_size_x-1 and tuple([coordinate[0]+1,coordinate[1]]) not in the_agent.world_model:
                if coordinate[0]+1 != tuple(the_agent.location):
                    foundResultCoordinates.append([coordinate[0]+1,coordinate[1]])
            
            if coordinate[0]-1 > 0 and tuple([coordinate[0]-1,coordinate[1]]) not in the_agent.world_model:
                if coordinate[0]-1 != tuple(the_agent.location):
                    foundResultCoordinates.append([coordinate[0]-1,coordinate[1]])
            
            if coordinate[1]+1 < the_agent.world_size_y-1 and tuple([coordinate[0],coordinate[1]+1]) not in the_agent.world_model:
                if coordinate[1]+1 != tuple(the_agent.location):
                    foundResultCoordinates.append([coordinate[0],coordinate[1]+1])
                
            if coordinate[1]-1 > 0 and tuple([coordinate[0],coordinate[1]-1]) not in the_agent.world_model:
                if coordinate[1]-1 != tuple(the_agent.location):
                    foundResultCoordinates.append([coordinate[0],coordinate[1]-1])
                
        """
        # also add current agent neighbour tiles as well (in order to update results when agent is moving)
        if the_agent.location[0]+1 < the_agent.world_size_x-1 and tuple([the_agent.location[0]+1,the_agent.location[1]]) not in the_agent.world_model:            
                foundResultCoordinates.append([the_agent.location[0]+1,the_agent.location[1]])
        
        if the_agent.location[0]-1 > 0 and tuple([the_agent.location[0]-1,the_agent.location[1]]) not in the_agent.world_model:            
                foundResultCoordinates.append([the_agent.location[0]-1,the_agent.location[1]])
            
        if the_agent.location[1]+1 < the_agent.world_size_y-1 and tuple([the_agent.location[0],the_agent.location[1]+1]) not in the_agent.world_model:            
                foundResultCoordinates.append([the_agent.location[0],the_agent.location[1]+1])
                
        if the_agent.location[1]-1 > 0 and tuple([the_agent.location[0],the_agent.location[1]-1]) not in the_agent.world_model:            
                foundResultCoordinates.append([the_agent.location[0],the_agent.location[1]-1])
        """
                
        
        # same as "getClosest" from here
        if foundResultCoordinates: # not empty
            theClosestOne = foundResultCoordinates[0]
            for coordinate in foundResultCoordinates:
                # measure distance for each found                 
                closestDistance = returnDistanceBetweenTwoPoints(the_agent.location,theClosestOne)
                # if better then previous assign current one
                if returnDistanceBetweenTwoPoints(the_agent.location,coordinate) < closestDistance:
                    theClosestOne = coordinate
            return theClosestOne
        else:
            return None
        

            
    
    def walkTowardsCoordinate(coordinate):  
        ''' returns the appropriate move action based on target coordinate '''

        # world edge cases
        if coordinate[0] == 0 and the_agent.location[0] > the_agent.world_size_x/2: # half the x world space
            return 'travel south'
        if coordinate[0] == the_agent.world_size_x-1 and the_agent.location[0] < the_agent.world_size_x/2: # half the x world space
            return 'travel north'
        
        if coordinate[0] != the_agent.location[0]:
            if coordinate[0] == 0 and the_agent.location[0] < the_agent.world_size_x/2: # half the x world space
                return 'travel north'
        
            if coordinate[0] == the_agent.world_size_x-1 and the_agent.location[0] > the_agent.world_size_x/2: # half the x world space
                return 'travel south'
        
        if coordinate[1] != the_agent.location[1]:
            if coordinate[1] == 0 and the_agent.location[1] < the_agent.world_size_y/2: # half the x world space
                return 'travel west'
            if coordinate[1] == the_agent.world_size_y-1 and the_agent.location[1] > the_agent.world_size_y/2: # half the x world space
                return 'travel east'
            
        if coordinate[1] == 0 and the_agent.location[1] > the_agent.world_size_y/2: # half the x world space
            return 'travel east'        
        
        if coordinate[1] == the_agent.world_size_y-1 and the_agent.location[1] < the_agent.world_size_y/2: # half the x world space
            return 'travel west'
        
        
        # regular cases
        if coordinate[0] > the_agent.location[0]:            
            return 'travel south'
        if coordinate[0] < the_agent.location[0]:            
            return 'travel north'
        if coordinate[1] < the_agent.location[1]:            
            return 'travel west'
        if coordinate[1] > the_agent.location[1]:            
            return 'travel east'
        
        
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # setup rules so the agent can pick depending on own state and world state
    
    # if health below x (ignore all critter and find place to stay)
    if the_agent.energy < 30 and the_agent.rest_amount < 3:
        #find the nearest hut (if there is any known until now)
        closestHutCoordinate = getClosest(Hut)
        if closestHutCoordinate is not None:
            # arrived at coordinate and hut is there return 'rest'
            if closestHutCoordinate == the_agent.location:               
                return 'rest'
            else: #else walk towards it
                return walkTowardsCoordinate(closestHutCoordinate)            
        
        #don't know location of any hut - find closest unexplored title and go there - explore
        else:
            closestUnknownTile = getClosestUnknownTile()
            if closestUnknownTile is not None:
                if settings.basic_debug:
                    print("In search modus (hut) - closest unknown tile:", closestUnknownTile)
                return walkTowardsCoordinate(closestUnknownTile)


    # if health ok (>= x)
    elif the_agent.energy >= 30:
        
        # could implement random here for variety 
        # otherwise agent will systematically go for (known) deer and then rabbit
        # (maybe later)
        
        # below is technically a waste of resources (they should be implemted in the their IF statements)
        # however for some other logic reasons decided to do it this way
        # for a larger setup this should be "re-shuffled"
        
        closestDeerCoordinate = getClosest(Deer)
        closestRabbitCoordinate = getClosest(Rabbit)
        
        #find the nearest Deer (if any known and until none left -> hunt rabbit -> explore)        
        if closestDeerCoordinate is not None:
            # arrived at coordinate and deer is there return 'rest'
            if closestDeerCoordinate == the_agent.location:   

                # double checking here
                for thing in things:
                    if thing.location == the_agent.location: # same location
                        if isinstance(thing, Deer):                            
                            the_agent.world_model[tuple(thing.location)] = 'empty'                            
                            return 'hunt deer'
                    
            else: #else walk towards it
                return walkTowardsCoordinate(closestDeerCoordinate)
        
        
        #find the nearest Rabbit        
        elif closestRabbitCoordinate is not None:
            # arrived at coordinate and deer is there return 'rest'
            if closestRabbitCoordinate == the_agent.location:   

                # double checking here
                for thing in things:
                     if thing.location == the_agent.location: # same location
                         if isinstance(thing, Rabbit):                            
                            the_agent.world_model[tuple(thing.location)] = 'empty'
                            return 'hunt rabbit'
                    
            else: #else walk towards it
                return walkTowardsCoordinate(closestRabbitCoordinate)
            
        # this should only run if no further rabbit or deer is known
        # go exploring
        
        else:
            closestUnknownTile = getClosestUnknownTile()
            if closestUnknownTile is not None:
                if settings.basic_debug:
                    print("In search modus (hunt) - closest unknown tile:", closestUnknownTile)
                return walkTowardsCoordinate(closestUnknownTile)

        
        # this is technically the end of the game
        # if this is returned either something is wrong or game is done
        return 'Game End - Do Nothing At All'
        
  




    


