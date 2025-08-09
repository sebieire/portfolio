import random
import pprint

from . import settings # import module
from . import thing_objects # import module
from . import huntsman # import module
from . import search_functions # import module
from . import search_classes # import module

from . thing_objects import Thing, Hut # import classes
from . huntsman import Huntsman # import classes
from . search_functions import breadth_first_graph_search, depth_first_graph_search, depth_limited_search
from . search_classes import FindingHutProblem, CustomDirectedGraph

# ---------------------------------------------------------------------------
# - - = ENVIRONMENT (MAIN CLASS)  = - -
# ---------------------------------------------------------------------------
class World():
    
    def __init__(self, size_x=10, size_y=10):
        
        self.things = []
        self.agents = []
        self.searchedNodes = []
        self.used_locations = [] # to prevent a location being used twice when starting
        #self.critter_count = 0 # add 1 for every critter added and subtract one for every critter hunted (if "back to" 0 game ends)     # !-!
        self.rounds_played = 0 # add 1 for every round played (required for statistics later)
        
        # world size & starting coordinates
        self.world_size_x = size_x
        self.world_size_y = size_y
        
        # DATA variables (for statistics)
        self.data_agent_location_tracker = {} # tracks agents movements throughout the world
        self.data_hut_locations = [] # keep track of where the huts are in the world
        #self.data_deer_locations = [] # keep track of where the deer are in the world      # !-!
        #self.data_rabbit_locations = [] # keep track of where the deer are in the world    # !-!
        self.data_agent_starting_location = [] # useful for when using cloned world
        #self.data_critter_count = {} # additional critter development tracker over time    # !-!
        
        # setup & display  world grid
        self.world_grid = []
        self.initiate_world_grid(self.world_size_x, self.world_size_y)
        
        if(settings.basic_debug):
            self.display_world_grid()  # optional
    # <----- end constructor ----->
       

    
    def initiate_world_grid(self, world_x, world_y):
        ''' Initiates 2D World Grid '''
        
        tempList = []
        for i in range(world_x):
            if world_y is None: # equal square (x * x)
                for j in range(world_x):               
                    tempList.append([i,j])
            else: # rectangle with different length sides (x * y)
                for j in range(world_y):
                    tempList.append([i,j])        
        
        self.world_grid = tempList
        
        # initialise data variable grid (dictionary) to track agents location
        list_tuples = []
        for item in self.world_grid:            
            list_tuples.append(tuple(item))
        self.data_agent_location_tracker = {key: 0 for key in list_tuples}
        
        
    # <----- end initiate world grid ----->    
    
        
    def display_world_grid(self):
        ''' Displays The World Grid '''        
        print("The World Grid is layed out as follows:") 
        for row in range(self.world_size_y):
            for coordinate in self.world_grid:
                if coordinate[1] == row:
                    print(coordinate, end = '')
            print("") # line break
    # <----- end display world grid ----->
    
    
    def display_world_grid_with_things(self):
        ''' Displays The World Grid With Things In Coordinates'''
        print("Current Position Of Things") 
        for row in range(self.world_size_y):
            for coordinate in self.world_grid:
                things_at_location = self.list_things_at(coordinate, Thing)
                if coordinate[1] == row:
                    if len(things_at_location) != 0:                        
                        print("  ",things_at_location[0].__class__.__name__, "\t|", end = '') # print first item in list of things
                    else:
                        print('\tx\t|', end = '')
            print("") # line break
    # <----- end display world grid with things ----->
            


    # percept now obsolete -> world fully observable from get go
    
    # PERCEPT METHOD 
    def percept(self,agent):   
        ''' returns next Node state  '''        
        
        return (self.searchedNodes[self.rounds_played], agent)
    
        """
        things = self.list_things_at(agent.location) # returns things at current location
        
        if settings.full_debug:
            print("THINGS:",things, "SELF REST AMOUNT", agent.rest_amount)        
        
        return (things, agent)
        """
    
        
    
    
    # EXECUTE METHOD
    def execute_action(self, agent, action): 
        ''' changes the environment based on what the agents behaviour (program) is '''
        
        # DEBUG
        if settings.basic_debug:
            print("In Execute -->")
            print("Huntsman is at (pre-action):", agent.location)
            print("Huntsman has", agent.energy, "energy left (pre-action).")
            print("Huntsman has a performance score of:", agent.performance)
            print("Huntsman decides to:", action, "on this day.")            
            

             
        
        # move towards a specific direction
        if action == "travel north" or action == "travel east" or  action == "travel south" or action == "travel west":
            
            agent.travel(action)            
            agent.reset_time_based_stats()        
       
                    
        # Action == Rest
        if action == "rest" and agent.rest_amount < 2:
            items = self.list_things_at(agent.location, tclass=Hut)
            if len(items) != 0:
                if agent.rest(items[0]): # rest
                    if settings.game_info: 
                        print(str(agent)[1:-1], "decided to rest in a", str(items[0])[1:-1], "at location:", agent.location )  
                    
        # DEBUG
        if settings.basic_debug:
                print("Huntsman is currently at (post-action):", agent.location)
                print("Current Agent World Model (post-action):", agent.world_model)
        
        # DATA TRACKING: update agent location tracker (does not distinguish between agents! assuming only 1)
        # --> tracks location tile after something happens ("end of the day")
        self.data_agent_location_tracker[tuple(agent.location)] += 1   
       
        
    
    # STEP (changes for Search implemented)
    def step(self):
        """Runs the environment for one time step."""
        
        actions = []
        for agent in self.agents:            
                
            # return which action to take (through program(percepts))            
            actions.append(agent.program(self.percept(agent)))
            self.rounds_played += 1 # keep track how many rounds have been played
            
            # was nested inside this previously - now obsolete!
            """
            if agent.alive: # technically don't need this check here (duplicate)
                pass
            
            else:
                pass
                #actions.append("No Action - Lost In The Woods!") # changed from "" to be more clear
            """
        
        for (agent, action) in zip(self.agents, actions): # for agent & associated action (zip!)                
            self.execute_action(agent, action) # -> execute
    
        


    
    # RUN
    def run(self, steps=1000):
        """Run the Environment for given number of time steps."""        
        
        
        ################### EXECUTE SEARCH (get all list of all nodes tuplets)                
        finalNode = self.run_A_B_search(self.agents[0].location, self.things[1].location)
        
        if finalNode == "cutoff" or finalNode == None:            
            if finalNode == "cutoff":
                print("\n=== ERROR! === \nDepth Limited Search - Cutoff Reached (consider expanding limit!)")
            else:
                print("No Solution! Node = None")
                self.searchedNodes = None            
        else:            
            self.searchedNodes.extend(finalNode.solution())
            settings.nodeSolution = finalNode
                
        
        # - - - - - - - - - - - -
        
        #for step in range(steps):
        for step in range(len(self.searchedNodes)):            
            
            if(settings.visualise_map):
                self.display_world_grid_with_things() # Optional - Shows Each Step Visually
                
            if settings.basic_debug:
                print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - \nStep", step)
                
            if self.is_done():
                if settings.game_info:                
                    print("Game Stoped")
                break
            self.step()
        
        if(settings.visualise_map):           
            self.display_world_grid_with_things() # final visualisation at the end
            
    
    
    def run_A_B_search(self, start_location, target_location):
        
        """ CUSTOM: runs search algorithm """
        coordinatesDict = {} # the world coordinates (used below for search nodes - states)

        # initialise world size empty dict
        for x in range(0,settings.world_x):
            for y in range(0, settings.world_y):
                coordinatesDict[tuple([x,y])] = {}        
        
        # populate world size dict
        # position[0] is the step_cost (using list here to have expansion option later! )
        
        for key_tuple in coordinatesDict:
            innerDict = {}
            for x in range(0,settings.world_x):
                for y in range(0, settings.world_y):
                    # get rid of the same current coordinate from coordinatesDict for this particular "node-chain"
                    if not (key_tuple[0] == x and key_tuple[1] == y):
                        #only take direct neighbor grid
                        if abs(key_tuple[0]-x) <= 1 and abs(key_tuple[1]-y) <=1:
                            # get rid of (1,1) distance diagonal neigbor as well
                            if not (abs(key_tuple[0]-x) == 1 and abs(key_tuple[1]-y) ==1):
                                innerDict[tuple([x,y])] = [1] # ++++++++++++++++++++++++++++++++++ VALUES
                               
                    
                    # WORLD IS ROUND ! (special cases) ---->
                    
                    # X either 0 or settings.world_x-1
                    if key_tuple[0] == 0 and key_tuple[1] == y:
                            innerDict[tuple([settings.world_x-1,y])] = [1] # ++++++++++++++++++++++++++++++++++ VALUES
                    if key_tuple[0] == settings.world_x-1 and key_tuple[1] == y:
                            innerDict[tuple([0,y])] = [1] # ++++++++++++++++++++++++++++++++++ VALUES
                    
                    # Y either 0 or settings.world_y-1
                    if key_tuple[0] == x and key_tuple[1] == 0:
                            innerDict[tuple([x,settings.world_y-1])] = [1] # ++++++++++++++++++++++++++++++++++ VALUES
                    if key_tuple[0] == x and key_tuple[1] == settings.world_y-1:
                            innerDict[tuple([x,0])] = [1] # ++++++++++++++++++++++++++++++++++ VALUES
                   
            coordinatesDict[key_tuple] = innerDict
        
            
        if settings.searchShowCoordDict:
            print("Full Coordinate Listing:")
            pprint.pprint(coordinatesDict)    
    
    
        # SEARCH --> SETUP LOCATIONS / GRID MAP / PROBLEM
        the_agent_coord = tuple(start_location)        
        the_hut_coord = tuple(target_location)
        grid_map_graph = CustomDirectedGraph(coordinatesDict)
        navigation_problem = FindingHutProblem(the_agent_coord, the_hut_coord, grid_map_graph)
        
        # SEARCH TYPE -->
        
        if settings.searchAlgorithm == 'BFS':
            # BFS
            node = breadth_first_graph_search(navigation_problem)
        
        elif settings.searchAlgorithm == 'DFS':
            #DFS
            node = depth_first_graph_search(navigation_problem)
        
        elif settings.searchAlgorithm == 'DLS':
            # DLS
            node = depth_limited_search(navigation_problem)
            
        else: # default to BFS
            node = breadth_first_graph_search(navigation_problem)
        
        return node
            

    def default_location(self,thing):
        ''' sets a default location if none is given '''
        random_location_slot = random.choice(self.world_grid)
        emergency_counter = 0
        #try only 10 times or else this can go on forever in small environments where there is no space!
        while random_location_slot in self.used_locations and emergency_counter <= 10:
            if settings.full_debug:
                print("ASSIGNING NEW RANDOM COORDINATES (duplicate) !")
                print("Slot to re-roll:",random_location_slot)
            random_location_slot = random.choice(self.world_grid)
            emergency_counter += 1
            if settings.basic_debug and emergency_counter == 10:
                print("WARNING! Duplicate positioning detected. World too small or too many things in it!")
        return random_location_slot
    
    
    def exogenous_change(self):
        """If there is spontaneous change in the world, override this."""
        # not implemented for now (optional later)
        pass
        
    def is_done(self):
        ''' returns True no more Nodes to go through '''
        if self.rounds_played < len(self.searchedNodes):
            return False
        else:
            return True
            
        """
        for agent in self.agents:
            if isinstance(agent, Huntsman):
                if agent.alive:
                    return False # all good - keep going
                else:
                    if settings.basic_debug:
                        print("Locations Agent has been:", self.data_agent_location_tracker)
                    return True # agent not alive or critter count <= 0
        """
                
    
    
    def list_things_at(self, location, tclass=Thing):
        """Return all things exactly at a given location."""
        return [thing for thing in self.things
                if thing.location == location and isinstance(thing, tclass)]
            

    def some_things_at(self, location, tclass=Thing):
        """Return true if at least one of the things at location
        is an instance of class tclass (or a subclass)."""
        return self.list_things_at(location, tclass) != []    
  
    
    def add_thing(self, thing, location=None):
        """Add a thing to the environment, setting its location. For
        convenience, if thing is an agent program we make a new agent
        for it."""
        
        # just a backup case:
        if not isinstance(thing, Thing):
            thing = Huntsman(thing)
            
        # if already in list
        if thing in self.things:
            print("Can't add the same thing twice")
        
        # do stuff / add thing --->
        else:
            # adding specified or default location slot (atrribute) to Thing
            thing.location = location if location is not None else self.default_location(thing)
            self.used_locations.append(thing.location) # add location to a list (needed to prevent 2 things in same place)
            self.things.append(thing) # add thing
            
            # HUNTSMAN <<<-----------
            if isinstance(thing, Huntsman):
                thing.performance = 0 # define Agents initial values
                thing.energy = 100
                thing.world_size_x = self.world_size_x
                thing.world_size_y = self.world_size_y                
                self.data_agent_starting_location.extend(thing.location) # just list
                if settings.game_info:
                    print("Hunter starts at:", thing.location)
                self.agents.append(thing) # append Agent to list of agents in World
            
            # HUT <<<-----------
            elif isinstance(thing, Hut):
                self.data_hut_locations.append(thing.location) # just to track hut locations            

            else:
                if settings.basic_debug:
                    print(thing.__class__.__name__ ,"is at:", thing.location)
                
    
    def delete_thing(self, thing):   ### IMPORTANT TO DELETE ANYTHING ELSE BEING HELD WITH IT
        """Remove a thing from the environment."""
        try:
            self.things.remove(thing)
        except ValueError as e:
            print(e)
            print("  in Environment delete_thing")
            print("  Thing to be removed: {} at {}".format(thing, thing.location))
            print("  from list: {}".format([(thing, thing.location) for thing in self.things]))
        if thing in self.agents:
            self.agents.remove(thing)
    

