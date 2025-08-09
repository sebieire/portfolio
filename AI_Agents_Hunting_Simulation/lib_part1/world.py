import random

from . import settings # import module
from . import thing_objects # import module
from . import huntsman # import module

from . thing_objects import Thing, Rabbit, Deer, Hut, Obstacle # import classes
from . huntsman import Huntsman # import classes

# ---------------------------------------------------------------------------
# - - = ENVIRONMENT (MAIN CLASS)  = - -
# ---------------------------------------------------------------------------
class World():
    
    def __init__(self, size_x=10, size_y=10):
        
        self.things = []
        self.agents = []
        self.used_locations = [] # to prevent a location being used twice when starting
        self.critter_count = 0 # add 1 for every critter added and subtract one for every critter hunted (if "back to" 0 game ends)        
        self.rounds_played = 0 # add 1 for every round played (required for statistics later)
        
        # world size & starting coordinates
        self.world_size_x = size_x
        self.world_size_y = size_y     
                
        # perception distance allowed in the environment
        self.perceptible_distance = 1 # 1 tile in all directions next to current one
        
        # DATA variables (for statistics)
        self.data_agent_location_tracker = {} # tracks agents movements throughout the world
        self.data_hut_locations = [] # keep track of where the huts are in the world
        self.data_agent_starting_location = [] # useful for when using cloned world
        self.data_critter_count = {} # additional critter development tracker over time
        
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
            

    
    # PERCEPT METHOD - list things at location and return those
    def percept(self,agent):   # required for Environment run() -> step()
        ''' return list of everything in current agent location '''
        things = self.list_things_at(agent.location) # returns things at current location
        
        if settings.full_debug:
            print("THINGS:",things, "SELF REST AMOUNT", agent.rest_amount)
 
        if settings.agent.__name__ == "utility_based_agent": # utility agent
            return (self.things_near(agent.location),agent)
        
        else: # other agents
            return (things, agent)        
        
    
    
    # EXECUTE METHOD
    def execute_action(self, agent, action): # where all the good stuff happens
        ''' changes the environment based on what the agents behaviour (program) is '''
        
        # DEBUG
        if settings.basic_debug:
            print("In Execute -->")
            print("Huntsman is at (pre-action):", agent.location)
            print("Huntsman has", agent.energy, "energy left (pre-action).")
            print("Huntsman has a performance score of:", agent.performance)
            print("Huntsman decides to:", action, "on this day.")            
            

        # DATA TRACKING: update agent location tracker (does not distinguish between agents! assuming only 1)
        # --> tracks location tile before anything happens ("morning of the day")
        self.data_agent_location_tracker[tuple(agent.location)] += 1        
        
        # move towards a specific direction
        if action == "travel north" or action == "travel east" or  action == "travel south" or action == "travel west":
            
            agent.travel(action)            
            agent.reset_time_based_stats()
        
        # Action == Rabbit Hunting
        elif action == "hunt rabbit":
            items = self.list_things_at(agent.location, tclass=Rabbit)            
            if len(items) != 0:
                if agent.hunt_rabbit(items[0]): # hunt rabbit
                    if settings.game_info:
                        print(str(agent)[1:-1], "hunted a", str(items[0])[1:-1], "at location:", agent.location )
                    self.delete_thing(items[0])
                    self.critter_count -= 1
                    self.data_critter_count[agent.data_time_steps] = self.critter_count # note critter count (data) at this point in time
                    agent.reset_time_based_stats()
        
        # Action == Deer Hunting
        elif action == "hunt deer":
            items = self.list_things_at(agent.location, tclass=Deer)
            if len(items) != 0:
                if agent.hunt_deer(items[0]): # hunt deer
                    if settings.game_info: 
                        print(str(agent)[1:-1], "hunted a", str(items[0])[1:-1], "at location:", agent.location )
                    self.delete_thing(items[0])
                    self.critter_count -= 1
                    self.data_critter_count[agent.data_time_steps] = self.critter_count # note critter count (data) at this point in time
                    agent.reset_time_based_stats()             
                    
        # Action == Rest
        elif action == "rest" and agent.rest_amount < 2:
            items = self.list_things_at(agent.location, tclass=Hut)
            if len(items) != 0:
                if agent.rest(items[0]): # rest
                    if settings.game_info: 
                        print(str(agent)[1:-1], "decided to rest in a", str(items[0])[1:-1], "at location:", agent.location )  
                    
        # DEBUG
        if settings.basic_debug:
                print("Huntsman is currently at (post-action):", agent.location)
                print("Current Agent World Model (post-action):", agent.world_model)
        
        
        
        
    # STEP
    def step(self):
        """Run the environment for one time step. If the
        actions and exogenous changes are independent, this method will
        do. If there are interactions between them, you'll need to
        override this method."""
        if not self.is_done():
            actions = []
            for agent in self.agents:
                if agent.alive: # technically don't need this check here (duplicate)
                    # return which action to take (through program(percepts))
                    actions.append(agent.program(self.percept(agent))) # self.percept returns Things at Agent Location & Agent
                    self.rounds_played += 1 # keep track how many rounds have been played
                    
                else:
                    actions.append("No Action - Lost In The Woods!") # changed from "" to be more clear
            
            for (agent, action) in zip(self.agents, actions): # for agent & associated action (zip!)                
                self.execute_action(agent, action) # -> execute
            
            self.exogenous_change() # independant exogenous change        
        

    # RUN
    def run(self, steps=100):
        """Run the Environment for given number of time steps."""        
        
        for step in range(steps):
            
            if(settings.visualise_map):
                self.display_world_grid_with_things() # Optional - Shows Each Step Visually
                
            if settings.basic_debug:
                print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - \nStep", step)
            if self.is_done():
                if settings.game_info:
                    print("Either Huntsman is lost in the woods or no more critters around. -> current critter count: ", self.critter_count) # agent dead or 0 critters
                break
            self.step()
        
        if(settings.visualise_map):           
            self.display_world_grid_with_things() # final visualisation at the end

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
        ''' returns True if agent is dead or no further critters are available '''
        for agent in self.agents:
            if isinstance(agent, Huntsman):
                if agent.alive and self.critter_count > 0:
                    return False # all good - keep going
                else:
                    if settings.basic_debug:
                        print("Locations Agent has been:", self.data_agent_location_tracker)
                    return True # agent not alive or critter count <= 0
                
    
    
    def list_things_at(self, location, tclass=Thing):
        """Return all things exactly at a given location."""
        return [thing for thing in self.things
                if thing.location == location and isinstance(thing, tclass)]
        
        
    def things_near(self, location, radius=None):
        """Return all things within radius of location - adapted method
        This will also return things away radius steps away diagonally!
        """
        
        if radius is None:
            radius = self.perceptible_distance
            
        surroundingsThings=[]
        for thing in self.things:
            # regular (next to each other)
            if (location[0] - thing.location[0])**2 <= radius and (location[1] - thing.location[1])**2 <= radius: 
                surroundingsThings.append(thing)
            # world edge (opposites X)
            elif location[0] == 0 and thing.location[0] == self.world_size_x-1 and (location[1] - thing.location[1])**2 <= radius:
                surroundingsThings.append(thing)
            # world edge (opposites X - other way round)
            elif location[0] == self.world_size_x-1 and thing.location[0] == 0 and (location[1] - thing.location[1])**2 <= radius:
                surroundingsThings.append(thing)
            # world edge (opposites Y)
            elif (location[0] - thing.location[0])**2 <= radius and location[1] == 0 and thing.location[1] == self.world_size_y-1:
                surroundingsThings.append(thing)
            # world edge (opposites Y - other way round)
            elif (location[0] - thing.location[0])**2 <= radius and location[1] == self.world_size_y-1 and thing.location[1] == 0:
                surroundingsThings.append(thing)
                
        return surroundingsThings
            

    def some_things_at(self, location, tclass=Thing):
        """Return true if at least one of the things at location
        is an instance of class tclass (or a subclass)."""
        return self.list_things_at(location, tclass) != []
    
    
    def move_to(self, thing, destination):
        """Move a thing to a new location. Returns True on success or False if there is an Obstacle.
        If thing is holding anything, they move with him."""
        thing.bump = self.some_things_at(destination, Obstacle)
        if not thing.bump:
            thing.location = destination
            for t in thing.holding:
                self.delete_thing(t)
                self.add_thing(t, destination)
                t.location = destination
        return thing.bump
    
    
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
                #self.data_agent_starting_location.append(thing.location) # list in list (old)
                self.data_agent_starting_location.extend(thing.location) # just list
                if settings.game_info:
                    print("Hunter starts at:", thing.location)
                self.agents.append(thing) # append Agent to list of agents in World
            
            # HUT <<<-----------
            elif isinstance(thing, Hut):
                self.data_hut_locations.append(thing.location) # just to track hut locations
            
            # RABBIT or DEER <<<-----------
            elif isinstance(thing, Rabbit) or isinstance(thing, Deer):
                self.critter_count += 1 # add 1 for each critter added to the world
                self.data_critter_count[0] = self.critter_count # updates so that at start we have full critter count
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
    

