import random

from . import settings # import module
from . import thing_objects # import module
from . import huntsman # import module

from . thing_objects import Thing, Rabbit, Deer, Deer_Tracks, Hut, Obstacle # import classes
from . huntsman import Huntsman # import classes
from . aima_KB_logic import PropDefiniteKB, pl_fc_entails, expr
#from . aima_KB_logic import *
# from . aimapython.logic import PropDefiniteKB, pl_fc_entails, expr

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
        
        # ADDED IN PART 3        
        self.deer_track_percepts = [] # kind of doubled up here from settings but easier to use internally
        self.agent_knowledge_base = PropDefiniteKB() # initialize KB
        self.all_clauses = []
        self.all_tracks_found = False
        
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
        
        self.data_agent_location_tracker[tuple(agent.location)] += 1        
        
        # move towards a specific direction
        if action == "travel north" or action == "travel east" or  action == "travel south" or action == "travel west":
            
            agent.travel(action)
            agent.reset_time_based_stats()        
        
        
        
    # STEP
    def step(self):
        """Run the environment for one time step. If the
        actions and exogenous changes are independent, this method will
        do. If there are interactions between them, you'll need to
        override this method."""
        if not self.is_done():
            actions = []
            for agent in self.agents:
                
                actions.append(agent.program()) # self.percept trivial (will only run random movement here)
                self.rounds_played += 1 # keep track how many rounds have been played
                
                # NEW in V3
                # for every step -> check if location already in KB or if part of percepts & update KB
                self.query_percepts_and_update(self.agent_knowledge_base, agent.location)
            
            for (agent, action) in zip(self.agents, actions): # for agent & associated action (zip!)                
                self.execute_action(agent, action) # -> execute            
        

    # RUN
    def run(self, steps=100):
        """Run the Environment for given number of time steps."""                
        
        # NEW in V3
        # init clauses
        self.init_clauses(settings.deer_init_starting_locations) 
        
        # add to KB
        self.inform_knowledge_base_of_success_clauses(self.agent_knowledge_base, self.all_clauses, False)
        
        
        for step in range(steps):
            
            if(settings.visualise_map):
                self.display_world_grid_with_things() # Optional - Shows Each Step Visually
                
            if settings.basic_debug:
                print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - \nStep", step)
            if self.is_done():
                if settings.game_info:
                    print("All Tracks Have Been Found") # agent dead or 0 critters
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
   
        
    def is_done(self):
        ''' returns True if all deer tracks have been found which is set to True once SUCCESS is entailed! '''
        
        if self.all_tracks_found:
            return True
        else:
            return False
        
    
    
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
            
            #print("ADDED THING AT LOCATION:" + str(thing.__class__.__name__) + str(thing.location))
            
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
                
            # DEER TRACKS (Added in V3)
            elif isinstance(thing, Deer_Tracks): 
                # adding a percept of tracks for each track here <-----------------------------------------
                self.deer_track_percepts.append( ("TRACKS" + str(thing.location[0]) + str(thing.location[1])) )
                
                
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
    
    
    # ---> NEW IN V3 (forward chaining demo)
            
    def init_clauses(self, deer_locations):
        """ set up the 'success' logical clauses based on deer & tracks location """
        
        # placeholders
        deer_strings = []
        track_strings = []
        
        # create strings
        for deer_loc in deer_locations:
            
            # create deer strings
            deer_strings.append( ("DEER" + str(deer_loc[0]) + str(deer_loc[1])) )
            
            # create track strings
            track_strings.append( ("TRACKS" + str(deer_loc[0]-1) + str(deer_loc[1])) ) # north
            track_strings.append( ("TRACKS" + str(deer_loc[0]) + str(deer_loc[1]+1)) ) # west
            track_strings.append( ("TRACKS" + str(deer_loc[0]+1) + str(deer_loc[1])) ) # south
            track_strings.append( ("TRACKS" + str(deer_loc[0]) + str(deer_loc[1]-1)) ) # east
        
        # create clauses
        success_condition = "("        
        four_tracks_counter = 0
        for deer_string in deer_strings:
            
            # success clause
            success_condition = success_condition + deer_string + " & "
            
            # deer clause
            deer_condition = "("
            for i in range(4): # 4 tracks per deer
                deer_condition = deer_condition + track_strings[four_tracks_counter] + " & "
                four_tracks_counter += 1
            
            deer_condition = deer_condition.rstrip('& ')
            deer_condition = deer_condition + ")==>" + deer_string
            self.all_clauses.append(deer_condition)
            
        success_condition = success_condition.rstrip('& ')
        success_condition = success_condition + ")==>SUCCESS"
        
        self.all_clauses.append(success_condition)
        
    
    def inform_knowledge_base_of_success_clauses(self, KB, success_clauses, output=False):
        """ simple function to include known success clauses into agent KB """
        
        for clause in success_clauses:
            KB.tell(expr(clause))
            
        if output:
            print("\nKnowledge Base After Added Success Clauses:", KB.clauses)
            
        
        
    def add_to_knowledge_base(self, KB, agent_location):
        """ update the KB """
        
        query_string = "TRACKS" + str(agent_location[0]) + str(agent_location[1])
        
        if query_string in self.deer_track_percepts and not pl_fc_entails(KB, expr(query_string)):
            
            KB.tell(expr(query_string))            
        
        else:
            no_track_string = "notracks" + str(agent_location[0])+ str(agent_location[1])
            
            if not no_track_string in self.all_clauses: # make sure it isn't in all clauses already
                
                self.all_clauses.append(no_track_string)
                
                if not pl_fc_entails(KB, expr(no_track_string)):
                    KB.tell(expr(no_track_string))                
    
    
    def query_percepts_and_update(self, KB, agent_location):
        
        track_query_string = "TRACKS" + str(agent_location[0]) + str(agent_location[1])        
        
        if not pl_fc_entails(KB, expr(track_query_string)):
            self.add_to_knowledge_base(KB, agent_location)
        
        if pl_fc_entails(KB, expr('SUCCESS')):
            self.all_tracks_found = True            
       