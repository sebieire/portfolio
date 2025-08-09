import collections
from . import settings # import module
from . import thing_objects # import module
from . thing_objects import Thing, Rabbit, Deer, Hut # import classes

# ---------------------------------------------------------------------------
# - - = PRIMARY THING (AGENT / MAIN CLASS)  = - -
# ---------------------------------------------------------------------------


class Huntsman(Thing):
    ''' The Huntsman (Agent) '''
    
    def __init__(self,program=None):
        self.alive = True
       
        self.performance = 0
        self.energy = 100
        self.location = [] # empty location list
        
        self.bump = False # not currently used
        self.holding = [] # not currently used
        
        # warning message (requires program)
        if program is None or not isinstance(program, collections.abc.Callable):
            print("Can't find a valid program for:", self.__class__.__name__)
        
        self.program = program # holds agent function of choice (e.g. simple_reflex_agent)
        self.world_model = {} # internal model of the world (required for model agent and beyond)        
        self.rest_amount = 0 # count "rested" (maximum of 2 is allowed at any one time - functions as a cooldown)
        
        
        # aquired values from "world" when adding Huntsman
        # required for travel, as well as model agent and beyond 
        # agent needs to know at least what size the world is --> as it is round
        self.world_size_x = 0 # initial 
        self.world_size_y = 0 # initial
        
        
        # DATA VARIABLES
        self.data_time_steps = 0 # regular counter
        self.data_agent_travels = {} # dict
        self.data_agent_deaths = {} # dict        
        self.data_energy_level = {}
        self.data_performance_level = {}
        self.data_critters_killed = {}
        self.data_days_rested = {}
        self.data_performance_threshold_10 = False
        self.data_performance_threshold_20 = False
        self.data_performance_threshold_30 = False
        self.data_performance_threshold_40 = False
        self.data_performance_threshold_50 = False
        self.data_performance_threshold_60 = False
        self.data_performance_threshold_70 = False
        self.data_performance_threshold_80 = False
        self.data_performance_threshold_90 = False
        self.data_performance_threshold_100 = False
        
    
    def reset_time_based_stats(self):
        ''' Time Related Resets Counter '''
        
        # Adjust Rest Allowance
        if self.rest_amount > 0:
            self.rest_amount = self.rest_amount - 1 
    
    def update_statistic_and_data_variables(self):
        
        self.data_energy_level[self.data_time_steps] = self.energy
        self.data_performance_level[self.data_time_steps] = self.performance            
        self.data_time_steps += 1
        
        if self.performance >= 10:            
            self.data_performance_threshold_10 = True
        if self.performance >= 20:            
            self.data_performance_threshold_20 = True
        if self.performance >= 30:            
            self.data_performance_threshold_30 = True
        if self.performance >= 40:            
            self.data_performance_threshold_40 = True
        if self.performance >= 50:            
            self.data_performance_threshold_50 = True
        if self.performance >= 60:            
            self.data_performance_threshold_60 = True
        if self.performance >= 70:            
            self.data_performance_threshold_70 = True
        if self.performance >= 80:            
            self.data_performance_threshold_80 = True
        if self.performance >= 90:            
            self.data_performance_threshold_90 = True
        if self.performance >= 100:            
            self.data_performance_threshold_100 = True
            
    
    def travel(self, direction):
        ''' moves the agent to an adjacent world grid element
            THE WORLD IS ROUND 
            going outside the scope will result returning on the "opposite" end of the world
            
            top left is [0,0] !!!! ------->>>> so x and y axis are realistically inverted
            bottom right is [world_x , world_y]
            '''        
        
        if settings.full_debug:
            print("Running Travel")
        
        if direction == "travel east":
            if(self.location[1] >= self.world_size_y-1):
                newLocation = [self.location[0],0]
            else:
                newLocation = [self.location[0],self.location[1]+1]
            
        elif direction == "travel north":
            if(self.location[0] == 0):
                newLocation = [self.world_size_x-1,self.location[1]]
            else:
                newLocation = [self.location[0]-1,self.location[1]]
            
        elif direction == "travel west":
            if(self.location[1] == 0):
                newLocation = [self.location[0],self.world_size_y-1]
            else:
                newLocation = [self.location[0],self.location[1]-1]
            
        elif direction == "travel south":
            if(self.location[0] >= self.world_size_x-1):
                newLocation = [0,self.location[1]]
            else:
                newLocation = [self.location[0]+1,self.location[1]]

        self.energy -= 3
        self.performance -= 1
        self.check_alive()
        self.location = newLocation
        
        #data vars        
        self.data_agent_travels[self.data_time_steps] = 1        
        self.update_statistic_and_data_variables()
    
    
    def can_grab(self,thing): # implement later
        pass
    
    def hunt_rabbit(self, thing):
        if isinstance(thing, Rabbit):
            self.energy -= 5
            self.check_alive()
            self.performance += 3
            #data vars            
            self.data_critters_killed[self.data_time_steps] = 1
            self.update_statistic_and_data_variables()

            return True
        return False
    
    def hunt_deer(self, thing):
        if isinstance(thing, Deer):
            self.energy -= 8
            self.check_alive()
            self.performance += 10
            #data vars            
            self.data_critters_killed[self.data_time_steps] = 1
            self.update_statistic_and_data_variables()

            return True
        return False
    
    def rest(self, thing):
        if isinstance(thing, Hut):
            self.rest_amount = self.rest_amount + 1
            self.energy += 20
            if self.energy > 100: # 100 is max
                self.energy = 100
            
            self.data_days_rested[self.data_time_steps] = 1
            self.update_statistic_and_data_variables()
            self.performance -= 2            
            return True
        return False
    
    
    def check_alive(self):
        if self.energy <= 0:
            self.alive = False
            self.data_agent_deaths[self.data_time_steps] = 1