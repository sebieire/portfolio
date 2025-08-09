# IMPORTS -------------------------------------------------------------------
import sys
sys.path.append('..')  # access parent folder if needed

import copy

from lib_part2 import settings
from lib_part2.thing_objects import Hut # Things
from lib_part2.huntsman import Huntsman # Huntsman (Thing)
from lib_part2.world import World # Environment
from lib_part2.statistical_data_functions import (accumulate_data, prep_accumulated_data, graph_all_data, graph_travel_and_rest_data,
                                            graph_death_data, graph_energy_data, graph_performance_data, graph_performance_histogram,
                                            show_heatmap, output_general_statistics, output_search_statistics)
#search add ons (A to B search)
from lib_part2.search_functions import breadth_first_graph_search, depth_first_graph_search, depth_limited_search, search_agent
from lib_part2.search_classes import FindingHutProblem, CustomDirectedGraph


# ---------------------------------------------------------------------------
# - - = INIT GLOBAL SETTINGS  = - -
# ---------------------------------------------------------------------------

#settings.init_settings(search_agent, 'DLS', 10) # options for 2nd param: 'BFS' , 'DFS', ' DLS' # options 3rd param = DFS limit ###################################################################
#settings.init_debug_settings() ###################################################################
#settings.init_statistics_data() ###################################################################


# ---------------------------------------------------------------------------
# - - = SETUP WORLD = - -
# ---------------------------------------------------------------------------
def setup_world(world, is_cloned=False):
    """
    sets up world / adds critters / adds agents - all parameters defined in settings
    """
    
    
    if not is_cloned: # if not a clone (original world)
        hunter = Huntsman(settings.agent) # passing in agent (Search technique in this case)
        world.add_thing(hunter)    
        
        all_huts = {}        
        
        
        # create all the huts
        for i in range(int(settings.world_huts)):
            all_huts["hut{0}".format(i)]= Hut()        
       
        for key in all_huts:
            world.add_thing(all_huts[key])
            
        world_clone = copy.deepcopy(world)    
        
        world.run(settings.steps_per_game)
    
    else:
        #make a copy of the world
        world_clone = copy.deepcopy(world)
        
        # retrieve Huntsman agent from the world so data can be collected (otherwise reference error)
        # is already present in the above "if" statement but not visible outside of it so need to get it from the world object
        for the_agent in world.agents:
            if isinstance(the_agent, Huntsman):
                hunter = the_agent
        # RUN
        world.run(settings.steps_per_game)
    
    # pass data for each game (cycle) played --> statistics
    accumulate_data(hunter, world)
    
    # returns the clone (IMPORTANT!)
    return world_clone
   
    
# ---------------------------------------------------------------------------
# - - = INITIALISE / RUN CYCLES / GATHER DATA = - -
# ---------------------------------------------------------------------------

def launch_world(cycles=1):
    ''' wrapper function to run defined number of cycles and evaluate data '''
    
    # runs same world during the loop
    if settings.cloning_enabled:
        for i in range(cycles):
            if settings.cycle_info:
                print("\n- - = = RUNNING CYCLE:", i+1, "= = - -")
            if i == 0: # first world must be original world (not a clone)
                world = World(settings.world_x,settings.world_y)
                cloned_world = setup_world(world, False) 
            else: # now use cloned world for all other iterations
                cloned_world = setup_world(cloned_world, True)
            
    # creates a new world each time during the loop
    else:
        for i in range(cycles):
            if settings.basic_debug:
                print("\n- - = = RUNNING CYCLE:", i+1, "= = - -")
            world = World(settings.world_x,settings.world_y)
            setup_world(world, False)
     
    if settings.data_info:
        print("Travels", settings.agent_travels)
        print("Agent Deaths", settings.agent_deaths)        
        print("Energy Level", settings.agent_energy_level)
        print("Performance Level", settings.agent_performance_level)        
        print("Days Rested", settings.agent_days_rested)
        
    # required for data! - preps all data into lists etc. required for graph functions
    prep_accumulated_data()
    settings.agent_starting_location = world.data_agent_starting_location 
    settings.huts_starting_locations = world.data_hut_locations    
    
    
    if settings.show_hut_location:
        print("The locations of Huts in the World is (row / col):", world.data_hut_locations)
    if settings.agent_first_spawn_location:
        print("Agent starting location was (row / col):", world.data_agent_starting_location)




# ---------------------------------------------------------------------------
# - - = LAUNCH WORLD = - -
# ---------------------------------------------------------------------------
        
# launches everythig -->
#launch_world(settings.number_of_runs) ###################################################################


# ---------------------------------------------------------------------------
# - - = STATISTICS - INCLUDING SEARCH (ADD ON) = - -
# ---------------------------------------------------------------------------

#output_general_statistics()
#output_search_statistics() ###################################################################

# ---------------------------------------------------------------------------
# - - = DISPLAY = - -
# ---------------------------------------------------------------------------

# display options -->

"""
graph_travel_and_rest_data()
graph_death_data()
graph_energy_data()
graph_performance_data()
graph_performance_histogram()
show_heatmap() ###################################################################
"""








        