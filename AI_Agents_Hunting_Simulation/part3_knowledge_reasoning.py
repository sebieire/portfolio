# IMPORTS -------------------------------------------------------------------
import sys
sys.path.append('..')  # access parent folder if needed

import copy

from lib_part3 import settings
from lib_part3.thing_objects import Deer, Deer_Tracks # Things
from lib_part3.huntsman import Huntsman # Huntsman (Thing)
from lib_part3.world import World # Environment
from lib_part3.agent_variants import random_movement_agent # Agent Types
from lib_part3.statistical_data_functions import (accumulate_data, prep_accumulated_data, graph_all_data, graph_travel_and_rest_data,
                                            graph_death_data, graph_energy_data, graph_performance_data, graph_performance_histogram,
                                            graph_critter_data, show_heatmap, output_general_statistics)



IDE_run = False # set to False if running from Jupyter Notebook

# ---------------------------------------------------------------------------
# - - = INIT GLOBAL SETTINGS  = - -
# ---------------------------------------------------------------------------

if IDE_run:
    settings.init_settings(random_movement_agent) # options are: random_movement_agent
    settings.init_debug_settings()
    settings.init_statistics_data()
    
# ---------------------------------------------------------------------------
# - - = SETUP WORLD = - -
# ---------------------------------------------------------------------------
def setup_world(world, is_cloned=False):
    """
    sets up world / adds critters / adds agents - all parameters defined in settings
    """
    
    if not is_cloned: # if not a clone (original world)
        hunter = Huntsman(settings.agent)
        world.add_thing(hunter, settings.hunter_init_starting_location) # here with starting location (= not random -> take out to be random)
    
        crit_split_deer = 2 # two deer in this scenario
        
        all_the_deer = {}
        all_the_deer_tracks = {}
        
        # create deer & deer tracks
        for i in range(int(crit_split_deer)):
            all_the_deer["deer{0}".format(i)] = Deer()
            # here assuming 4 tracks per Deer (North, South, East & West of Deer)
            all_the_deer_tracks["deer_track_" + str(i) + "_1"] = Deer_Tracks()
            all_the_deer_tracks["deer_track_" + str(i) + "_2"] = Deer_Tracks()
            all_the_deer_tracks["deer_track_" + str(i) + "_3"] = Deer_Tracks()
            all_the_deer_tracks["deer_track_" + str(i) + "_4"] = Deer_Tracks()        
        
        # add deer to the world
        deer_counter = 0
        for key in all_the_deer:
            if deer_counter < 2: # only 2 deer have specific coordinates in settings
                
                # only first deer added with specific location
                deer_loc = settings.deer_init_starting_locations[deer_counter]
                world.add_thing(all_the_deer[key], deer_loc)
                
                # add deer tracks to the world (works for 4 tracks for first deer only & ignores other deer - quick and dirty here for demo)
                track_loc = [deer_loc[0], deer_loc[1]+1]
                world.add_thing(all_the_deer_tracks['deer_track_' + str(deer_counter) + '_1'], track_loc)
                settings.deer_tracks_locations.append(track_loc)
                
                track_loc = [deer_loc[0], deer_loc[1]-1]
                world.add_thing(all_the_deer_tracks['deer_track_' + str(deer_counter) + '_2'], track_loc)
                settings.deer_tracks_locations.append(track_loc)
                
                track_loc = [deer_loc[0]+1, deer_loc[1]]
                world.add_thing(all_the_deer_tracks['deer_track_' + str(deer_counter) + '_3'], track_loc)
                settings.deer_tracks_locations.append(track_loc)
                
                track_loc = [deer_loc[0]-1, deer_loc[1]]
                world.add_thing(all_the_deer_tracks['deer_track_' + str(deer_counter) + '_4'], track_loc)
                settings.deer_tracks_locations.append(track_loc)
                
                deer_counter += 1
                
            else:
                world.add_thing(all_the_deer[key])
            
        
            
            
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
        print("World Critter Count", settings.world_critter_count)
        print("Critters Killed", settings.agent_critters_killed)
        print("Days Rested", settings.agent_days_rested)
        
    # required for data! - preps all data into lists etc. required for graph functions
    prep_accumulated_data()
    settings.agent_starting_location = world.data_agent_starting_location 
    settings.huts_starting_locations = world.data_hut_locations
    
    if settings.show_hut_location:
        print("The locations of Huts in the World is (row / col):", world.data_hut_locations)
    if settings.agent_first_spawn_location:
        print("Agent starting location was (row / col):", world.data_agent_starting_location)
        
    return world # for eval reference in V3

# ---------------------------------------------------------------------------
# - - = LAUNCH & DISPLAY = - -
# ---------------------------------------------------------------------------
        
# launches everythig -->
if IDE_run:
    world = launch_world(settings.number_of_runs)

# display options -->

"""
output_general_statistics()
graph_travel_and_rest_data()
graph_death_data()
graph_energy_data()
graph_performance_data()
graph_performance_histogram()
graph_critter_data()    
show_heatmap()
"""

if IDE_run:
    output_general_statistics()
    show_heatmap()