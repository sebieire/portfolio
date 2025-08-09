import numpy as np
import math

# ---------------------------------------------------------------------------
# - - = GLOBAL SETTINGS  = - -
# ---------------------------------------------------------------------------

def init_settings(agentType, cloning=True, worldX=6, worldY=6, gameSteps=500, runs=1, crit=1, huts=0):
    '''
        init all global settings for the simulation
        @param: requires at least the agent type to run, accepts additional setting parameters
    '''
    print("\n======= SETTINGS =======")    
    print("-> Initializing Global Settings")    
    
    global agent
    
    # WORLD SIZE
    global world_x
    global world_y
    
    # RUNS & CLONING OPTIONS
    global steps_per_game
    global number_of_runs
    global cloning_enabled # if True this will use the same world every time / False will reset world every time
    
    # CRITTER POPULATION / % SPLITS
    global crit_population # string
    global huts_density # string
    global deer_split # number 
    global rabbit_split # number
    
    # -----------------------------------------
    # PART 3 ADDITIONAL VARIABLES
    
    # static starting locations
    global hunter_init_starting_location
    #global deer_init_starting_location
    global deer_init_starting_locations
    global deer_tracks_locations
    
    hunter_init_starting_location = [0,0]
    #deer_init_starting_location = [int(worldX/2), int(worldY/2)]
    deer_init_starting_locations = [[1,3], [4,2]]  # IMPORTANT! --> make sure this is NOT on the edge of map & NOT overlapping for Demo
    deer_tracks_locations = []
    
    # -----------------------------------------
    
    crit_population = 2 #options are: high (60%), medium (30%), low (15%) or int
    huts_density = "high" #options are: high (15%), medium (8%), low (3%) - minimum is always 1!
    deer_split = 0.35  # percentage of deers from overall population
    rabbit_split = 0.65 # percentage of rabbits from overall population
    
    world_x = worldX # param
    world_y = worldY # param
    steps_per_game = gameSteps # param (how many times 1 game will run max)
    
    number_of_runs = runs # param
    cloning_enabled = cloning # param
    
    agent = agentType # param
    print("Agent Type Running:", agent.__name__ )    
    
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # automatically set - do not change below!
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    global world_crit
    global world_huts
    
    # critter density
    if crit_population == "high":
        world_crit = world_x * world_y * 0.6
    elif crit_population == "medium":
        world_crit = world_x * world_y * 0.3
    elif crit_population == "low":
        world_crit = world_x * world_y * 0.15
    elif crit_population == 1:
        world_crit = 1        
    else:
        world_crit = world_x * world_y * 0.3 # default fallback
        
    # hut density
    if huts_density == "high":
        world_huts = world_x * world_y * 0.15
    elif huts_density == "medium":
        world_huts = world_x * world_y * 0.08
    elif huts_density == "low":
        world_huts = math.ceil(world_x * world_y * 0.03)
    elif huts_density == 0:
        world_huts = 0
    else:
        world_huts = math.ceil(world_x * world_y * 0.03) # default
    
    
    
def init_debug_settings():
    print("-> Initializing Global Debug Settings")
    
    global full_debug # comprehensive dubg notes
    global basic_debug # simple debug notes    
    global visualise_map # visualises each grid for each step    
    global game_info # some general light info about the games running
    global cycle_info # prints when new cycle starts (1 line)
    global data_info # prints dictionary of data (this is not the graph - those will display anyway)
    global agent_first_spawn_location # Useful for when running with cloned world repeatedly
    global show_hut_location # Announce Hut Locations ( World tracks hut locations )    
        
    full_debug = False
    basic_debug = False
    visualise_map= False
    game_info = False
    cycle_info = False
    data_info = False
    
    agent_first_spawn_location = False
    show_hut_location = False
    
    
def init_statistics_data():
    print("-> Initializing Statistics Containers")
    
    global agent_travels
    global agent_deaths
    global agent_energy_level
    global agent_performance_level
    global agent_critters_killed
    global agent_days_rested    
    global agent_performance_threshold_10_cummulative
    global agent_performance_threshold_20_cummulative
    global agent_performance_threshold_30_cummulative
    global agent_performance_threshold_40_cummulative
    global agent_performance_threshold_50_cummulative
    global agent_performance_threshold_60_cummulative
    global agent_performance_threshold_70_cummulative
    global agent_performance_threshold_80_cummulative
    global agent_performance_threshold_90_cummulative
    global agent_performance_threshold_100_cummulative
    global world_critter_count
    global world_rounds_played_cummulative        
    
    global heatmap_list
    global agent_starting_location  # 'world' populated variable
    global huts_starting_locations # 'world' populated variable
    
    agent_travels = {}
    agent_deaths = {}
    agent_energy_level = {}
    agent_performance_level = {}
    agent_critters_killed = {}
    world_rounds_played_cummulative = 0
    agent_performance_threshold_10_cummulative = 0
    agent_performance_threshold_20_cummulative = 0
    agent_performance_threshold_30_cummulative = 0
    agent_performance_threshold_40_cummulative = 0
    agent_performance_threshold_50_cummulative = 0
    agent_performance_threshold_60_cummulative = 0
    agent_performance_threshold_70_cummulative = 0
    agent_performance_threshold_80_cummulative = 0
    agent_performance_threshold_90_cummulative = 0
    agent_performance_threshold_100_cummulative = 0
    agent_days_rested = {}
    world_critter_count = {}
    heatmap_list = np.zeros(shape=(world_x,world_y))
    
    # initialise world_critter_count positions here already (required!)
    for i in range (0, steps_per_game):
        world_critter_count[i] = 0
    