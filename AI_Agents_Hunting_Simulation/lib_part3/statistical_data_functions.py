import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from . import settings # import module

#from . import custom_heat_map
from . custom_heat_map import heatmap, annotate_heatmap


# ---------------------------------------------------------------------------
# - - = STATISTICAL DATA FUNCTIONS  = - -
# ---------------------------------------------------------------------------


# - - - ACCUMULATE - - -
def accumulate_data(hunter,the_world):
    ''' accumulates data over the amount of cycles runs (merges dictionary data)'''    
    
    settings.agent_travels = {key: settings.agent_travels.get(key, 0) + hunter.data_agent_travels.get(key, 0) for key in set(settings.agent_travels) | set(hunter.data_agent_travels)}        
    
    settings.agent_deaths = {key: settings.agent_deaths.get(key, 0) + hunter.data_agent_deaths.get(key, 0) for key in set(settings.agent_deaths) | set(hunter.data_agent_deaths)}    

    settings.agent_energy_level = {key: settings.agent_energy_level.get(key, 0) + hunter.data_energy_level.get(key, 0) for key in set(settings.agent_energy_level) | set(hunter.data_energy_level)}        

    settings.agent_performance_level = {key: settings.agent_performance_level.get(key, 0) + hunter.data_performance_level.get(key, 0) for key in set(settings.agent_performance_level) | set(hunter.data_performance_level)}    
    
    settings.agent_critters_killed = {key: settings.agent_critters_killed.get(key, 0) + hunter.data_critters_killed.get(key, 0) for key in set(settings.agent_critters_killed) | set(hunter.data_critters_killed)}
    
    settings.agent_days_rested = {key: settings.agent_days_rested.get(key, 0) + hunter.data_days_rested.get(key, 0) for key in set(settings.agent_days_rested) | set(hunter.data_days_rested)}
    
    #rounds played
    settings.world_rounds_played_cummulative += the_world.rounds_played
    
    # 10 performance treshold achieved
    if hunter.data_performance_threshold_10:
        settings.agent_performance_threshold_10_cummulative += 1
        
    # 20 performance treshold achieved
    if hunter.data_performance_threshold_20:
        settings.agent_performance_threshold_20_cummulative += 1
        
    # 30 performance treshold achieved
    if hunter.data_performance_threshold_30:
        settings.agent_performance_threshold_30_cummulative += 1
        
    # 40 performance treshold achieved
    if hunter.data_performance_threshold_40:
        settings.agent_performance_threshold_40_cummulative += 1
    
    # 50 performance treshold achieved
    if hunter.data_performance_threshold_50:
        settings.agent_performance_threshold_50_cummulative += 1
        
    # 60 performance treshold achieved
    if hunter.data_performance_threshold_60:
        settings.agent_performance_threshold_60_cummulative += 1
    
    # 70 performance treshold achieved
    if hunter.data_performance_threshold_70:
        settings.agent_performance_threshold_70_cummulative += 1
        
    # 80 performance treshold achieved
    if hunter.data_performance_threshold_80:
        settings.agent_performance_threshold_80_cummulative += 1
        
    # 90 performance treshold achieved
    if hunter.data_performance_threshold_90:
        settings.agent_performance_threshold_90_cummulative += 1
        
    # 100 performance treshold achieved
    if hunter.data_performance_threshold_100:
        settings.agent_performance_threshold_100_cummulative += 1
    
    
    # settings.world_critter_count -->
    killed_crits_cumulative = 0
    for game_step in range(0,settings.steps_per_game):
        settings.world_critter_count[game_step] += int(settings.world_crit)        
        if game_step in hunter.data_critters_killed:
            killed_crits_cumulative = killed_crits_cumulative + hunter.data_critters_killed[game_step]
            
        settings.world_critter_count[game_step] = settings.world_critter_count[game_step] - killed_crits_cumulative
    
    
    # create matching multidimensional list for heatmap    
    outer_list=[]    
    inner_list=[]
    for i in range(settings.world_x):
        for j in range(settings.world_y):
            inner_list.append(the_world.data_agent_location_tracker[i,j])        
        outer_list.append(inner_list)
        inner_list=[]  
    
    if settings.full_debug:
        print("Outer Heatmap List - Iteration:", outer_list)
        print("General Heatmap List - After Iteration:", settings.heatmap_list)
    settings.heatmap_list = np.add(settings.heatmap_list, outer_list)
    
    

def calc_average_value_per_list_item(the_list):
    ''' returns a list with each position being the average value for the whole run (item/cycles)'''
    
    new_list = []
    for item in the_list:
        item = item / settings.number_of_runs
        new_list.append(item)
    return new_list


# - - - PREP - - -
def prep_accumulated_data():
    ''' takes accumulated data and preps it for graphing / output '''
    
    global traveled, traveled_on_days    
    global agent_death, agent_death_on_days
    global energy_levels_per_day, energy_level_days
    global performance_value, performance_days
    global critter_numbers, critter_numbers_per_day    
    global critters_killed_on_day, critters_killed_days
    global rested, rested_on_days
    
    # Days Travelled
    traveled = [value for value in settings.agent_travels.values()] # values
    traveled_on_days = [value for value in settings.agent_travels] # keys
    
    # Agent Died
    if settings.agent_deaths: # not empty
        agent_death = [value for value in settings.agent_deaths.values()] # values (y)
        agent_death_on_days = [value for value in settings.agent_deaths] # keys = days (x)
    
    # Energy
    energy_levels_per_day = [value for value in settings.agent_energy_level.values()]
    energy_level_days = [value for value in settings.agent_energy_level]
    
    # Performance
    performance_value = [value for value in settings.agent_performance_level.values()]
    performance_days = [value for value in settings.agent_performance_level]
    
    # Critter Count
    critter_numbers = [value for value in settings.world_critter_count.values()]
    critter_numbers_per_day = [value for value in settings.world_critter_count]
    
    # Critters Hunted
    if settings.agent_critters_killed: # not empty
        critters_killed_on_day = [value for value in settings.agent_critters_killed.values()]
        critters_killed_days = [value for value in settings.agent_critters_killed]
    
    # Days Rested
    if settings.agent_days_rested: # not empty    
        rested = [value for value in settings.agent_days_rested.values()]
        rested_on_days = [value for value in settings.agent_days_rested]
    

# PRINT DATA SUMMARIES
def output_general_statistics():
    print("\n======= STATISTICS =======")
    #print("Total number of world runs:", settings.number_of_runs)
    print("Max total rounds for this run set:", settings.steps_per_game)
    print("Rounds per game played until all SUCCESS entailments (all deer tracks found) are valid:", round(settings.world_rounds_played_cummulative / settings.number_of_runs , 2 ) )
    
    """
    print("\n----- Agent Survival -----")
    if settings.agent_deaths: # not empty
        print("Agent deaths total:", sum(agent_death), "(Equals:", round(sum(agent_death)/settings.number_of_runs*100,2) , "% of total games)" )
        print("Agent survived (all rounds):", settings.number_of_runs - sum(agent_death), "(Equals:", round(100 - sum(agent_death)/settings.number_of_runs*100,2) , "% of total games)" )
    else:
        print("Agent survived 100% of runs.")
    
    print("\n----- Critter Stats -----")
    avg_crit_end = critter_numbers[-1]/settings.number_of_runs
    print("Average number of critters alive at end of game:", round(avg_crit_end,2), "\n(Equals:", round(avg_crit_end/settings.world_crit*100,2) , "% of total.)" )
    
    print("\n----- Performance Goals -----")
    print("The agent achieved the following performance scores:")
    print("10 points:", settings.agent_performance_threshold_10_cummulative, "times (", round(settings.agent_performance_threshold_10_cummulative/settings.number_of_runs*100,2) , "% of total rounds)" )
    print("20 points:", settings.agent_performance_threshold_20_cummulative, "times (", round(settings.agent_performance_threshold_20_cummulative/settings.number_of_runs*100,2) , "% of total rounds)" )
    print("30 points:", settings.agent_performance_threshold_30_cummulative, "times (", round(settings.agent_performance_threshold_30_cummulative/settings.number_of_runs*100,2) , "% of total rounds)" )
    print("40 points:", settings.agent_performance_threshold_40_cummulative, "times (", round(settings.agent_performance_threshold_40_cummulative/settings.number_of_runs*100,2) , "% of total rounds)" )
    print("50 points:", settings.agent_performance_threshold_50_cummulative, "times (", round(settings.agent_performance_threshold_50_cummulative/settings.number_of_runs*100,2) , "% of total rounds)" )
    print("60 points:", settings.agent_performance_threshold_60_cummulative, "times (", round(settings.agent_performance_threshold_60_cummulative/settings.number_of_runs*100,2) , "% of total rounds)" )
    print("70 points:", settings.agent_performance_threshold_70_cummulative, "times (", round(settings.agent_performance_threshold_70_cummulative/settings.number_of_runs*100,2) , "% of total rounds)" )
    print("80 points:", settings.agent_performance_threshold_80_cummulative, "times (", round(settings.agent_performance_threshold_80_cummulative/settings.number_of_runs*100,2) , "% of total rounds)" )
    print("90 points:", settings.agent_performance_threshold_90_cummulative, "times (", round(settings.agent_performance_threshold_90_cummulative/settings.number_of_runs*100,2) , "% of total rounds)" )
    print("100 points:", settings.agent_performance_threshold_100_cummulative, "times (", round(settings.agent_performance_threshold_100_cummulative/settings.number_of_runs*100,2) , "% of total rounds)" )
    """
    

# ALL GRAPHS
def graph_travel_and_rest_data():
    ''' chart for travel data'''
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Days Travelled (CUMULATIVE)
    plt.plot(traveled_on_days,traveled, 'b-')
    plt.ylabel('Traveled Steps (Cumulative)')
    plt.xlabel('Days / Steps')
    plt.axis([0,max(traveled_on_days)+2,0,max(traveled)+2]) # plt.axis([0,settings.steps_per_game,0,max(traveled)+2])
    
    plt.axis()
    plt.show()
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Days Rested (CUMULATIVE)
    if settings.agent_days_rested: # not empty        
        plt.plot(rested_on_days,rested, 'mo')
        plt.ylabel('Rested (Cumulative)')
        plt.xlabel('Days / Steps')
        plt.axis([0,max(rested_on_days)+2,0,max(rested)+2])
        plt.show()
    else:
        print("No Days Rested!")
    
    
def graph_death_data():
    ''' chart for death data'''
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Agent Died (CUMULATIVE)
    if settings.agent_deaths: # not empty
        plt.plot(agent_death_on_days,agent_death, 'ro')
        plt.ylabel('Agent Death (Cumulative)')
        plt.xlabel('Days / Steps')
        plt.axis([0,max(agent_death_on_days)+2,0,max(agent_death)+2]) 
        plt.axis()    
        plt.show()
    else:
        print("No Agent Deaths Occured!")
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -        
    # Agent Died Histogram (CUMMULATIVE)
    if settings.agent_deaths: # not empty
        day_list = agent_death_on_days
        occurance_list = agent_death
        bin_list = []
        
        for i in range(0,len(occurance_list)): # any of the 2 lists would work here (just about length)
            tempList = [day_list[i]] * occurance_list[i]
            bin_list.extend(tempList)
        
        #print("Agent Death on Days:", agent_death_on_days)
        #print("Agent Death:", agent_death)
        #print("bin_list:", bin_list)    
        
        #number_of_bins = 10
        number_of_bins = [0,10,20,30,40,50,60,70,80,90,100]
        
        #n, bins, patches = plt.hist(bin_list, number_of_bins, density=True, facecolor='r', alpha=0.6)
        
        n, bins, patches = plt.hist(bin_list, number_of_bins,  rwidth=0.95, facecolor='r', alpha=0.8)
        
        plt.xlabel('Days / Steps')
        plt.ylabel('Number Of Deaths')
        plt.title('Cummulative Number Of Deaths Histogram (Interval 10)')   
        plt.grid(True)
        plt.show()
    

def graph_energy_data():
    ''' chart for energy data'''
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Energy  (AVERAGE)    
    plt.plot(energy_level_days, calc_average_value_per_list_item(energy_levels_per_day),'g-')    
    plt.ylabel('Energy Levels (Average)')
    plt.xlabel('Days / Steps')
    plt.axis([0,settings.steps_per_game,0,100])
    plt.show() 
    

def graph_performance_data():
    ''' chart for performance data'''
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Performance (AVERAGE)    
    plt.plot(performance_days,calc_average_value_per_list_item(performance_value), 'r-')
    plt.ylabel('Performance Level (Average)')
    plt.xlabel('Days / Steps')
    plt.axis([0,settings.steps_per_game,min(calc_average_value_per_list_item(performance_value))-1,max(calc_average_value_per_list_item(performance_value))+1])
    plt.show()  
    
def graph_performance_histogram():
    ''' histogram for performance points achieved'''
    
    bin_list = []
    listOfThresholds = []
    listOfThresholds.append(settings.agent_performance_threshold_10_cummulative)
    listOfThresholds.append(settings.agent_performance_threshold_20_cummulative)
    listOfThresholds.append(settings.agent_performance_threshold_30_cummulative)
    listOfThresholds.append(settings.agent_performance_threshold_40_cummulative)
    listOfThresholds.append(settings.agent_performance_threshold_50_cummulative)
    listOfThresholds.append(settings.agent_performance_threshold_60_cummulative)
    listOfThresholds.append(settings.agent_performance_threshold_70_cummulative)
    listOfThresholds.append(settings.agent_performance_threshold_80_cummulative)
    listOfThresholds.append(settings.agent_performance_threshold_90_cummulative)
    listOfThresholds.append(settings.agent_performance_threshold_100_cummulative)
    
    localCounter = 0
    for i in range(10,101,10):
        tempList = [i] * listOfThresholds[localCounter]
        localCounter += 1
        bin_list.extend(tempList)    

    number_of_bins = [0,10,20,30,40,50,60,70,80,90,100]
    
    n, bins, patches = plt.hist(bin_list, number_of_bins,  rwidth=0.95, facecolor='g', alpha=0.8)
    
    plt.xlabel('Performance Points')
    plt.ylabel('Number Of Times Achieved')
    plt.title('Performance Measure Overall (Points Achieved)')
    plt.grid(True)
    plt.show()    

def graph_critter_data():
    ''' chart for critter data'''
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Critter Count (AVERAGE)  
    plt.plot(critter_numbers_per_day,calc_average_value_per_list_item(critter_numbers), 'y-')
    plt.ylabel('Critter Count (Average)')
    plt.xlabel('Days / Steps')
    #plt.axis([0,settings.steps_per_game, min(calc_average_value_per_list_item(critter_numbers))-1, max(calc_average_value_per_list_item(critter_numbers))+1])    
    plt.axis([0,settings.steps_per_game, 0, max(calc_average_value_per_list_item(critter_numbers))+1])    
    plt.show()
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Critters Hunted (CUMULATIVE)
    if settings.agent_critters_killed: # not empty
        
        plt.plot(critters_killed_days, critters_killed_on_day, 'co')
        plt.ylabel('Critters Hunted (Cumulative)')
        plt.xlabel('Days / Steps')
        plt.axis([0,settings.steps_per_game,0,max(critters_killed_on_day)+2])
        plt.show()
    else:
        print("No Critters Hunted!")
        
def show_heatmap():
    ''' steps heatmap '''
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # HEATMAP - Steps Per Coordinate (CUMULATIVE)    
    the_grid = np.array(settings.heatmap_list)
    fig, ax = plt.subplots()
    
    # enabled only if cloning is ON (otherwise this makes no sense)
    if settings.cloning_enabled:
        # highlight agent starting location
        ax.add_patch(Rectangle((settings.agent_starting_location[1]-0.5, settings.agent_starting_location[0]-0.5), 1, 1, fill=False, edgecolor='orange', lw=4))
        
        # highlight deer starting locations
        for deer_location in  settings.deer_init_starting_locations:
            ax.add_patch(Rectangle((deer_location[1]-0.5, deer_location[0]-0.5), 1, 1, fill=False, edgecolor='darkgreen', lw=6))
        
        
        # highlight deer track location
        for track_loc in settings.deer_tracks_locations:        
            ax.add_patch(Rectangle((track_loc[1]-0.5, track_loc[0]-0.5), 1, 1, fill=False, edgecolor='fuchsia', lw=1))
        
        # highlight huts
        for position in settings.huts_starting_locations:
            ax.add_patch(Rectangle((position[1]-0.5, position[0]-0.5), 1, 1, fill=False, edgecolor='fuchsia', lw=1))
    
    
    
    
    im, cbar = heatmap(the_grid, ax=ax, cmap="YlGnBu", cbarlabel="Days Spent On Tile (Cumulative)")  
    annotate_heatmap(im)
    fig.tight_layout()
    plt.show()
    
    

def graph_all_data():
    ''' does what it says '''
    graph_travel_and_rest_data()
    graph_death_data()
    graph_energy_data()
    graph_performance_data()
    graph_performance_histogram()
    graph_critter_data()    
    show_heatmap()
    
      
    
    
    
    
    
    
    
    
    
     
    