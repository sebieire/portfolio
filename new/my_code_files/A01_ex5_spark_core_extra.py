# --------------------------------------------------------
#
# PYTHON PROGRAM DEFINITION
#
# The knowledge a computer has of Python can be specified in 3 levels:
# (1) Prelude knowledge --> The computer has it by default.
# (2) Borrowed knowledge --> The computer gets this knowledge from 3rd party libraries defined by others
#                            (but imported by us in this program).
# (3) Generated knowledge --> The computer gets this knowledge from the new functions defined by us in this program.
#
# When launching in a terminal the command:
# user:~$ python3 this_file.py
# our computer first processes this PYTHON PROGRAM DEFINITION section of the file.
# On it, our computer enhances its Python knowledge from levels (2) and (3) with the imports and new functions
# defined in the program. However, it still does not execute anything.
#
# --------------------------------------------------------

import pyspark
import datetime
from math import cos, sin, atan2, radians, sqrt


# ------------------------------------------
# FUNCTION process_line
# ------------------------------------------
def process_line(line):
    # 1. We create the output variable
    res = ()

    # 2. We get the parameter list from the line
    params_list = line.strip().split(",")

    #(00) Date => The date of the measurement. String <%Y-%m-%d %H:%M:%S> (e.g., "2013-01-01 13:00:02").
    #(01) Bus_Line => The bus line. Int (e.g., 120).
    #(02) Bus_Line_Pattern => The pattern of bus stops followed by the bus. String (e.g., "027B1001"). It can be empty (e.g., "").
    #(03) Congestion => On whether the bus is at a traffic jam (No -> 0 and Yes -> 1). Int (e.g., 0).
    #(04) Longitude => Longitude position of the bus. Float (e.g., -6.269634).
    #(05) Latitude = > Latitude position of the bus. Float (e.g., 53.360504).
    #(06) Delay => Delay of the bus in seconds (negative if ahead of schedule). Int (e.g., 90).
    #(07) Vehicle => An identifier for the bus vehicle. Int (e.g., 33304)
    #(08) Closer_Stop => An idenfifier for the closest bus stop given the current bus position. Int (e.g., 7486). It can be no bus stop, in which case it takes value -1 (e.g., -1).
    #(09) At_Stop => On whether the bus is currently at the bus stop (No -> 0 and Yes -> 1). Int (e.g., 0).

    # 3. If the list contains the right amount of parameters
    if (len(params_list) == 10):
        # 3.1. We set the right type for the parameters
        params_list[1] = int(params_list[1])
        params_list[3] = int(params_list[3])
        params_list[4] = float(params_list[4])
        params_list[5] = float(params_list[5])
        params_list[6] = int(params_list[6])
        params_list[7] = int(params_list[7])
        params_list[8] = int(params_list[8])
        params_list[9] = int(params_list[9])

        # 3.2. We assign res
        #res = tuple(params_list)
        
        # altered - only used required params for smaller initial size
        res = tuple( (params_list[0], params_list[1], params_list[3], params_list[4], params_list[5], params_list[6], params_list[8],params_list[9]) )
        # omitting: pattern id and vehicle id

    # 4. We return res
    return res

  
# ------------------------------------------
# FUNCTION filter_day_congest_delay
# ------------------------------------------
def filter_day_congest_delay(element, max_delay):  
  
  # CUSTOM AUX FUNCTION
  # must return True for element to be included in new RDD
  
  # filters elements by:
  # - must be at stop (=1)
  # - week day (weekend not included)
  # - no congestion only
  # - no higher delay than 'max_delay'
  
  # input shape expected example: ('2013-01-01 08:06:02', 40, 0, -6.280644, 53.343063, -59, 1939, 1)
  
  # placeholder return var
  res = False
  flag_at_stop = False
  flag_date_ok = False
  flag_congestion = False
  flag_delay = False
  
  # Step 1: check at stop
  if element[7] == 1:
    flag_at_stop = True
  
  # speed up and ignore rest if first flag fails (saves some time)
  if flag_at_stop:
    
    # Step 2: check date
    # split date (into date & time) and check if both are valid 
    the_date = datetime.datetime.strptime(element[0], "%Y-%m-%d %H:%M:%S") # date with time
    weekday = the_date.weekday() # int value 0-6 for day of week

    if weekday < 5: # must be Mon - Fri (0-4)
      flag_date_ok = True

    # speed up and ignore rest if second flag fails (saves some time)
    if flag_date_ok:
      
      # Step 3: check congestion
      if element[2] == 0:
        flag_congestion = True

      # Step 4: check delay
      if element[5] < max_delay:
        flag_delay = True
  
  # Final Check
  if flag_at_stop and flag_date_ok and flag_congestion and flag_delay:
    res = True  
    
  return res


# ------------------------------------------
# FUNCTIONS base_combiner, merge_values, merge_combiners ---> task A & B combineByKey functions
# ------------------------------------------

# CUSTOM AUX FUNCTIONS
# input values to combineByKey: (279, 544) where the first part (= bus line or stop) is the key

def base_combiner(value):
  
  # NOTE: as outlined by specs any negative value is deemed to be 0 (= on time)
  
  # insure not negative  
  if value < 0:
    value = 0
  
  return (value, 1)

def merge_values(accum, new_value):
  
  # insure not negative
  if new_value < 0:
    new_value = 0
    
  res = (accum[0] + new_value, accum[1] + 1)
  
  return res
  
def merge_combiners(accum1, accum2):  
  
  res = (accum1[0] + accum2[0], accum1[1] + accum2[1] )  
    
  return res
  

###################### TASK A SPECIFIC FUNCTIONS ######################

# ------------------------------------------
# FUNCTION map_stop_delay_task_A
# ------------------------------------------
def map_stop_delay_task_A(element):  
  
  # CUSTOM AUX FUNCTION
  # map bus stop & delay
  
  return( (element[6],element[5]) )


###################### TASK B SPECIFIC FUNCTIONS ######################

# ------------------------------------------
# FUNCTION map_busLine_delay_task_B
# ------------------------------------------
def map_busLine_delay_task_B(element):  
  
  # CUSTOM AUX FUNCTION
  # map bus line & delay
  
  return( (element[1],element[5]) )


###################### TASK C SPECIFIC FUNCTIONS ######################

# ------------------------------------------
# FUNCTION filter_selected_bus_stops
# ------------------------------------------
def filter_selected_bus_stops(element, bus_stop_list):  
  
  # CUSTOM AUX FUNCTION
  # must return True for element to be included in new RDD
  
  # filters elements by:
  # - bus stop must be in bus_stop_list  
  
  # input shape expected example: ('2013-01-10 09:05:59', 40, 0, -6.258078, 53.339279, 544, 279, 1)
  
  # placeholder return var
  res = False
  
  if element[6] in bus_stop_list:
    res = True
    
  return res

# ------------------------------------------
# FUNCTION map_busStop_longLat
# ------------------------------------------
def map_busStop_longLat(element):  
  
  # CUSTOM AUX FUNCTION
  # map bus stop & (long, lat)
  # expected input: ('2013-01-10 09:05:59', 41, 0, -6.258078, 53.339279, 544, 279, 1)
  
  return( (element[6], (element[3], element[4]) ) )

# ------------------------------------------
# FUNCTIONS base_combiner_TaskC, merge_values_TaskC, merge_combiners_TaskC
# ------------------------------------------

# CUSTOM AUX FUNCTIONS
# input values to combineByKey: (279, (-6.258078, 53.339279)) where the first part (= bus stop) is the key

# NOTE: the below averaging method works for close proximities in 
# Lat & Long but is not possible for large distances or close to north & south pole

def base_combiner_TaskC(value):
  
  return ((value[0], value[1]), 1) # ((Long & Lat) + count)

def merge_values_TaskC(accum, new_value):
  
  sum_long = (accum[0][0] + new_value[0])
  sum_lat = (accum[0][1] + new_value[1])
  
  return ( (sum_long,sum_lat), accum[1] + 1  ) # ((Long & Lat) + count)
  
def merge_combiners_TaskC(accum1, accum2):  
  
  res = ( (accum1[0][0] + accum2[0][0], accum1[0][1] + accum2[0][1]), accum1[1]+ accum2[1] )
    
  return res

# ------------------------------------------
# FUNCTION distance_bundle_calculation
# ------------------------------------------
def distance_bundle_calculation(resVal_T3, proximity_radius_km ): 
  
  # CUSTOM AUX FUNCTION
  # merges 'bus stops' by given proximity_radius if applicable
  # returns a dictionary {0: [ [bus stop(s) list], (Lat, Long)  ] }  
  
  # input shape expected example: resVal_T3 [(244, (-6.458078, 53.439279)),....]
  
  res_dict = {}
  index = 0
  
  earth_radius = 6371 # km
  
  list_length = len(resVal_T3)
  
  # for each item in the list
  for i in range(list_length):
    # go through each item in the list
    for j in range(list_length):
      
      # make sure they are not the same items
      if i != j:       
        
        # get Long & Lat for each of the 2 current i & j resVal_T3 bus stops
        longitude_1 = resVal_T3[i][1][0]
        latitude_1 = resVal_T3[i][1][1]
        
        longitude_2 = resVal_T3[j][1][0]
        latitude_2 = resVal_T3[j][1][1]
        
        # calculate distance between the 2
        rad_long_1 = radians(longitude_1)
        rad_lat_1 = radians(latitude_1)
        
        rad_long_2 = radians(longitude_2)
        rad_lat_2 = radians(latitude_2)
        
        diff_lon = rad_long_2 - rad_long_1
        diff_lat = rad_lat_2 - rad_lat_1
        
        ar = sin(diff_lat / 2)**2 + cos(rad_lat_1) * cos(rad_lat_2) * sin(diff_lon / 2)**2
        c = 2 * atan2(sqrt(ar), sqrt(1 - ar))        
        distance = earth_radius * c
        
        # optional debug output
        #print("Distance:", distance, " i:", resVal_T3[i][0], " j:", resVal_T3[j][0])        
        
        ########################### ------> WITHIN DISTANCE
        # if distance is within radius -> create or append to proximity group (of bus stops)
        if distance <= proximity_radius_km:
          
          # not empty dict -> check a few things
          if res_dict:
          
            # make sure bus stops aren't already in it!
            bus_stop_i_already_contained = False
            bus_stop_j_already_contained = False
            
            # will contain key for res_dict if found
            res_dict_i_key = 0 
            res_dict_j_key = 0 
            
            for key in res_dict: # {0: [ [bus stop(s) list], (Lat, Long)  ] }              
              
              if resVal_T3[i][0] in res_dict[key][0]: # example: if '244' in {0: [ [264,244], ()  ] ==> True
                bus_stop_i_already_contained = True
                res_dict_i_key = key
                
              if resVal_T3[j][0] in res_dict[key][0]:
                bus_stop_j_already_contained = True
                res_dict_j_key = key
            
            # case 1: neither is yet included (add both together as a group)
            if not bus_stop_i_already_contained and not bus_stop_j_already_contained:
              # add both bus stops              
              res_dict[index] = [ [ resVal_T3[i][0], resVal_T3[j][0] ] , () ]
              index += 1
              
            # case 2: i is included but not j
            if bus_stop_i_already_contained and not bus_stop_j_already_contained:
              # add j to i into the correct res_dict[res_dict_i_key]
              res_dict[res_dict_i_key][0].append(resVal_T3[j][0])              
                  
            # case 3: j is included but not i
            if bus_stop_j_already_contained and not bus_stop_i_already_contained:
              # add i to j into the correct res_dict[res_dict_j_key]
              res_dict[res_dict_j_key][0].append(resVal_T3[i][0])
              
            # case 4: j and i are included but are seperately included (CHECK!)
            # NOTE: this is a corner case that happens if both are cycled independently through an 
            # "outside distance" bus stop and then added as individuals instead (happens often)
            if bus_stop_j_already_contained and bus_stop_i_already_contained:
              # first check that they are NOT in the same key
              if res_dict_i_key != res_dict_j_key:
                # now check that one of them is of length 1 and move it to the other
                # if both are > len 1 = special case.... they are paired with others
                # don't do anything about that for now
                
                # case for i: if exists and len = 1
                if res_dict[res_dict_i_key] and len(res_dict[res_dict_i_key][0]) == 1:
                  # add to i to j
                  res_dict[res_dict_j_key][0].append(resVal_T3[i][0])
                  # delete dict[i]
                  res_dict.pop(res_dict_i_key)
                  
                # case for j: if exists and len = 1
                if res_dict[res_dict_j_key] and len(res_dict[res_dict_j_key][0]) == 1:
                  # add to j to i
                  res_dict[res_dict_i_key][0].append(resVal_T3[j][0])
                  # delete dict[j]
                  res_dict.pop(res_dict_j_key)

          
          # empty dict
          else:
            # put in first pair of bus stops and Lat & Long            
            res_dict[index] = [ [ resVal_T3[i][0], resVal_T3[j][0] ] , () ]
            index += 1
        
        ########################### ------> OUTSIDE DISTANCE
        # if distance is outside radius
        else:
          
          # not empty dict
          if res_dict:
          
            # make sure bus stops aren't already included
            bus_stop_i_already_contained = False
            bus_stop_j_already_contained = False
            
            for key in res_dict: # {0: [ [bus stop(s) list], (Lat, Long)  ] }              
              
              if resVal_T3[i][0] in res_dict[key][0]: # example: if '244' in {0: [ [264,244], ()  ] ==> True
                bus_stop_i_already_contained = True                
                
              if resVal_T3[j][0] in res_dict[key][0]:
                bus_stop_j_already_contained = True                
            
            # case 1: neither is yet included (add 2 entities together)
            if not bus_stop_i_already_contained and not bus_stop_j_already_contained:
              # add both bus stops 
              res_dict[index] = [ [ resVal_T3[i][0] ] , () ]
              index += 1
              res_dict[index] = [ [ resVal_T3[j][0] ] , () ]
              index += 1
              
            # case 2: i is included but not j
            if bus_stop_i_already_contained and not bus_stop_j_already_contained:
              # add j as a new entity
              res_dict[index] = [ [ resVal_T3[j][0] ] , () ]
              index += 1                  
                  
            # case 3: j is included but not i
            if bus_stop_j_already_contained and not bus_stop_i_already_contained:
              # add i as a new entity
              res_dict[index] = [ [ resVal_T3[i][0] ] , () ]
              index += 1
          
          # empty dict - create 2 separate entities
          else:
            res_dict[index] = [ [ resVal_T3[i][0] ] , () ]
            index += 1
            res_dict[index] = [ [ resVal_T3[j][0] ] , () ]
            index += 1
  
  # now calculate the average Long & Lat for the grouped bus stops and add to each bus_stop_group
  
  # IS NOW:
  # dict = {0: [ [bus stop list], ( )  ] , 1: .... }  
  # resVal_T3 = [ (244, (-6.458078, 53.439279)) , ....]
  
  # GOAL:
  # dict = {0: [ [bus stop list], (Avg Lat, Avg Long)  ] , 1: .... } 
  
  for key in res_dict:
    
    long_sum = 0
    lat_sum = 0    
    count = 0
    
    # go through every key in res_dict (obtain bus stops list)    
        
    #correlate with resVal_T3 bus stops
    for element in resVal_T3:
      # if bus_stop in bus_stop list
      if element[0] in res_dict[key][0]: 
        long_sum = long_sum + element[1][0]
        lat_sum = lat_sum + element[1][1]
        count += 1
        
    # calc and assign the average long, lat values to dict
    res_dict[key][1] = (round(lat_sum/count,6), round(long_sum/count,6)) 
    # exchanged long & lat positions here (easier to copy for like Google Maps and stuff)  
  
  # return dict
  return res_dict
    

  
# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(sc, my_dataset_dir, max_sec_delay_threshold, num_bus_stops, num_bus_lines, proximity_radius_km):
  
    """
      TASK: Own Scenario:
      
      A) Find the top number of bus stops that show the highest average delay of arrival during weekdays when there is no congestion.
      Only consider records that indicate 'at stop' = 1.
      
      Take into account the 'max_sec_delay_threshold' as the ceiling for attributed delay and discard the rest.
      NOTE: The maximum delay in the dataset found is 116122 seconds and there are many more with anywhere
      from 10k-40k+ seconds of delay that are obviously incorrect (does this indicate a parked bus or in the garage?). 
      
      Regard any negative delay (early arrival) as 0, indicating an 'on time' arrival instead so as to not skew the negative values.
      
      (Warning: Bus stops and bus lines are represented repeatedly with the same data over and over. Cummulative results will not work but averaging 
      through division by count should give a relatively accurate picture instead....one can only hope... ha ha ha)   :D 
      
      B) What top number of bus lines are most affected by the delay? Use the same settings as in task A.
      
      C) In relation to findings in Task A (top number of delayed bus stops): cross reference the bus latitude and longitude coordinates with 
      associated delay of bus stops. Use a pre-defined area of proximity (radius) to bundle the bus stops into groups (promity to each other).
      
      Evaluate (Google map location) if most stops (aka bus locations) with a higher delay are within the same area in the city or if the opposite is the case.
      
      The former could indicate related traffic issues based on a few key locations only while the latter could indicate traffic 
      issues in various locations possibly unrelated to each other (e.g.: city center vs multiple different locations).
      
      (Warning: Stops have different Lat & Long values attributed to them. Ideally use the average proximity instead as applicable.)
            
    """
    
    ##################################################
    # Additional Settings:

    run_task_A = True
    run_task_B = True
    run_task_C = True # MUST RUN TASK A TO WORK !
    
    output_A = True
    output_B = True
    output_C = True
    output_C_coordinates_only = True # for easy copy paste
    output_settings = True

    # results dir     
    result_dir_base = "FileStore/tables/6_Assignments/"

    # Utilize Test Data? (assuming here databricks only and NOT local files)
    use_test_data = False
    if use_test_data:
      my_dataset_dir = "/FileStore/tables/6_Assignments/ex5_small_dataset/"
      result_dir = result_dir_base + "results/ex5_small_dataset/"
    else:
      result_dir = result_dir_base + "results/ex5/"
      
    ##################################################
    
    if output_settings:
      print("\n[PARAMETERS SET]: \n* max seconds treshold per element taken into account:", max_sec_delay_threshold,
            "\n* return num of bus stops:", num_bus_stops,
            "\n* return num of bus lines:", num_bus_lines,
            "\n* proximity radius (km):", proximity_radius_km)
    
    # - - - PREP - - -
  
    # 1. Operation C1: 'textFile' to load the dataset into an RDD
    
    inputRDD = sc.textFile(my_dataset_dir)
    
    # 2. Transformations / Process / Output
    
    # -------------> process each line and omitting pattern id and vehicle id
    preProcessedRDD = inputRDD.map(process_line)
    # example -> one element will now look like:  ('2013-01-01 08:06:02', 40, 0, -6.280644, 53.343063, -59, 1939, 1)
    
    # -------------> filter by date, congestion and delay
    date_congestion_delay_filteredRDD = preProcessedRDD.filter(lambda element: filter_day_congest_delay(element, max_sec_delay_threshold) )
    # example -> one element will now look like: ('2013-01-10 09:05:59', 40, 0, -6.258078, 53.339279, 544, 279, 1)
    
    # persist (will be required again)
    date_congestion_delay_filteredRDD.persist()
    
    
    # - - - TASK A - - -
    if run_task_A:
    
      # -------------> map bus stop & respective delay for that day/time
      busStopDelayRDD = date_congestion_delay_filteredRDD.map(map_stop_delay_task_A)
      # example -> one element will now look like: (279, 544)

      # -------------> combine by key: (key (sum of delays, count)) AND map to a tuplet (easier to sort) AND sort by delay high to low
      combinedTaskA_RDD = busStopDelayRDD.combineByKey(base_combiner, merge_values, merge_combiners)\
      .map(lambda item: (item[0], item[1][0] / item[1][1]) )\
      .sortBy(lambda item: item[1] * (-1))
      # example -> one element will now look like - after combine: (279, (655, 2)) after map: (279, 327.5)    

      # -------------> OUTPUT TASK A
      # with the assumption is that the output can now be formatted, unlike in Ex1 - Ex4

      # take only 'num_bus_stops' highest
      task_A_combi_busLine_delay = combinedTaskA_RDD.take(num_bus_stops)
      task_A_result_bus_stop_list = [] # required for Task C
      
      if output_A:
        print("\n[TASK A] - The", num_bus_stops, "highest bus stops with the most delay on average (weekdays and no congestion):\n")
        
      for item in task_A_combi_busLine_delay:
        if output_A:
          print("Bus stop:", item[0], " with an average delay of:", int(item[1]), "seconds (equal to", round(item[1]/60, 2), "minutes or", round(item[1]/3600, 2) ,"hours).")
        task_A_result_bus_stop_list.append(item[0])
    
    
    # - - - TASK B - - - (similar to A)
    if run_task_B:
      
      # -------------> map bus line & respective delay for that day/time
      busLineDelayRDD = date_congestion_delay_filteredRDD.map(map_busLine_delay_task_B)
      # example -> one element will now look like: (40, 544)
      
      # -------------> combine by key: (key (sum of delays, count)) AND map to a tuplet (easier to sort) AND sort by delay high to low
      combinedTaskB_RDD = busLineDelayRDD.combineByKey(base_combiner, merge_values, merge_combiners)\
      .map(lambda item: (item[0], item[1][0] / item[1][1]) )\
      .sortBy(lambda item: item[1] * (-1))
      # example -> one element will now look like -  after map: (41, 272.0)
      
      # -------------> OUTPUT TASK B
      # with the assumption is that the output can now be formatted, unlike in Ex1 - Ex4

      # take only 'num_bus_stops' highest
      items = combinedTaskB_RDD.take(num_bus_lines)
      
      if output_B:

        print("\n[TASK B] - The", num_bus_lines, "highest bus lines with the most delay on average (weekdays and no congestion):\n")
        for item in items:
          print("Bus line:", item[0], " with an average delay of:", int(item[1]), "seconds (equal to", round(item[1]/60, 2), "minutes or", round(item[1]/3600, 2) ,"hours).")
   
  
    # - - - TASK C - - - (works only with Task A)
    if run_task_C and run_task_A:
      
      # -------------> filter by task_A_result_bus_stop_list (only use stops that are in the list from Task A)
      task_A_busStop_filteredRDD = date_congestion_delay_filteredRDD.filter(lambda element: filter_selected_bus_stops(element, task_A_result_bus_stop_list) )
      # example -> one element will now look like: ('2013-01-10 09:05:59', 41, 0, -6.258078, 53.339279, 544, 279, 1)
      
      # -------------> map bus_stop & (Long, Lat)
      map_busStop_longLat_RDD = task_A_busStop_filteredRDD.map(map_busStop_longLat)
      # example -> one element will now look like: (279, (-6.258078, 53.339279))
      
      # -------------> combine by key and map: (key (avg. Long, avg. Lat))
      combinedTaskC_RDD = map_busStop_longLat_RDD.combineByKey(base_combiner_TaskC, merge_values_TaskC, merge_combiners_TaskC)\
      .map(lambda item: (item[0], (item[1][0][0] / item[1][1] ,  item[1][0][1] / item[1][1]) ) )
      # example -> one element will now look like before map: (279, ((-12.516156, 106.678558), 2)) and after map: (279, (-6.258078, 53.339279))
      
      # -------------> GET DATA / OUTPUT TASK C
      # with the assumption is that the output can now be formatted, unlike in Ex1 - Ex4
      resVal_T3 = combinedTaskC_RDD.collect()
      
      # bundle all bus stop data corresponding to promixity (GPS coordinates)
      bundled_bus_stops_GPS_proxy = distance_bundle_calculation(resVal_T3, proximity_radius_km )
      # example -> {0: [[244], (53.439279, -6.458078)], 1: [[264], (53.639279, -6.658078)], 3: [[1939, 335, 279], (53.342481, -6.269452)]}      
      
      # NOTE: for larger samples, keys will not always be continous aka 1,2,3,4,... but will have gaps due to the way the function is implemented  
      
      if output_C:
        
        print("\n[TASK C] - Bundled bus stop data (top", num_bus_stops , "stops) according to their promixity to each other (range of", proximity_radius_km, 
              "km) with their respective average delay (weekdays no congestion).\n")

        for key in bundled_bus_stops_GPS_proxy:
          
          # calculate the average delay of each of the bundled bus_stops from task A
          current_bus_lines_avg_delay = 0
          avg_delay_count = 0
          for taskA_item in task_A_combi_busLine_delay: #[ (279, 327.5), .....]            
            # crossreference if bus_line is in bundle and add
            if taskA_item[0] in bundled_bus_stops_GPS_proxy[key][0]:
              current_bus_lines_avg_delay = current_bus_lines_avg_delay + taskA_item[1]
              avg_delay_count += 1
              
          # output    
          print("Bus stop(s):", bundled_bus_stops_GPS_proxy[key][0], "with (avg.) location:", bundled_bus_stops_GPS_proxy[key][1],
               "and average delay of:", int(current_bus_lines_avg_delay/avg_delay_count), "seconds." )     

    
      if output_C_coordinates_only:
        print("\n[TASK C] Copy Paste Coordinates:")
        
        print("\nLat / Long:")
        
        for key in bundled_bus_stops_GPS_proxy:
          
          print(bundled_bus_stops_GPS_proxy[key][1][0],",", bundled_bus_stops_GPS_proxy[key][1][1], ",", sep="")
        
        """
        print("\nLong / Lat:")
        
        for key in bundled_bus_stops_GPS_proxy:
          print(bundled_bus_stops_GPS_proxy[key][1][1],",", bundled_bus_stops_GPS_proxy[key][1][0])
        """

# --------------------------------------------------------
#
# PYTHON PROGRAM EXECUTION
#
# Once our computer has finished processing the PYTHON PROGRAM DEFINITION section its knowledge is set.
# Now its time to apply this knowledge.
#
# When launching in a terminal the command:
# user:~$ python3 this_file.py
# our computer finally processes this PYTHON PROGRAM EXECUTION section, which:
# (i) Specifies the function F to be executed.
# (ii) Define any input parameter such this function F has to be called with.
#
# --------------------------------------------------------
if __name__ == '__main__':
  
    # 1. We use as many input arguments as needed
    max_sec_delay_threshold = 3600 # = 1h
    num_bus_stops = 50 # x: show top x number bus stops with highest delay (30)
    num_bus_lines = 10 # x: show top x number bus lines with highest delay (10)
    proximity_radius_km = 5 # radius in km that allows association with other Lat/Long coordinates

    # 2. Local or Databricks
    local_False_databricks_True = True

    # 3. We set the path to my_dataset and my_result
    my_local_path = "../../../3_Code_Examples/L09-25_Spark_Environment/"
    my_databricks_path = "/"
    my_dataset_dir = "FileStore/tables/6_Assignments/my_dataset_complete/"

    if local_False_databricks_True == False:
        my_dataset_dir = my_local_path + my_dataset_dir
    else:
        my_dataset_dir = my_databricks_path + my_dataset_dir

    # 4. We configure the Spark Context
    sc = pyspark.SparkContext.getOrCreate()
    sc.setLogLevel('WARN')
    print("\n\n\n")        

    # 5. We call to our main function
    my_main(sc, my_dataset_dir, max_sec_delay_threshold, num_bus_stops, num_bus_lines, proximity_radius_km)
