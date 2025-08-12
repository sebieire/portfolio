#!/usr/bin/env python3
"""
Spark Core implementation for analyzing Dublin Bus GPS data.
Exercise 1: Calculate average bus delay by hour for specific stops.
"""

import pyspark
import datetime

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
        res = tuple(params_list)

    # 4. We return res
    return res

  
# ------------------------------------------
# FUNCTION my_filter_function
# ------------------------------------------
def my_filter_function(my_str_tuple):
  
  # CUSTOM AUX FUNCTION
  # function must return True in order to be applied to RRD.filter
  
  # one input element equals for example: ('2013-01-09 08:00:36', 40, '015B1002', 0, -6.258078, 53.339279, 300, 33488, 279, 1)
  
  # placeholder return var
  res = False
  
  # reformat hours list to ints
  int_hours = []
  for an_hour in hours_list:
    int_hours.append(int(an_hour))
  
  # Step 1: split date (into date & time) and check if both are valid 
  the_date = datetime.datetime.strptime(my_str_tuple[0], "%Y-%m-%d %H:%M:%S") # date with time
  weekday = the_date.weekday() # int value 0-6 for day of week
  hour = the_date.hour
  if weekday < 5: # must be Mon - Fri (0-4)
    if hour in int_hours: # spec: must be between 7:01 and 9:59
      
      # Step 2: check if busline num is correct
      if my_str_tuple[1] == bus_line:
        # check if bus stop is correct and if at the stop
        if my_str_tuple[8] == bus_stop and my_str_tuple[9] == 1:
          res = True
      
  return res
  
# ------------------------------------------
# FUNCTION my_map_function
# ------------------------------------------
def my_map_function(my_str_tuple):
  
  # CUSTOM AUX FUNCTION
  # one input element equals for example: ('2013-01-09 08:00:36', 40, '015B1002', 0, -6.258078, 53.339279, 300, 33488, 279, 1)
  
  # split first part into date and time
  date_time = my_str_tuple[0].split(" ")
  
  # split time into elements
  the_hour = date_time[1].split(":")  
  
  return (the_hour[0], my_str_tuple[6]) 
  
  
# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(sc, my_dataset_dir, bus_stop, bus_line, hours_list):
    
    """
      TASK:  Compute the average delay of "bus_line" vehicles when stopping at "bus_stop" for each hour of 
      "hours_list" during weekdays (you must discard any measurement taking place on a Saturday or
      Sunday).
    """
  
    ##################################################
    # Additional Settings:
    
    # results dir
    save_results_to_file = False
    result_dir_base = "FileStore/tables/results/"
    
    # Utilize Test Data?
    use_test_data = False
    if use_test_data:
      my_dataset_dir = "/FileStore/tables/ex1_small_dataset/"
      result_dir = result_dir_base + "results/ex1_small_dataset/"
    else:
      result_dir = result_dir_base + "results/ex1/"      
  
    ##################################################
  
    # 1. Operation C1: 'textFile' to load the dataset into an RDD
    inputRDD = sc.textFile(my_dataset_dir)    
    
    # 2. Transformations:
    
    # -------------> process each line
    processedRDD = inputRDD.map(process_line)
    # example -> one element will now look like: ('2013-01-09 08:00:36', 40, '015B1002', 0, -6.258078, 53.339279, 300, 33488, 279, 1)
    
    # -------------> filter for data that is required
    filteredRDD = processedRDD.filter(my_filter_function)
    # example -> one element will now (still) look like: ('2013-01-09 08:00:36', 40, '015B1002', 0, -6.258078, 53.339279, 300, 33488, 279, 1)
    
    # -------------> map and return only values per line that are required
    remappedRDD = filteredRDD.map(my_map_function) # return just tuples of ('hour', 'delay')
    # example -> one element will now look like: (8, 300)
    
    # -------------> combine by key
    combinedRDD = remappedRDD.combineByKey(lambda item_value: (item_value, 1),
                                        lambda accum, new_item_val: (accum[0] + new_item_val, accum[1] + 1),
                                        lambda final_accum1, final_accum2: (final_accum1[0] + final_accum2[0], final_accum1[1] + final_accum2[1])
                                       )
    # example -> one element will now look like: ('09', (150, 3))
    
    # -------------> calculate values map again and round - (this step is required only as the specs say that resVAL output CANNOT be altered)
    calculatedRDD = combinedRDD.map(lambda item: ( item[0], round(item[1][0]/item[1][1],2) ) )
    # example -> one element will now look like: ('09', 50.0), ('08', 66.67)
    
    # -------------> sort RDD (by increasing order of delay) as is required in Ass1 specs
    solutionRDD = calculatedRDD.sortBy(lambda item: item[1])
        
    # 3. Output / Collect 
    
    # Operation A1: 'collect' to get all results    
    resVAL = solutionRDD.collect()
    for item in resVAL:
        print(item)
        
    # 4. Save to File (Optional Flag)
    
    if save_results_to_file:
      dbutils.fs.rm(result_dir, True)
      solutionRDD.coalesce(1).saveAsTextFile(result_dir)       
    
    
# Main execution
if __name__ == '__main__':
    # 1. We use as many input arguments as needed
    bus_stop = 279
    bus_line = 40
    hours_list = ["07", "08", "09"]    

    # 2. Configure dataset path
    my_dataset_dir = "/FileStore/tables/my_dataset_complete/"

    # 3. We configure the Spark Context
    sc = pyspark.SparkContext.getOrCreate()
    sc.setLogLevel('WARN')
    print("\n\n\n")

    # 5. We call to our main function
    my_main(sc, my_dataset_dir, bus_stop, bus_line, hours_list)
