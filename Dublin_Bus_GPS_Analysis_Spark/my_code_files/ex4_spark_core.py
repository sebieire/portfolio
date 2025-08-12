#!/usr/bin/env python3
"""
Spark implementation for analyzing Dublin Bus GPS data.
Exercise 4: Find next bus and travel path simulation.
"""

import pyspark
import time
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
        #res = tuple(params_list)
        
        # altered - only used required params for smaller initial size
        res = tuple( (params_list[0], params_list[7],params_list[8],params_list[9]) ) 

    # 4. We return res
    return res


# ------------------------------------------
# FUNCTION filter_byTime_busStop
# ------------------------------------------
def filter_byTime_atStop(item, seconds_horizon, current_time):
  
  # CUSTOM AUX FUNCTION
  # must return True for element to be included
  # filters elements by time (only include delta time elements) and if atStop is 1 (= True)
  
  res = False
  within_time_horizon = False
  is_at_stop = False
  
  # 1. Only include anything that is within 'seconds_horizon' from 'current_time' - discard the rest
  start_time = datetime.datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
  end_time = start_time + datetime.timedelta(seconds = seconds_horizon)
  
  item_time = datetime.datetime.strptime(item[0], "%Y-%m-%d %H:%M:%S") # current cycled str tuple item
  
  if item_time >= start_time and item_time <= end_time:
    within_time_horizon = True
    
  # 2. Only include anything that is "at stop" = 1  
  if item[3] == 1:
    is_at_stop = True
    
  # 3. Conclude  
  if within_time_horizon and is_at_stop:
    res = True
    
  return res

# ------------------------------------------
# FUNCTION filter_byBusStop
# ------------------------------------------
def filter_byBusStop(item, bus_stop):
  
  # CUSTOM AUX FUNCTION
  # must return True for element to be included
  # filters elements by given bus_stop
  
  res = False
  
  if item[2] == bus_stop:
    res = True
    
  return res


# ------------------------------------------
# FUNCTION filter_byVehicleNumber
# ------------------------------------------
def filter_byVehicleNumber(item, vehicle_number):
  
  # CUSTOM AUX FUNCTION
  # must return True for element to be included
  # filters elements by given vehicle_number
  
  res = False
  
  if item[1] == vehicle_number:
    res = True
    
  return res

# ------------------------------------------
# FUNCTION map_solution_output
# ------------------------------------------
def map_solution_output(item):
  
  # CUSTOM AUX FUNCTION
  # maps solution output  
  # expected input: ('2013-01-10 09:05:59', 33500, 1935, 1)
  
  
  # this should suffice for last solution
  # return (item[0], item[2])
  # but need to mold into a shape that matches Ex4 output instead:    
  
  return (item[1], (item[0], item[2]) )  
  
  
# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(sc, my_dataset_dir, current_time, current_stop, seconds_horizon):
  
    """
      TASK: Compute the first bus stopping at "current_stop" and the list of bus stops it bring us within that time
      "seconds_horizon".
    """
  
    ##################################################
    # Additional Settings:
    
    # results dir
    save_results_to_file = False    
    result_dir_base = "FileStore/tables/"
    
    # Utilize Test Data? (assuming here databricks only and NOT local files)
    use_test_data = False
    if use_test_data:
      my_dataset_dir = "/FileStore/tables/ex4_small_dataset/"
      result_dir = result_dir_base + "results/ex4_small_dataset/"
    else:
      result_dir = result_dir_base + "results/ex4/"      
  
    ##################################################
  
    # 1. Operation C1: 'textFile' to load the dataset into an RDD
    inputRDD = sc.textFile(my_dataset_dir)

    
    # 2. Transformations
    
    # -------------> process each line (function included in original file)
    processedRDD = inputRDD.map(process_line)
    # example -> one element will now look like: ('2013-01-10 08:00:59', 33488, 1935, 1)
    
    # -------------> filter for data that is required (remove "At Stop" = 0 AND anything that is outside seconds_horizon)
    timeFilteredRDD = processedRDD.filter(lambda element: filter_byTime_atStop(element, seconds_horizon, current_time)).sortBy(lambda item: item[0])
    # example -> one element will now look like: ('2013-01-10 09:10:59', 33600, 1935, 1)
    
    # persist (will be required once more later)
    timeFilteredRDD.persist()
    
    # -------------> filter for bus stop data --> current_stop = 1935 --> then getting the correct vehicle number
    busStopFilteredRDD = timeFilteredRDD.filter(lambda element: filter_byBusStop(element, current_stop)).sortBy(lambda item: item[0])
    # example -> one element will now look like: ('2013-01-10 09:05:59', 33500, 1935, 1)
    
    # collect the values here to obtain the first vehicle
    resVal_correctBusStop = busStopFilteredRDD.collect() # spark streaming issue down the line?
    # get vehicle number from first element
    vehicle_number = resVal_correctBusStop[0][1]
    
    # -------------> filter for vehicle number from the original "timeFilteredRDD"
    vehicleFilteredRDD = timeFilteredRDD.filter(lambda element: filter_byVehicleNumber(element, vehicle_number)).sortBy(lambda item: item[0])
    # example -> one element will now look like: ('2013-01-10 09:05:59', 33500, 1935, 1)    
    
    # -------------> map the solution output
    mappedSolutionRDD = vehicleFilteredRDD.map(map_solution_output) #.sortBy(lambda item: item[1])  # not needed here (done below)
    
    # -------------> solution RDD (this is technically only an extra step in oder to format this into the required solution output format)    
    solutionRDD = mappedSolutionRDD.groupByKey().mapValues(lambda values : list(sorted(values)) ) # mapValues is important here (display as list & sort in order)
    
        
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
    current_time = "2013-01-10 08:59:59"
    current_stop = 1935
    seconds_horizon = 1800

    # 2. Local or Databricks
    local_False_databricks_True = True

    # 3. We set the path to my_dataset and my_result
    my_local_path = ""
    my_databricks_path = "/"
    my_dataset_dir = "FileStore/tables/my_dataset_complete/"

    if local_False_databricks_True == False:
        my_dataset_dir = my_local_path + my_dataset_dir
    else:
        my_dataset_dir = my_databricks_path + my_dataset_dir

    # 3. We configure the Spark Context
    sc = pyspark.SparkContext.getOrCreate()
    sc.setLogLevel('WARN')
    print("\n\n\n")

    # 5. We call to our main function
    my_main(sc, my_dataset_dir, current_time, current_stop, seconds_horizon)
