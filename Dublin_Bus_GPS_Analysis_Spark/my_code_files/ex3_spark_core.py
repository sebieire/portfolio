#!/usr/bin/env python3
"""
Spark implementation for analyzing Dublin Bus GPS data.
Exercise 3: Identify congestion patterns by day and hour.
"""

import pyspark

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
# FUNCTION retrieve_required_info (map) function
# ------------------------------------------
def retrieve_required_info(str_tuple):
  
  # CUSTOM AUX FUNCTION
  # one input element equals for example: ('2013-01-01 09:15:36', 41, '015B1002', 0, -6.258078, 53.339279, 200, 33145, 122, 1)
  # goal: return tuple with first element = day, second element = hour. third element = congestion
  
  # split first item into components (need to isolate the days & hours)
  date_time = str_tuple[0].split(" ")
  
  # split date into elements
  the_day = date_time[0].split("-") # 0 = year, 1 = month, 2 = day (all strings)
  the_hour = date_time[1].split(":")
  
  #return(the_day[2], the_hour[0], str_tuple[3]) # day, hour & congestion
  combined_day_hour = the_day[2] + "-" + the_hour[0]
  return(combined_day_hour, str_tuple[3]) # (day + hour combined, congestion)

# ------------------------------------------
# FUNCTIONS combiner, merge_values, merge_combiners ---> combineByKey functions
# ------------------------------------------
# input values to combineByKey: ('01-09', 0) where first part is key (makes it easier down the line)
def combiner(value):
  
  res = {0:0, 1:0}
  if value == 0:
    res[0] += 1
  elif value == 1:
    res[1] += 1
    
  return res

def merge_values(accum, new_value):
  
  if new_value == 0:
    accum[0] += 1
  elif new_value == 1:
    accum[1] += 1
    
  return accum
  
def merge_combiners(accum1, accum2):  
  
  for key in accum2:
    accum1[key] = accum1[key] + accum2[key]
    
  return accum1

# ------------------------------------------
# FUNCTION calculate_data (map) function
# ------------------------------------------
def calculate_data(item):
  
  # CUSTOM AUX FUNCTION
  # reshapes data back into what is required and calculates results for each
  # expected input item: ('14-11', {0: 88438, 1: 1091})
  
  # caculate the percentage of congestion
  the_sum = item[1][0] + item[1][1]
  percent_congested = round( (item[1][1] / the_sum) * 100, 2)
  
  # reshape into required data structure
  day_time = item[0].split("-")
  the_day = day_time[0]
  the_time = day_time[1]
  
  return (the_day, the_time, percent_congested)


# ------------------------------------------
# FUNCTION filter_by_threshold
# ------------------------------------------
def filter_by_threshold(item, threshold_percent):
  
  # CUSTOM AUX FUNCTION
  # expected input item: ('03', '09', 100.0)
  
  res = False
  
  # determin it item passes the threshold value
  if item[2] > threshold_percent:
    res = True  
  
  return res
  

  
# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(sc, my_dataset_dir, threshold_percentage):
  
    """
      TASK: Compute the concrete days and hours having a percentage of measurements reporting congestions
      above "threshold_percentage".
    """
  
    ##################################################
    # Additional Settings:
    
    # results dir
    save_results_to_file = False
    result_dir_base = "FileStore/tables/"
    
    # Utilize Test Data? (assuming here databricks only and NOT local files)
    use_test_data = False
    if use_test_data:
      my_dataset_dir = "/FileStore/tables/ex3_small_dataset/"
      result_dir = result_dir_base + "results/ex3_small_dataset/"
    else:
      result_dir = result_dir_base + "results/ex3/"      
  
    ##################################################
  
    # 1. Operation C1: 'textFile' to load the dataset into an RDD
    inputRDD = sc.textFile(my_dataset_dir)

    # 2. Transformations
    
    # -------------> process each line (function included in original file)
    processedRDD = inputRDD.map(process_line)
    # example -> one element will now look like: ('2013-01-01 09:15:36', 41, '015B1002', 0, -6.258078, 53.339279, 200, 33145, 122, 1)    
    
    # -------------> retrieve (map) required info only and make into a 2 item tuple
    remappedRDD = processedRDD.map(retrieve_required_info)
    # example -> one element will now look like: ('01-09', 1), ('02-09', 0),....
    
    # -------------> combine by key
    combinedRDD = remappedRDD.combineByKey(combiner, merge_values, merge_combiners)
    # example -> one element will now look like: ('14-11', {0: 88438, 1: 1091}), ('13-20', {0: 46739, 1: 526}), ('01-23', {0: 35551, 1: 467}), ....
    
    # -------------> calculate solutions (map) and reshape into required structure
    calculatedRDD = combinedRDD.map(calculate_data)
    # example -> output is: ('03', '09', 100.0), ('01', '09', 33.33), ('02', '09', 66.67)
    
    # -------------> filter for data that is required according to "threshold_percentage"
    solutionRDD = calculatedRDD.filter(lambda item : filter_by_threshold(item, threshold_percentage)).sortBy(lambda item: item[2] * (-1))
    # example -> one element will now look like: ('28', '01', 45.0), ....
    

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
    threshold_percentage = 10.0 # 10.0 for full dataset & 35.0 for test set

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
    my_main(sc, my_dataset_dir, threshold_percentage)
