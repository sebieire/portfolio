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
# FUNCTION my_filter_data_function
# ------------------------------------------
def my_filter_data_function(my_str_tuple):  
  
  # CUSTOM AUX FUNCTION
  # function must return True in order to be applied to RRD.filter  
    
  # placeholder return var
  res = False
  
  if my_str_tuple[7] == vehicle_id:
    res = True
    
  return res

# ------------------------------------------
# FUNCTION my_map_function
# ------------------------------------------
def my_map_function(my_str_tuple):
  
  # CUSTOM AUX FUNCTION
  # one input element equals for example: ('2013-01-04 09:00:36', 25, '015B1002', 0, -6.258078, 53.339279, 544, 33145, 279, 1)
  # goal: return tuple with first element = day, second element = bus line
  
  # split first item into components (need to isolate the days)
  date_time = my_str_tuple[0].split(" ")
  
  # split date into elements
  the_day = date_time[0].split("-") # 0 = year, 1 = month, 2 = day (all strings)
  
  return(the_day[2], my_str_tuple[1]) # day & busline   


# ------------------------------------------
# FUNCTION my_filter_results function
# ------------------------------------------
def my_filter_results(element, length):  
  
  # CUSTOM AUX FUNCTION
  # function must return True in order to be applied to RRD.filter
  
  # filters results based on value list length for each key
  # assumes that very first element is longest (assumes sorted by length already in previous steps)
      
  # placeholder
  res = False
  
  # determine if element value list is required length
  if len(element[1]) == length:
    res = True
    
  return res


# ------------------------------------------
# FUNCTIONS combiner, merge_values, merge_combiners ---> combineByKey functions
# ------------------------------------------
def combiner(value):
  
  res = [value]
  
  return res

def merge_values(accum, new_value):
  
  if new_value not in accum:
    accum.append(new_value)
    
  return sorted(accum)
  
def merge_combiners(accum1, accum2):
  
  res = []  
  
  for value in accum1:
    res.append(value)   
    
  for value in accum2:
    if value not in res:
      res.append(value)
  
  return sorted(res)  
  

# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(sc, my_dataset_dir, vehicle_id):
  
    """
      TASK: Compute the day(s) of the month in which this "vehicle_id" is serving the highest amount of
      different bus lines, and the IDs of such bus lines.
    """
  
    ##################################################
    # Additional Settings:
    
    # results dir
    save_results_to_file = False
    result_dir_base = "FileStore/tables/6_Assignments/"
    
    # Utilize Test Data? (assuming here databricks only and NOT local files)
    use_test_data = False
    if use_test_data:
      my_dataset_dir = "/FileStore/tables/6_Assignments/ex2_small_dataset/"
      result_dir = result_dir_base + "results/ex2_small_dataset/"
    else:
      result_dir = result_dir_base + "results/ex2/"      
  
    ##################################################
    
  
    # 1. Operation C1: 'textFile' to load the dataset into an RDD
    inputRDD = sc.textFile(my_dataset_dir)

    # 2. Transformations
    
    # -------------> process each line
    processedRDD = inputRDD.map(process_line)
    # example -> one element will now look like: ('2013-01-04 09:00:36', 25, '015B1002', 0, -6.258078, 53.339279, 544, 33145, 279, 1)
    
    # -------------> filter for data that is required (= get entries with the correct vehicle_id only)
    filteredRDD = processedRDD.filter(my_filter_data_function)
    # example -> one element will now (still) look like: ('2013-01-04 09:00:36', 25, '015B1002', 0, -6.258078, 53.339279, 544, 33145, 279, 1)
    
    # -------------> map and return only values per line that are required
    remappedRDD = filteredRDD.map(my_map_function) # return just tuples of (day_string, bus_line)
    # example -> one element will now look like: ('09', 66)
    # for large set it will contain all the duplicates! e.g.: ('02', 171), ('02', 171), ('02', 171)......)
    
    # -------------> combine by key
    combinedRDD = remappedRDD.combineByKey(combiner, merge_values, merge_combiners)
    # example -> one element will now look like: ('28', [171, 140, 13])
    
    # -------------> combined Values - sort keys by length of their value lists!
    combinedSortedValuesRDD = combinedRDD.sortBy(lambda item: len(item[1]) * (-1))
    # example -> one element will now look like: ('28', [13, 140, 171])            
    
    # -------------> get first item to get its value list LENGTH (that's the maximum length of any value list)
    first_item = combinedSortedValuesRDD.take(1) # first_item: [('02', [38, 41, 171])]
    value_list_max_length = len(first_item[0][1]) # len = 3 in this example
    
    # -------------> filter: only return items with same list length (lambda -> function) + sort by Key
    solutionRDD = combinedSortedValuesRDD.filter(lambda element: my_filter_results(element, value_list_max_length)).sortByKey()  
    
    
    # 3. Output / Collect
    
    # Operation A1: 'collect' to get all results        
    resVAL = solutionRDD.collect()
    for item in resVAL:
        print(item)
        
    # 4. Save to File (Optional Flag)    
    
    if save_results_to_file:
      dbutils.fs.rm(result_dir, True)
      solutionRDD.coalesce(1).saveAsTextFile(result_dir)  
    

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
    vehicle_id = 33145

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
    my_main(sc, my_dataset_dir, vehicle_id)
