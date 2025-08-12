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
import pyspark.sql.functions

# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(spark, my_dataset_dir, bus_stop, bus_line, hours_list):  
  
    """
      TASK:  Compute the average delay of "bus_line" vehicles when stopping at "bus_stop" for each hour of 
      "hours_list" during weekdays (you must discard any measurement taking place on a Saturday or
      Sunday).
    """

    ##################################################
    # Additional Settings:

    # results dir
    save_results_to_file = False
    result_dir_base = "FileStore/tables/6_Assignments/"

    # Utilize Test Data? (assuming here databricks only and NOT local files)
    use_test_data = False
    if use_test_data:
      my_dataset_dir = "/FileStore/tables/6_Assignments/ex1_small_dataset/"
      result_dir = result_dir_base + "results/ex1_sql_small_dataset/"
    else:
      result_dir = result_dir_base + "results/ex1_sql/"  

    ##################################################

    # 1. We define the Schema of our DF.
    my_schema = pyspark.sql.types.StructType(
        [pyspark.sql.types.StructField("date", pyspark.sql.types.StringType(), False),
         pyspark.sql.types.StructField("busLineID", pyspark.sql.types.IntegerType(), False),
         pyspark.sql.types.StructField("busLinePatternID", pyspark.sql.types.StringType(), False),
         pyspark.sql.types.StructField("congestion", pyspark.sql.types.IntegerType(), False),
         pyspark.sql.types.StructField("longitude", pyspark.sql.types.FloatType(), False),
         pyspark.sql.types.StructField("latitude", pyspark.sql.types.FloatType(), False),
         pyspark.sql.types.StructField("delay", pyspark.sql.types.IntegerType(), False),
         pyspark.sql.types.StructField("vehicleID", pyspark.sql.types.IntegerType(), False),
         pyspark.sql.types.StructField("closerStopID", pyspark.sql.types.IntegerType(), False),
         pyspark.sql.types.StructField("atStop", pyspark.sql.types.IntegerType(), False)
         ])

    # 2. Operation C2: 'read' to create the DataFrame from the dataset and the schema
    inputDF = spark.read.format("csv") \
        .option("delimiter", ",") \
        .option("quote", "") \
        .option("header", "false") \
        .schema(my_schema) \
        .load(my_dataset_dir)

    # at this point DF looks like:
    #+-------------------+---------+----------------+----------+---------+--------+-----+---------+------------+------+
    #|               date|busLineID|busLinePatternID|congestion|longitude|latitude|delay|vehicleID|closerStopID|atStop|
    #+-------------------+---------+----------------+----------+---------+--------+-----+---------+------------+------+
    #|2013-01-09 08:00:36|       40|        015B1002|         0|-6.258078|53.33928|  300|    33488|         279|     1|
    #...
    
    
    # Modify / Transform DFs 
    
    # ----------------------------------------->
    # remove unneccessary columns
    lightInputDF = inputDF.drop('busLinePatternID').drop('congestion').drop('longitude').drop('latitude').drop('vehicleID')
    
    # at this point DF looks like:
    #+-------------------+---------+-----+------------+------+
    #|               date|busLineID|delay|closerStopID|atStop|
    #+-------------------+---------+-----+------------+------+
    #|2013-01-09 08:00:36|       40|  300|         279|     1|    
    
    
    # Part A ----------------------------------------->
    
    # Split 'date' into 2 parts and re-inject as 'date_aux' & 'time_aux' / drop 'date' column afterwards    
    split_col_date = pyspark.sql.functions.split(lightInputDF['date'], ' ')
    
    aux_date_DF = lightInputDF.withColumn('date_aux', split_col_date.getItem(0))
    aux_date_DF2 = aux_date_DF.withColumn('time_aux', split_col_date.getItem(1)).drop('date')   
    
    # insert int value for 'day of the week' and then drop the 'date_aux' col
    # !!!! NOTE: it appears that Sunday is 1 (1 = Sun, 2 = Mon, 3 = Tue, ... , 7 = Sat)
    aux_date_DF3 = aux_date_DF2.withColumn('day_int', pyspark.sql.functions.dayofweek('date_aux') ).drop('date_aux')    
    
    # at this point DF looks like:
    #+---------+-----+------------+------+--------+-------+
    #|busLineID|delay|closerStopID|atStop|time_aux|day_int|
    #+---------+-----+------------+------+--------+-------+
    #|       40|  300|         279|     1|08:00:36|      4|
    
    # Split 'time_aux' into 3 parts and re-inject 'hour' / then drop 'time_aux'    
    split_col_time = pyspark.sql.functions.split(aux_date_DF3['time_aux'], ':')
    
    reducedDateTimeDF = aux_date_DF3.withColumn('hour', split_col_time.getItem(0)).drop('time_aux')
    
    reducedDateTimeDF.persist()
    
    # at this point DF looks like:
    #+---------+-----+------------+------+-------+----+
    #|busLineID|delay|closerStopID|atStop|day_int|hour|
    #+---------+-----+------------+------+-------+----+
    #|       40|  620|         279|     1|      4|  08|   
    
        
    # Part B ----------------------------------------->
    
    # filter out weekend days
    min_day_int = 2 # Monday
    max_day_int = 6 # Friday    
    filterNoWeekendDF = reducedDateTimeDF.filter(reducedDateTimeDF["day_int"] >= min_day_int).filter(reducedDateTimeDF["day_int"] <= max_day_int)
    
    # select only rows that match the given 'hours_list' (aka: 07, 08, 09)
    matchHoursListDF = filterNoWeekendDF.select( pyspark.sql.functions .col("*")).where(filterNoWeekendDF["hour"].isin(hours_list) )
    
    # filter out any atStop = 0 (meaning: 'not at stop')
    filterOnlyAtStopDF = matchHoursListDF.filter(reducedDateTimeDF["atStop"] == 1)
    
    # filter equal to 'bus_stop'
    filterBusStopDF = filterOnlyAtStopDF.filter(filterOnlyAtStopDF["closerStopID"] == bus_stop)
    
    # filter equal to 'bus_line'
    filterBusLineDF = filterBusStopDF.filter(filterBusStopDF["busLineID"] == bus_line)
    
    
    # Part C ----------------------------------------->
    
    # group by hour and avg    
    groupedDF = filterBusLineDF.groupBy(["hour"]).agg( {"delay" : "avg"} )
    
    # rename , round and order
    solutionDF = groupedDF.withColumnRenamed('avg(delay)', 'averageDelay') \
    .withColumn('averageDelay', pyspark.sql.functions.round(pyspark.sql.functions.col('averageDelay'),2)) \
    .orderBy(pyspark.sql.functions.col('averageDelay').asc())
    
     # at this point DF looks like:
    #+----+------------+
    #|hour|averageDelay|
    #+----+------------+
    #|  09|        50.0|
    #|  08|       66.67|
    #+----+------------+
    
    #solutionDF.show()    
    
    # 3. Output / Collect
    
    # Operation A1: 'collect' to get all results
    resVAL = solutionDF.collect()
    for item in resVAL:
        print(item)
    
    
    # 4. Save to File (Optional Flag)
    
    if save_results_to_file:
      dbutils.fs.rm(result_dir, True)
      outputRDD = solutionDF.rdd.map(list)
      outputRDD.coalesce(1).saveAsTextFile(result_dir)

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
    bus_stop = 279
    bus_line = 40
    hours_list = ["07", "08", "09"] # ["07", "08", "09"]

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

    # 4. We configure the Spark Session
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    print("\n\n\n")

    # 5. We call to our main function
    my_main(spark, my_dataset_dir, bus_stop, bus_line, hours_list)
