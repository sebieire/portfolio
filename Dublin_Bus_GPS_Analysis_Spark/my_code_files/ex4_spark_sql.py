#!/usr/bin/env python3
"""
Spark implementation for analyzing Dublin Bus GPS data.
Exercise 4: Find next bus and travel path simulation.
"""

import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T
import time
import datetime

# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(spark, my_dataset_dir, current_time, current_stop, seconds_horizon):
  
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
      result_dir = result_dir_base + "results/ex4_sql_small_dataset/"
    else:
      result_dir = result_dir_base + "results/ex4_sql/"  

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
    #+-------------------+---------+----------------+----------+---------+---------+-----+---------+------------+------+
    #|               date|busLineID|busLinePatternID|congestion|longitude| latitude|delay|vehicleID|closerStopID|atStop|
    #+-------------------+---------+----------------+----------+---------+---------+-----+---------+------------+------+
    #|2013-01-10 08:00:59|       40|        015B1002|         0|-6.258078| 53.33928|  544|    33488|        1935|     1|
    #|2013-01-10 10:00:59|       40|        015B1002|         0|-6.258078| 53.33928|  544|    33488|        1935|     1|
    #....

    
    # Modify / Transform DFs
    
    # ----------------------------------------->    
    # remove unneccessary columns and filter out 'atStop' is 0
    lightInputDF = inputDF.drop('busLineID').drop('busLinePatternID').drop('congestion').drop('longitude').drop('latitude').drop('delay').filter(F.col('atStop') == 1)
    
    # at this point DF looks like:
    #+-------------------+---------+------------+------+
    #|               date|vehicleID|closerStopID|atStop|
    #+-------------------+---------+------------+------+
    #|2013-01-10 08:00:59|    33488|        1935|     1|
    #|2013-01-10 10:00:59|    33488|        1935|     1|
    #|2013-01-10 09:05:59|    33488|         279|     1|
    #|2013-01-10 09:05:59|    33500|        1935|     1|
    #|2013-01-10 09:10:59|    33600|        1935|     1|
    #|2013-01-10 09:15:59|    33500|         244|     1|
    #|2013-01-10 09:25:59|    33500|         264|     1|
    #|2013-01-10 09:35:59|    33500|         284|     1|
    #+-------------------+---------+------------+------+
    
    
    # Part A ----------------------------------------->
    
    # set start & end time (2013-01-10 08:59:59) & (2013-01-10 09:29:59)
    start_time = datetime.datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
    end_time = start_time + datetime.timedelta(seconds = seconds_horizon)
        
    # convert date column to timestamp / drop 'atStop' cols / insert 'end_time' as constant & calculate 'end_time' - 'timestamp' in seconds
    timeStampDifferenceDF = lightInputDF.withColumn('timestamp', F.to_timestamp( F.unix_timestamp(F.col('date'), 'yyyy-MM-dd HH:mm:ss') ) ) \
    .withColumn('end_time', F.lit(end_time) ) \
    .withColumn('diff_seconds', F.col('end_time').cast(T.LongType()) - F.col('timestamp').cast(T.LongType()) ) \
    .drop('atStop')
   
    #timeStampDifferenceDF.show()
    
    # at this point DF looks like (+ date column!!!!):
    #+---------+------------+-------------------+-------------------+------------+
    #|vehicleID|closerStopID|          timestamp|           end_time|diff_seconds|
    #+---------+------------+-------------------+-------------------+------------+
    #|    33488|        1935|2013-01-10 08:00:59|2013-01-10 09:29:59|        5340|
    #|    33488|        1935|2013-01-10 10:00:59|2013-01-10 09:29:59|       -1860|
    #|    33488|         279|2013-01-10 09:05:59|2013-01-10 09:29:59|        1440|
    #|    33500|        1935|2013-01-10 09:05:59|2013-01-10 09:29:59|        1440|
    #|    33600|        1935|2013-01-10 09:10:59|2013-01-10 09:29:59|        1140|
    #|    33500|         244|2013-01-10 09:15:59|2013-01-10 09:29:59|         840|
    #|    33500|         264|2013-01-10 09:25:59|2013-01-10 09:29:59|         240|
    #|    33500|         284|2013-01-10 09:35:59|2013-01-10 09:29:59|        -360|
    #+---------+------------+-------------------+-------------------+------------+
    
    # filter by time (0 - seconds_horizon) / order by 'diff_seconds'
    filter_orderTimeStampsDF = timeStampDifferenceDF.filter( (timeStampDifferenceDF["diff_seconds"] >= 0) & (timeStampDifferenceDF["diff_seconds"] <= seconds_horizon) ) \
    .orderBy(F.col('diff_seconds').desc())
    
    # persist
    filter_orderTimeStampsDF.persist()
    
    # at this point DF looks like (+ date column!!!!):
    #+---------+------------+-------------------+-------------------+------------+
    #|vehicleID|closerStopID|          timestamp|           end_time|diff_seconds|
    #+---------+------------+-------------------+-------------------+------------+
    #|    33488|         279|2013-01-10 09:05:59|2013-01-10 09:29:59|        1440|
    #|    33500|        1935|2013-01-10 09:05:59|2013-01-10 09:29:59|        1440|
    #|    33600|        1935|2013-01-10 09:10:59|2013-01-10 09:29:59|        1140|
    #|    33500|         244|2013-01-10 09:15:59|2013-01-10 09:29:59|         840|
    #|    33500|         264|2013-01-10 09:25:59|2013-01-10 09:29:59|         240|
    #+---------+------------+-------------------+-------------------+------------+    
    
    # filter by current stop (current_stop)
    currentStopFilter = filter_orderTimeStampsDF.filter( F.col("closerStopID") == current_stop)
    
    # at this point DF looks like (+ date column!!!!):
    #+---------+------------+-------------------+-------------------+------------+
    #|vehicleID|closerStopID|          timestamp|           end_time|diff_seconds|
    #+---------+------------+-------------------+-------------------+------------+
    #|    33500|        1935|2013-01-10 09:05:59|2013-01-10 09:29:59|        1440|
    #|    33600|        1935|2013-01-10 09:10:59|2013-01-10 09:29:59|        1140|
    #+---------+------------+-------------------+-------------------+------------+
    
    # get first item (required vehicle id)
    vehicle_id = currentStopFilter.select(F.col('vehicleID')).collect()[0][0]
    
    # filter for the vehicle / drop not required columns
    filter_vehicleID  = filter_orderTimeStampsDF.filter( F.col("vehicleID") == vehicle_id).drop('timestamp').drop('end_time').drop('diff_seconds')
    filter_vehicleID.persist()
    
    # at this point DF looks like:
    #+-------------------+---------+------------+
    #|               date|vehicleID|closerStopID|
    #+-------------------+---------+------------+
    #|2013-01-10 09:05:59|    33500|        1935|
    #|2013-01-10 09:15:59|    33500|         244|
    #|2013-01-10 09:25:59|    33500|         264|
    #+-------------------+---------+------------+

    
    
    ###################################################################################
    ## DANGER ZONE! -- THIS CODE IS CRAP
    
    aux1DF = filter_vehicleID.select( F.col('date').alias('time') , F.col('closerStopID').alias('stop'))
    aux1DF.persist()
    
    aux1DF_list_time = aux1DF.agg(F.collect_list('time').alias('time')).collect()
    aux2DF_list_stop = aux1DF.agg(F.collect_list('stop').alias('stop')).collect()
    
    #aux2DF = filter_vehicleID.select( F.col('vehicleID')).withColumn('stations', DOESNT_WORK_!!!! )
    
    # was hoping to insert the below Dataframe into a new 'aux2DF' dataframe in order to get the proper solution.... 
    # but doesn't work
    # don't know how to merge them properly.
    
    """
    +-------------------+----+
    |               time|stop|
    +-------------------+----+
    |2013-01-10 09:05:59|1935|
    |2013-01-10 09:15:59| 244|
    |2013-01-10 09:25:59| 264|
    +-------------------+----+"""
    
    # technically I want to do this:
    # random values but correct structure ---> how to do this?
    
    """+----------+-------------------------------------------+
    |vehicle_id|stations                                   |
    +----------+-------------------------------------------+
    |33500     |[[xx, xx], [xx, xx], [xx, xx]]
    +----------+-------------------------------------------+"""
    
    
    ###################################################################################
    
    # the wrong format but numbers are correct - almost there
    solutionDF = filter_vehicleID.select( F.col('vehicleID'), F.col('date').alias('time'), F.col('closerStopID').alias('stop') )
        
    
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

    # 3. We configure the Spark Session
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    print("\n\n\n")

    # 5. We call to our main function
    my_main(spark, my_dataset_dir, current_time, current_stop, seconds_horizon)
