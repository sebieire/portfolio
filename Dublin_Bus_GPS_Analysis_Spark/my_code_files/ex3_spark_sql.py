#!/usr/bin/env python3
"""
Spark implementation for analyzing Dublin Bus GPS data.
Exercise 3: Identify congestion patterns by day and hour.
"""

import pyspark
import pyspark.sql.functions as F

# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(spark, my_dataset_dir, threshold_percentage):
  
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
      result_dir = result_dir_base + "results/ex3_sql_small_dataset/"
    else:
      result_dir = result_dir_base + "results/ex3_sql/"  

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
    #+------------------+---------+----------------+----------+---------+---------+-----+---------+------------+------+
    #               date|busLineID|busLinePatternID|congestion|longitude| latitude|delay|vehicleID|closerStopID|atStop|
    #+------------------+---------+----------------+----------+---------+---------+-----+---------+------------+------+
    #2013-01-16 08:00:01|       40|            null|         0|-6.395583|53.352367|    0|    33341|        4795|     0|
    #2013-01-16 08:00:01|       27|        00270001|         0|-6.401166|53.279118| -542|    33232|        2355|     1|
    #....
    
    
    # Modify / Transform DFs
    
    # ----------------------------------------->
    # remove unneccessary columns
    lightInputDF = inputDF.drop('busLineID').drop('busLinePatternID').drop('longitude').drop('latitude').drop('delay').drop('vehicleID').drop('closerStopID').drop('atStop')
    
    # at this point DF looks like:
    #+-------------------+----------+
    #|               date|congestion|
    #+-------------------+----------+
    #|2013-01-01 09:15:36|         0|
    #|2013-01-01 09:30:36|         0|
    #|2013-01-01 09:45:36|         1|
    #....
    
    
    # Part A ----------------------------------------->
    
    # Split 'date' into 2 parts and re-inject as 'date_aux' & 'time_aux' / drop 'date' column afterwards
    split_col_date = pyspark.sql.functions.split(lightInputDF['date'], ' ')
    
    aux_date_DF = lightInputDF.withColumn('date_aux', split_col_date.getItem(0))
    aux_date_DF2 = aux_date_DF.withColumn('time_aux', split_col_date.getItem(1)).drop('date')   
    
    # at this point DF looks like:
    #+----------+----------+--------+
    #|congestion|  date_aux|time_aux|
    #+----------+----------+--------+
    #|         0|2013-01-01|09:15:36|
    #|         0|2013-01-01|09:30:36|
    #|         1|2013-01-01|09:45:36|
    #....    
    
    # Split 'date_aux' & time_aux into their parts and re-inject / drop 'date_aux' & 'time_aux' columns afterwards
    split_col_day = pyspark.sql.functions.split(aux_date_DF2['date_aux'], '-') # column object
    split_col_time = pyspark.sql.functions.split(aux_date_DF2['time_aux'], ':') # column object
    
    aux_date_DF3 = aux_date_DF2.withColumn('day', split_col_day.getItem(2)).drop('date_aux')
    aux_date_DF4 = aux_date_DF3.withColumn('hour', split_col_time.getItem(0)).drop('time_aux')    
    
    
    # at this point DF looks like:
    #+----------+---+----+
    #|congestion|day|hour|
    #+----------+---+----+
    #|         0| 01|  09|
    #|         0| 01|  09|
    #|         1| 01|  09|
    #....
    
    # combined day-hour DF
    dayHourConcatDF = aux_date_DF4.select( F.concat(F.col('day'), F.lit("-"), F.col('hour') ).alias("day_time"), F.col('congestion')  )
    
    # at this point DF looks like:
    #+--------+----------+
    #|day_time|congestion|
    #+--------+----------+
    #|   01-09|         0|
    #|   01-09|         0|
    #|   01-09|         1|
    
    
    
    # Part B ----------------------------------------->
    
    # group data accordingly as required
    groupedDF = dayHourConcatDF.groupBy(["day_time"]).agg( {"day_time": "count", "congestion" : "sum"} )        
    
    # at this point DF looks like:
    #+--------+---------------+---------------+
    #|day_time|count(day_time)|sum(congestion)|
    #+--------+---------------+---------------+
    #|   01-09|              3|              1|
    #|   03-09|              3|              3|
    #|   02-09|              3|              2|
    #+--------+---------------+---------------+
    
    # calculate the average congestion in percent and drop 2 columns
    calculationDF = groupedDF.withColumn("percentage",  F.round( F.col( 'sum(congestion)' ) / F.col( 'count(day_time)' ) * 100 , 2 )  ).drop('count(day_time)').drop('sum(congestion)')
    
    # at this point DF looks like:
    #+--------+----------+
    #|day_time|percentage|
    #+--------+----------+
    #|   01-09|      33.0|
    #|   03-09|     100.0|
    #|   02-09|      67.0|
    #+--------+----------+
    
    # filter out anything that is below the given 'threshold_percentage' (NOTE: threshold is given as 10.0 - but for test set supposed to be 35.0)
    filterThresholdDF = calculationDF.filter(calculationDF["percentage"] > threshold_percentage)
    filterThresholdDF.persist()
    
    
    
    # Part C ----------------------------------------->
    
    # Split Day & Time Again
    splitDayHourCol = F.split( filterThresholdDF['day_time'], '-' )
    
    aux_splitDayHourDF = filterThresholdDF.withColumn("day", splitDayHourCol.getItem(0) )    
    splitDayHourDF = aux_splitDayHourDF.withColumn("hour", splitDayHourCol.getItem(1) ).drop('day_time')  
    
    # at this point DF looks like:
    #+----------+---+----+
    #|percentage|day|hour|
    #+----------+---+----+
    #|      33.0| 01|  09|
    #|     100.0| 03|  09|
    #|      67.0| 02|  09|
    #+----------+---+----+
    
    # order columns properly
    solutionDF = splitDayHourDF.select('day', 'hour', 'percentage').orderBy(F.col('percentage').desc())
    # solutionDF.show() # if 'threshold_percentage' is set to 35 !
    
    # at this point DF looks like:
    #+---+----+----------+
    #|day|hour|percentage|
    #+---+----+----------+
    #| 03|  09|     100.0|
    #| 02|  09|      67.0|
    #+---+----+----------+
    
    
    
    # 3. Output / Collect   
    
    """
      Row(day='03', hour='09', percentage=100.0)
      Row(day='02', hour='09', percentage=67.0)
    """
    
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
    threshold_percentage = 10.0 # 10.0 for real data & 35.0 for test set (?)

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
    my_main(spark, my_dataset_dir, threshold_percentage)
