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
def my_main(spark, my_dataset_dir, vehicle_id):
  
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
      result_dir = result_dir_base + "results/ex2_sql_small_dataset/"
    else:
      result_dir = result_dir_base + "results/ex2_sql/"  

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
    #|2013-01-04 09:00:36|       25|        015B1002|         0|-6.258078|53.33928|  544|    33145|         279|     1|

    
    # Modify / Transform DFs 
    
    # ----------------------------------------->
    # remove unneccessary columns
    lightInputDF = inputDF.drop('busLinePatternID').drop('congestion').drop('longitude').drop('latitude').drop('delay').drop('closerStopID').drop('atStop')
    
    # at this point DF looks like:
    #+-------------------+---------+---------+
    #|               date|busLineID|vehicleID|
    #+-------------------+---------+---------+
    #|2013-01-04 09:00:36|       25|    33145|
    
    
    # Part A ----------------------------------------->
    
    # Split 'date' into 2 parts and re-inject first as 'date_aux' 
    split_col_date = pyspark.sql.functions.split(lightInputDF['date'], ' ') # column object    
    aux_date_DF = lightInputDF.withColumn('date_aux', split_col_date.getItem(0))
    
    # Split 'date_aux' into 2 parts and re-inject third as 'day' / drop 'date' column afterwards
    split_col_day = pyspark.sql.functions.split(aux_date_DF['date_aux'], '-') # column object
    aux_date_DF2 = aux_date_DF.withColumn('day', split_col_day.getItem(2)).drop('date').drop('date_aux')
    
    # at this point DF looks like:
    #+---------+---------+---+
    #|busLineID|vehicleID|day|
    #+---------+---------+---+
    #|       25|    33145| 04|
    
    
    # Part B ----------------------------------------->
    
    # filer for only 'vehicle_id' / then drop 'vehicleID' col / then drop duplicates (as list will contain lots of them)
    vehicleFilteredDF = aux_date_DF2.select( pyspark.sql.functions .col("*")).where(aux_date_DF2["vehicleID"] == vehicle_id).drop('vehicleID').dropDuplicates()
    
    # at this point DF looks like:
    #+---------+---+
    #|busLineID|day|
    #+---------+---+
    #|      171| 24|
    #|       39| 18|
    #|       40| 10|
    # ...
    
    # group by day and make list out of bus lines / rename column
    aux_groupedDayBusDF = vehicleFilteredDF.groupBy("day").agg(pyspark.sql.functions.collect_list('busLineID')) #
    groupedDayBusDF = aux_groupedDayBusDF.withColumnRenamed("collect_list(busLineID)", "busLineIDs")    
    
    #groupedDayBusDF.show()
    
    # at this point DF looks like:
    #+---+----------------+
    #|day|      busLineIDs|
    #+---+----------------+
    #| 04|            [25]|
    #| 08|      [120, 122]|
    #| 09|    [25, 66, 67]|
    #...    
    
    
    # Part C ----------------------------------------->
    
    # add column that indicates array len from 'sortedBusLineIDs' column
    aux_dayBusLengthDF = groupedDayBusDF.withColumn('array_len', pyspark.sql.functions.size( pyspark.sql.functions.col('busLineIDs') )  )
    
    #aux_dayBusLengthDF.show()
    
    # at this point DF looks like:
    #+---+----------------+---------+
    #|day|      busLineIDs|array_len|
    #+---+----------------+---------+
    #| 04|            [25]|        1|
    #| 08|      [120, 122]|        2|
    #| 09|    [25, 66, 67]|        3|
    #...
    
    # select only rows that match the max array_len value
    selectByMaxDF = aux_dayBusLengthDF.select(aux_dayBusLengthDF['day'], aux_dayBusLengthDF['busLineIDs']) \
    .where(aux_dayBusLengthDF['array_len'] == aux_dayBusLengthDF.agg( {'array_len' : 'max'}).collect()[0][0])
    
    #selectByMaxDF.show()
    
    # at this point DF looks like:
    #+---+----------------+
    #|day|      busLineIDs|
    #+---+----------------+
    #| 09|    [25, 66, 67]|
    #| 17|    [39, 67, 66]|
    #+---+----------------+
    
    # sort buslineIDs array
    sortedBusLinesDF = selectByMaxDF.select(pyspark.sql.functions.col('day'),pyspark.sql.functions.sort_array(selectByMaxDF['busLineIDs']).alias('sortedBusLineIDs'))
    
    # sort by day
    solutionDF = sortedBusLinesDF.orderBy(pyspark.sql.functions.col('day').asc())
    
    # solutionDF.show()
    
    # at this point DF looks like:
    #+---+----------------+
    #|day|sortedBusLineIDs|
    #+---+----------------+
    #| 09|    [25, 66, 67]|
    #| 17|    [39, 66, 67]|
    #+---+----------------+
    
    
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

    # 4. We configure the Spark Session
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    print("\n\n\n")

    # 5. We call to our main function
    my_main(spark, my_dataset_dir, vehicle_id)
