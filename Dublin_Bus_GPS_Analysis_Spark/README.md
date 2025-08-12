# Dublin Bus GPS Data Analysis with Apache Spark

A comprehensive big data analytics project analyzing Dublin Bus GPS data from January 2013 using Apache Spark Core and Spark SQL. This project demonstrates distributed data processing techniques to extract meaningful insights from large-scale transportation data.

## Project Overview

This project analyzes over 40 million GPS measurements from Dublin buses to understand patterns in:
- Bus delay patterns by time and location
- Traffic congestion analysis
- Vehicle routing and service patterns
- Geographic distribution of delays

## Features

### Exercise 1: Average Delay Analysis
- Calculate average bus delay by hour for specific stops
- Filter weekday data only
- Identify optimal travel times

### Exercise 2: Multi-Line Service Analysis
- Track vehicles serving multiple bus lines
- Identify days with maximum line coverage
- Analyze vehicle utilization patterns

### Exercise 3: Congestion Pattern Detection
- Identify high-congestion periods by day and hour
- Calculate congestion percentages above thresholds
- Map temporal traffic patterns

### Exercise 4: Next Bus Finder
- Simulate real-time next bus arrival
- Track bus journey paths within time windows
- Provide stop-by-stop journey tracking

### Exercise 5: Geographic Delay Analysis (Advanced)
- Correlate delays with GPS coordinates
- Group delays by geographic proximity
- Identify traffic hotspots across Dublin

## Technologies Used

- **Apache Spark Core**: RDD-based distributed processing
- **Apache Spark SQL**: DataFrame operations for structured data analysis
- **PySpark**: Python API for Spark
- **Python 3.x**: Core programming language

## Dataset

The project uses Dublin Bus GPS sample data containing:
- Date and time of measurements
- Bus line and vehicle identifiers
- GPS coordinates (latitude/longitude)
- Delay information (seconds ahead/behind schedule)
- Congestion indicators
- Stop proximity data

Dataset structure: 744 files (one per hour interval) containing measurements with 10 fields per record.

## Project Structure

```
Dublin_Bus_GPS_Analysis_Spark/
├── my_code_files/
│   ├── ex1_spark_core.py      # Exercise 1: Spark Core implementation
│   ├── ex1_spark_sql.py       # Exercise 1: Spark SQL implementation
│   ├── ex2_spark_core.py      # Exercise 2: Spark Core implementation
│   ├── ex2_spark_sql.py       # Exercise 2: Spark SQL implementation
│   ├── ex3_spark_core.py      # Exercise 3: Spark Core implementation
│   ├── ex3_spark_sql.py       # Exercise 3: Spark SQL implementation
│   ├── ex4_spark_core.py      # Exercise 4: Spark Core implementation
│   ├── ex4_spark_sql.py       # Exercise 4: Spark SQL implementation
│   └── ex5_spark_core_extra.py # Exercise 5: Advanced geographic analysis
├── gitignore/                  # Files excluded from repository
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore configuration
└── README.md                   # This file
```

## Usage

Each exercise can be run independently with configurable parameters:

```python
# Example: Running Exercise 1
python3 my_code_files/ex1_spark_core.py

# Key parameters (configurable in each file):
# - bus_stop: Target bus stop ID
# - bus_line: Target bus line number
# - hours_list: Hours to analyze
# - threshold_percentage: Congestion threshold
```

## Key Insights

The analysis reveals:
- Peak congestion occurs during traditional rush hours
- Certain geographic areas show consistently higher delays
- Bus vehicles frequently serve multiple lines throughout the day
- Delay patterns vary significantly by day of week

## Technical Highlights

- **Distributed Processing**: Efficient handling of 3GB+ dataset
- **Optimization Techniques**: combineByKey for aggregations, proper filtering strategies
- **Dual Implementation**: Both RDD and DataFrame approaches for comparison
- **Geographic Analysis**: Distance calculations using Haversine formula
- **Performance Tuning**: Appropriate partitioning and caching strategies

## Requirements

- Python 3.x (tested with Python 3.12)
- Apache Spark 3.x
- PySpark

Note: This project requires Apache Spark integration to run properly. The code was originally developed and tested on Databricks platform.

## Installation

```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt

# Note: Apache Spark must be installed separately
# Visit https://spark.apache.org for installation instructions
```

## Future Enhancements

- Real-time streaming analysis using Spark Streaming
- Machine learning models for delay prediction
- Interactive visualization dashboard
- Integration with current Dublin Bus APIs

## Author

[sebieire](https://github.com/sebieire/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original dataset provided by Dublin City Council (Insight Project)
- Data source: Irish Government Open Data Portal