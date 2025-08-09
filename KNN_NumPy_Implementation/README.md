# k-Nearest Neighbors Implementation

A comprehensive, pure NumPy implementation of k-Nearest Neighbors for both classification and regression tasks. This project demonstrates advanced ML concepts without relying on high-level libraries like scikit-learn.

**Original Implementation: 2019**

## Key Features

### 🎯 Core Highlights
- **No loops in distance calculations** - Fully vectorized NumPy implementation
- **Multiple distance metrics** - Euclidean and Manhattan
- **Distance-weighted voting** - Improved accuracy through weighted neighbors
- **Advanced feature selection** - Achieved 95% accuracy on regression tasks
- **Tie-breaking mechanism** - Handles equal vote scenarios in classification

## Performance Results

### Classification (10 features, 3 classes)
- **Basic k-NN**: 89.5% accuracy (k=1)
- **Optimized k-NN**: 93% accuracy (k=9, Euclidean, weighted)
- **Best configuration**: Distance-weighted Euclidean with k=8-9

### Regression (12 features)
- **Basic approach**: 85.07% R² score
- **With feature selection**: **95.03% R² score** (8 features dropped)
- **Optimal k value**: 10-15 for regression tasks

## Project Structure

```
├── knn_basic.py         # Basic k-NN implementation
├── knn_enhanced.py      # Enhanced with multiple metrics and weighting
├── knn_regression.py    # Regression implementation with feature selection
├── requirements.txt     # Dependencies
└── data/
    ├── classification/
    │   ├── trainingData.csv  # 4000 samples, 10 features, 3 classes
    │   └── testData.csv      # 1000 samples
    └── regression/
        ├── trainingData.csv  # 6400 samples, 12 features
        └── testData.csv      # 1600 samples
```

## Implementation Details

### 1. Basic k-NN (`knn_basic.py`)
- Pure NumPy Euclidean distance calculation
- No loops in `calculateDistances` function
- Automatic tie-breaking for equal votes

### 2. Enhanced k-NN (`knn_enhanced.py`)
Features multiple improvements:
- **Distance metrics**: Euclidean, Manhattan, Modified Euclidean (with feature dropping)
- **Weighted voting**: 1/d weighting with configurable power
- **Experiment mode**: Automated hyperparameter tuning
- **Comprehensive analysis**: Graphs and performance metrics

Configuration options:
```python
kNN_K_AMOUNT = 10                    # k value
DISTANCE_FUNCTION = 1                # 1=Euclidean, 2=Euclidean with drop, 3=Manhattan
weightedDistanceCalculationEnabled = True
VALUE_OF_N = 1                       # Power for distance weighting
```

### 3. k-NN Regression (`knn_regression.py`)
Advanced regression implementation:
- **R² metric**: Custom implementation without sklearn
- **Feature selection**: Drop list capability for noise reduction
- **Key finding**: Features 6, 7, 9, 11 were critical; dropping others improved accuracy by 10%

## Usage

### Basic Classification
```python
python knn_basic.py
# Runs with k=1 by default, modify kNN_K_AMOUNT for different k values
```

### Enhanced Classification with Experiments
```python
# For single run:
python knn_enhanced.py

# For full experiment (modify in code):
singleRunEnabled = False
fullExperiment = True
K_RANGE = 30  # Test k values from 1 to 30
```

### Regression
```python
python knn_regression.py
# Includes feature selection experiments
```

## Key Insights

### Distance Calculation (No Loops!)
The core innovation is the fully vectorized distance calculation:
```python
euclidDistancesArray = np.sqrt(np.sum((npArr2D_trainingData[:,0:10] - 
                                       npArr1D_queryInstance[0:10])**2,axis=1))
```

### Optimal k Values
- **Classification**: k=6-10 (peaks around 8-9)
- **Regression**: k=10-15 (requires more neighbors for stability)

### Feature Selection Impact
Through systematic testing, discovered that removing 8 out of 12 features increased regression accuracy from 85% to 95% - demonstrating the importance of feature selection in ML.

## Requirements

- Python 3.7+ (tested with Python 3.12)
- NumPy
- Matplotlib (for visualization)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Technical Notes

- All distance calculations use vectorized NumPy operations
- No high-level ML libraries (scikit-learn, etc.) are used
- Implements everything from scratch for educational purposes
- Includes comprehensive debugging and visualization options

## Experiments Included

The code includes extensive experiment capabilities:
- Hyperparameter grid search
- Distance metric comparison
- Feature importance analysis
- Performance visualization

## Author

[Sebastian Eire](https://github.com/sebieire/)

## License

MIT License - See LICENSE file for details