# Terrorism ML Prediction

A machine learning project that analyzes the Global Terrorism Database to predict which terrorist group was responsible for attacks based on various features including location, attack type, and casualties.

## Overview

This project uses machine learning techniques to classify terrorist attacks by predicting the responsible group. The implementation demonstrates comprehensive data preprocessing, feature engineering, outlier detection, and multiple classification algorithms with hyperparameter optimization.

## Features

### Data Preprocessing Pipeline
- **Manual preprocessing**: Handling missing values with domain-specific logic
- **Feature selection**: Selecting most relevant columns from 100+ features
- **Outlier detection**: Both univariate and multivariate (DBSCAN) approaches
- **Data scaling**: MinMax and Standardization options
- **Imbalance handling**: SMOTE for addressing class imbalance

### Machine Learning Models
- Decision Tree Classifier
- Linear SVC
- K-Nearest Neighbors
- Random Forest Classifier
- Stochastic Gradient Descent
- Gradient Boosting Classifier
- Bagging Classifier

### Advanced Techniques
- **Hyperparameter optimization**: GridSearchCV for optimal model parameters
- **Cross-validation**: K-fold validation for robust evaluation
- **Feature importance analysis**: Understanding which features drive predictions
- **Comprehensive evaluation**: Multiple metrics including accuracy, precision, recall, F1-score

## Performance Highlights

The project achieves excellent performance on the terrorism classification task:

- **Top 3 terrorist groups**: Near-perfect accuracy (99-100%)
- **15 groups classification**: ~95% accuracy with balanced precision/recall
- **Scalable architecture**: Can handle any number of target groups

Key findings:
- Location features (Country, City, Region) are most predictive
- Attack type and casualties provide significant signal
- Models maintain high performance even with increased class complexity

## Dataset

This project uses the Global Terrorism Database (GTD) from Kaggle, containing information on over 180,000 terrorist attacks worldwide from 1970 to 2017. The dataset is maintained by the National Consortium for the Study of Terrorism and Responses to Terrorism (START).

**Note**: The dataset file `globalterrorismdb_0718dist.csv` needs to be downloaded separately from [Kaggle](https://www.kaggle.com/START-UMD/gtd) and placed in the project directory.

## Project Structure

```
.
├── terrorism_prediction.py    # Main implementation file
├── requirements.txt           # Python dependencies
├── README.md                 # This file
└── globalterrorismdb_0718dist.csv  # Dataset (needs to be downloaded)
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from terrorism_prediction import run

# Run the complete analysis pipeline
run()
```

### Configurable Parameters

```python
# Set number of terrorist groups to classify
numberOfMostActiveGroups = 3  # Can be changed to any number

# Choose experiment type
experiment_one()  # Basic pipeline without hyperparameter optimization
experiment_two(runHyperParamOpt=True)  # Full pipeline with optimization
```

### Custom Analysis

```python
from terrorism_prediction import preprocess_initial, inital_multi_model_accuracy_evaluation

# Preprocess data for top 5 groups
df_processed = preprocess_initial(numberActiveGroups=5, showInfo=True)

# Evaluate multiple models
models = [DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier()]
inital_multi_model_accuracy_evaluation(models, train_features, train_labels)
```

## Technical Implementation

### Preprocessing Pipeline
1. **Column selection**: Reduces 100+ columns to 20 most relevant features
2. **Missing value handling**: Context-aware imputation (0 for casualties, 'unknown' for locations)
3. **Normalization**: Consistent string formatting (lowercase) for categorical variables
4. **Encoding**: Ordinal encoding for categorical features
5. **Scaling**: Choice of MinMax or Standard scaling
6. **Outlier removal**: Combined univariate and multivariate detection

### Feature Engineering
- Extracts year, month, day from date information
- Combines latitude/longitude for geographical analysis
- Creates derived features from attack characteristics

### Model Evaluation
- K-fold cross-validation for robust performance estimates
- Comprehensive metrics: accuracy, precision, recall, F1-score
- Confusion matrices for detailed error analysis
- Feature importance visualization

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Imbalanced-learn

## Results

The project demonstrates that terrorist group attribution can be predicted with high accuracy when sufficient data is available. Key insights:

- Geographic location is the strongest predictor
- Attack methodology provides secondary signal
- Model performance scales well with increased complexity
- Ensemble methods (Random Forest, Gradient Boosting) perform best

## Author

[sebieire](https://github.com/sebieire/)

## License

MIT License

---

*Note: This project is for educational and research purposes only. The analysis of terrorism data is intended to demonstrate machine learning techniques and should not be used for any harmful purposes.*