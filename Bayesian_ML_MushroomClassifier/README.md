# Bayesian Networks and ML Mushroom Classifier

A comprehensive implementation of knowledge representation concepts through Bayesian Networks and Naive Bayes classification, featuring practical applications in health risk analysis and mushroom edibility prediction.

## Overview

This project demonstrates advanced knowledge representation techniques through two main components:

1. **Bayesian Network for Health Risk Analysis**: Models the relationship between lifestyle factors (smoking, alcohol, exercise, weight) and residual life expectancy using probabilistic reasoning
2. **Naive Bayes Mushroom Classifier**: Implements a custom Naive Bayes algorithm to classify mushrooms as edible or poisonous based on physical characteristics

## Features

### Part 1: Bayesian Network for RLE (Residual Life Expectancy)
- **Probabilistic Modeling**: Constructs a Bayesian network with 7 key variables affecting life expectancy
- **Conditional Probability Tables**: Detailed CPTs based on real health study data
- **Inference Engine**: Uses enumeration and likelihood weighting for probability queries
- **Health Risk Assessment**: Calculates potential years of life lost based on lifestyle factors

### Part 2: Naive Bayes Mushroom Classification
- **Custom Implementation**: Built-from-scratch Naive Bayes classifier with fit/predict methods
- **Probability Calculations**: 
  - Prior probabilities for each class
  - Evidence probabilities for all features
  - Likelihood calculations using Bayes theorem
- **Data Processing**: Handles categorical features from UCI Mushroom Dataset
- **Performance Evaluation**: Achieves 99.7% accuracy on test set

## Performance Highlights

### Bayesian Network Results
- Male smoker RLE loss: ~8.65 years (text reference: 9.4 years)
- Female smoker RLE loss: ~6.01 years (text reference: 5.3 years)
- All risk factors combined: 15.91 years (male), 10.73 years (female)
- Healthy lifestyle: <1 year RLE loss

### Mushroom Classifier Performance
- **Accuracy**: 99.7% on test set
- **Confusion Matrix**: Only 2 misclassifications out of 813 test samples
- **Prior Probabilities**: Balanced dataset (47.8% poisonous, 52.2% edible)
- **Robust Feature Processing**: Handles 22 categorical features effectively

## Project Structure

```
.
├── bayes_network_rle.py           # Bayesian network implementation
├── naive_bayes_classifier.py      # Data processing and probability calculations
├── knowledge_representation_notebook.ipynb  # Main Jupyter notebook with analysis
├── lib/
│   ├── custom_naive_bayes_learner.py  # Custom Naive Bayes class
│   ├── settings.py                     # Global settings
│   └── __init__.py
├── lib_aima/                      # AIMA library components
│   ├── learning.py
│   ├── probability.py
│   └── utils.py
├── data/
│   ├── agaricus-lepiota.data     # Main mushroom dataset
│   └── agaricus-lepiota-expanded.data
└── images/                        # Visualization assets
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

### Running the Bayesian Network

```python
from bayes_network_rle import get_RLE_bayes_network
from lib_aima.probability import enumeration_ask, likelihood_weighting

# Get the network
network = get_RLE_bayes_network()

# Query for male smoker RLE loss
query = likelihood_weighting('RLE_Loss', 
    {'Is_Smoker': True, 'Is_Female': False}, 
    network)
years_lost = query[True] * 17  # Max years for male
print(f"RLE Loss: {years_lost:.2f} years")
```

### Using the Mushroom Classifier

```python
from naive_bayes_classifier import read_in_dataFrame, get_train_test_split_data
from lib.custom_naive_bayes_learner import NaiveBayesLearner

# Load and split data
data = read_in_dataFrame("./data/agaricus-lepiota.data", column_names=headers)
train_set, test_set = get_train_test_split_data(data, percentage_train_split=0.9)

# Train classifier
NBL = NaiveBayesLearner(train_set)
NBL.fit_data()

# Make predictions
predicted_class = NBL.predict_class_from_features(feature_dict)
```

### Interactive Notebook

The project includes a comprehensive Jupyter notebook that walks through both implementations with visualizations and detailed explanations:

```bash
jupyter notebook knowledge_representation_notebook.ipynb
```

## Technical Implementation

### Bayesian Network Design
- **Variables**: 7 nodes representing health factors and outcomes
- **Structure**: Directed acyclic graph with conditional dependencies
- **Inference**: Forward propagation using AIMA probability algorithms
- **Validation**: Results align with published health study data

### Naive Bayes Algorithm
- **Assumption**: Conditional independence of features given class
- **Laplace Smoothing**: Handles zero probabilities
- **Logarithmic Computation**: Prevents underflow for probability products
- **Feature Processing**: Handles categorical data without encoding

## Dataset

### Mushroom Dataset
- **Source**: UCI Machine Learning Repository
- **Size**: 8,124 samples with 22 categorical features
- **Classes**: Binary (edible/poisonous)
- **Features**: Cap shape, odor, gill size, stalk shape, habitat, etc.

### Health Data
- Based on German cohort study (Li et al., 2014)
- Lifestyle risk factors and residual life expectancy at age 40

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- scikit-learn
- jupyter

## Results Analysis

### Key Insights from Bayesian Network
1. Smoking has the highest impact on life expectancy reduction
2. Combined risk factors show multiplicative effects
3. Gender significantly influences RLE loss patterns
4. Model predictions align well with empirical health data

### Mushroom Classification Findings
1. Odor is the most predictive feature for toxicity
2. Spore print color and gill size are strong indicators
3. High accuracy achievable with simple probabilistic model
4. Robust performance across different train/test splits

## Author

[sebieire](https://github.com/sebieire/)

## License

MIT License

---

*Note: This project was developed in 2020 to demonstrate knowledge representation concepts. The health predictions are based on statistical models and should not be used for medical advice. The mushroom classifier is for educational purposes only - never use ML models alone to determine mushroom edibility.*