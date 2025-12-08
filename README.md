# Wine Quality Classification

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Machine learning project comparing multiple classification algorithms to predict wine quality based on physicochemical properties.

## Overview

This project tackles the wine quality classification problem using an ensemble approach. The model classifies wines as "good" (quality ≥ 7) or "bad" (quality < 7) based on 13 physicochemical features.

## Features

- Comparative analysis of 4 different ML algorithms
- Comprehensive data exploration with correlation heatmaps
- ROC-AUC evaluation for model selection
- Feature importance analysis
- Hyperparameter-tuned Random Forest model

## Tech Stack

- **Python 3.8+**
- **scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Matplotlib & Seaborn** - Data visualization

## Dataset

**Wine Quality Dataset**
- 1,143 samples
- 13 features (physicochemical properties)
- Binary target: Quality ≥ 7 (Good) vs Quality < 7 (Bad)

**Features:**
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free/Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol content

## Models Compared

| Algorithm | ROC-AUC (Validation) | Test Accuracy |
|---|---|---|
| Logistic Regression | 72.6% | - |
| SVM (RBF Kernel) | 50.0% | - |
| XGBoost | 74.8% | - |
| **Random Forest** | **80.0%** | **91.3%** |

**Winner:** Random Forest (selected as the best model)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shreyasnimbalkar/wine-model.git
cd wine-model
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook Untitled3.ipynb
```

## Usage

Open and run the notebook:
```python
# Load the dataset
df = pd.read_csv('WineQT.csv')

# Train-test split (90-10)
X_train, X_test, y_train, y_test = train_test_split(...)

# Train Random Forest model
rf_model.fit(X_train, y_train)

# Evaluate
accuracy = rf_model.score(X_test, y_test)
```

## Results

- **Best Model:** Random Forest
- **Test Accuracy:** 91.3%
- **ROC-AUC Score:** 80.0%
- **Train-Test Split:** 90-10

The Random Forest model significantly outperformed other algorithms, showing robust performance in distinguishing between good and bad quality wines.

## Data Visualization

The project includes:
- Feature correlation heatmaps
- Class distribution analysis
- ROC-AUC comparison charts
- Feature importance plots

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## License

MIT License

## Author

**Shreyas Nimbalkar**
- GitHub: [@shreyasnimbalkar](https://github.com/shreyasnimbalkar)
- Email: shreyasgnimbalkarwork@gmail.com
- LinkedIn: [Shreyas Nimbalkar](https://www.linkedin.com/in/shreyas-nimbalkar-713bb9267/)
