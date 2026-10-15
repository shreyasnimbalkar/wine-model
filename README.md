# Wine quality classification

A small scikit-learn project: predict whether a wine is "good" (quality 7 or
higher) or not from its chemical measurements. I compared four classifiers and kept
a random forest.

## Data

`WineQT.csv`, 1,143 wines and 13 columns: 11 measurements (fixed acidity, volatile
acidity, citric acid, residual sugar, chlorides, free and total sulfur dioxide,
density, pH, sulphates, alcohol), the quality score, and an `Id` column. There are no
missing values. The CSV is not in the repo; the notebook uploads it in Colab.

## Approach

1. Look at the data: quality counts, volatile acidity and citric acid by quality, and
   a correlation heatmap.
2. Turn quality into a label: 1 if quality >= 7, else 0.
3. Split 90/10 with `random_state=3` (1,028 training rows, 115 test rows).
4. Train logistic regression, XGBoost, an RBF SVM and a random forest, and score each
   with ROC-AUC on the test split.
5. Retrain a random forest and report its accuracy on the same test split.

## Results

ROC-AUC on the 115 test rows (computed from predicted labels, not probabilities):

| Model | Train | Test |
|---|---|---|
| Logistic regression | 0.605 | 0.726 |
| XGBoost | 1.000 | 0.748 |
| SVM (RBF) | 0.500 | 0.500 |
| Random forest | 1.000 | 0.800 |

The final random forest got 91.3% accuracy on the test split.

Things I would fix if I came back to it:

- Most wines are below 7, so accuracy flatters the model. ROC-AUC is the better
  number to look at.
- The `Id` column is left in as a feature. It should be dropped.
- `MinMaxScaler` is imported but never used. The SVM probably scores 0.5 because the
  features aren't scaled, so it predicts one class for everything.
- XGBoost and the random forest score 1.0 on training data, so they overfit; a
  cross-validated search over tree depth would help.
- The notebook prints ROC-AUC under the label "Accuracy".

## How to run

The notebook was written for Google Colab (it uses `google.colab.files.upload()` to
load the CSV). Open `Untitled3.ipynb` in Colab, run the cells, and upload `WineQT.csv`
when asked.

To run it locally instead, remove the upload cell, put `WineQT.csv` next to the
notebook, and install:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

## Files

- `Untitled3.ipynb`: the whole project, from loading the data to a single-wine
  prediction at the end
- `LICENSE`: MIT
