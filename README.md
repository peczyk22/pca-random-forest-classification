# PCA and Random Forest for Multiclass Classification

A machine learning project investigating whether **Principal Component Analysis (PCA)** can improve the performance of a **Random Forest classifier** on an imbalanced multiclass dataset.

This project was completed for the **Mathematical Modelling in Machine Learning** course as part of the **BSc in Mathematical and Computing Sciences for Artificial Intelligence** at **Bocconi University**.

**Author:** Jan Petsch

---

## Project Overview

The dataset consists of **1,100 samples** with **30 features** and a target variable containing **5 imbalanced classes**.

The main objective of the project was to determine:

- whether applying PCA is beneficial when using a Random Forest classifier;
- the optimal number of principal components;
- how class imbalance should be handled;
- and which Random Forest hyperparameters provide the best predictive performance.

The analysis aims to balance predictive performance with computational efficiency rather than exhaustively exploring every possible combination of parameters.

---

## Methodology

The project follows the general workflow:

**Exploratory Data Analysis → PCA Analysis → Random Forest Baseline → Class Imbalance Strategies → Hyperparameter Tuning → Final Model Evaluation**

### 1. Exploratory Data Analysis

The dataset was first examined to understand:

- feature distributions;
- class frequencies and imbalance;
- correlations between features;
- and the overall structure of the classification problem.

### 2. Principal Component Analysis

Both scaled and non-scaled PCA configurations were investigated.

The number of principal components was evaluated using stratified cross-validation together with Random Forest performance, rather than relying exclusively on explained variance.

The selected PCA configuration retained:

**14 principal components**, explaining approximately **96.57% of the original variance**.

### 3. Handling Class Imbalance

Several approaches for dealing with the imbalanced target classes were compared:

- Baseline Random Forest
- Class weighting
- Random oversampling
- SMOTE

The methods were evaluated using stratified cross-validation, with particular attention paid to F1 scores due to the unequal class distribution.

**SMOTE** was ultimately selected for the final modelling strategy.

### 4. Hyperparameter Optimization

Random Forest hyperparameters were first explored individually to identify promising regions of the parameter space.

This was followed by:

- `RandomizedSearchCV` for broader exploration;
- `GridSearchCV` for more focused optimization.

The optimization focused primarily on F1-based performance in order to account for the class imbalance.

---

## Final Model

The final modelling strategy combines:

- **PCA with 14 components**
- **SMOTE**
- **Random Forest classification**

The selected Random Forest configuration was:

```python
RandomForestClassifier(
    n_estimators=574,
    max_depth=50,
    max_features="sqrt",
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
```

### Final Performance

| Metric | Score |
|---|---:|
| Accuracy | **0.7364** |
| Weighted F1 | **0.7309** |
| Macro F1 | **0.6793** |

The final model also improved performance on the minority classes compared with the initial Random Forest baseline.

---

## Conclusion

For this dataset, applying PCA before Random Forest classification proved beneficial.

A non-standardized PCA representation using **14 principal components** provided a useful trade-off between dimensionality reduction and predictive performance.

Combining this representation with **SMOTE** and a tuned **Random Forest classifier** resulted in the strongest overall model identified during the analysis.

The final approach achieved approximately **73.6% classification accuracy**, while providing more balanced predictive performance across the five target classes.

---

## Technologies

The project was implemented in Python using:

- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- imbalanced-learn
- SciPy
- Jupyter Notebook

---

## Repository Structure

```text
pca-random-forest-classification/
│
├── Model_final.ipynb     # Complete analysis and modelling
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
└── .gitignore
```

---

## Data

The dataset used for this project was provided as part of the university coursework and is therefore **not included in this repository**.

The notebook contains the complete methodology, experiments, visualizations, model comparisons, and final evaluation.

---

## Author

**Jan Petsch**

BSc in Mathematical and Computing Sciences for Artificial Intelligence  
Bocconi University

Course: **Mathematical Modelling in Machine Learning**
