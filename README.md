# Wine Quality Analysis Using Machine Learning

## 1. Project Overview

This project applies three machine learning approaches to wine-quality data:

1. **Classification** — predicts whether a wine belongs to the lower-quality or higher-quality class.
2. **Regression** — predicts the numerical wine-quality score.
3. **Clustering** — groups wines with similar physicochemical properties without using quality labels during clustering.

The notebooks include dataset inspection, exploratory data analysis (EDA), data cleaning, feature engineering, preprocessing, model comparison, evaluation metrics, visualizations, and hyperparameter tuning where implemented.

## 2. Problem Statement

Wine quality is commonly assessed through sensory evaluation. This project explores how measurable physicochemical properties can be used by machine learning models to classify wine quality, estimate a numerical quality score, and discover groups of similar wines.

The three tasks answer different questions:

- **Classification:** Which quality class does a wine belong to?
- **Regression:** What numerical quality score is predicted for a wine?
- **Clustering:** Which wines have similar chemical characteristics, without using their quality labels to create the groups?

## 3. Dataset Description

The notebooks use wine-quality datasets containing physicochemical measurements such as:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Wine type, where present

The quality score is the target for regression. The classification notebooks convert quality into a binary target using the rule `quality >= 6` for the higher-quality class and `quality < 6` for the lower-quality class. The clustering workflow deliberately excludes both `quality` and `quality_class` when fitting clusters.

### Dataset-file note

The current notebooks reference different input CSV files:

- `classification.ipynb` loads `wine-quality-white-and-red.csv`.
- `regression.ipynb` loads `winequality-red.csv`.
- `review2.ipynb` loads `winequality-white.csv` using `sep=';'`.

Therefore, the current results are based on different dataset selections and should not be interpreted as a direct, like-for-like comparison across all three tasks. Keep the CSV files in the project directory with these exact names, or update the paths in the notebooks. If the project is intended to use one consistent dataset throughout, standardize the dataset and rerun the notebooks before reporting cross-task conclusions.

## 4. Project Structure

Example folder structure:

```text
wine-quality-ml/
├── wine-quality-white-and-red.csv
├── winequality-red.csv
├── winequality-white.csv
├── classification.ipynb
├── regression.ipynb
├── review2.ipynb
└── README.md
```

Only include the CSV files that you actually use, and retain the filenames expected by the corresponding notebooks. `review2.ipynb` contains the Review 2 classification comparison and the clustering workflow.

## 5. Classification

### Objective

Predict a binary wine-quality class using physicochemical properties. The target rule used in the notebooks is:

- `quality >= 6` → higher-quality class
- `quality < 6` → lower-quality class

### Algorithms implemented

The consolidated classification notebook evaluates these 10 algorithms:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes
4. Decision Tree
5. Support Vector Machine (SVM)
6. Random Forest
7. AdaBoost
8. Gradient Boosting
9. Bagging with a Decision Tree
10. Multi-Layer Perceptron (MLP) Classifier

### Workflow

- Inspect the dataset, data types, missing values, duplicates, and target distribution.
- Explore feature distributions, correlations, and relationships with wine quality.
- Remove duplicate rows and inspect outliers using the IQR method.
- Engineer additional features, including sulfur-dioxide ratio, acidity balance, sugar-to-alcohol ratio, and squared alcohol.
- Use preprocessing pipelines for imputation, scaling, and categorical encoding where appropriate.
- Split the data into training and test sets; the classification workflow uses a stratified split.
- Train and compare all 10 models.
- Evaluate predictions using accuracy, weighted precision, weighted recall, weighted F1-score, confusion matrices, and ROC-AUC.
- Use cross-validation and `GridSearchCV` for model selection and tuning.

### Classification results

The following values are from the consolidated comparison output in `classification.ipynb`. Values are rounded to three decimal places.

| Algorithm | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| SVM | 0.777 | 0.805 | 0.850 | 0.775 | 0.832 |
| Random Forest | 0.774 | 0.808 | 0.839 | 0.773 | 0.833 |
| Gradient Boosting | 0.765 | 0.797 | 0.838 | 0.763 | 0.826 |
| Bagging (Decision Tree) | 0.762 | 0.801 | 0.826 | 0.761 | 0.828 |
| KNN | 0.758 | 0.786 | 0.844 | 0.755 | 0.818 |
| AdaBoost | 0.754 | 0.789 | 0.829 | 0.751 | 0.812 |
| Logistic Regression | 0.748 | 0.782 | 0.829 | 0.745 | 0.815 |
| MLP Classifier | 0.733 | 0.783 | 0.793 | 0.733 | 0.785 |
| Decision Tree | 0.727 | 0.782 | 0.781 | 0.727 | 0.767 |
| Gaussian Naive Bayes | 0.690 | 0.781 | 0.701 | 0.694 | 0.757 |

**Classification observation:** SVM has the highest accuracy and weighted F1-score in this initial comparison. Random Forest has the highest ROC-AUC in the same table. The tuned Random Forest in this notebook used `n_estimators=200`, `max_depth=None`, and `min_samples_leaf=2`; its reported cross-validation ROC-AUC was approximately `0.833` and test ROC-AUC approximately `0.832`.

`review2.ipynb` also includes a separate 10-model classification comparison on the white-wine dataset. Its results differ because it uses a different input dataset. Use the results from the notebook and dataset required for your final submission.

## 6. Regression

### Objective

Predict the numerical `quality` score from physicochemical properties.

### Algorithms implemented

The regression notebook compares 10 models:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Elastic Net
5. Polynomial Regression
6. KNN Regressor
7. Support Vector Regression (SVR)
8. Random Forest Regressor
9. Gradient Boosting Regressor
10. Decision Tree Regressor

### Workflow

- Inspect the dataset and examine quality distribution and feature relationships.
- Remove exact duplicate rows.
- Inspect numerical outliers using the IQR method; the notebook retains them rather than automatically removing potentially valid observations.
- Engineer additional features: free-to-total sulfur-dioxide ratio, acidity balance, sugar-to-alcohol ratio, and squared alcohol.
- Apply preprocessing pipelines with median imputation and standard scaling for numerical features, and most-frequent imputation and one-hot encoding for categorical features when present.
- Compare 10 regression models using a train/test split and five-fold cross-validation.
- Tune selected models with `GridSearchCV`.
- Inspect residuals, predicted-versus-actual plots, and feature importance where available.

### Regression results

The following results are from the output saved in `regression.ipynb`.

| Model | CV RMSE | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|---:|
| Random Forest | 0.694 | 0.690 | 0.527 | 0.399 |
| Gradient Boosting | 0.702 | 0.703 | 0.541 | 0.376 |
| Polynomial Regression | 0.715 | 0.704 | 0.548 | 0.375 |
| SVR | 0.725 | 0.716 | 0.534 | 0.353 |
| Ridge | 0.735 | 0.719 | 0.555 | 0.348 |
| Linear Regression | 0.735 | 0.719 | 0.555 | 0.347 |
| Lasso | 0.735 | 0.719 | 0.556 | 0.347 |
| Elastic Net | 0.735 | 0.721 | 0.559 | 0.343 |
| KNN | 0.723 | 0.731 | 0.562 | 0.325 |
| Decision Tree | 0.964 | 0.933 | 0.644 | -0.100 |

**Regression observation:** Random Forest has the lowest test RMSE and MAE and the highest test R² in the saved comparison. Its test R² of approximately `0.399` indicates that the model explains around 40% of the variation in the held-out quality scores; there is still substantial variation that the model does not explain.

## 7. Clustering

### Objective

Discover groups of wines with similar physicochemical measurements using unsupervised learning. The clustering workflow excludes `quality` and `quality_class` from the features used to fit the clusters.

### Algorithms implemented

- **K-Means:** partitions observations into a chosen number of clusters.
- **Agglomerative Hierarchical Clustering:** builds clusters by successively merging observations or groups.

The notebook standardizes the clustering features, examines K-Means inertia for values of `k` from 2 through 10, and uses `k=3` for the reported comparison. It also produces a sampled hierarchical dendrogram and PCA and t-SNE visualizations.

### Clustering evaluation

Clustering does not use classification accuracy as its main evaluation metric. The notebook reports:

- **Silhouette Score:** higher is generally better; measures how well-separated clusters are.
- **Davies-Bouldin Index:** lower is generally better; measures within-cluster scatter relative to separation.
- **Calinski-Harabasz Index:** higher is generally better; compares between-cluster dispersion with within-cluster dispersion.

### Clustering results

The saved output reports a clustering feature matrix with **3,961 rows and 15 features**.

| Algorithm | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index |
|---|---:|---:|---:|
| K-Means (`k=3`) | 0.1580 | 2.0305 | 812.5270 |
| Agglomerative Hierarchical (`k=3`) | 0.1048 | 2.4407 | 616.4181 |

**Clustering observation:** K-Means performs better than Agglomerative Hierarchical Clustering on all three reported internal metrics for this run. However, the silhouette scores are relatively low, so the clusters should be interpreted as imperfect groupings rather than clearly separated natural categories. Use the elbow curve and PCA/t-SNE plots to support the final interpretation.

## 8. Environment Setup

### Requirements

- Python 3.10 or later recommended
- Jupyter Notebook or Jupyter support in Visual Studio Code

Install the libraries used by the notebooks:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

## 9. How to Run the Project

1. Place the notebook files and the required dataset CSV files in the project directory.
2. Confirm that each CSV filename matches the path used in its notebook.
3. Open a terminal in the project directory.
4. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open and run the notebooks in order:
   - `classification.ipynb` — combined red/white dataset classification.
   - `regression.ipynb` — regression workflow.
   - `review2.ipynb` — Review 2 classification comparison and clustering.
6. Run the cells from top to bottom so that imports, dataset loading, preprocessing, model fitting, metrics, and plots are generated in order.

If using VS Code, open the project folder, open a notebook, select a Python kernel, and run the cells from top to bottom.

## 10. Evaluation Metrics Summary

| Task | Main metrics | Purpose |
|---|---|---|
| Classification | Accuracy, precision, recall, weighted F1-score, ROC-AUC, confusion matrix | Evaluate class predictions and class separation |
| Regression | RMSE, MAE, R² | Measure prediction error and explained variance |
| Clustering | Silhouette score, Davies-Bouldin index, Calinski-Harabasz index | Evaluate cluster cohesion and separation |

## 11. Conclusion

This project compares supervised and unsupervised machine learning techniques on wine-quality measurements. Classification compares 10 algorithms for binary quality prediction, regression compares 10 models for numerical quality estimation, and clustering compares K-Means with Agglomerative Hierarchical Clustering.

In the saved classification comparison, SVM has the strongest accuracy and weighted F1-score, while Random Forest has the highest ROC-AUC. In regression, Random Forest achieves the strongest reported test metrics. In clustering, K-Means has better internal evaluation scores than Agglomerative Hierarchical Clustering, although the relatively low silhouette score suggests limited cluster separation.

These findings apply to the datasets and settings used in the individual notebooks. Since the notebooks currently load different CSV files, ensure the dataset choice is consistent before making direct comparisons across tasks.

## 12. Future Improvements

- Standardize the dataset source across notebooks if a unified comparison is required, then rerun all experiments.
- Compare tuned models using consistent validation procedures.
- Investigate class imbalance and report class-specific classification metrics.
- Explore alternative clustering settings and validate the choice of cluster count.
- Interpret cluster profiles using feature summaries rather than relying only on two-dimensional plots.
- Save fitted models and add a simple interface for new wine-quality predictions.
