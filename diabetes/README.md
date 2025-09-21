# 🩺 Diabetes Prediction & Model Comparison

This project is the result of **comprehensive experiments and evaluations** on various **machine learning** and **deep learning** architectures for predicting diabetes using the **Pima Indians Diabetes Dataset**.  
The pipeline includes **data cleaning**, **outlier detection**, **feature scaling**, **missing value imputation**, exploratory data analysis, and **multi‑model performance benchmarking**.

> For each major step (e.g., selecting algorithms, tuning hyperparameters, building ANN architectures), **multiple configurations and parameter ranges** were tested systematically to ensure the balance between **accuracy**, **generalization**, and **interpretability**.

---

## 📊 Dataset
Dataset used:  
[🔗 Pima Indians Diabetes Database on Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

## 📦 Libraries Used

### Core Libraries
```python
pandas, numpy, matplotlib, seaborn
```

### Machine Learning
```python
scikit-learn, xgboost
```

### Deep Learning
```python
tensorflow, keras
```

### Additional Preprocessing
```python
KNNImputer from scikit-learn
IsolationForest from scikit-learn
```

---

## ⚙️ Project Workflow Overview

1. **Import Libraries** → Load all the necessary tools for data processing, visualization, and modeling.  
2. **Load Dataset** → Read the `diabetes.csv` file and check the shape, preview, and data types.  
3. **Outlier Detection** → Use `IsolationForest` to identify and analyze potential outliers.  
4. **Data Visualization**:  
   - Histograms before and after preprocessing.  
   - PCA projection plots for outlier visualization.  
   - Target variable distribution plots.  
5. **Data Cleaning**:  
   - Replace invalid zero values in specific medical measurements with `NaN`.  
   - Impute missing values using **KNNImputer**.  
6. **Feature Scaling** → Apply **StandardScaler** after imputation to normalize features.  
7. **Feature Correlation Analysis** → Generate a correlation heatmap.  
8. **Feature Importance (Random Forest)** → Identify the most predictive features (e.g., *Glucose* found as the most important).  
9. **Model Comparison via GridSearchCV**:  
   - **Random Forest Classifier**  
   - **XGBoost Classifier**  
   - **Gradient Boosting Classifier**  
   - Evaluate based on accuracy and optimal hyperparameters.  
10. **Artificial Neural Network (ANN)** Experiments:  
    - Multiple architectures tested (varying layers, neurons, dropouts).  
    - Early Stopping and validation split used.  
11. **Additional Model (SVM)** → For performance comparison.  
12. **Final Best Model** → **GradientBoostingClassifier** with parameters:
```python
learning_rate = 0.05
n_estimators = 100
```

---

## 🚀 Results
The **Gradient Boosting Classifier** achieved the highest accuracy in the tested configurations for this dataset.  
However, remember:

> **Important Note:**  
> While **Gradient Boosting** can sometimes perform exceptionally well — even with a relatively small dataset — this is **not always guaranteed**. Performance depends on factors like **data quality**, **feature relevance**, **noise**, and **hyperparameter tuning**. Always validate and compare your results with multiple models.

---

## 📌 Key Takeaways
- Proper **data preprocessing** and **cleaning** have a significant impact on accuracy.  
- Outlier detection and correct imputation strategies prevent bias in modeling.  
- Multiple models should be compared before final selection, even if one is often favored in theory.  

---

