# **Unsupervised Coffee Health Data Clustering and Classification**  

## **Introduction**  

This repository presents a complete, end‑to‑end machine learning pipeline applied to a **synthetic coffee health dataset**. The aim of this work is to demonstrate the process of **unsupervised clustering** combined with **supervised classification**, starting from raw data and concluding with a deep learning neural network (ANN).  

The project was deliberately designed as a **practical, reproducible case study** for other developers, students, and researchers who wish to understand and replicate the full journey of:  

- Data loading and exploration  
- Outlier detection and removal  
- Encoding and scaling features  
- Applying K‑Means clustering with evaluation metrics  
- Transforming clusters into target labels for classification tasks  
- Performing feature importance analysis  
- Comparing multiple supervised algorithms  
- Building and fine‑tuning an Artificial Neural Network (ANN)  

Alongside the code, this README incorporates **practical lessons learned** during experimentation, such as the decision to **avoid SMOTE oversampling** in certain scenarios to minimize overfitting risk, as well as insights from **learning rate behavior analysis** where validation accuracy occasionally surpassed training accuracy without causing serious overfitting.  

---

## **Data Overview**  

The dataset, `synthetic_coffee_health_10000.csv`, contains diverse features that mimic the relationship between coffee consumption factors and various health indicators. Key variables include both numerical and categorical values, ranging from demographic information to physical measurements.  

The **`ID`** column was dropped early in preprocessing since it serves no analytical purpose in clustering or classification.  

---

## **Workflow Summary**  

This project was structured into distinct, well‑defined stages, each implemented within the Jupyter Notebook with clear, numbered sections:  

1. **Data Loading & Exploration**  
   - Load CSV data with `pandas`.  
   - Inspect structure, data types, and unique value counts.  
   - Identify duplicate rows.  

2. **Outlier Detection**  
   - Use **Isolation Forest** (`sklearn.ensemble.IsolationForest`) on integer features.  
   - Label normal samples as `1` and anomalies as `-1`.  
   - Remove outliers to create a cleaner dataset for clustering.  
   - Calculate and visualize the outlier percentage.  

   _Rationale_: Outliers, if left unchecked, can distort clustering shapes and reduce the accuracy of downstream supervised models.  

3. **Encoding Categorical Features**  
   - Apply **Label Encoding** to convert categorical text into numerical integers.  
   - Important step before scaling, as K‑Means and neural networks work on numerical data.  

4. **Feature Scaling**  
   - Normalize features with **StandardScaler** to prevent features with large numeric ranges from dominating clustering behavior.  

5. **K‑Means Clustering**  
   - Initial 4‑cluster fitting.  
   - Fine‑tuning the number of clusters based on **Silhouette Score**.  
   - PCA reduction to 2D for visualizing cluster separations.  
   - Analyze the distribution of records in each cluster.  

6. **Train–Test Split for Classification Task**  
   - Treat chosen clusters as supervised learning target labels.  
   - Split into train and test sets (80% / 20%).  

7. **Class Imbalance Handling (Optional)**  
   - While SMOTE can balance class distributions, it was **intentionally avoided** here to reduce the risk of the classifier **overfitting to synthetic data patterns**.  

8. **Feature Importance Analysis**  
   - Use **RandomForestClassifier** to determine which features most influence the classification of clusters.  

9. **Model Benchmarking**  
   - Evaluate:  
     - **RandomForestClassifier**  
     - **XGBoostClassifier**  
     - **GradientBoostingClassifier**  
   - Apply **GridSearchCV** for hyperparameter tuning.  
   - Compare test accuracies in a performance chart.  

10. **Final Model Selection**  
    - Based on performance, the best‑tuned RandomForest model was trained on the full training data.  
    - Accuracy and sample predictions were reported.  

11. **Artificial Neural Network (ANN)**  
    - Built a custom multi‑layer architecture with dropout for regularization.  
    - Tested various layer configurations (details omitted here for clarity, but experiments covered neuron count and activation function variations).  
    - Monitored training/validation accuracy.  
    - Applied **EarlyStopping** to prevent overfitting.  

    _Key Observation_: In some experiments, the validation accuracy slightly exceeded training accuracy. Based on careful inspection and prior tests, this did not indicate harmful overfitting — possible reasons include regularization effects, stochastic training behavior, or particularities of the validation split.  

---

## **Results**  

- **Silhouette Score** helped confirm cluster separation quality.  
- **Random Forest Feature Importance** provided interpretability by ranking the influential attributes.  
- **Model comparison** revealed strong performance for RandomForest after tuning.  
- **ANN model** achieved competitive accuracy with proper regularization and early stopping.  

---

## **Lessons Learned**  

- Outlier detection early in the workflow significantly improved cluster quality.  
- Avoiding SMOTE in certain unsupervised→supervised transitions can help prevent artificially inflated model accuracy.  
- Learning rate and regularization choices in ANN can produce validation metrics slightly better than training metrics — a behavior that requires cautious interpretation but is not inherently problematic.  
- Visualization with PCA is an indispensable tool both for intuitively validating clusters and explaining results to non‑technical audiences.  

---
 

