# 🧠 Brain Tumor Detection & Model Benchmarking

This repository contains a **comprehensive experimental framework** for detecting brain tumors from genomic features, leveraging both **classic machine learning algorithms** and **custom-designed deep neural networks**.  
The project focuses on **data preprocessing**, **feature selection**, **dimensionality reduction**, **class balancing**, and **multi‑model performance comparison**—particularly under **small dataset constraints**, where overfitting is a primary concern.

Our methodology systematically tests multiple configurations for each step—ranging from **hyperparameter tuning** in classic classifiers to **architectural variations** in neural networks—to ensure the optimal balance between **accuracy**, **generalization**, and **interpretability**.

---

## 📊 Dataset
Dataset used:  
The `brain_tumor.csv` dataset containing labeled genomic profiles of tumor and normal samples, with target variable `y`:
- `normal` → 0  
- `tumor` → 1  

*Note:* The dataset is relatively balanced, so oversampling techniques like SMOTE are applied strategically during training, not for correcting severe imbalance.

---

## 📦 Libraries Used

### Core Libraries
```python
pandas, numpy, matplotlib, seaborn
```

### Machine Learning
```python
scikit-learn, imbalanced-learn
```

### Deep Learning
```python
tensorflow, keras
```

---

## ⚙️ Project Workflow Overview

1. **Import Libraries** → Load required packages for data preprocessing, visualization, machine learning, and deep learning.  
2. **Load Dataset** → Read `brain_tumor.csv` and inspect its shape, columns, and missing values.  
3. **Target Distribution Analysis** → Visualize `y` to confirm balanced class representation.  
4. **Feature/Target Separation** → Remove unnecessary identifier columns and encode `y` into binary integer form.  
5. **Machine Learning Pipeline**:  
   - **StandardScaler** for feature normalization.  
   - **SelectKBest (ANOVA F-test)** to retain top 15 features.  
   - **PCA** to reduce feature space to 5 principal components for noise reduction and computational efficiency.  
   - **Logistic Regression** as a baseline classic model.  
6. **Train/Test Splitting** → Stratified split (80% train, 20% test) for consistent class proportions.  
7. **Evaluation of Logistic Regression** → Capture baseline accuracy for comparison.  
8. **Feature Importance Analysis (Random Forest)** → Identify top genomic features differentiating tumor and normal samples and visualize the top 20 contributors.  
9. **Class Balancing (SMOTE)** → Synthetically oversample minority class to ensure balanced learning.  
10. **Custom Neural Network Design**:  
    - Flexible architecture with variable hidden layers, neurons, dropout rates, and activations.  
    - Regularization via dropout and **EarlyStopping** to prevent overfitting.  
    - Trained with Adam optimizer and binary crossentropy loss.  
    - Validation split for in-training performance monitoring.  
11. **Neural Network Evaluation** → Analyze training vs. validation performance; ensure high test accuracy without signs of overfitting.  
12. **Small Dataset Handling Strategy**:  
    - Compare neural network results with a well-regularized **classic model** (Gradient Boosting) as an overfitting safeguard.  
    - If NN accuracy ≈ Classic model accuracy → NN generalizes well.  
13. **Gradient Boosting Classifier**:  
    - Parameters:
```python
learning_rate = 0.05
n_estimators = 100
```
    - Selected as one of the best-performing models in terms of accuracy and robustness to small datasets.

---

## 🚀 Results

- **Logistic Regression (Baseline)**: Provided interpretable performance foundation.  
- **Random Forest (Feature Importance)**: Revealed strong contribution from selected genomic features.  
- **Custom Neural Network**: Achieved high accuracy with careful regularization and architectural tuning.  
- **Gradient Boosting Classifier**: Matched or exceeded deep learning performance—strong candidate for deployment in small dataset scenarios.

> **Important Note:**  
> While **Gradient Boosting** can often deliver excellent performance on small datasets, **this is not guaranteed**. Factors such as **data quality**, **feature relevance**, **label noise**, and **hyperparameter optimization** all influence the outcome. Always benchmark against multiple models and validate results before drawing conclusions.

---

## 📌 Key Takeaways

- Combining **classic** and **deep learning** models provides more reliable evaluation, particularly on limited data.  
- Strategic **feature selection** and **dimensionality reduction** can significantly boost performance.  
- Class balancing with SMOTE and careful regularization are critical in avoiding overfitting.  
- Performance benchmarking helps ensure that complex models like neural networks are truly learning patterns—not just memorizing noise.

---

📁 *This README is designed to both document our methodology thoroughly and guide future researchers on applying similar techniques to small, sensitive biomedical datasets.*

---
