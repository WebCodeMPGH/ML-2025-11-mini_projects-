# 🍄 Mushroom Classification Project

This project focuses on building and evaluating machine learning models to classify mushrooms as **edible** or **poisonous** using the [Mushroom Dataset from Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification).

---

## 📂 Project Workflow

### 1. Data Preprocessing
- Checked for missing values (**none found**).
- Checked for duplicate rows (**none found**).
- Explored class distribution → data is already balanced, so **no SMOTE** needed.
- Encoded categorical features using **LabelEncoder**.

### 2. Exploratory Data Analysis (EDA)
- Visualized **class distribution** using Seaborn.
- Generated **feature correlation heatmap**.
- Ranked **feature importance** using Random Forest.

### 3. Modeling Approaches

#### **A. Deep Learning Approach**
- Built an **Artificial Neural Network (ANN)** with `TensorFlow/Keras`.
- Added **Dropout layers** & **L2 regularization** to prevent overfitting.
- Used **EarlyStopping** to optimize training.
- Evaluated accuracy on test data.

#### **B. Classical Machine Learning Approach**
- Models tested:
  - Random Forest
  - Gradient Boosting
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
- **Grid Search + Cross Validation** for hyperparameter tuning.
- Compared test accuracy with ANN.

### 4. Results
- Final comparison between ANN and the best classical model to find the most efficient method for mushroom classification.

---

## 📊 Technologies Used
- **Python 3.x**
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- TensorFlow / Keras

---

## 📎 Dataset
**Source**: [Kaggle - Mushroom Classification Dataset](https://www.kaggle.com/datasets/uciml/mushroom-classification)  
Contains categorical attributes describing physical characteristics of mushrooms and their classification as edible or poisonous.

---

