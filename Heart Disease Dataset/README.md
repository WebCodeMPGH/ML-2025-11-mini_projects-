

---

## 📌 Project Title  
**Heart Disease Prediction using EDA, Isolation Forest, Feature Importance & Artificial Neural Networks**

---

## 📖 Overview  
This project focuses on predicting **heart disease** from patient health records using **data preprocessing, exploratory data analysis (EDA), outlier detection, feature selection, and deep learning**.  
The workflow blends **statistical analysis**, **machine learning techniques**, and **Artificial Neural Networks (ANNs)** to build both **multi-class** and **binary classification** models.

The project is structured as a **step-by-step Jupyter Notebook** (exported to Python script here), making it easy to follow for learners, researchers, and data enthusiasts.

---

## 🗂️ Dataset Source  
The dataset used in this project is the **Heart Disease Dataset** available on **Kaggle**:  

🔗 **[Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)**  


---

## 🔍 Project Workflow  

### 1. **Loading and Exploring Data**
- Read the CSV file into a Pandas DataFrame.
- Checked dataset shape, data types, and previewed records.

### 2. **Exploratory Data Analysis (EDA)**
- Examined the distribution of target classes.
- Generated descriptive statistics.
- Created **visualizations** (histograms, correlation heatmaps, count plots) using **Matplotlib** and **Seaborn**.

### 3. **Outlier Detection with Isolation Forest**
- Used `IsolationForest` (Sklearn) to identify and remove potential outliers.
- Visualized outlier vs. normal data distribution.

### 4. **Feature Scaling**
- Applied `StandardScaler` to normalize features for neural network training.

### 5. **Feature Importance via Random Forest**
- Ranked features by importance to help understand which factors are most influential for predicting heart disease.
- Identified `thal`, `thalach`, and `oldpeak` as top predictors.

### 6. **Model Building – Artificial Neural Networks**
Implemented two ANN models with TensorFlow/Keras:  
1. **Multi-Class Model** – Dense(2, activation='softmax'), loss: `sparse_categorical_crossentropy`.  
2. **Binary Classification Model** – Dense(1, activation='sigmoid'), loss: `binary_crossentropy`.  

Both models include:
- Multiple Dense layers with **ReLU activation**.
- **Dropout layers** for regularization.
- **EarlyStopping** callback to prevent overfitting.

### 7. **Evaluation**
- Printed **model accuracy** on the test set.
- Plotted **training vs. validation accuracy** over epochs.

---

## 📦 Dependencies  
To run this project, install the following Python packages:  
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```


---

## 📊 Example Visualizations  
- Distribution of target variable.  
- Isolation Forest outlier detection results.  
- Correlation heatmap of features.  
- Random Forest feature importance chart.  
- Training vs. validation accuracy plots for ANN models.

---

## 🙌 Acknowledgment  
Dataset obtained from **Kaggle**. Many thanks to the dataset provider for making this research possible.

