# %% [markdown]
# IMPORTATION :

# %%
# Import core data manipulation libraries
import pandas as pd   # For loading and processing data in DataFrame format
import numpy as np    # For numerical operations
import matplotlib.pyplot as plt   # For plotting and charts
import seaborn as sns  # For statistical data visualization
import os              # For operating system functions (not used here)
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Preprocessing tools
from sklearn.model_selection import train_test_split            # For splitting data into training/testing sets

# %% [markdown]
# import CSV

# %%
# Load the dataset from CSV file into a pandas DataFrame
df = pd.read_csv(r"D:\pyhton\#\project 1\machin learning\csv\insurance.csv")

# Display the number of rows and columns in the dataset
print('\nNumber of rows and columns in the data set: ', df.shape)
print('')

# Show the first few rows for a quick preview
df.head()

# %% [markdown]
# check that the lines don't have any non value 

# %%
# Check for missing values in each column
df.isnull().sum()

# %%
# Get summary statistics for numerical columns (mean, std, min, max, quartiles)
df.describe()

# %%
# Show column data types and non-null counts
df.info()

# %%
# Print column names and types, then make a pairplot of all variables
print("\nData types:\n", df.dtypes)
sns.pairplot(df)  # Pairwise scatterplots to visualize relationships and distributions
plt.show()

# %% [markdown]
# ## LabelEncoder

# %%
# Encode categorical variables into numerical labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Loop through each categorical feature and encode it
for c in ["sex", "region", "smoker"]:
    le.fit(df[c])  # Learn mapping from categories to integers
    df[c] = le.fit_transform(df[c])  # Apply transformation to replace categories with integers

df.head()

# %% [markdown]
# # clean outliers:

# %%
from scipy import stats

# Compute z-scores for the BMI column (absolute value)
z_scores_f = abs(stats.zscore(df['bmi']))

# Define z-score threshold for detecting outliers
threshold = 3

# Create a boolean mask where True means "BMI is within threshold"
mask1 = z_scores_f < threshold

# Apply the mask to filter DataFrame and remove rows where BMI is an outlier
df_clean = df[mask1]
df_clean.head()
df_clean.shape  # Show how many rows remain after filtering

# %% [markdown]
# # explain feature and target :

# %%
# Separate independent variables (features) and dependent variable (target)
X = df_clean.drop("charges", axis=1)   # Features: all columns except 'charges'
y = df_clean["charges"]                # Target: 'charges'

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ## StandardScaler

# %%
# Standardize features by removing mean and scaling to unit variance
scaler = StandardScaler()

# Fit the scaler on training data and transform it
X_train = scaler.fit_transform(X_train)

# Use the same scaler to transform the testing data
X_test = scaler.transform(X_test)

# %% [markdown]
# # use polynomial 

# %%
# Create polynomial features up to degree 2 by default
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures()
poly.fit_transform(X_train)  # Fit on training data (output is ignored here)

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

results = []  # Store (degree, R² score) pairs

# Loop over polynomial degrees 1 to 14
for degree in range(1, 15):
    poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
    
    # Transform training and testing data with the current polynomial degree
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Create and fit a linear regression model on polynomial-transformed features
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Predict on the transformed test data
    y_pred = model.predict(X_test_poly)
    
    # Calculate R² score for current degree
    r2 = r2_score(y_test, y_pred)
    
    # Save the results
    results.append((degree, r2))
    
    print(f"sucsessfuly add {degree} to path ")

# %%
# Show the list of (degree, R² score) results
results

# %%
# Select the degree with the highest R² score
best_degree, best_r2 = max(results, key=lambda x: x[1])
best_degree

# %%
# Rebuild polynomial features with the best degree found
poly = PolynomialFeatures(degree=best_degree, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train a final Linear Regression model with the selected polynomial features
model = LinearRegression()
model.fit(X_train_poly, y_train)

# %%
# Print best degree and corresponding R² score
print(best_degree, best_r2)

# %%
# Plot R² scores for each degree
degrees = [deg for deg, _ in results]
r2_scores = [r2 for _, r2 in results]

plt.figure(figsize=(8, 5))
plt.plot(degrees, r2_scores, marker='o', color='b')
plt.scatter(best_degree, best_r2, color='red', s=100, label=f"Best Degree = {best_degree}")
plt.title("Polynomial Degree vs R² Score (Cleaned & Scaled)")
plt.xlabel("Polynomial Degree")
plt.ylabel("R² Score")
plt.grid(True)
plt.legend()
plt.show()

print(f"Best Degree: {best_degree}, Best R²: {best_r2:.4f}")

# %%
# Build polynomial features object again with best degree (without bias term)
poly = PolynomialFeatures(degree=best_degree, include_bias=False)
poly.fit(X_train)

# Get names of all generated polynomial feature terms
feature_names = poly.get_feature_names_out(X.columns)
print(feature_names)

# 1. Get feature names from the polynomial transformer
feature_names = poly.get_feature_names_out(X.columns)

# 2. Get coefficients from the trained regression model
coefs = model.coef_

# 3. Sort coefficients by absolute value to find most important features
abs_coefs = np.abs(coefs)
top_idx = np.argsort(abs_coefs)[::-1][:10]  # Indices of top 10 features

# 4. Create a list of top feature names with their coefficients
top_features = [(feature_names[i], coefs[i]) for i in top_idx]
print("Top Polynomial Features:")
for name, weight in top_features:
    print(f"{name}: {weight:.4f}")
