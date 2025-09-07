# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset with the new filename 'data.csv'
data = pd.read_csv('data.csv', encoding='latin1')
print(data.dtypes)

# Data Cleaning and Preparation
data['timestamp'] = data['timestamp'].str.replace('AP', '')
data['timestamp'] = data['timestamp'].str.replace('AM', '')

# Corrected date parsing
data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce', dayfirst=True)

data.dropna(axis=0, inplace=True)

column_1 = data['timestamp']
data1 = pd.DataFrame({
    "year": column_1.dt.year,
    "month": column_1.dt.month,
    "day": column_1.dt.day,
    "hour": column_1.dt.hour,
    "dayofyear": column_1.dt.dayofyear,
    "week": column_1.dt.isocalendar().week,
    "weekofyear": column_1.dt.isocalendar().week,
    "dayofweek": column_1.dt.dayofweek,
    "weekday": column_1.dt.weekday,
    "quarter": column_1.dt.quarter,
})

data.drop(['timestamp'], axis=1, inplace=True)
data = pd.concat([data1, data], axis=1)

# Handle potential missing values after concatenation just in case
data.dropna(inplace=True)

print(data.info())

# --- FIX: Use the final 'data' DataFrame for plotting ---
plt.figure(figsize=(10, 8))
sns.violinplot(x='act379', y='hour', data=data, hue='act379', palette='winter_r', legend=False)
plt.figure(figsize=(10, 8))
sns.violinplot(x='act13', y='hour', data=data, hue='act13', palette='winter_r', legend=False)
plt.figure(figsize=(10, 8))
sns.violinplot(x='act279', y='hour', data=data, hue='act279', palette='winter_r', legend=False)
plt.figure(figsize=(10, 8))
sns.violinplot(x='act323', y='hour', data=data, hue='act323', palette='winter_r', legend=False)
plt.figure(figsize=(10, 8))
sns.violinplot(x='act363', y='hour', data=data, hue='act363', palette='winter_r', legend=False)
plt.figure(figsize=(10, 8))
sns.violinplot(x='act302', y='hour', data=data, hue='act302', palette='winter_r', legend=False)

# Model Training
X = data.drop(['act379'], axis=1)
y = data['act379']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train, y_train)

# Corrected feature importance plotting
importances = rfc.feature_importances_
indices = np.argsort(importances)
features = X.columns
plt.figure(figsize=(10, 20))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()

# Save the trained model
print("Saving model to disk")
joblib.dump(rfc, 'rf_model')
print("Model saved successfully")

# Make predictions (optional, for verification)
y_pred = rfc.predict(X_test)
print("\n--- Model Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))