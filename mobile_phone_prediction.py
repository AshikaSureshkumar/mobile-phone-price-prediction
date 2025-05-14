# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("dataset.csv")

# Basic EDA
print(df.info())
print(df.describe())
print(df['price_range'].value_counts())

# Optional: Visualize feature importance later
# sns.heatmap(df.corr(), annot=True, fmt=".2f")
# plt.show()

# Features and Target
X = df.drop("price_range", axis=1)
y = df["price_range"]

# Feature Scaling (Recommended for models sensitive to scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model: Random Forest (you can swap with others like SVM, XGBoost, etc.)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importance Plot
importances = model.feature_importances_
features = df.columns[:-1]
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title('Feature Importance')
plt.show()
