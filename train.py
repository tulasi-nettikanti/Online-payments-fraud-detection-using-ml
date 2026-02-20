# train.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("data/fraud.csv")

# Drop unnecessary columns
data = data.drop(["nameOrig", "nameDest"], axis=1)

# Encode categorical column
le = LabelEncoder()
data["type"] = le.fit_transform(data["type"])

# Features and Target
X = data.drop("isFraud", axis=1)
y = data["isFraud"]

# Train-test split (stratified to preserve fraud ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize Random Forest with imbalance handling
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",   # Important for fraud detection
    random_state=42,
    n_jobs=-1
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model/model1.pkl")
print("Model saved successfully!")
