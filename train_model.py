# ================================
# STEP 1: Import Required Libraries
# ================================

import pandas as pd                      # For data handling
import numpy as np                       # For numerical operations
import joblib                            # To save model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# ================================
# STEP 2: Load Dataset
# ================================

# Load your cleaned dataset
data = pd.read_csv("D:/ML/ML project/insurance_fraud_data_cleaned.csv")

# Display first few rows (optional check)
print("Dataset Loaded Successfully")
print(data.head())


# ================================
# STEP 3: Separate Features & Target
# ================================

# Change 'fraud' to your actual target column name if different
target_column = "fraud reported"

X = data.drop(target_column, axis=1)
y = data[target_column]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# ================================
# STEP 4: Train-Test Split
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Maintains class balance
)


# ================================
# STEP 5: Create ML Pipeline
# ================================

# We use pipeline so preprocessing + model are saved together

pipeline = Pipeline([
    ("scaler", StandardScaler()),            # Feature scaling
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])


# ================================
# STEP 6: Train Model
# ================================

pipeline.fit(X_train, y_train)

print("\nModel Training Completed")


# ================================
# STEP 7: Evaluate Model
# ================================

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)


# ================================
# STEP 8: Save Model as model.pkl
# ================================

joblib.dump(pipeline, "model.pkl")

print("\nmodel.pkl saved successfully!")
