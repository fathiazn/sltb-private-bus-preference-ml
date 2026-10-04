import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from preprocessing import load_and_preprocess_data


DATA_PATH = "data/bus_preference_data.xlsx"

# Load data
X, y, label_encoders, target_encoder = load_and_preprocess_data(DATA_PATH)

# Same split used throughout the project
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Create models
models = {
    "decision_tree": DecisionTreeClassifier(
        random_state=42
    ),

    "logistic_regression": LogisticRegression(
        max_iter=1000
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "xgboost": XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )
}

# Create models directory
os.makedirs("models", exist_ok=True)

# Train and save
for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    file_path = f"models/{name}.joblib"

    joblib.dump(model, file_path)

    print(f"Saved: {file_path}")

# Save preprocessing information
joblib.dump(
    label_encoders,
    "models/label_encoders.joblib"
)

joblib.dump(
    target_encoder,
    "models/target_encoder.joblib"
)

print("\nAll models saved successfully!")