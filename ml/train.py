
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "telco_churn.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# Load data
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

y = df["Churn"].map({"Yes": 1, "No": 0})

if y.isnull().any():
    raise ValueError("Unexpected values in Churn column.")

X = df.drop(columns=["Churn", "customerID"])

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()


# 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)


def make_pipeline(classifier, scale_numeric=True):
    if scale_numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


# Candidate models
candidates = {
    "Logistic Regression": make_pipeline(
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
    ),
    "Random Forest": make_pipeline(
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        scale_numeric=False,
    ),
}


# Train and compare using validation ROC-AUC
results = {}
trained_models = {}

for name, model in candidates.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_prob)

    results[name] = {"validation_roc_auc": float(auc)}
    trained_models[name] = model

    print(f"{name} validation ROC-AUC: {auc:.4f}")


# Select model based on validation ROC-AUC
best_name = max(
    results,
    key=lambda name: results[name]["validation_roc_auc"],
)
best_model = trained_models[best_name]

print(f"\nSelected model: {best_name}")


# Select threshold on validation data to maximize F1
val_prob = best_model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.20, 0.801, 0.01)
threshold_scores = [
    f1_score(y_val, (val_prob >= t).astype(int), zero_division=0)
    for t in thresholds
]

best_threshold = float(thresholds[int(np.argmax(threshold_scores))])

print(f"Selected threshold: {best_threshold:.2f}")
print(
    "Validation F1 at selected threshold:",
    round(max(threshold_scores), 4),
)


# Final evaluation on untouched test data
test_prob = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

metrics = {
    "selected_model": best_name,
    "validation_results": results,
    "decision_threshold": best_threshold,
    "test_accuracy": float(accuracy_score(y_test, test_pred)),
    "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
    "test_precision_churn": float(
        precision_score(y_test, test_pred, zero_division=0)
    ),
    "test_recall_churn": float(
        recall_score(y_test, test_pred, zero_division=0)
    ),
    "test_f1_churn": float(
        f1_score(y_test, test_pred, zero_division=0)
    ),
    "test_confusion_matrix": confusion_matrix(
        y_test, test_pred, labels=[0, 1]
    ).tolist(),
    "test_classification_report": classification_report(
        y_test,
        test_pred,
        target_names=["No Churn", "Churn"],
        output_dict=True,
        zero_division=0,
    ),
}

print("\nTEST RESULTS")
print("Accuracy:", round(metrics["test_accuracy"], 4))
print("ROC-AUC:", round(metrics["test_roc_auc"], 4))
print("Churn precision:", round(metrics["test_precision_churn"], 4))
print("Churn recall:", round(metrics["test_recall_churn"], 4))
print("Churn F1:", round(metrics["test_f1_churn"], 4))
print("\nConfusion matrix:")
print(metrics["test_confusion_matrix"])
print("\nClassification report:")
print(classification_report(
    y_test,
    test_pred,
    target_names=["No Churn", "Churn"],
    zero_division=0,
))


# Save the model and metadata
joblib.dump(best_model, MODEL_DIR / "churn_pipeline.joblib")

with open(REPORT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open(REPORT_DIR / "model_config.json", "w") as f:
    json.dump({
        "model_name": best_name,
        "decision_threshold": best_threshold,
    }, f, indent=2)

print("\nSaved model: models/churn_pipeline.joblib")
print("Saved metrics: reports/metrics.json")
print("Saved configuration: reports/model_config.json")
