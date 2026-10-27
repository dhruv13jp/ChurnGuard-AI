
from pathlib import Path
import json

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT_DIR / "ml" / "models" / "churn_pipeline.joblib"
CONFIG_PATH = ROOT_DIR / "ml" / "reports" / "model_config.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. Run ml/train.py first."
    )

model = joblib.load(MODEL_PATH)

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)
    THRESHOLD = float(config["decision_threshold"])
else:
    THRESHOLD = 0.5


def predict_customer(customer: dict) -> dict:
    # The training pipeline expects the original dataset columns.
    customer_df = pd.DataFrame([customer])

    expected_columns = list(
        model.named_steps["preprocessor"].feature_names_in_
    )
    customer_df = customer_df[expected_columns]

    probability = float(model.predict_proba(customer_df)[0, 1])
    prediction = int(probability >= THRESHOLD)

    if probability >= 0.70:
        risk = "High"
    elif probability >= 0.35:
        risk = "Medium"
    else:
        risk = "Low"

    # These are transparent business heuristics, not SHAP explanations.
    factors = []

    if customer["Contract"] == "Month-to-month":
        factors.append("Month-to-month contract")

    if customer["tenure"] <= 12:
        factors.append("Short customer tenure")

    if customer["PaymentMethod"] == "Electronic check":
        factors.append("Electronic check payment method")

    if customer["TechSupport"] == "No":
        factors.append("No technical support")

    if customer["MonthlyCharges"] >= 80:
        factors.append("High monthly charges")

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_level": risk,
        "decision_threshold": THRESHOLD,
        "risk_factors_to_review": factors,
    }
