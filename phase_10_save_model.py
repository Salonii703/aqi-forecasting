import joblib
from pathlib import Path

MODEL_DIR = Path("models")

model_path = MODEL_DIR / "xgb_aqi.pkl"
features_path = MODEL_DIR / "feature_cols.pkl"

if not model_path.exists() or not features_path.exists():
    raise FileNotFoundError(
        "Saved model not found. Run phase_7_models.py first."
    )

best_xgb = joblib.load(model_path)
feature_cols = joblib.load(features_path)

print("Model loaded successfully!")
print("Model type:", type(best_xgb).__name__)
print("Number of features:", len(feature_cols))
print("Features:", feature_cols)