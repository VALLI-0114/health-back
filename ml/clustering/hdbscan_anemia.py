from flask import Blueprint, request, jsonify, send_file
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
import threading
import joblib
import warnings

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

anemia_bp = Blueprint("anemia", __name__)

MODEL_PATH = "anaemia_xgb_model.pkl"
model = None

# -------------------------------
# Train / Load ML Model
# -------------------------------
def train_or_load_model():
    global model

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Anaemia ML model loaded")
        return

    print("⚡ Training Anaemia ML model...")

    np.random.seed(42)
    data = []

    for _ in range(1500):
        age = np.random.randint(5, 25)
        bmi = np.random.uniform(14, 30)
        hb = np.random.uniform(6, 15)

        tiredness = np.random.randint(0, 101)
        weakness = np.random.randint(0, 101)
        dizziness = np.random.randint(0, 101)
        breathlessness = np.random.randint(0, 101)

        risk_score = (
            (hb < 10) +
            (bmi < 18.5) +
            (tiredness > 60) +
            (weakness > 60) +
            (dizziness > 50)
        )

        label = 1 if risk_score >= 2 else 0

        data.append([
            age, bmi, hb,
            tiredness, weakness, dizziness, breathlessness,
            label
        ])

    df = pd.DataFrame(data, columns=[
        "age", "bmi", "hemoglobin",
        "tiredness", "weakness", "dizziness", "breathlessness",
        "label"
    ])

    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)

    print("✅ Anaemia ML model trained & saved")

threading.Thread(target=train_or_load_model, daemon=True).start()

# -------------------------------
# ML Prediction Core
# -------------------------------
def ml_predict(payload):
    global model
    while model is None:
        pass

    symptoms = payload.get("symptoms", {})

    features = [
        payload.get("age", 0),
        payload.get("bmi", 0),
        payload.get("hemoglobin", 0),
        symptoms.get("tiredness", 0),
        symptoms.get("weakness", 0),
        symptoms.get("dizziness", 0),
        symptoms.get("breathlessness", 0),
    ]

    proba = float(model.predict_proba([features])[0][1])

    if proba < 0.3:
        level = "Low"
    elif proba < 0.6:
        level = "Moderate"
    else:
        level = "High"

    return {
        "risk_score": round(proba * 100),
        "risk_level": level,
        "anaemia_status": "Detected" if proba >= 0.5 else "Normal",
        "hemoglobin": payload.get("hemoglobin"),
        "risk_factors": [
            k for k, v in symptoms.items() if v >= 60
        ],
        "recommendations": (
            ["Maintain iron-rich diet"]
            if level == "Low"
            else ["Iron supplements", "Doctor consultation"]
            if level == "Moderate"
            else ["Immediate medical evaluation"]
        ),
        "medical_note": "ML-based screening, not a medical diagnosis",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# ===============================
# ✅ API USED BY YOUR REACT UI
# ===============================
@anemia_bp.route("/check", methods=["POST"])
def check_anemia():
    try:
        payload = request.json
        if not payload:
            return jsonify({"error": "Invalid input"}), 400
        return jsonify(ml_predict(payload))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===============================
# ✅ CSV BULK CHECK (LAKHS)
# ===============================
@anemia_bp.route("/check_csv", methods=["POST"])
def check_anemia_csv():
    try:
        if "file" not in request.files:
            return jsonify({"error": "CSV file missing"}), 400

        df = pd.read_csv(request.files["file"])

        required = ["age", "bmi", "hemoglobin"]
        for col in required:
            if col not in df.columns:
                return jsonify({"error": f"Missing column: {col}"}), 400

        results = []
        for _, row in df.iterrows():
            payload = row.to_dict()
            payload["symptoms"] = {
                "tiredness": row.get("tiredness", 0),
                "weakness": row.get("weakness", 0),
                "dizziness": row.get("dizziness", 0),
                "breathlessness": row.get("breathlessness", 0),
            }
            results.append(ml_predict(payload))

        df["RiskLevel"] = [r["risk_level"] for r in results]
        df["RiskScore"] = [r["risk_score"] for r in results]
        df["AnaemiaStatus"] = [r["anaemia_status"] for r in results]

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(tmp.name, index=False)

        return send_file(
            tmp.name,
            as_attachment=True,
            download_name="anaemia_ml_results.csv"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
