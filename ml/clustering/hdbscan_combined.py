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

combined_bp = Blueprint("combined", __name__)

MODEL_PATH = "combined_xgb_model.pkl"
model = None

# =========================================
# 🔹 TRAIN / LOAD SUPERVISED MODEL
# =========================================
def train_or_load_model():
    global model

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Combined Health ML model loaded")
        return

    print("⚡ Training Combined Health ML model...")

    np.random.seed(42)
    data = []

    for _ in range(3000):
        age = np.random.randint(18, 50)
        height = np.random.uniform(150, 180)
        weight = np.random.uniform(45, 90)
        bmi = weight / ((height / 100) ** 2)
        hb = np.random.uniform(7, 15)

        cycle_length = np.random.randint(21, 50)
        bleeding_days = np.random.randint(3, 10)

        tiredness = np.random.randint(0, 101)
        dizziness = np.random.randint(0, 101)
        hairfall = np.random.randint(0, 101)
        irregular_periods = np.random.randint(0, 101)
        acne = np.random.randint(0, 101)
        pelvic_pain = np.random.randint(0, 101)

        heavy_periods = np.random.randint(0, 2)
        poor_diet = np.random.randint(0, 2)
        sleep_hours = np.random.uniform(5, 9)

        # -------- SUPERVISED LABEL --------
        risk_score = (
            (hb < 10) +
            (bmi > 25) +
            (irregular_periods > 60) +
            (acne > 60) +
            (tiredness > 60) +
            (heavy_periods == 1)
        )

        label = 1 if risk_score >= 3 else 0  # 1 = High combined risk

        data.append([
            age, height, weight, bmi, hb,
            cycle_length, bleeding_days,
            tiredness, dizziness, hairfall,
            irregular_periods, acne, pelvic_pain,
            heavy_periods, poor_diet, "sleep_hours",
            label
        ])

    df = pd.DataFrame(data, columns=[
        "age", "height", "weight", "bmi", "hemoglobin",
        "cycle_length", "bleeding_days",
        "tiredness", "dizziness", "hairfall",
        "irregular_periods", "acne", "pelvic_pain",
        "heavy_periods", "poor_diet", "sleep_hours",
        "label"
    ])

    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)

    print("✅ Combined Health ML model trained & saved")

threading.Thread(target=train_or_load_model, daemon=True).start()


# =========================================
# 🔹 INDIVIDUAL SCORE CALCULATORS
# =========================================
def calculate_anaemia_score(payload):
    """Calculate individual anaemia score (0-100)"""
    score = 0
    hb = float(payload.get("hemoglobin", 12))
    bmi = float(payload.get("bmi", 0))
    
    # Hemoglobin scoring (40 points max)
    if hb < 7:
        score += 40
    elif hb < 10:
        score += 30
    elif hb < 12:
        score += 20
    elif hb > 16:
        score += 15
    
    # Symptoms (30 points max)
    tiredness = int(payload.get("tiredness", 0))
    dizziness = int(payload.get("dizziness", 0))
    hairfall = int(payload.get("hairfall", 0))
    
    score += (tiredness / 100) * 15
    score += (dizziness / 100) * 10
    score += (hairfall / 100) * 5
    
    # Lifestyle factors (20 points max)
    if payload.get("heavy_periods", False):
        score += 10
    if payload.get("poor_diet", False):
        score += 10
    
    # BMI consideration (10 points max)
    if bmi < 16:
        score += 15
    elif bmi < 18.5:
        score += 10
    
    return min(round(score, 1), 100)


def calculate_pcod_score(payload):
    """Calculate individual PCOD score (0-100)"""
    score = 0
    bmi = float(payload.get("bmi", 0))
    
    # BMI scoring (25 points max)
    if bmi < 16:
        score += 20
    elif bmi < 18.5:
        score += 12
    elif bmi >= 30:
        score += 25
    elif bmi >= 25:
        score += 18
    
    # Cycle regularity (25 points max)
    cycle_length = int(payload.get("cycle_length", 28))
    if cycle_length > 35:
        score += 10
    elif cycle_length < 21:
        score += 8
    
    # Symptoms (30 points max)
    irregular_periods = int(payload.get("irregular_periods", 0))
    acne = int(payload.get("acne", 0))
    pelvic_pain = int(payload.get("pelvic_pain", 0))
    
    score += (irregular_periods / 100) * 15
    score += (acne / 100) * 10
    score += (pelvic_pain / 100) * 5
    
    return min(round(score, 1), 100)


def get_risk_level(score):
    """Get risk level from score"""
    if score >= 75:
        return "Critical"
    elif score >= 50:
        return "High"
    elif score >= 25:
        return "Moderate"
    else:
        return "Low"


# =========================================
# 🔹 ML PREDICTION CORE (FIXED)
# =========================================
def ml_predict(payload):
    global model
    
    print("\n" + "="*70)
    print("🔍 ML PREDICT CALLED")
    print("="*70)
    print(f"Input: age={payload.get('age')}, hb={payload.get('hemoglobin')}, bmi={payload.get('bmi')}")
    
    # Wait for model to load
    while model is None:
        pass

    # Extract symptoms from nested structure if present
    if 'symptoms' in payload and isinstance(payload['symptoms'], dict):
        symptoms = payload['symptoms']
        payload['tiredness'] = symptoms.get('tiredness', 0)
        payload['dizziness'] = symptoms.get('dizziness', 0)
        payload['hairfall'] = symptoms.get('hairfall', 0)
        payload['irregular_periods'] = symptoms.get('irregular_periods', 0)
        payload['acne'] = symptoms.get('acne', 0)
        payload['pelvic_pain'] = symptoms.get('pelvic_pain', 0)
        print("✓ Extracted symptoms from nested structure")

    # Calculate individual scores using rule-based approach
    anaemia_score = calculate_anaemia_score(payload)
    pcod_score = calculate_pcod_score(payload)
    
    anaemia_risk = get_risk_level(anaemia_score)
    pcod_risk = get_risk_level(pcod_score)
    
    print(f"📊 Individual Scores:")
    print(f"   Anaemia: {anaemia_score}% ({anaemia_risk})")
    print(f"   PCOD: {pcod_score}% ({pcod_risk})")

    # Prepare features for ML model
    features = [
        float(payload.get("age", 0)),
        float(payload.get("height", 0)),
        float(payload.get("weight", 0)),
        float(payload.get("bmi", 0)),
        float(payload.get("hemoglobin", 0)),
        float(payload.get("cycle_length", 28)),
        float(payload.get("bleeding_days", 5)),
        float(payload.get("tiredness", 0)),
        float(payload.get("dizziness", 0)),
        float(payload.get("hairfall", 0)),
        float(payload.get("irregular_periods", 0)),
        float(payload.get("acne", 0)),
        float(payload.get("pelvic_pain", 0)),
        int(payload.get("heavy_periods", False)),
        int(payload.get("poor_diet", False)),
        float(payload.get("sleep_hours", 7)),
    ]

    # Get ML model prediction
    try:
        proba = float(model.predict_proba([features])[0][1])
        ml_score = round(proba * 100, 1)
        print(f"   ML Combined: {ml_score}%")
    except Exception as e:
        print(f"⚠️ ML prediction failed: {e}, using rule-based")
        ml_score = round((anaemia_score * 0.4) + (pcod_score * 0.6), 1)
    
    # Use weighted average of individual scores for combined
    combined_score = round((anaemia_score * 0.4) + (pcod_score * 0.6), 1)
    overall_risk = get_risk_level(combined_score)
    
    print(f"   Final Combined: {combined_score}% ({overall_risk})")
    print("="*70 + "\n")

    # Determine status
    if overall_risk == "Critical":
        status = "Critical Risk: Immediate Medical Attention Required"
    elif overall_risk == "High":
        status = "High Risk: Medical Consultation Recommended"
    elif overall_risk == "Moderate":
        status = "Moderate Risk: Monitoring & Lifestyle Changes Recommended"
    else:
        status = "Low Risk: Maintain Healthy Lifestyle"

    # Generate recommendations
    recommendations = []
    if overall_risk in ["Critical", "High"]:
        recommendations.append("🏥 URGENT: Consult a doctor immediately")
        if anaemia_score >= 50:
            recommendations.append("🩺 Hematologist visit for anemia management")
        if pcod_score >= 50:
            recommendations.append("👩‍⚕️ Gynecologist consultation for PCOD screening")
    elif overall_risk == "Moderate":
        recommendations.append("👨‍⚕️ Schedule medical checkup within 2 weeks")
        recommendations.append("🥗 Focus on balanced, nutritious diet")
    else:
        recommendations.append("✅ Continue healthy lifestyle practices")
        recommendations.append("📅 Annual health checkups recommended")

    # Build response with ALL required fields
    hb = float(payload.get("hemoglobin", 12))
    bmi = float(payload.get("bmi", 0))
    
    return {
        "success": True,
        
        # Combined scores (multiple field names for compatibility)
        "combined_score": combined_score,
        "combined_risk_score": combined_score,
        
        # Status fields
        "final_status": status,
        "combined_health_status": status,
        "overall_risk": overall_risk,
        "risk_level": overall_risk,
        
        # ✅ CRITICAL: Individual condition details with ACTUAL SCORES
        "anaemia": {
            "risk_score": anaemia_score,
            "risk_level": anaemia_risk,
            "hemoglobin_value": f"{hb} g/dL",
            "hemoglobin_status": "Low" if hb < 12 else "Normal" if hb <= 16 else "High",
        },
        
        "pcod": {
            "risk_score": pcod_score,
            "risk_level": pcod_risk,
            "bmi": bmi,
            "bmi_status": "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight",
        },
        
        # Other fields
        "recommendations": recommendations,
        "medical_note": f"ML-based screening showing {overall_risk} risk (Hemoglobin: {hb} g/dL, BMI: {bmi})",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# =========================================
# ✅ SINGLE CHECK API
# =========================================
@combined_bp.route("/check", methods=["POST"])
def check_combined():
    try:
        payload = request.json
        if not payload:
            return jsonify({"error": "Invalid input"}), 400
        
        result = ml_predict(payload)
        return jsonify(result)
    except Exception as e:
        print(f"❌ Error in check_combined: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# =========================================
# ✅ CSV BULK CHECK
# =========================================
@combined_bp.route("/check_csv", methods=["POST"])
def check_combined_csv():
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
            results.append(ml_predict(row.to_dict()))

        df["RiskLevel"] = [r["risk_level"] for r in results]
        df["RiskScore"] = [r["combined_risk_score"] for r in results]
        df["HealthStatus"] = [r["combined_health_status"] for r in results]
        df["AnaemiaScore"] = [r["anaemia"]["risk_score"] for r in results]
        df["PCODScore"] = [r["pcod"]["risk_score"] for r in results]

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(tmp.name, index=False)

        return send_file(
            tmp.name,
            as_attachment=True,
            download_name="combined_health_ml_results.csv"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@combined_bp.route("/test", methods=["GET"])
def test_route():
    """Test endpoint"""
    return jsonify({
        "success": True,
        "message": "ML-based combined check with individual scoring",
        "version": "6.0 - Fixed with anaemia/pcod scores",
        "model_loaded": model is not None,
    }), 200