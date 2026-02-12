from flask import Blueprint, request, jsonify, send_file
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
import threading
import joblib
import warnings

try:
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML libraries not available, using rule-based only")

warnings.filterwarnings("ignore")

pcod_bp = Blueprint("pcod", __name__)

MODEL_PATH = "pcod_xgb_model.pkl"
model = None

# =========================================
# 🔹 TRAIN / LOAD SUPERVISED ML MODEL
# =========================================
def train_or_load_model():
    global model
    
    if not ML_AVAILABLE:
        print("⚠️ Skipping ML model (libraries not available)")
        return

    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("✅ PCOD ML model loaded")
            return
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
            return

    print("⚡ Training PCOD ML model...")

    try:
        np.random.seed(42)
        data = []

        for _ in range(2000):
            age = np.random.randint(18, 45)
            bmi = np.random.uniform(18, 35)
            cycle_length = np.random.randint(21, 50)
            irregular_periods = np.random.randint(0, 101)
            acne = np.random.randint(0, 101)
            excessive_hair = np.random.randint(0, 101)
            hair_loss = np.random.randint(0, 101)
            weight_gain = np.random.randint(0, 2)
            fertility_issues = np.random.randint(0, 2)

            risk_score = (
                (bmi > 25) +
                (cycle_length > 35) +
                (irregular_periods > 60) +
                (acne > 60) +
                (excessive_hair > 60) +
                (weight_gain == 1)
            )

            label = 1 if risk_score >= 3 else 0

            data.append([
                age, bmi, cycle_length, irregular_periods,
                acne, excessive_hair, hair_loss,
                weight_gain, fertility_issues,
                label
            ])

        df = pd.DataFrame(data, columns=[
            "age", "bmi", "cycle_length", "irregular_periods",
            "acne", "excessive_hair", "hair_loss",
            "weight_gain", "fertility_issues",
            "label"
        ])

        X = df.drop("label", axis=1)
        y = df["label"]

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42
        )

        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)

        print("✅ PCOD ML model trained & saved")
    except Exception as e:
        print(f"⚠️ Model training failed: {e}")

threading.Thread(target=train_or_load_model, daemon=True).start()

# =========================================
# 🔹 RULE-BASED PREDICTION (ALWAYS WORKS)
# =========================================
def calculate_pcod_risk(payload):
    """Rule-based PCOD risk calculation - always available"""
    
    print("\n" + "="*70)
    print("🔍 PCOD RISK CALCULATION")
    print("="*70)
    print(f"Input payload keys: {list(payload.keys())}")
    
    # Extract symptoms from nested structure OR top level
    symptoms = payload.get("symptoms", {})
    
    # Helper function to get value from either nested or top level
    def get_symptom(key, default=0):
        # Try nested symptoms first
        if symptoms and key in symptoms:
            return int(symptoms[key])
        # Try top level
        if key in payload:
            val = payload[key]
            return int(val) if val not in [None, '', True, False] else (100 if val is True else 0)
        return default
    
    # Extract all values with defaults
    age = int(payload.get("age", 25))
    bmi = float(payload.get("bmi", 22))
    cycle_length = int(payload.get("cycle_length", 28))
    bleeding_days = int(payload.get("bleeding_days", 5))
    cycle_regularity = payload.get("cycle_regularity", "regular")
    
    # Symptoms
    irregular_periods = get_symptom("irregular_periods", 0)
    acne = get_symptom("acne", 0)
    excessive_hair = get_symptom("excessive_hair", 0)
    hair_loss = get_symptom("hair_loss", 0)
    dark_skin_patches = get_symptom("dark_skin_patches", 0)
    mood_swings = get_symptom("mood_swings", 0)
    fatigue = get_symptom("fatigue", 0)
    pelvic_pain = get_symptom("pelvic_pain", 0)
    
    # Boolean risk factors
    weight_gain = payload.get("weight_gain", False)
    difficulty_losing_weight = payload.get("difficulty_losing_weight", False)
    fertility_issues = payload.get("fertility_issues", False)
    pcos_family_history = payload.get("pcos_family_history", False)
    
    print(f"\n📊 Extracted values:")
    print(f"   Age: {age}, BMI: {bmi}")
    print(f"   Cycle: {cycle_regularity}, length={cycle_length} days")
    print(f"   Symptoms: irregular={irregular_periods}%, acne={acne}%, hair={excessive_hair}%")
    print(f"   Risk factors: weight_gain={weight_gain}, fertility={fertility_issues}")

    # Calculate PCOD risk score (0-100)
    score = 0
    risk_factors = []
    
    # 1. BMI scoring (25 points max)
    if bmi < 16:
        score += 20
        risk_factors.append(f"Severely underweight (BMI: {bmi:.1f})")
    elif bmi < 18.5:
        score += 12
        risk_factors.append(f"Underweight (BMI: {bmi:.1f})")
    elif bmi >= 30:
        score += 25
        risk_factors.append(f"Obesity (BMI: {bmi:.1f}) - major PCOD risk")
    elif bmi >= 25:
        score += 18
        risk_factors.append(f"Overweight (BMI: {bmi:.1f}) - increased PCOD risk")
    
    # 2. Cycle regularity (25 points max)
    if cycle_regularity in ["very_irregular", "highly_irregular"]:
        score += 25
        risk_factors.append("Very irregular menstrual cycles")
    elif cycle_regularity in ["often_irregular", "irregular"]:
        score += 20
        risk_factors.append("Irregular menstrual cycles")
    
    if cycle_length > 35:
        score += 10
        risk_factors.append(f"Long cycle length ({cycle_length} days)")
    elif cycle_length < 21:
        score += 8
        risk_factors.append(f"Short cycle length ({cycle_length} days)")
    
    # 3. Symptoms severity (35 points max)
    score += (irregular_periods / 100) * 15
    if irregular_periods > 70:
        risk_factors.append(f"Severe menstrual irregularity ({irregular_periods}%)")
    
    score += (acne / 100) * 10
    if acne > 60:
        risk_factors.append(f"Persistent acne ({acne}%)")
    
    score += (excessive_hair / 100) * 8
    if excessive_hair > 60:
        risk_factors.append(f"Excessive hair growth/hirsutism ({excessive_hair}%)")
    
    score += (pelvic_pain / 100) * 5
    if pelvic_pain > 60:
        risk_factors.append(f"Chronic pelvic pain ({pelvic_pain}%)")
    
    score += (dark_skin_patches / 100) * 3
    if dark_skin_patches > 60:
        risk_factors.append(f"Dark skin patches/acanthosis nigricans ({dark_skin_patches}%)")
    
    # 4. Additional risk factors (15 points max)
    if weight_gain:
        score += 5
        risk_factors.append("Recent unexplained weight gain")
    
    if difficulty_losing_weight:
        score += 5
        risk_factors.append("Difficulty losing weight - possible insulin resistance")
    
    if fertility_issues:
        score += 8
        risk_factors.append("Fertility issues - possible anovulation")
    
    if pcos_family_history:
        score += 7
        risk_factors.append("Family history of PCOS - genetic predisposition")
    
    # Cap at 100
    pcod_score = min(round(score, 1), 100)
    
    # Determine risk level
    if pcod_score >= 75:
        level = "Critical"
    elif pcod_score >= 50:
        level = "High"
    elif pcod_score >= 25:
        level = "Moderate"
    else:
        level = "Low"
    
    print(f"\n📊 FINAL CALCULATION:")
    print(f"   PCOD Score: {pcod_score}% ({level})")
    print(f"   Risk factors identified: {len(risk_factors)}")
    print("="*70 + "\n")

    # Generate recommendations
    recommendations = []
    if level in ["Critical", "High"]:
        recommendations.append("🏥 URGENT: Consult a gynecologist immediately")
        recommendations.append("🔬 Hormonal panel testing (FSH, LH, Testosterone, Insulin)")
        recommendations.append("🩺 Pelvic ultrasound to check for ovarian cysts")
        if bmi >= 25:
            recommendations.append("⚖️ Weight management program recommended")
        recommendations.append("💊 Discuss treatment options with your doctor")
    elif level == "Moderate":
        recommendations.append("👩‍⚕️ Schedule gynecologist appointment within 2 weeks")
        recommendations.append("🥗 Focus on balanced diet (low glycemic index foods)")
        recommendations.append("💪 Regular exercise (150 min/week moderate activity)")
        if bmi >= 23:
            recommendations.append("⚖️ Consider weight management")
        recommendations.append("📊 Monitor menstrual cycle patterns")
    else:
        recommendations.append("✅ Continue healthy lifestyle practices")
        recommendations.append("🥗 Maintain balanced, nutritious diet")
        recommendations.append("💪 Regular physical activity (30 min daily)")
        recommendations.append("📅 Annual gynecological checkups")
        recommendations.append("😴 Ensure adequate sleep (7-8 hours)")

    return {
        "risk_score": pcod_score,
        "pcod_risk_score": pcod_score,
        "risk_level": level,
        "pcod_status": "High Risk" if pcod_score >= 50 else "Moderate Risk" if pcod_score >= 25 else "Low Risk",
        "risk_factors": risk_factors if risk_factors else ["No significant risk factors identified"],
        "recommendations": recommendations,
        "medical_note": f"Rule-based PCOD screening showing {level} risk (Score: {pcod_score}%). This is a screening tool, not a medical diagnosis. Please consult a qualified gynecologist for proper evaluation and treatment.",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# =========================================
# ✅ SINGLE CHECK API
# =========================================
@pcod_bp.route("/check", methods=["POST", "OPTIONS"])
def check_pcod():
    """PCOD risk check endpoint"""
    
    # Handle CORS preflight
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    try:
        print("\n" + "="*80)
        print("🌸 PCOD CHECK ENDPOINT CALLED")
        print("="*80)
        
        payload = request.json
        if not payload:
            print("❌ No payload received")
            return jsonify({"error": "No data provided"}), 400
        
        print(f"📥 Received payload with {len(payload)} fields")
        
        # Use rule-based calculation (always works)
        result = calculate_pcod_risk(payload)
        
        print(f"✅ Returning result: {result['risk_level']} risk, score={result['risk_score']}%")
        print("="*80 + "\n")
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"\n❌ ERROR in check_pcod:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n")
        
        return jsonify({
            "error": "Failed to process PCOD check",
            "message": str(e)
        }), 500

# =========================================
# ✅ CSV BULK CHECK
# =========================================
@pcod_bp.route("/check_csv", methods=["POST"])
def check_pcod_csv():
    try:
        if "file" not in request.files:
            return jsonify({"error": "CSV file missing"}), 400

        df = pd.read_csv(request.files["file"])

        required = ["age", "bmi"]
        for col in required:
            if col not in df.columns:
                return jsonify({"error": f"Missing column: {col}"}), 400

        results = []
        for _, row in df.iterrows():
            results.append(calculate_pcod_risk(row.to_dict()))

        df["PCOD_Status"] = [r["pcod_status"] for r in results]
        df["RiskLevel"] = [r["risk_level"] for r in results]
        df["RiskScore"] = [r["risk_score"] for r in results]

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(tmp.name, index=False)

        return send_file(
            tmp.name,
            as_attachment=True,
            download_name="pcod_analysis_results.csv"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@pcod_bp.route("/test", methods=["GET"])
def test_route():
    """Test endpoint"""
    return jsonify({
        "success": True,
        "message": "PCOD check with bulletproof validation",
        "version": "3.0 - Production ready",
        "ml_available": ML_AVAILABLE,
        "model_loaded": model is not None if ML_AVAILABLE else False,
    }), 200