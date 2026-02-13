"""
PCOD Routes - ML-BASED VERSION (Like Anemia)
✅ XGBoost ML Model
✅ Risk Factors Calculation
✅ Calibrated Probabilities
✅ Professional Implementation
✅ FIXED: Ensures all 3 classes in training data
✅ IMPROVED: Better synthetic data generation
✅ IMPROVED: Threshold-based risk mapping
✅ IMPROVED: Medical safety overrides
"""

from flask import Blueprint, request, jsonify, send_file
from datetime import datetime
import pandas as pd
import tempfile
import os
import joblib
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from extensions import db
from ml.models.pcod import PCODCheck
from ml.routes.auth import token_required

warnings.filterwarnings("ignore")

pcod_bp = Blueprint("pcod", __name__, url_prefix="/api/pcod")

# ======================================================
# 🔹 ML MODEL SETUP
# ======================================================
MODEL_PATH = "pcod_xgb_model.pkl"
pcod_model = None


def train_or_load_pcod_model():
    """Train or load the PCOD XGBoost model"""
    global pcod_model

    if os.path.exists(MODEL_PATH):
        try:
            pcod_model = joblib.load(MODEL_PATH)
            print("✅ PCOD ML model loaded")
            print(f"📊 Model classes: {pcod_model.classes_}")
            
            # ✅ CRITICAL FIX: Validate model has all 3 classes
            if not np.array_equal(pcod_model.classes_, np.array([0, 1, 2])):
                print("⚠️ WARNING: Model missing classes. Retraining...")
                # Delete corrupted model and retrain
                os.remove(MODEL_PATH)
                return train_or_load_pcod_model()
            
            return
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}. Retraining...")

    print("⚡ Training PCOD ML model...")

    # ======================================================
    # IMPROVED SYNTHETIC TRAINING DATA (MEDICALLY REALISTIC)
    # ======================================================

    np.random.seed(42)
    data = []

    for _ in range(4000):

        age = np.random.randint(15, 45)
        bmi = np.random.uniform(16, 40)
        cycle_length = np.random.randint(21, 60)
        bleeding_days = np.random.randint(2, 10)

        irregular_periods = np.random.randint(0, 101)
        excessive_hair = np.random.randint(0, 101)
        acne = np.random.randint(0, 101)
        weight_gain = np.random.randint(0, 101)
        mood_swings = np.random.randint(0, 101)

        # --------------------------------------------------
        # IMPROVED RISK LOGIC
        # --------------------------------------------------

        risk_score = 0

        # Cycle-based dominance
        if cycle_length > 40:
            risk_score += 2
        elif cycle_length > 35:
            risk_score += 1

        # BMI factor (both obese and lean PCOS)
        if bmi > 30:
            risk_score += 2
        elif bmi > 25:
            risk_score += 1
        elif bmi < 18.5 and cycle_length > 35:
            risk_score += 2  # Lean PCOS

        # Hormonal symptoms
        if acne > 70:
            risk_score += 1
        if excessive_hair > 70:
            risk_score += 1
        if weight_gain > 70:
            risk_score += 1

        # Final label
        if risk_score >= 4:
            label = 2  # High
        elif risk_score >= 2:
            label = 1  # Moderate
        else:
            label = 0  # Low

        data.append([
            age, bmi, cycle_length, bleeding_days,
            irregular_periods, excessive_hair,
            acne, weight_gain, mood_swings,
            label
        ])

    df = pd.DataFrame(data, columns=[
        "age", "bmi", "cycle_length", "bleeding_days",
        "irregular_periods", "excessive_hair",
        "acne", "weight_gain", "mood_swings",
        "label"
    ])
    
    # Verify all classes present
    class_counts = df['label'].value_counts().sort_index()
    print(f"📊 Training data class distribution:")
    print(f"   Low Risk (0): {class_counts.get(0, 0)} samples")
    print(f"   Moderate Risk (1): {class_counts.get(1, 0)} samples")
    print(f"   High Risk (2): {class_counts.get(2, 0)} samples")

    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # ✅ Stratified split
    )

    # Train XGBoost model
    pcod_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3,
        eval_metric="mlogloss",
        random_state=42
    )

    pcod_model.fit(X_train, y_train)
    
    print(f"📊 Trained model classes: {pcod_model.classes_}")
    print(f"📈 Feature importance: {pcod_model.feature_importances_}")
    
    # ✅ CRITICAL VALIDATION: Ensure model learned all 3 classes
    if not np.array_equal(pcod_model.classes_, np.array([0, 1, 2])):
        raise ValueError(f"❌ CRITICAL: Model only learned classes {pcod_model.classes_}. Expected [0, 1, 2]!")
    
    joblib.dump(pcod_model, MODEL_PATH)

    print("✅ PCOD ML model trained & saved with all 3 classes")


def get_pcod_recommendations(level):
    """Get recommendations based on ML prediction"""
    mapping = {
        "Low": [
            "Maintain healthy weight through balanced diet",
            "Regular exercise (30 minutes, 5 days/week)",
            "Annual gynecological checkup recommended",
            "Monitor menstrual cycle regularity"
        ],
        "Moderate": [
            "Consult gynecologist for comprehensive evaluation",
            "Consider hormonal panel blood tests",
            "Weight management program if BMI > 25",
            "Low glycemic index diet recommended",
            "Regular exercise routine essential",
            "Track menstrual cycles closely"
        ],
        "High": [
            "Immediate gynecological consultation required",
            "Ultrasound scan (pelvic) recommended",
            "Comprehensive hormonal panel testing",
            "Fasting insulin and glucose testing",
            "Medical management may be necessary",
            "Lifestyle modification program essential"
        ]
    }
    return mapping.get(level, [])


def get_pcod_status(cycle_length, bmi):
    """Get PCOD status classification"""
    high_risk_count = 0
    
    if bmi > 30:
        high_risk_count += 2
    elif bmi > 25:
        high_risk_count += 1
    
    if cycle_length > 40:
        high_risk_count += 2
    elif cycle_length > 35:
        high_risk_count += 1
    
    if high_risk_count >= 3:
        return "PCOD Likely - Medical Evaluation Required"
    elif high_risk_count >= 2:
        return "High Risk - Consultation Recommended"
    elif high_risk_count >= 1:
        return "Moderate Risk - Monitoring Required"
    else:
        return "Low Risk - Regular Screening Advised"


def get_pcod_risk_factors(age, bmi, cycle_length, bleeding_days, symptoms, 
                          family_history=False, irregular_periods=False, 
                          excessive_hair_growth=False, acne=False):
    """
    Calculate PCOD risk factors based on patient data
    
    Args:
        age: Patient age
        bmi: Body Mass Index
        cycle_length: Menstrual cycle length in days
        bleeding_days: Duration of bleeding in days
        symptoms: Dictionary of symptom severity scores (0-100)
        family_history: Family history of PCOD/diabetes
        irregular_periods: Self-reported irregular periods
        excessive_hair_growth: Hirsutism present
        acne: Acne/skin issues present
    
    Returns:
        List of risk factor strings
    """
    factors = []

    # ========== MENSTRUAL CYCLE FACTORS ==========
    if cycle_length > 35:
        factors.append("Prolonged menstrual cycle (oligomenorrhea)")
    elif cycle_length < 21:
        factors.append("Abnormally short menstrual cycle")
    
    if bleeding_days > 7:
        factors.append("Prolonged menstrual bleeding")
    elif bleeding_days < 2:
        factors.append("Very short bleeding duration")
    
    if irregular_periods:
        factors.append("Irregular menstrual periods")

    # ========== BMI-BASED FACTORS ==========
    if bmi > 30:
        factors.append("Obesity (BMI > 30)")
    elif bmi > 25:
        factors.append("Overweight (BMI 25-30)")
    elif bmi < 18.5:
        factors.append("Underweight (Low BMI)")

    # ========== AGE-BASED FACTORS ==========
    if 15 <= age <= 25:
        factors.append("Peak age range for PCOD onset (15-25)")
    elif age < 15:
        factors.append("Early onset (adolescent)")
    elif age > 40:
        factors.append("Age over 40 with PCOD symptoms")

    # ========== CLINICAL PRESENTATION ==========
    if family_history:
        factors.append("Family history of PCOD/diabetes")
    
    if excessive_hair_growth:
        factors.append("Excessive hair growth (hirsutism)")
    
    if acne:
        factors.append("Acne and skin issues")

    # ========== SYMPTOM-BASED FACTORS ==========
    weight_gain = symptoms.get("weight_gain", 0)
    if weight_gain > 60:
        factors.append("Significant weight gain")
    
    hair_loss = symptoms.get("hair_loss", 0)
    if hair_loss > 60:
        factors.append("Severe hair loss/thinning")
    
    facial_hair = symptoms.get("facial_hair", 0)
    if facial_hair > 60:
        factors.append("Excessive facial hair growth")
    
    acne_severity = symptoms.get("acne", 0)
    if acne_severity > 60:
        factors.append("Severe acne")
    
    mood_swings = symptoms.get("mood_swings", 0)
    if mood_swings > 70:
        factors.append("Severe mood swings")
    
    fatigue = symptoms.get("fatigue", 0)
    if fatigue > 70:
        factors.append("Chronic fatigue")
    
    skin_darkening = symptoms.get("skin_darkening", 0)
    if skin_darkening > 50:
        factors.append("Skin darkening (acanthosis nigricans)")
    
    pelvic_pain = symptoms.get("pelvic_pain", 0)
    if pelvic_pain > 60:
        factors.append("Chronic pelvic pain")

    # ========== COMBINED RISK FACTORS ==========
    if bmi > 30 and cycle_length > 35:
        factors.append("Combined obesity and irregular cycles")
    
    if bmi > 25 and (excessive_hair_growth or acne):
        factors.append("Metabolic and hyperandrogenism signs")
    
    if family_history and bmi > 25:
        factors.append("Genetic predisposition with elevated BMI")
    
    if cycle_length > 35 and (facial_hair > 60 or acne_severity > 60):
        factors.append("Irregular cycles with androgen excess")

    # ========== METABOLIC RISK ==========
    if bmi > 30 and age > 30:
        factors.append("Increased metabolic syndrome risk")
    
    if weight_gain > 60 and fatigue > 70:
        factors.append("Signs of insulin resistance")

    # Ensure we always return a list (never None or empty)
    if not factors:
        factors = ["No significant PCOD risk factors detected"]

    return factors


def apply_probability_calibration(probabilities, temperature=1.3):
    """
    Apply temperature scaling to soften overconfident probabilities
    
    Temperature > 1: Makes model less confident (softens probabilities)
    Temperature < 1: Makes model more confident (sharpens probabilities)
    Temperature = 1: No change
    
    Args:
        probabilities: Raw model probabilities [P(Low), P(Moderate), P(High)]
        temperature: Calibration parameter (default 1.3 for medical uncertainty)
    
    Returns:
        Calibrated probabilities that sum to 1.0
    """
    epsilon = 1e-9
    soft_logits = np.log(probabilities + epsilon) / temperature
    soft_probs = np.exp(soft_logits)
    soft_probs = soft_probs / np.sum(soft_probs)
    return soft_probs


# Load model SYNCHRONOUSLY at startup
print("🔄 Loading PCOD ML model at startup...")
train_or_load_pcod_model()
print("✅ PCOD ML model ready for predictions")


# ======================================================
# 🔹 SINGLE USER CHECK (ML-POWERED)
# ✅ USES AUTHENTICATED USER
# Endpoint: POST /api/pcod/check
# ======================================================
@pcod_bp.route("/check", methods=["POST"])
@token_required
def check_pcod(current_user):
    """
    Perform ML-based PCOD check and save to database
    
    ✅ NOW USES AUTHENTICATED USER - NO NEED TO SEND user_id
    ✅ NOW USES XGBOOST ML MODEL - NOT RULE-BASED
    ✅ FIXED: Cycle-length and BMI dominant predictions
    ✅ FIXED: Calibrated probabilities (realistic confidence levels)
    ✅ FIXED: Risk factors included in response
    ✅ FIXED: Ensures model has all 3 classes
    ✅ IMPROVED: Threshold-based risk mapping
    ✅ IMPROVED: Medical safety overrides
    
    Expected JSON payload:
    {
        "age": 22,
        "bmi": 28.5,
        "cycle_length": 40,
        "bleeding_days": 6,
        "family_history": true,
        "irregular_periods": true,
        "excessive_hair_growth": true,
        "acne": true,
        "symptoms": {
            "weight_gain": 70,
            "facial_hair": 60,
            "acne": 65,
            "mood_swings": 50,
            "fatigue": 40
        }
    }
    """
    try:
        # Ensure model is loaded before prediction
        if pcod_model is None:
            print("⚠️ Model not loaded! Attempting to load now...")
            train_or_load_pcod_model()
            if pcod_model is None:
                return jsonify({
                    "error": "ML model not available. Please try again later."
                }), 503

        data = request.json or {}
        
        user_id = current_user.id
        
        print(f"\n🔍 PCOD CHECK DEBUG (ML-POWERED):")
        print(f"   Authenticated user: {current_user.full_name} (ID: {user_id})")
        print(f"   Request data: {data}")
        
        # Validate required fields
        if not data.get("age"):
            return jsonify({"error": "age is required"}), 400
        
        if not data.get("bmi"):
            return jsonify({"error": "bmi is required"}), 400
        
        if not data.get("cycle_length"):
            return jsonify({"error": "cycle_length is required"}), 400
        
        if not data.get("bleeding_days"):
            return jsonify({"error": "bleeding_days is required"}), 400

        # ========== EXTRACT INPUT DATA ==========
        symptoms = data.get("symptoms", {})
        
        # Extract numeric values
        age = int(data.get("age"))
        bmi = float(data.get("bmi"))
        cycle_length = int(data.get("cycle_length"))
        bleeding_days = int(data.get("bleeding_days"))
        
        # Extract optional risk factors
        family_history = data.get("family_history", False)
        irregular_periods = data.get("irregular_periods", False)
        excessive_hair_growth = data.get("excessive_hair_growth", False)
        acne = data.get("acne", False)

        # Build feature vector for ML model
        feature_vector = np.array([[
            age,
            bmi,
            cycle_length,
            bleeding_days,
            int(symptoms.get("irregular_periods", 0)),
            int(symptoms.get("excessive_hair", 0)),
            int(symptoms.get("acne", 0)),
            int(symptoms.get("weight_gain", 0)),
            int(symptoms.get("mood_swings", 0))
        ]])

        print(f"📊 Feature vector: {feature_vector}")

        # ========== ML PREDICTION ==========
        prediction = pcod_model.predict(feature_vector)[0]
        raw_probabilities = pcod_model.predict_proba(feature_vector)[0]

        print(f"🤖 ML Prediction: {prediction}")
        print(f"📈 Raw Probabilities: {raw_probabilities}")
        print(f"📊 Model classes: {pcod_model.classes_}")

        # ✅ CRITICAL FIX: Handle models with only 2 classes
        if len(raw_probabilities) == 2:
            print("⚠️ WARNING: Model only has 2 classes. Retraining...")
            # Delete and retrain model
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            train_or_load_pcod_model()
            
            # Retry prediction with new model
            prediction = pcod_model.predict(feature_vector)[0]
            raw_probabilities = pcod_model.predict_proba(feature_vector)[0]
            print(f"📈 New Raw Probabilities: {raw_probabilities}")

        # Apply probability calibration (IMPROVED: temperature 1.3)
        calibrated_probabilities = apply_probability_calibration(raw_probabilities, temperature=1.3)
        
        print(f"🎯 Calibrated Probabilities (Low/Moderate/High): {calibrated_probabilities}")

        # ========== IMPROVED THRESHOLD-BASED RISK MAPPING ==========
        high_prob = calibrated_probabilities[2]
        moderate_prob = calibrated_probabilities[1]
        low_prob = calibrated_probabilities[0]

        if high_prob >= 0.55:
            risk_level = "High"
            risk_score = round(high_prob * 100, 2)

        elif moderate_prob >= 0.45:
            risk_level = "Moderate"
            risk_score = round(moderate_prob * 100, 2)

        else:
            risk_level = "Low"
            risk_score = round(low_prob * 100, 2)

        # ======================================================
        # MEDICAL SAFETY OVERRIDE (CRITICAL FOR PRODUCTION)
        # ======================================================

        if cycle_length > 35 and (
            symptoms.get("acne", 0) > 75 or
            symptoms.get("excessive_hair", 0) > 75 or
            symptoms.get("pelvic_pain", 0) > 75
        ):
            risk_level = "High"
            risk_score = max(risk_score, 70)
            print("⚠️ MEDICAL SAFETY OVERRIDE: Risk elevated to High")
        
        # Get PCOD status
        pcod_status = get_pcod_status(cycle_length, bmi)

        # ========== CALCULATE RISK FACTORS ==========
        risk_factors = get_pcod_risk_factors(
            age=age,
            bmi=bmi,
            cycle_length=cycle_length,
            bleeding_days=bleeding_days,
            symptoms=symptoms,
            family_history=family_history,
            irregular_periods=irregular_periods,
            excessive_hair_growth=excessive_hair_growth,
            acne=acne
        )

        print(f"✅ Risk Level: {risk_level}, Risk Score: {risk_score}%, Status: {pcod_status}")
        print(f"📋 Risk Factors ({len(risk_factors)}): {risk_factors[:3]}...")

        # ========== SAVE TO DATABASE ==========
        new_check = PCODCheck(
            user_id=user_id,
            age=age,
            bmi=bmi,
            cycle_length=cycle_length,
            bleeding_days=bleeding_days,
            symptoms=symptoms,
            result=pcod_status,
            risk_score=int(risk_score),
            risk_factors=risk_factors,
            created_at=datetime.utcnow()
        )
        
        db.session.add(new_check)
        db.session.commit()
        
        print(f"✅ Saved ML-powered PCOD check for user: {current_user.full_name} (ID: {new_check.id})")

        # ========== RETURN RESPONSE ==========
        return jsonify({
            "success": True,
            "check_id": new_check.id,
            "user_name": current_user.full_name,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "pcod_status": pcod_status,
            "age": age,
            "bmi": bmi,
            "cycle_length": cycle_length,
            "bleeding_days": bleeding_days,
            "symptoms": symptoms,
            "risk_factors": risk_factors,
            "model_confidence": {
                "low": round(float(calibrated_probabilities[0] * 100), 2),
                "moderate": round(float(calibrated_probabilities[1] * 100), 2),
                "high": round(float(calibrated_probabilities[2] * 100), 2)
            },
            "prediction_method": "XGBoost ML Model (Calibrated, Threshold-Based, Medical Override)",
            "recommendations": get_pcod_recommendations(risk_level),
            "medical_note": "ML-based screening using calibrated XGBoost. Not a medical diagnosis.",
            "timestamp": new_check.created_at.isoformat() + "Z"
        }), 201
        
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error in check_pcod: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to process PCOD check: {str(e)}"
        }), 500


# ======================================================
# 🔹 GET USER HISTORY
# Endpoint: GET /api/pcod/history
# ======================================================
@pcod_bp.route("/history", methods=["GET"])
@token_required
def get_pcod_history(current_user):
    """Get PCOD check history for authenticated user with full details including risk factors"""
    try:
        limit = request.args.get("limit", 10, type=int)
        
        checks = PCODCheck.query.filter_by(user_id=current_user.id)\
            .order_by(PCODCheck.created_at.desc())\
            .limit(limit)\
            .all()
        
        # Use to_dict_full() to include risk_factors
        history = [check.to_dict_full() for check in checks]
        
        return jsonify({
            "success": True,
            "total": len(history),
            "history": history
        }), 200
        
    except Exception as e:
        print(f"Error in get_pcod_history: {str(e)}")
        return jsonify({
            "error": f"Failed to retrieve history: {str(e)}"
        }), 500


# ======================================================
# 🔹 GET LATEST CHECK
# Endpoint: GET /api/pcod/latest
# ======================================================
@pcod_bp.route("/latest", methods=["GET"])
@token_required
def get_latest_check(current_user):
    """Get the most recent PCOD check for authenticated user with full details"""
    try:
        latest_check = PCODCheck.query.filter_by(user_id=current_user.id)\
            .order_by(PCODCheck.created_at.desc())\
            .first()
        
        if latest_check:
            return jsonify({
                "success": True,
                "latest_check": latest_check.to_dict_full()
            }), 200
        else:
            return jsonify({
                "success": True,
                "latest_check": None,
                "message": "No PCOD checks found for this user"
            }), 200
            
    except Exception as e:
        print(f"Error in get_latest_check: {str(e)}")
        return jsonify({
            "error": f"Failed to retrieve latest check: {str(e)}"
        }), 500


# ======================================================
# 🔹 GET USER STATISTICS
# Endpoint: GET /api/pcod/stats
# ======================================================
@pcod_bp.route("/stats", methods=["GET"])
@token_required
def get_user_stats(current_user):
    """Get PCOD statistics for authenticated user"""
    try:
        checks = PCODCheck.query.filter_by(user_id=current_user.id)\
            .order_by(PCODCheck.created_at.desc())\
            .limit(100)\
            .all()
        
        if not checks:
            return jsonify({
                "success": True,
                "stats": {
                    "total_checks": 0,
                    "average_risk_score": None,
                    "average_bmi": None,
                    "average_cycle_length": None,
                    "trend": None,
                    "latest_result": None,
                    "latest_risk_level": None
                }
            }), 200
        
        # Calculate statistics
        total_checks = len(checks)
        avg_risk = sum(c.risk_score for c in checks if c.risk_score) / total_checks
        avg_bmi = sum(c.bmi for c in checks if c.bmi) / total_checks
        avg_cycle = sum(c.cycle_length for c in checks if c.cycle_length) / total_checks
        
        # Determine trend based on cycle length
        trend = "stable"
        if total_checks >= 6:
            recent_avg = sum(c.cycle_length for c in checks[:3] if c.cycle_length) / 3
            previous_avg = sum(c.cycle_length for c in checks[3:6] if c.cycle_length) / 3
            
            if recent_avg < previous_avg - 3:
                trend = "improving"
            elif recent_avg > previous_avg + 3:
                trend = "worsening"
        
        return jsonify({
            "success": True,
            "stats": {
                "total_checks": total_checks,
                "average_risk_score": round(avg_risk, 2),
                "average_bmi": round(avg_bmi, 2),
                "average_cycle_length": round(avg_cycle, 1),
                "trend": trend,
                "latest_result": checks[0].result,
                "latest_risk_level": checks[0].result,
                "first_check_date": checks[-1].created_at.isoformat() + "Z",
                "last_check_date": checks[0].created_at.isoformat() + "Z"
            }
        }), 200
        
    except Exception as e:
        print(f"Error in get_user_stats: {str(e)}")
        return jsonify({
            "error": f"Failed to retrieve statistics: {str(e)}"
        }), 500


# ======================================================
# 🔹 DELETE CHECK
# Endpoint: DELETE /api/pcod/check/<check_id>
# ======================================================
@pcod_bp.route("/check/<int:check_id>", methods=["DELETE"])
@token_required
def delete_check(current_user, check_id):
    """Delete a specific PCOD check (only own checks)"""
    try:
        check = PCODCheck.query.get(check_id)
        
        if not check:
            return jsonify({
                "error": "Check not found"
            }), 404
        
        # Security: Only allow users to delete their own checks
        if check.user_id != current_user.id:
            return jsonify({
                "error": "Unauthorized to delete this check"
            }), 403
        
        db.session.delete(check)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Check deleted successfully"
        }), 200
        
    except Exception as e:
        db.session.rollback()
        print(f"Error in delete_check: {str(e)}")
        return jsonify({
            "error": f"Failed to delete check: {str(e)}"
        }), 500


# ======================================================
# 🔹 CSV BULK CHECK (ML-POWERED)
# Endpoint: POST /api/pcod/check_csv
# ======================================================
@pcod_bp.route("/check_csv", methods=["POST"])
def check_pcod_csv():
    """
    Bulk PCOD check from CSV file using ML model
    Optionally save results to database if user_id is provided
    """
    if "file" not in request.files:
        return jsonify({"error": "CSV file missing"}), 400

    df = pd.read_csv(request.files["file"])

    # Required base columns
    required_cols = ["age", "bmi", "cycle_length", "bleeding_days"]
    for col in required_cols:
        if col not in df.columns:
            return jsonify({"error": f"Missing column: {col}"}), 400

    risk_scores = []
    risk_levels = []
    statuses = []
    
    # Check if user_id column exists for database save
    save_to_db = "user_id" in df.columns

    for _, row in df.iterrows():
        # Build feature vector
        age = int(row.get("age", 25))
        bmi = float(row.get("bmi", 22))
        cycle_length = int(row.get("cycle_length", 28))
        bleeding_days = int(row.get("bleeding_days", 5))
        
        feature_vector = np.array([[
            age,
            bmi,
            cycle_length,
            bleeding_days,
            int(row.get("irregular_periods", 0)),
            int(row.get("excessive_hair", 0)),
            int(row.get("acne", 0)),
            int(row.get("weight_gain", 0)),
            int(row.get("mood_swings", 0))
        ]])

        # ML prediction
        prediction = pcod_model.predict(feature_vector)[0]
        raw_probabilities = pcod_model.predict_proba(feature_vector)[0]
        
        # Apply calibration (IMPROVED: temperature 1.3)
        calibrated_probabilities = apply_probability_calibration(raw_probabilities, temperature=1.3)
        
        # Use threshold-based mapping
        high_prob = calibrated_probabilities[2]
        moderate_prob = calibrated_probabilities[1]
        low_prob = calibrated_probabilities[0]

        if high_prob >= 0.55:
            risk_level = "High"
            risk_score = round(high_prob * 100, 2)
        elif moderate_prob >= 0.45:
            risk_level = "Moderate"
            risk_score = round(moderate_prob * 100, 2)
        else:
            risk_level = "Low"
            risk_score = round(low_prob * 100, 2)
        
        status = get_pcod_status(cycle_length, bmi)
        
        risk_scores.append(risk_score)
        risk_levels.append(risk_level)
        statuses.append(status)
        
        # Save to database if user_id is present
        if save_to_db and pd.notna(row.get("user_id")):
            try:
                symptoms = {
                    "irregular_periods": int(row.get("irregular_periods", 0)),
                    "excessive_hair": int(row.get("excessive_hair", 0)),
                    "acne": int(row.get("acne", 0)),
                    "weight_gain": int(row.get("weight_gain", 0)),
                    "mood_swings": int(row.get("mood_swings", 0)),
                }
                
                new_check = PCODCheck(
                    user_id=int(row["user_id"]),
                    age=age,
                    bmi=bmi,
                    cycle_length=cycle_length,
                    bleeding_days=bleeding_days,
                    symptoms=symptoms,
                    result=status,
                    risk_score=int(risk_score),
                    created_at=datetime.utcnow()
                )
                db.session.add(new_check)
            except Exception as e:
                print(f"Error saving row to database: {str(e)}")
                continue
    
    # Commit all database saves
    if save_to_db:
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"Error committing to database: {str(e)}")

    # Append results to CSV
    df["RiskScore"] = risk_scores
    df["RiskLevel"] = risk_levels
    df["PCODStatus"] = statuses

    # Save and return CSV
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)

    return send_file(
        tmp.name,
        as_attachment=True,
        download_name="pcod_ml_results.csv",
        mimetype="text/csv"
    )


# ======================================================
# 🔹 GET ALL CHECKS (ADMIN/ANALYTICS)
# Endpoint: GET /api/pcod/all
# ======================================================
@pcod_bp.route("/all", methods=["GET"])
def get_all_checks():
    """Get all PCOD checks with optional filtering"""
    try:
        # Optional filters
        risk_level = request.args.get("risk_level")
        result = request.args.get("result")
        limit = request.args.get("limit", 100, type=int)
        offset = request.args.get("offset", 0, type=int)
        
        query = PCODCheck.query
        
        if risk_level:
            query = query.filter_by(result=risk_level)
        
        if result:
            query = query.filter_by(result=result)
        
        total_count = query.count()
        
        checks = query.order_by(PCODCheck.created_at.desc())\
            .limit(limit)\
            .offset(offset)\
            .all()
        
        data = [check.to_dict() for check in checks]
        
        return jsonify({
            "success": True,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "data": data
        }), 200
        
    except Exception as e:
        print(f"Error in get_all_checks: {str(e)}")
        return jsonify({
            "error": f"Failed to retrieve checks: {str(e)}"
        }), 500
