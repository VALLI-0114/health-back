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
from ml.models.anaemia import AnaemiaCheck
from ml.routes.auth import token_required

warnings.filterwarnings("ignore")

anemia_bp = Blueprint("anaemia", __name__, url_prefix="/api/anaemia")

# ======================================================
# 🔹 ML MODEL SETUP
# ======================================================
MODEL_PATH = "anaemia_xgb_model.pkl"
anaemia_model = None


def train_or_load_anaemia_model():
    """Train or load the Anaemia XGBoost model"""
    global anaemia_model

    if os.path.exists(MODEL_PATH):
        try:
            anaemia_model = joblib.load(MODEL_PATH)
            print("✅ Anaemia ML model loaded")
            print(f"📊 Model classes: {anaemia_model.classes_}")
            return
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}. Retraining...")
            # Continue to training if load fails

    print("⚡ Training Anaemia ML model...")

    np.random.seed(42)
    data = []

    # Generate synthetic training data
    for _ in range(3000):
        age = np.random.randint(15, 60)
        bmi = np.random.uniform(16, 35)
        hb = np.random.uniform(6, 16)
        tiredness = np.random.randint(0, 101)
        weakness = np.random.randint(0, 101)
        dizziness = np.random.randint(0, 101)
        breathlessness = np.random.randint(0, 101)

        # Make hemoglobin the PRIMARY factor
        if hb < 7:
            label = 2  # High Risk - Severe anemia
        elif hb < 10:
            label = 1  # Moderate Risk
        elif hb < 12:
            label = 1  # Moderate Risk - Mild anemia
        else:
            label = 0  # Low Risk - Normal

        # Increase severity if critical symptoms present
        if hb < 10 and (tiredness > 70 or dizziness > 70 or breathlessness > 70):
            label = 2  # Upgrade to High Risk
        
        # Underweight + low Hb is serious
        if bmi < 18.5 and hb < 10:
            label = 2  # High Risk

        # 🔥 FIX: Add 8% label noise for realism (prevents overconfidence)
        if np.random.rand() < 0.08:
            label = np.random.choice([0, 1, 2])

        data.append([
            age, bmi, hb,
            tiredness, weakness,
            dizziness, breathlessness,
            label
        ])

    df = pd.DataFrame(data, columns=[
        "age", "bmi", "hemoglobin",
        "tiredness", "weakness",
        "dizziness", "breathlessness",
        "label"
    ])

    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔥 FIX: Reduced model complexity to prevent overconfidence
    anaemia_model = XGBClassifier(
        n_estimators=100,        # ↓ Fewer trees (was 300)
        max_depth=3,             # ↓ Shallower trees (was 6)
        learning_rate=0.1,       # ↑ Faster learning (was 0.05)
        subsample=0.8,           # ↓ Sample 80% of data per tree
        colsample_bytree=0.8,    # ↓ Sample 80% of features per tree
        reg_lambda=3,            # ↑ L2 regularization to prevent overfitting
        eval_metric="mlogloss",
        random_state=42
    )

    anaemia_model.fit(X_train, y_train)
    
    # Verify model classes
    print(f"📊 Model classes: {anaemia_model.classes_}")
    if not np.array_equal(anaemia_model.classes_, np.array([0, 1, 2])):
        print("⚠️ WARNING: Model classes are not [0, 1, 2]. Predictions may be incorrect!")
    
    joblib.dump(anaemia_model, MODEL_PATH)

    print("✅ Anaemia ML model trained & saved")


def get_ml_recommendations(level):
    """Get recommendations based on ML prediction"""
    mapping = {
        "Low": [
            "Maintain balanced diet",
            "Regular health checkups",
            "Include iron-rich foods in diet"
        ],
        "Moderate": [
            "Iron-rich foods (spinach, red meat, legumes)",
            "Doctor consultation recommended",
            "Consider iron supplements",
            "Retest in 4-6 weeks"
        ],
        "High": [
            "Immediate medical evaluation required",
            "Iron supplementation required",
            "Complete blood count (CBC) test recommended",
            "Diet rich in iron, vitamin B12, and folic acid"
        ]
    }
    return mapping.get(level, [])


def get_hemoglobin_status(hb):
    """Get hemoglobin status classification"""
    if hb < 7:
        return "Severe Anaemia"
    elif hb < 10:
        return "Moderate Anaemia"
    elif hb < 12:
        return "Mild Anaemia"
    return "Normal"


def get_risk_factors(age, bmi, hb, symptoms, heavy_periods=False, poor_diet=False):
    """
    Calculate risk factors based on patient data
    
    Args:
        age: Patient age
        bmi: Body Mass Index
        hb: Hemoglobin level
        symptoms: Dictionary of symptom severity scores (0-100)
        heavy_periods: Whether patient has heavy menstrual bleeding
        poor_diet: Whether patient has iron-poor diet
    
    Returns:
        List of risk factor strings
    """
    factors = []

    # Hemoglobin-based factors
    if hb < 7:
        factors.append("Severely low hemoglobin level")
    elif hb < 10:
        factors.append("Moderately low hemoglobin level")
    elif hb < 12:
        factors.append("Low hemoglobin level")

    # BMI-based factors
    if bmi < 18.5:
        factors.append("Underweight (Low BMI)")
    elif bmi > 30:
        factors.append("Obesity (High BMI)")

    # Age-based factors
    if age < 18:
        factors.append("Adolescent (increased iron needs)")
    elif age > 50:
        factors.append("Age over 50 (increased risk)")

    # Menstrual factors
    if heavy_periods:
        factors.append("Heavy menstrual bleeding")

    # Dietary factors
    if poor_diet:
        factors.append("Iron-poor diet")

    # Symptom-based factors
    if symptoms.get("tiredness", 0) > 60:
        factors.append("Severe tiredness")
    
    if symptoms.get("weakness", 0) > 60:
        factors.append("Severe weakness")
    
    if symptoms.get("dizziness", 0) > 60:
        factors.append("Frequent dizziness")
    
    if symptoms.get("breathlessness", 0) > 60:
        factors.append("Breathlessness")

    # Combined risk factors
    if bmi < 18.5 and hb < 10:
        factors.append("Combined low weight and low hemoglobin")
    
    if hb < 10 and (symptoms.get("tiredness", 0) > 70 or symptoms.get("dizziness", 0) > 70):
        factors.append("Low hemoglobin with severe symptoms")

    # Ensure we always return a list (never None or empty)
    if not factors:
        factors = ["No significant clinical risk factors detected"]

    return factors


def apply_probability_calibration(probabilities, temperature=1.8):
    """
    Apply temperature scaling to soften overconfident probabilities
    
    Temperature > 1: Makes model less confident (softens probabilities)
    Temperature < 1: Makes model more confident (sharpens probabilities)
    Temperature = 1: No change
    
    Args:
        probabilities: Raw model probabilities [P(Low), P(Moderate), P(High)]
        temperature: Calibration parameter (default 1.8 for medical uncertainty)
    
    Returns:
        Calibrated probabilities that sum to 1.0
    """
    # Prevent log(0) by adding small epsilon
    epsilon = 1e-9
    
    # Apply temperature scaling
    soft_logits = np.log(probabilities + epsilon) / temperature
    soft_probs = np.exp(soft_logits)
    
    # Normalize to sum to 1
    soft_probs = soft_probs / np.sum(soft_probs)
    
    return soft_probs


# Load model SYNCHRONOUSLY at startup (no background thread)
print("🔄 Loading Anaemia ML model at startup...")
train_or_load_anaemia_model()
print("✅ Anaemia ML model ready for predictions")


# ======================================================
# 🔹 SINGLE USER CHECK (ML-POWERED)
# ✅ USES AUTHENTICATED USER
# Endpoint: POST /api/anaemia/check
# ======================================================
@anemia_bp.route("/check", methods=["POST"])
@token_required
def check_anemia(current_user):
    """
    Perform ML-based anemia check and save to database
    
    ✅ NOW USES AUTHENTICATED USER - NO NEED TO SEND user_id
    ✅ NOW USES XGBOOST ML MODEL - NOT RULE-BASED
    ✅ FIXED: Hemoglobin-dominant predictions
    ✅ FIXED: Synchronous model loading (no race conditions)
    ✅ FIXED: Calibrated probabilities (realistic confidence levels)
    ✅ FIXED: Risk factors included in response
    
    Expected JSON payload:
    {
        "hemoglobin": 11.5,
        "bmi": 22.5,
        "age": 25,
        "symptoms": {
            "tiredness": 70,
            "weakness": 50,
            "dizziness": 30,
            "breathlessness": 20
        },
        "heavy_periods": false,
        "poor_diet": true
    }
    """
    try:
        # Ensure model is loaded before prediction
        if anaemia_model is None:
            print("⚠️ Model not loaded! Attempting to load now...")
            train_or_load_anaemia_model()
            if anaemia_model is None:
                return jsonify({
                    "error": "ML model not available. Please try again later."
                }), 503

        data = request.json or {}
        
        user_id = current_user.id
        
        print(f"\n🔍 ANEMIA CHECK DEBUG (ML-POWERED):")
        print(f"   Authenticated user: {current_user.full_name} (ID: {user_id})")
        print(f"   Request data: {data}")
        
        # Validate required fields
        if not data.get("hemoglobin"):
            return jsonify({"error": "hemoglobin level is required"}), 400
        
        if not data.get("age"):
            return jsonify({"error": "age is required"}), 400
        
        if not data.get("bmi"):
            return jsonify({"error": "bmi is required"}), 400

        # ========== EXTRACT INPUT DATA ==========
        symptoms = data.get("symptoms", {})
        
        # Extract numeric values
        age = int(data.get("age"))
        bmi = float(data.get("bmi"))
        hb = float(data.get("hemoglobin"))
        
        # Extract optional risk factors
        heavy_periods = data.get("heavy_periods", False)
        poor_diet = data.get("poor_diet", False)

        # Build feature vector for ML model
        feature_vector = np.array([[
            age,
            bmi,
            hb,
            int(symptoms.get("tiredness", 0)),
            int(symptoms.get("weakness", 0)),
            int(symptoms.get("dizziness", 0)),
            int(symptoms.get("breathlessness", 0))
        ]])

        print(f"📊 Feature vector: {feature_vector}")

        # ========== ML PREDICTION ==========
        prediction = anaemia_model.predict(feature_vector)[0]
        raw_probabilities = anaemia_model.predict_proba(feature_vector)[0]

        print(f"🤖 ML Prediction: {prediction}")
        print(f"📈 Raw Probabilities (Low/Moderate/High): {raw_probabilities}")

        # 🔥 Apply probability calibration to prevent overconfidence
        calibrated_probabilities = apply_probability_calibration(raw_probabilities, temperature=1.8)
        
        print(f"🎯 Calibrated Probabilities (Low/Moderate/High): {calibrated_probabilities}")

        # 🔥 IMPROVED: Use highest probability class for consistent risk level and score
        max_index = np.argmax(calibrated_probabilities)
        
        risk_mapping = {
            0: "Low",
            1: "Moderate",
            2: "High"
        }
        
        risk_level = risk_mapping[max_index]
        
        # Show probability of the predicted risk level (consistent with prediction)
        risk_score = round(float(calibrated_probabilities[max_index] * 100), 2)
        
        # Get hemoglobin status
        anaemia_status = get_hemoglobin_status(hb)

        # ========== CALCULATE RISK FACTORS ==========
        risk_factors = get_risk_factors(
            age=age,
            bmi=bmi,
            hb=hb,
            symptoms=symptoms,
            heavy_periods=heavy_periods,
            poor_diet=poor_diet
        )

        # 🔍 DEBUG: Print risk factors calculation
        print("=" * 60)
        print("=== RISK FACTORS DEBUG ===")
        print(f"Input data:")
        print(f"  - Age: {age}")
        print(f"  - BMI: {bmi}")
        print(f"  - Hemoglobin: {hb}")
        print(f"  - Heavy Periods: {heavy_periods}")
        print(f"  - Poor Diet: {poor_diet}")
        print(f"  - Symptoms: {symptoms}")
        print(f"Calculated risk factors ({len(risk_factors)} total):")
        for i, factor in enumerate(risk_factors, 1):
            print(f"  {i}. {factor}")
        print("=" * 60)

        print(f"✅ Risk Level: {risk_level}, Risk Score: {risk_score}%, Status: {anaemia_status}")
        print(f"⚠️ Risk Factors: {risk_factors}")

        # ========== SAVE TO DATABASE ==========
        # ✅ Model now has risk_factors column - safe to save
        new_check = AnaemiaCheck(
            user_id=user_id,
            age=age,
            bmi=bmi,
            hemoglobin=hb,
            symptoms=symptoms,
            result=anaemia_status,
            risk_level=risk_level,
            risk_factors=risk_factors,  # ✅ Now saving risk factors to database
            created_at=datetime.utcnow()
        )
        
        db.session.add(new_check)
        db.session.commit()
        
        print(f"✅ Saved ML-powered anemia check for user: {current_user.full_name} (ID: {new_check.id})")

        # ========== RETURN RESPONSE ==========
        return jsonify({
            "success": True,
            "check_id": new_check.id,
            "user_name": current_user.full_name,
            "risk_score": risk_score,  # Probability of the predicted risk level
            "risk_level": risk_level,
            "anaemia_status": anaemia_status,
            "hemoglobin": hb,
            "bmi": bmi,
            "age": age,
            "symptoms": symptoms,
            "risk_factors": risk_factors,  # 👈 RISK FACTORS ADDED HERE
            "model_confidence": {
                "low": round(float(calibrated_probabilities[0] * 100), 2),
                "moderate": round(float(calibrated_probabilities[1] * 100), 2),
                "high": round(float(calibrated_probabilities[2] * 100), 2)
            },
            "prediction_method": "XGBoost ML Model (Calibrated)",
            "recommendations": get_ml_recommendations(risk_level),
            "medical_note": "ML-based screening using calibrated XGBoost with hemoglobin-dominant predictions. Not a medical diagnosis.",
            "timestamp": new_check.created_at.isoformat() + "Z"
        }), 201
        
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error in check_anemia: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to process anemia check: {str(e)}"
        }), 500


# ======================================================
# 🔹 GET USER HISTORY
# Endpoint: GET /api/anaemia/history
# ======================================================
@anemia_bp.route("/history", methods=["GET"])
@token_required
def get_anemia_history(current_user):
    """Get anemia check history for authenticated user with full details including risk factors"""
    try:
        limit = request.args.get("limit", 10, type=int)
        
        checks = AnaemiaCheck.query.filter_by(user_id=current_user.id)\
            .order_by(AnaemiaCheck.created_at.desc())\
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
        print(f"Error in get_anemia_history: {str(e)}")
        return jsonify({
            "error": f"Failed to retrieve history: {str(e)}"
        }), 500


# ======================================================
# 🔹 GET LATEST CHECK
# Endpoint: GET /api/anaemia/latest
# ======================================================
@anemia_bp.route("/latest", methods=["GET"])
@token_required
def get_latest_check(current_user):
    """Get the most recent anemia check for authenticated user with full details"""
    try:
        latest_check = AnaemiaCheck.query.filter_by(user_id=current_user.id)\
            .order_by(AnaemiaCheck.created_at.desc())\
            .first()
        
        if latest_check:
            return jsonify({
                "success": True,
                "latest_check": latest_check.to_dict_full()  # Include risk_factors
            }), 200
        else:
            return jsonify({
                "success": True,
                "latest_check": None,
                "message": "No anemia checks found for this user"
            }), 200
            
    except Exception as e:
        print(f"Error in get_latest_check: {str(e)}")
        return jsonify({
            "error": f"Failed to retrieve latest check: {str(e)}"
        }), 500


# ======================================================
# 🔹 GET USER STATISTICS
# Endpoint: GET /api/anaemia/stats
# ======================================================
@anemia_bp.route("/stats", methods=["GET"])
@token_required
def get_user_stats(current_user):
    """Get anemia statistics for authenticated user"""
    try:
        checks = AnaemiaCheck.query.filter_by(user_id=current_user.id)\
            .order_by(AnaemiaCheck.created_at.desc())\
            .limit(100)\
            .all()
        
        if not checks:
            return jsonify({
                "success": True,
                "stats": {
                    "total_checks": 0,
                    "average_hemoglobin": None,
                    "average_bmi": None,
                    "trend": None,
                    "latest_result": None,
                    "latest_risk_level": None
                }
            }), 200
        
        # Calculate statistics
        total_checks = len(checks)
        avg_hemoglobin = sum(c.hemoglobin for c in checks if c.hemoglobin) / total_checks
        avg_bmi = sum(c.bmi for c in checks if c.bmi) / total_checks
        
        # Determine trend
        trend = "stable"
        if total_checks >= 6:
            recent_avg = sum(c.hemoglobin for c in checks[:3] if c.hemoglobin) / 3
            previous_avg = sum(c.hemoglobin for c in checks[3:6] if c.hemoglobin) / 3
            
            if recent_avg > previous_avg + 0.5:
                trend = "improving"
            elif recent_avg < previous_avg - 0.5:
                trend = "declining"
        
        return jsonify({
            "success": True,
            "stats": {
                "total_checks": total_checks,
                "average_hemoglobin": round(avg_hemoglobin, 2),
                "average_bmi": round(avg_bmi, 2),
                "trend": trend,
                "latest_result": checks[0].result,
                "latest_risk_level": checks[0].risk_level,
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
# Endpoint: DELETE /api/anaemia/check/<check_id>
# ======================================================
@anemia_bp.route("/check/<int:check_id>", methods=["DELETE"])
@token_required
def delete_check(current_user, check_id):
    """Delete a specific anemia check (only own checks)"""
    try:
        check = AnaemiaCheck.query.get(check_id)
        
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
# Endpoint: POST /api/anaemia/check_csv
# ======================================================
@anemia_bp.route("/check_csv", methods=["POST"])
def check_anemia_csv():
    """
    Bulk anemia check from CSV file using ML model
    Optionally save results to database if user_id is provided
    """
    if "file" not in request.files:
        return jsonify({"error": "CSV file missing"}), 400

    df = pd.read_csv(request.files["file"])

    # Required base columns
    required_cols = ["age", "bmi", "hemoglobin"]
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
        hb = float(row.get("hemoglobin", 12))
        
        feature_vector = np.array([[
            age,
            bmi,
            hb,
            int(row.get("tiredness", 0)),
            int(row.get("weakness", 0)),
            int(row.get("dizziness", 0)),
            int(row.get("breathlessness", 0))
        ]])

        # ML prediction
        prediction = anaemia_model.predict(feature_vector)[0]
        raw_probabilities = anaemia_model.predict_proba(feature_vector)[0]
        
        # Apply calibration
        calibrated_probabilities = apply_probability_calibration(raw_probabilities, temperature=1.8)
        
        # Use highest probability class for consistent results
        max_index = np.argmax(calibrated_probabilities)
        risk_mapping = {0: "Low", 1: "Moderate", 2: "High"}
        risk_level = risk_mapping[max_index]
        risk_score = round(float(calibrated_probabilities[max_index] * 100), 2)
        status = get_hemoglobin_status(hb)
        
        risk_scores.append(risk_score)
        risk_levels.append(risk_level)
        statuses.append(status)
        
        # Save to database if user_id is present
        if save_to_db and pd.notna(row.get("user_id")):
            try:
                symptoms = {
                    "tiredness": int(row.get("tiredness", 0)),
                    "weakness": int(row.get("weakness", 0)),
                    "dizziness": int(row.get("dizziness", 0)),
                    "breathlessness": int(row.get("breathlessness", 0)),
                }
                
                new_check = AnaemiaCheck(
                    user_id=int(row["user_id"]),
                    age=age,
                    bmi=bmi,
                    hemoglobin=hb,
                    symptoms=symptoms,
                    result=status,
                    risk_level=risk_level,
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
    df["AnaemiaStatus"] = statuses

    # Save and return CSV
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)

    return send_file(
        tmp.name,
        as_attachment=True,
        download_name="anaemia_ml_results.csv",
        mimetype="text/csv"
    )


# ======================================================
# 🔹 GET ALL CHECKS (ADMIN/ANALYTICS)
# Endpoint: GET /api/anaemia/all
# ======================================================
@anemia_bp.route("/all", methods=["GET"])
def get_all_checks():
    """Get all anemia checks with optional filtering"""
    try:
        # Optional filters
        risk_level = request.args.get("risk_level")
        result = request.args.get("result")
        limit = request.args.get("limit", 100, type=int)
        offset = request.args.get("offset", 0, type=int)
        
        query = AnaemiaCheck.query
        
        if risk_level:
            query = query.filter_by(risk_level=risk_level)
        
        if result:
            query = query.filter_by(result=result)
        
        total_count = query.count()
        
        checks = query.order_by(AnaemiaCheck.created_at.desc())\
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