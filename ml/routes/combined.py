# ml/routes/combined.py
"""
Combined Health Check Routes - ML-POWERED VERSION
✅ Uses actual XGBoost models for Anemia and PCOD
✅ Authenticated user support
✅ Calibrated probabilities
✅ Professional implementation
"""

from flask import Blueprint, request, jsonify
from extensions import db
from ml.models.combined import CombinedCheck
from ml.routes.auth import token_required
from datetime import datetime
import numpy as np
import joblib
import os

combined_bp = Blueprint('combined', __name__, url_prefix='/api/combined')

# ======================================================
# 🔹 ML MODELS - Load existing trained models
# ======================================================
ANAEMIA_MODEL_PATH = "anaemia_xgb_model.pkl"
PCOD_MODEL_PATH = "pcod_xgb_model.pkl"

anaemia_model = None
pcod_model = None


def load_ml_models():
    """Load the trained ML models for anemia and PCOD"""
    global anaemia_model, pcod_model
    
    try:
        if os.path.exists(ANAEMIA_MODEL_PATH):
            anaemia_model = joblib.load(ANAEMIA_MODEL_PATH)
            print("✅ Loaded Anemia ML model")
        else:
            print("⚠️ Anemia model not found. Please train it first.")
            
        if os.path.exists(PCOD_MODEL_PATH):
            pcod_model = joblib.load(PCOD_MODEL_PATH)
            print("✅ Loaded PCOD ML model")
        else:
            print("⚠️ PCOD model not found. Please train it first.")
            
    except Exception as e:
        print(f"❌ Error loading ML models: {e}")


def apply_probability_calibration(probabilities, temperature=1.8):
    """Apply temperature scaling to soften overconfident probabilities"""
    epsilon = 1e-9
    soft_logits = np.log(probabilities + epsilon) / temperature
    soft_probs = np.exp(soft_logits)
    soft_probs = soft_probs / np.sum(soft_probs)
    return soft_probs


def get_hemoglobin_status(hb):
    """Get hemoglobin status classification"""
    if hb < 7:
        return "Severe Anaemia"
    elif hb < 10:
        return "Moderate Anaemia"
    elif hb < 12:
        return "Mild Anaemia"
    return "Normal"


def get_bmi_status(bmi):
    """Get BMI status classification"""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def calculate_anaemia_ml(hemoglobin, age, bmi, symptoms, heavy_periods=False, poor_diet=False):
    """
    Calculate anemia risk using ML model
    
    Returns:
        dict with risk_score, risk_level, hemoglobin_status, risk_factors
    """
    if anaemia_model is None:
        raise Exception("Anemia ML model not loaded")
    
    # Build feature vector for anemia model
    feature_vector = np.array([[
        age,
        bmi,
        hemoglobin,
        int(symptoms.get("tiredness", 0)),
        int(symptoms.get("weakness", 0)),
        int(symptoms.get("dizziness", 0)),
        int(symptoms.get("breathlessness", 0))
    ]])
    
    # ML prediction
    prediction = anaemia_model.predict(feature_vector)[0]
    raw_probabilities = anaemia_model.predict_proba(feature_vector)[0]
    
    # Apply calibration
    calibrated_probabilities = apply_probability_calibration(raw_probabilities, temperature=1.8)
    
    # Use highest probability class
    max_index = np.argmax(calibrated_probabilities)
    risk_mapping = {0: "Low", 1: "Moderate", 2: "High"}
    risk_level = risk_mapping[max_index]
    risk_score = round(float(calibrated_probabilities[max_index] * 100), 2)
    
    # Get hemoglobin status
    hb_status = get_hemoglobin_status(hemoglobin)
    
    # Calculate risk factors
    risk_factors = []
    
    if hemoglobin < 7:
        risk_factors.append("Severely low hemoglobin level")
    elif hemoglobin < 10:
        risk_factors.append("Moderately low hemoglobin level")
    elif hemoglobin < 12:
        risk_factors.append("Low hemoglobin level")
    
    if bmi < 18.5:
        risk_factors.append("Underweight (Low BMI)")
    
    if heavy_periods:
        risk_factors.append("Heavy menstrual bleeding")
    
    if poor_diet:
        risk_factors.append("Iron-poor diet")
    
    if symptoms.get("tiredness", 0) > 60:
        risk_factors.append("Severe tiredness")
    
    if symptoms.get("weakness", 0) > 60:
        risk_factors.append("Severe weakness")
    
    if symptoms.get("dizziness", 0) > 60:
        risk_factors.append("Frequent dizziness")
    
    if symptoms.get("breathlessness", 0) > 60:
        risk_factors.append("Breathlessness")
    
    if bmi < 18.5 and hemoglobin < 10:
        risk_factors.append("Combined low weight and low hemoglobin")
    
    if not risk_factors:
        risk_factors = ["No significant anemia risk factors detected"]
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "hemoglobin_status": hb_status,
        "risk_factors": risk_factors,
        "model_confidence": {
            "low": round(float(calibrated_probabilities[0] * 100), 2),
            "moderate": round(float(calibrated_probabilities[1] * 100), 2),
            "high": round(float(calibrated_probabilities[2] * 100), 2)
        }
    }


def calculate_pcod_ml(age, bmi, cycle_length, bleeding_days, symptoms):
    """
    Calculate PCOD risk using ML model
    
    Returns:
        dict with risk_score, risk_level, bmi_status, risk_factors
    """
    if pcod_model is None:
        raise Exception("PCOD ML model not loaded")
    
    # Build feature vector for PCOD model
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
    
    # ML prediction
    prediction = pcod_model.predict(feature_vector)[0]
    raw_probabilities = pcod_model.predict_proba(feature_vector)[0]
    
    # Handle 2-class models (auto-retrain if needed)
    if len(raw_probabilities) == 2:
        print("⚠️ PCOD model only has 2 classes. Expected 3 classes.")
        # Pad with zeros for missing class
        raw_probabilities = np.append(raw_probabilities, 0)
    
    # Apply calibration
    calibrated_probabilities = apply_probability_calibration(raw_probabilities, temperature=1.8)
    
    # Use highest probability class
    max_index = np.argmax(calibrated_probabilities)
    risk_mapping = {0: "Low", 1: "Moderate", 2: "High"}
    risk_level = risk_mapping[max_index]
    risk_score = round(float(calibrated_probabilities[max_index] * 100), 2)
    
    # Get BMI status
    bmi_status = get_bmi_status(bmi)
    
    # Calculate risk factors
    risk_factors = []
    
    if cycle_length > 35:
        risk_factors.append("Prolonged menstrual cycle (oligomenorrhea)")
    elif cycle_length < 21:
        risk_factors.append("Abnormally short menstrual cycle")
    
    if bleeding_days > 7:
        risk_factors.append("Prolonged menstrual bleeding")
    
    if bmi > 30:
        risk_factors.append("Obesity (BMI > 30)")
    elif bmi > 25:
        risk_factors.append("Overweight (BMI 25-30)")
    elif bmi < 18.5:
        risk_factors.append("Underweight (Low BMI)")
    
    if 15 <= age <= 25:
        risk_factors.append("Peak age range for PCOD onset (15-25)")
    
    if symptoms.get("irregular_periods", 0) > 60:
        risk_factors.append("Significant menstrual irregularity")
    
    if symptoms.get("acne", 0) > 60:
        risk_factors.append("Severe acne")
    
    if symptoms.get("excessive_hair", 0) > 60:
        risk_factors.append("Excessive hair growth (hirsutism)")
    
    if symptoms.get("mood_swings", 0) > 70:
        risk_factors.append("Severe mood swings")
    
    if bmi > 30 and cycle_length > 35:
        risk_factors.append("Combined obesity and irregular cycles")
    
    if not risk_factors:
        risk_factors = ["No significant PCOD risk factors detected"]
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "bmi_status": bmi_status,
        "risk_factors": risk_factors,
        "model_confidence": {
            "low": round(float(calibrated_probabilities[0] * 100), 2),
            "moderate": round(float(calibrated_probabilities[1] * 100), 2),
            "high": round(float(calibrated_probabilities[2] * 100), 2)
        }
    }


def calculate_combined_analysis_ml(anaemia_result, pcod_result, hemoglobin, bmi, age):
    """
    Combine ML-based anemia and PCOD risk assessments
    
    Args:
        anaemia_result: Dict from calculate_anaemia_ml()
        pcod_result: Dict from calculate_pcod_ml()
        hemoglobin: Hemoglobin level
        bmi: Body Mass Index
        age: Patient age
    
    Returns:
        Combined analysis with overall risk, recommendations, etc.
    """
    # Calculate combined score (weighted average)
    combined_score = (anaemia_result['risk_score'] * 0.5 + pcod_result['risk_score'] * 0.5)
    
    # Determine overall risk (take the maximum)
    risk_levels = ['Low', 'Moderate', 'High']
    anaemia_level_idx = risk_levels.index(anaemia_result['risk_level'])
    pcod_level_idx = risk_levels.index(pcod_result['risk_level'])
    overall_risk_idx = max(anaemia_level_idx, pcod_level_idx)
    overall_risk = risk_levels[overall_risk_idx]
    
    # Combine all risk factors
    all_risk_factors = anaemia_result['risk_factors'] + pcod_result['risk_factors']
    
    # Generate ML-aware recommendations
    recommendations = []
    
    # Anemia-specific recommendations
    if anaemia_result['risk_level'] == 'High':
        recommendations.append("⚠️ HIGH ANEMIA RISK: Immediate medical evaluation required")
        recommendations.append("🥩 Increase iron-rich foods: red meat, spinach, lentils, fortified cereals")
        recommendations.append("💊 Iron supplementation may be necessary (consult doctor)")
        recommendations.append("🔬 Complete blood count (CBC) test recommended")
    elif anaemia_result['risk_level'] == 'Moderate':
        recommendations.append("📋 MODERATE ANEMIA RISK: Schedule blood test soon")
        recommendations.append("🥗 Focus on iron-rich diet: leafy greens, beans, nuts")
        recommendations.append("🍊 Consume vitamin C with iron-rich foods for better absorption")
    else:
        recommendations.append("✅ LOW ANEMIA RISK: Maintain balanced diet with iron-rich foods")
    
    # PCOD-specific recommendations
    if pcod_result['risk_level'] == 'High':
        recommendations.append("⚠️ HIGH PCOD RISK: Consult gynecologist for evaluation")
        recommendations.append("🏃‍♀️ Regular exercise essential (30-45 minutes daily)")
        recommendations.append("🍽️ Low glycemic index diet recommended")
        recommendations.append("🔬 Hormonal panel and pelvic ultrasound may be needed")
    elif pcod_result['risk_level'] == 'Moderate':
        recommendations.append("⚡ MODERATE PCOD RISK: Monitor menstrual cycles closely")
        recommendations.append("🚶‍♀️ Maintain regular physical activity")
        recommendations.append("🥗 Balanced diet with whole grains and vegetables")
    else:
        recommendations.append("✅ LOW PCOD RISK: Continue healthy lifestyle habits")
    
    # General health recommendations
    recommendations.append("💧 Stay well-hydrated (8-10 glasses of water daily)")
    recommendations.append("😴 Ensure 7-8 hours of quality sleep")
    recommendations.append("🧘‍♀️ Practice stress management techniques (yoga, meditation)")
    recommendations.append("📅 Schedule regular health checkups")
    
    # Special combined risks
    if anaemia_result['risk_level'] in ['High', 'Moderate'] and pcod_result['risk_level'] in ['High', 'Moderate']:
        recommendations.insert(0, "⚠️ DUAL RISK DETECTED: Both anemia and PCOD require attention")
        recommendations.insert(1, "🏥 Comprehensive medical evaluation strongly recommended")
    
    # Final status message
    if overall_risk == "High":
        final_status = "⚠️ HIGH RISK - Medical consultation strongly recommended"
    elif overall_risk == "Moderate":
        final_status = "⚡ MODERATE RISK - Monitor health and consider medical consultation"
    else:
        final_status = "✅ LOW RISK - Continue healthy lifestyle habits"
    
    # Medical note
    medical_note = (
        f"ML-based combined screening using XGBoost models. "
        f"Anemia: {anaemia_result['hemoglobin_status']} (Hb: {hemoglobin} g/dL, Risk: {anaemia_result['risk_level']}). "
        f"PCOD: {pcod_result['bmi_status']} (BMI: {bmi}, Risk: {pcod_result['risk_level']}). "
        f"This is a screening tool, NOT a clinical diagnosis. Consult healthcare professionals for evaluation."
    )
    
    return {
        "combined_score": round(combined_score, 1),
        "overall_risk": overall_risk,
        "final_status": final_status,
        "all_risk_factors": all_risk_factors,
        "recommendations": recommendations,
        "medical_note": medical_note
    }


# Load models at startup
print("🔄 Loading ML models for combined check...")
load_ml_models()
print("✅ ML models ready for combined predictions")


@combined_bp.route('/check', methods=['POST'])
@token_required
def create_combined_check(current_user):
    """
    Create ML-powered combined health check using authenticated user
    ✅ Uses actual XGBoost models for both Anemia and PCOD
    ✅ Calibrated probabilities
    ✅ Professional risk assessment
    """
    try:
        data = request.get_json()
        
        user_id = current_user.id
        
        print(f"\n📥 COMBINED ML CHECK for user: {current_user.full_name} (ID: {user_id})")
        print(f"📦 Request data: {data}")
        
        # Validate required fields
        required_fields = ['age', 'height', 'weight', 'bmi', 'hemoglobin', 'cycle_length', 'bleeding_days']
        missing_fields = [field for field in required_fields if field not in data or data[field] == '']
        
        if missing_fields:
            return jsonify({
                "success": False,
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Ensure models are loaded
        if anaemia_model is None or pcod_model is None:
            return jsonify({
                "success": False,
                "error": "ML models not available. Please ensure models are trained."
            }), 503
        
        # Extract data
        age = int(data['age'])
        height = float(data['height'])
        weight = float(data['weight'])
        bmi = float(data['bmi'])
        hemoglobin = float(data['hemoglobin'])
        cycle_length = int(data['cycle_length'])
        bleeding_days = int(data['bleeding_days'])
        
        # Lifestyle factors
        heavy_periods = data.get('heavy_periods', False)
        poor_diet = data.get('poor_diet', False)
        
        # Symptoms
        anaemia_symptoms = data.get('anaemia_symptoms', {})
        pcod_symptoms = data.get('pcod_symptoms', {})
        
        print("🧮 Calculating ML-based anemia risk...")
        # Calculate Anemia Risk using ML
        anaemia_result = calculate_anaemia_ml(
            hemoglobin=hemoglobin,
            age=age,
            bmi=bmi,
            symptoms=anaemia_symptoms,
            heavy_periods=heavy_periods,
            poor_diet=poor_diet
        )
        
        print("🧮 Calculating ML-based PCOD risk...")
        # Calculate PCOD Risk using ML
        pcod_result = calculate_pcod_ml(
            age=age,
            bmi=bmi,
            cycle_length=cycle_length,
            bleeding_days=bleeding_days,
            symptoms=pcod_symptoms
        )
        
        print("🧮 Calculating combined ML analysis...")
        # Calculate Combined Analysis
        combined_analysis = calculate_combined_analysis_ml(
            anaemia_result=anaemia_result,
            pcod_result=pcod_result,
            hemoglobin=hemoglobin,
            bmi=bmi,
            age=age
        )
        
        print("💾 Saving to database...")
        # Save to database
        combined_check = CombinedCheck(
            user_id=user_id,
            age=age,
            gender=data.get('gender', 'female'),
            
            # Anemia data
            hemoglobin=hemoglobin,
            anemia_risk_level=anaemia_result['risk_level'],
            
            # PCOD data
            bmi=bmi,
            cycle_length=cycle_length,
            pcod_risk_level=pcod_result['risk_level'],
            
            # Overall
            result=combined_analysis['final_status'],
            overall_risk=combined_analysis['overall_risk'],
        )
        
        db.session.add(combined_check)
        db.session.commit()
        
        print(f"✅ Saved ML-powered combined check for user: {current_user.full_name} (ID: {combined_check.id})")
        
        # Build response
        response_data = {
            "success": True,
            "message": "ML-powered combined health check completed successfully",
            "user_name": current_user.full_name,
            
            # Individual condition results with ML confidence
            "anaemia": {
                "risk_score": anaemia_result['risk_score'],
                "risk_level": anaemia_result['risk_level'],
                "hemoglobin_value": f"{hemoglobin} g/dL",
                "hemoglobin_status": anaemia_result['hemoglobin_status'],
                "risk_factors": anaemia_result['risk_factors'],
                "model_confidence": anaemia_result['model_confidence']
            },
            
            "pcod": {
                "risk_score": pcod_result['risk_score'],
                "risk_level": pcod_result['risk_level'],
                "bmi": bmi,
                "bmi_status": pcod_result['bmi_status'],
                "risk_factors": pcod_result['risk_factors'],
                "model_confidence": pcod_result['model_confidence']
            },
            
            # Combined ML analysis
            "combined_score": combined_analysis['combined_score'],
            "overall_risk": combined_analysis['overall_risk'],
            "final_status": combined_analysis['final_status'],
            
            "combined_analysis": {
                "all_risk_factors": combined_analysis['all_risk_factors'],
                "recommendations": combined_analysis['recommendations']
            },
            
            "recommendations": combined_analysis['recommendations'],
            "medical_note": combined_analysis['medical_note'],
            
            # Database record info
            "record_id": combined_check.id,
            "created_at": combined_check.created_at.isoformat(),
            
            # ML metadata
            "prediction_method": "XGBoost ML Models (Calibrated)",
            "models_used": ["Anemia XGBoost", "PCOD XGBoost"]
        }
        
        print("📤 Sending ML response")
        return jsonify(response_data), 201
        
    except ValueError as ve:
        print("❌ Validation error:", str(ve))
        return jsonify({
            "success": False,
            "error": f"Invalid input values: {str(ve)}"
        }), 400
        
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error in ML combined check: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@combined_bp.route('/history', methods=['GET'])
@token_required
def get_user_history(current_user):
    """Get user's health check history for authenticated user"""
    try:
        limit = request.args.get('limit', 10, type=int)
        checks = CombinedCheck.query.filter_by(user_id=current_user.id)\
            .order_by(CombinedCheck.created_at.desc())\
            .limit(limit)\
            .all()
        
        return jsonify({
            "success": True,
            "count": len(checks),
            "data": [check.to_dict() for check in checks]
        }), 200
        
    except Exception as e:
        print(f"Error in get_user_history: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@combined_bp.route('/latest', methods=['GET'])
@token_required
def get_latest_check(current_user):
    """Get user's most recent health check for authenticated user"""
    try:
        check = CombinedCheck.query.filter_by(user_id=current_user.id)\
            .order_by(CombinedCheck.created_at.desc())\
            .first()
        
        if not check:
            return jsonify({
                "success": False,
                "message": "No health checks found for this user"
            }), 404
        
        return jsonify({"success": True, "data": check.to_dict()}), 200
        
    except Exception as e:
        print(f"Error in get_latest_check: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@combined_bp.route('/stats', methods=['GET'])
@token_required
def get_user_stats(current_user):
    """Get statistical summary for authenticated user"""
    try:
        checks = CombinedCheck.query.filter_by(user_id=current_user.id).all()
        
        if not checks:
            return jsonify({
                "success": False,
                "message": "No health checks found"
            }), 404
        
        total_checks = len(checks)
        latest_check = max(checks, key=lambda x: x.created_at)
        
        risk_counts = {
            "low": sum(1 for c in checks if c.overall_risk == "Low"),
            "moderate": sum(1 for c in checks if c.overall_risk == "Moderate"),
            "high": sum(1 for c in checks if c.overall_risk == "High"),
        }
        
        avg_hb = sum(c.hemoglobin for c in checks if c.hemoglobin) / sum(1 for c in checks if c.hemoglobin) if any(c.hemoglobin for c in checks) else None
        avg_bmi = sum(c.bmi for c in checks if c.bmi) / sum(1 for c in checks if c.bmi) if any(c.bmi for c in checks) else None
        
        return jsonify({
            "success": True,
            "data": {
                "total_checks": total_checks,
                "latest_check_date": latest_check.created_at.isoformat(),
                "current_risk": latest_check.overall_risk,
                "risk_distribution": risk_counts,
                "averages": {
                    "hemoglobin": round(avg_hb, 2) if avg_hb else None,
                    "bmi": round(avg_bmi, 2) if avg_bmi else None
                }
            }
        }), 200
        
    except Exception as e:
        print(f"Error in get_user_stats: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500