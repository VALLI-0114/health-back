import os
import sys
from flask import Blueprint, request, jsonify, send_file
from datetime import datetime, timedelta
import traceback
import pandas as pd
import numpy as np
import tempfile
from functools import wraps
from sqlalchemy import func
from sqlalchemy.orm import joinedload

# Path setup
backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

from extensions import db
from ml.routes.auth import token_required, admin_required
from ml.models.anaemia import AnaemiaCheck
from ml.models.pcod import PCODCheck
from ml.models.combined import CombinedCheck
from ml.models.user import User

# Import model loader
try:
    from ml.model_loader import load_model, predict_disease, get_model_info
    USING_MODEL_LOADER = True
except ImportError:
    print("⚠️ model_loader.py not found, using legacy loading")
    USING_MODEL_LOADER = False

admin_bp = Blueprint("admin", __name__)

# ================= ML MODEL LOADING =================
if USING_MODEL_LOADER:
    try:
        disease_model, label_encoder = load_model(backend_root)
    except Exception as e:
        print(f"⚠️ Warning: Could not load ML model via model_loader: {e}")
        disease_model = None
        label_encoder = None
else:
    try:
        import joblib
        
        MODEL_PATH = os.path.join(backend_root, "ml", "models", "disease_model.pkl")
        ENCODER_PATH = os.path.join(backend_root, "ml", "models", "label_encoder.pkl")
        
        disease_model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        
        print(f"✅ ML model loaded from {MODEL_PATH}")
        print(f"✅ Label encoder loaded from {ENCODER_PATH}")
        print(f"✅ Model classes: {label_encoder.classes_}")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not load ML model: {e}")
        traceback.print_exc()
        disease_model = None
        label_encoder = None


# ================= HELPER FUNCTIONS =================
def get_date_filter(period):
    """Get datetime filter based on period"""
    now = datetime.utcnow()
    if period == 'today':
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == 'week':
        return now - timedelta(days=7)
    elif period == 'month':
        return now - timedelta(days=30)
    else:  # 'all'
        return datetime.min

def safe_get_blockchain_count(model_class):
    try:
        if hasattr(model_class, 'blockchain_hash'):
            return model_class.query.filter(
                model_class.blockchain_hash.isnot(None)
            ).count()
        elif hasattr(model_class, 'blockchain_verified'):
            return model_class.query.filter(
                model_class.blockchain_verified == True
            ).count()
    except Exception as e:
        print(f"⚠️ Blockchain count error for {model_class.__name__}: {e}")
        db.session.rollback()
    return 0

def safe_get_verified_status(check_obj):
    """Safely get verification status"""
    if hasattr(check_obj, 'blockchain_hash'):
        return bool(check_obj.blockchain_hash)
    elif hasattr(check_obj, 'blockchain_verified'):
        return bool(check_obj.blockchain_verified)
    return False

def normalize_risk_level(value):
    """
    🔧 FINAL FIX: Normalize risk level values from database to match frontend
    
    Database values → Frontend expects:
    - "High", "Severe"           → "High Risk"
    - "Moderate", "Medium"       → "Medium Risk"  
    - "Low", "Normal"            → "Low Risk"
    - "Mild Anaemia"             → "Medium Risk"
    - "Severe Anaemia"           → "High Risk"
    """
    if not value:
        return "Low Risk"
    
    value_lower = str(value).lower().strip()
    
    # Map database values to frontend expectations
    if value_lower in ['high', 'severe', 'critical', 'severe anaemia', 'severe anemia']:
        return "High Risk"
    elif value_lower in ['moderate', 'medium', 'mild', 'mild anaemia', 'mild anemia']:
        return "Medium Risk"
    elif value_lower in ['low', 'normal', 'healthy']:
        return "Low Risk"
    
    # Default mapping
    return "Low Risk"

def get_risk_column_value(check_obj, prefer_risk_level=True):
    """
    🔧 CRITICAL FIX: Get risk value from correct column based on table
    
    Table structure discovered:
    - anaemia_checks: HAS both 'result' and 'risk_level' columns
    - pcod_checks: HAS only 'result' column (NO risk_level!)
    - combined_checks: Unknown, check both
    
    Priority:
    1. Try risk_level if prefer_risk_level=True
    2. Fallback to result
    3. Fallback to overall_risk (for combined)
    """
    if prefer_risk_level and hasattr(check_obj, 'risk_level'):
        value = getattr(check_obj, 'risk_level', None)
        if value:
            return value
    
    # Try result column
    if hasattr(check_obj, 'result'):
        value = getattr(check_obj, 'result', None)
        if value:
            return value
    
    # Last resort for combined checks
    if hasattr(check_obj, 'overall_risk'):
        value = getattr(check_obj, 'overall_risk', None)
        if value:
            return value
    
    return None

def allow_options(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == "OPTIONS":
            response = jsonify({"status": "ok"})
            response.headers.add("Access-Control-Allow-Origin", request.headers.get("Origin", "*"))
            response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
            response.headers.add("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
            response.headers.add("Access-Control-Allow-Credentials", "true")
            return response, 200
        return f(*args, **kwargs)
    return decorated_function


# ================= DASHBOARD =================
@admin_bp.route("/dashboard", methods=["GET"])
@token_required
@admin_required
def get_dashboard(current_user):
    """
    🔧 FINAL FIX: Dashboard with correct column usage for each table
    
    DISCOVERED TABLE STRUCTURE:
    - AnaemiaCheck: has BOTH 'result' and 'risk_level'
    - PCODCheck: has ONLY 'result' (NO risk_level!)
    - CombinedCheck: TBD (checking both)
    """
    try:
        print(f"\n📊 Admin Dashboard requested by admin {current_user.id}")
        
        period = request.args.get('period', 'week')
        date_filter = get_date_filter(period)
        print(f"📅 Period: {period}, Date filter: {date_filter}")
        
        # Basic counts
        total_users = db.session.query(func.count(User.id)).scalar()
        
        anemia_checks = db.session.query(func.count(AnaemiaCheck.id)).filter(
            AnaemiaCheck.created_at >= date_filter
        ).scalar()
        
        pcod_checks = db.session.query(func.count(PCODCheck.id)).filter(
            PCODCheck.created_at >= date_filter
        ).scalar()
        
        combined_checks = db.session.query(func.count(CombinedCheck.id)).filter(
            CombinedCheck.created_at >= date_filter
        ).scalar()
        
        blockchain_records = (
            safe_get_blockchain_count(AnaemiaCheck) +
            safe_get_blockchain_count(PCODCheck) +
            safe_get_blockchain_count(CombinedCheck)
        )
        
        # Fetch recent activity with joinedload
        recent_anemia = (
            db.session.query(AnaemiaCheck)
            .options(joinedload(AnaemiaCheck.user))
            .filter(AnaemiaCheck.created_at >= date_filter)
            .order_by(AnaemiaCheck.created_at.desc())
            .limit(5)
            .all()
        )
        
        recent_pcod = (
            db.session.query(PCODCheck)
            .options(joinedload(PCODCheck.user))
            .filter(PCODCheck.created_at >= date_filter)
            .order_by(PCODCheck.created_at.desc())
            .limit(5)
            .all()
        )
        
        recent_combined = (
            db.session.query(CombinedCheck)
            .options(joinedload(CombinedCheck.user))
            .filter(CombinedCheck.created_at >= date_filter)
            .order_by(CombinedCheck.created_at.desc())
            .limit(5)
            .all()
        )
        
        # Build recent activity list
        recent_activity = []
        
        for check in recent_anemia:
            try:
                user_name = check.user.full_name if check.user else "Unknown"
                result_text = get_risk_column_value(check, prefer_risk_level=True)
                result_text = normalize_risk_level(result_text)
                
                recent_activity.append({
                    'id': check.id,
                    'type': 'anemia',
                    'userId': check.user_id,
                    'userName': user_name,
                    'result': result_text,
                    'timestamp': check.created_at.isoformat() + 'Z',
                    'verified': safe_get_verified_status(check)
                })
            except Exception as e:
                print(f"⚠️ Error processing anemia check {check.id}: {e}")
        
        for check in recent_pcod:
            try:
                user_name = check.user.full_name if check.user else "Unknown"
                # PCOD has NO risk_level column, use result only
                result_text = get_risk_column_value(check, prefer_risk_level=False)
                result_text = normalize_risk_level(result_text)
                
                recent_activity.append({
                    'id': check.id,
                    'type': 'pcod',
                    'userId': check.user_id,
                    'userName': user_name,
                    'result': result_text,
                    'timestamp': check.created_at.isoformat() + 'Z',
                    'verified': safe_get_verified_status(check)
                })
            except Exception as e:
                print(f"⚠️ Error processing PCOD check {check.id}: {e}")
        
        for check in recent_combined:
            try:
                user_name = check.user.full_name if check.user else "Unknown"
                result_text = get_risk_column_value(check, prefer_risk_level=True)
                result_text = normalize_risk_level(result_text)
                
                recent_activity.append({
                    'id': check.id,
                    'type': 'combined',
                    'userId': check.user_id,
                    'userName': user_name,
                    'result': result_text,
                    'timestamp': check.created_at.isoformat() + 'Z',
                    'verified': safe_get_verified_status(check)
                })
            except Exception as e:
                print(f"⚠️ Error processing combined check {check.id}: {e}")
        
        # Sort by timestamp
        recent_activity.sort(
            key=lambda x: datetime.fromisoformat(x['timestamp'].rstrip('Z')),
            reverse=True
        )
        recent_activity = recent_activity[:10]
        
        # 🔧 CRITICAL FIX: Cluster statistics with CORRECT column usage
        print("📊 Calculating cluster statistics...")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CLEAN & CORRECT CLUSTER STATISTICS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # ---------- ANEMIA (Flexible pattern matching for descriptive strings) ----------
        # Using ILIKE for partial matching in case database contains descriptive strings
        anemia_high = db.session.query(func.count(AnaemiaCheck.id)).filter(
            AnaemiaCheck.created_at >= date_filter,
            AnaemiaCheck.risk_level.ilike('%High%')
        ).scalar()

        anemia_medium = db.session.query(func.count(AnaemiaCheck.id)).filter(
            AnaemiaCheck.created_at >= date_filter,
            AnaemiaCheck.risk_level.ilike('%Moderate%')
        ).scalar()

        anemia_low = db.session.query(func.count(AnaemiaCheck.id)).filter(
            AnaemiaCheck.created_at >= date_filter,
            AnaemiaCheck.risk_level.ilike('%Low%')
        ).scalar()


        # ---------- PCOD (Flexible pattern matching for descriptive strings) ----------
        # Database contains values like "High Risk - Consultation Recommended"
        # so we use ILIKE for partial matching instead of exact match
        pcod_high = db.session.query(func.count(PCODCheck.id)).filter(
            PCODCheck.created_at >= date_filter,
            PCODCheck.result.ilike('%High%')
        ).scalar()

        pcod_medium = db.session.query(func.count(PCODCheck.id)).filter(
            PCODCheck.created_at >= date_filter,
            PCODCheck.result.ilike('%Moderate%')
        ).scalar()

        pcod_low = db.session.query(func.count(PCODCheck.id)).filter(
            PCODCheck.created_at >= date_filter,
            PCODCheck.result.ilike('%Low%')
        ).scalar()


        cluster_stats = {
            'anemiaHighRisk': anemia_high,
            'anemiaMediumRisk': anemia_medium,
            'anemiaLowRisk': anemia_low,
            'pcodHighRisk': pcod_high,
            'pcodMediumRisk': pcod_medium,
            'pcodLowRisk': pcod_low
        }
        
        print(f"✅ Cluster stats calculated:")
        print(f"   Anemia - High: {anemia_high}, Medium: {anemia_medium}, Low: {anemia_low}")
        print(f"   PCOD - High: {pcod_high}, Medium: {pcod_medium}, Low: {pcod_low}")
        print(f"✅ Dashboard data compiled successfully")
        
        return jsonify({
            'success': True,
            'totalUsers': total_users,
            'anemiaChecks': anemia_checks,
            'pcodChecks': pcod_checks,
            'combinedChecks': combined_checks,
            'blockchainRecords': blockchain_records,
            'recentActivity': recent_activity,
            'clusterStats': cluster_stats
        }), 200
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in get_dashboard:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(traceback.format_exc())
        
        db.session.rollback()
        
        return jsonify({
            'success': False,
            'message': f'Dashboard error: {str(e)}',
            'error_type': type(e).__name__
        }), 500


# ================= EXPORT ROUTE =================
@admin_bp.route("/export", methods=["GET"])
@token_required
@admin_required
def export_dashboard(current_user):
    """🔧 FINAL FIX: Export with correct column usage"""
    try:
        print(f"\n📥 Export requested by admin {current_user.id}")
        
        period = request.args.get('period', 'week')
        date_filter = get_date_filter(period)
        
        # Use joinedload for efficient queries
        anemia_checks = (
            db.session.query(AnaemiaCheck)
            .options(joinedload(AnaemiaCheck.user))
            .filter(AnaemiaCheck.created_at >= date_filter)
            .all()
        )
        
        pcod_checks = (
            db.session.query(PCODCheck)
            .options(joinedload(PCODCheck.user))
            .filter(PCODCheck.created_at >= date_filter)
            .all()
        )
        
        combined_checks = (
            db.session.query(CombinedCheck)
            .options(joinedload(CombinedCheck.user))
            .filter(CombinedCheck.created_at >= date_filter)
            .all()
        )
        
        # Build DataFrames
        anemia_data = []
        for check in anemia_checks:
            result = get_risk_column_value(check, prefer_risk_level=True)
            result = normalize_risk_level(result)
            
            anemia_data.append({
                'ID': check.id,
                'User': check.user.full_name if check.user else "Unknown",
                'Age': check.age,
                'BMI': check.bmi,
                'Hemoglobin': check.hemoglobin,
                'Risk Level': result,
                'Verified': safe_get_verified_status(check),
                'Date': check.created_at.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        pcod_data = []
        for check in pcod_checks:
            # PCOD uses only 'result' column
            result = get_risk_column_value(check, prefer_risk_level=False)
            result = normalize_risk_level(result)
            
            pcod_data.append({
                'ID': check.id,
                'User': check.user.full_name if check.user else "Unknown",
                'Age': check.age,
                'BMI': check.bmi,
                'Cycle Length': getattr(check, 'cycle_length', 'N/A'),
                'Bleeding Days': getattr(check, 'bleeding_days', 'N/A'),
                'Risk Level': result,
                'Verified': safe_get_verified_status(check),
                'Date': check.created_at.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        combined_data = []
        for check in combined_checks:
            result = get_risk_column_value(check, prefer_risk_level=True)
            result = normalize_risk_level(result)
            
            combined_data.append({
                'ID': check.id,
                'User': check.user.full_name if check.user else "Unknown",
                'Age': check.age,
                'BMI': check.bmi,
                'Hemoglobin': getattr(check, 'hemoglobin', 'N/A'),
                'Cycle Length': getattr(check, 'cycle_length', 'N/A'),
                'Risk Level': result,
                'Verified': safe_get_verified_status(check),
                'Date': check.created_at.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Create Excel file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        
        with pd.ExcelWriter(tmp.name, engine='openpyxl') as writer:
            summary_df = pd.DataFrame({
                'Metric': ['Total Users', 'Anemia Checks', 'PCOD Checks', 'Combined Checks', 'Period'],
                'Value': [
                    User.query.count(),
                    len(anemia_checks),
                    len(pcod_checks),
                    len(combined_checks),
                    period.upper()
                ]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            if anemia_data:
                pd.DataFrame(anemia_data).to_excel(writer, sheet_name='Anemia Checks', index=False)
            if pcod_data:
                pd.DataFrame(pcod_data).to_excel(writer, sheet_name='PCOD Checks', index=False)
            if combined_data:
                pd.DataFrame(combined_data).to_excel(writer, sheet_name='Combined Checks', index=False)
        
        print(f"✅ Export generated successfully")
        
        return send_file(
            tmp.name,
            as_attachment=True,
            download_name=f"admin_report_{period}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        print(f"\n❌ Export error: {e}")
        print(traceback.format_exc())
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


# ================= BULK UPLOAD =================
@admin_bp.route("/bulk-upload", methods=["POST", "OPTIONS"])
@allow_options
@token_required
@admin_required
def bulk_upload(current_user):
    """🔧 FINAL FIX: Bulk upload with correct column usage for each table"""
    
    try:
        file = request.files.get("file")
        save_to_db = request.form.get("save_to_db", "false").lower() == "true"

        if not file:
            return jsonify({"error": "File is required"}), 400

        print(f"📤 Bulk upload: user={current_user.id}, save_to_db={save_to_db}")
        
        # Read file
        try:
            filename = file.filename.lower()
            
            if filename.endswith(".xlsx"):
                df = pd.read_excel(file, engine='openpyxl')
                print(f"📊 Loaded {len(df)} rows from Excel")
            elif filename.endswith(".csv"):
                temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                file.save(temp_csv.name)
                temp_csv.close()
                
                file_size_mb = os.path.getsize(temp_csv.name) / (1024 * 1024)
                
                if file_size_mb > 10:
                    print(f"📊 Large file detected ({file_size_mb:.1f}MB) - using chunked processing")
                    chunks = []
                    for chunk in pd.read_csv(temp_csv.name, chunksize=5000):
                        chunks.append(chunk)
                    df = pd.concat(chunks, ignore_index=True)
                    print(f"✅ Loaded {len(df)} rows in chunks")
                else:
                    df = pd.read_csv(temp_csv.name)
                    print(f"📊 Loaded {len(df)} rows from CSV")
                
                os.unlink(temp_csv.name)
            else:
                return jsonify({
                    "error": "Invalid file format",
                    "details": "Only .xlsx or .csv files are allowed"
                }), 400
                
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return jsonify({
                "error": f"Failed to read file: {str(e)}",
                "details": "Please ensure the file is a valid Excel or CSV file"
            }), 400
        
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Validate required columns
        required_cols = [
            "roll_no", "age", "height_cm", "weight_kg", "hemoglobin",
            "tiredness", "weakness", "pale_skin", "dizziness", "breathless",
            "hair_fall", "headache", "cold_hand", "pica", "chest_pain", "palpitations"
        ]
        
        missing = [c for c in required_cols if c not in df.columns]
        
        if missing:
            return jsonify({
                "error": "Missing required columns",
                "missing_columns": missing,
                "required_columns": required_cols,
                "found_columns": list(df.columns)
            }), 400
        
        # Check ML model
        if disease_model is None or label_encoder is None:
            return jsonify({
                "error": "ML model not loaded properly",
                "details": "Both disease_model.pkl and label_encoder.pkl must be present in ml/models/"
            }), 500

        print("🧹 Cleaning and preparing data...")
        
        # Clean data
        df = df.fillna(0)
        df["roll_no"] = df["roll_no"].astype(str).str.strip().str.upper()
        df["roll_no"] = df["roll_no"].replace({"0": "UNKNOWN", "": "UNKNOWN"})
        
        # Normalize binary columns
        binary_cols = [
            "tiredness", "weakness", "pale_skin", "dizziness",
            "breathless", "hair_fall", "headache",
            "cold_hand", "pica", "chest_pain", "palpitations"
        ]
        
        for col in binary_cols:
            df[col] = df[col].replace({
                "Yes": 1, "No": 0, "yes": 1, "no": 0,
                "YES": 1, "NO": 0, True: 1, False: 0,
                "true": 1, "false": 0, "True": 1, "False": 0,
                "1": 1, "0": 0
            })
        
        # Calculate BMI
        df["bmi"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)
        df["bmi"] = df["bmi"].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        print("✅ Data cleaned and normalized")
        
        # ML PREDICTION
        print(f"🤖 Running ML prediction on {len(df)} records...")
        
        feature_columns = [
            "age", "height_cm", "weight_kg", "hemoglobin",
            "tiredness", "weakness", "pale_skin", "dizziness", "breathless",
            "hair_fall", "headache", "cold_hand", "pica", "chest_pain", "palpitations",
            "bmi"
        ]
        
        try:
            X = df[feature_columns].astype(float).values
            
            pred_classes = disease_model.predict(X)
            pred_labels = label_encoder.inverse_transform(pred_classes)
            
            if hasattr(disease_model, "predict_proba"):
                probabilities = disease_model.predict_proba(X)
                confidences = (probabilities.max(axis=1) * 100).round(2)
            else:
                confidences = np.full(len(df), 100.0)
            
            df["Prediction"] = pred_labels
            df["Confidence (%)"] = confidences
            df["Status"] = "OK"
            
            print(f"✅ ML prediction completed: {len(df)} records processed")
            
        except Exception as ml_error:
            print(f"❌ ML prediction failed: {ml_error}")
            traceback.print_exc()
            return jsonify({
                "error": "ML prediction failed",
                "details": str(ml_error)
            }), 500
        
        df["Is_Safe"] = (df["Prediction"] == "Safe").astype(int)
        df["Is_Anemia"] = (df["Prediction"] == "Anemia").astype(int)
        df["Is_PCOD"] = (df["Prediction"] == "PCOD").astype(int)
        
        # BULK DATABASE OPERATIONS
        saved_count = 0
        
        if save_to_db:
            print("💾 Preparing bulk database insert...")
            
            roll_numbers = df["roll_no"].unique()
            users = User.query.filter(User.roll_no.in_(roll_numbers)).all()
            users_map = {str(u.roll_no).strip().upper(): u.id for u in users}
            
            print(f"✅ Found {len(users_map)} users in database")
            
            anemia_records = []
            pcod_records = []
            combined_records = []
            
            rows = df.to_dict('records')
            statuses = []
            
            for row in rows:
                roll_no = row["roll_no"]
                user_id = users_map.get(roll_no)
                
                if not user_id:
                    statuses.append("OK - Not Saved (User Not Found)")
                    continue
                
                prediction = row["Prediction"]
                age = float(row["age"])
                bmi = float(row["bmi"])
                
                if prediction == "Anemia":
                    hgb = float(row["hemoglobin"])
                    if hgb < 10:
                        risk_value = "High"
                    elif hgb < 12:
                        risk_value = "Moderate"
                    else:
                        risk_value = "Low"
                    
                    # 🔧 FIX: Anemia has risk_level column - use it!
                    anemia_records.append(AnaemiaCheck(
                        user_id=user_id,
                        age=age,
                        bmi=bmi,
                        hemoglobin=hgb,
                        risk_level=risk_value,  # ✅ Use risk_level
                        result=prediction,       # Also set result for compatibility
                        created_at=datetime.utcnow()
                    ))
                    statuses.append("OK - Saved")
                
                elif prediction == "PCOD":
                    cycle_len = row.get("cycle_length", 28)
                    if cycle_len > 35 or cycle_len < 21:
                        risk_value = "High"
                    elif cycle_len > 32 or cycle_len < 24:
                        risk_value = "Moderate"
                    else:
                        risk_value = "Low"
                    
                    # 🔧 FIX: PCOD has NO risk_level column - use only result!
                    pcod_records.append(PCODCheck(
                        user_id=user_id,
                        age=age,
                        bmi=bmi,
                        cycle_length=row.get("cycle_length", 28),
                        bleeding_days=row.get("bleeding_days", 5),
                        result=risk_value,  # ✅ PCOD uses 'result' ONLY
                        created_at=datetime.utcnow()
                    ))
                    statuses.append("OK - Saved")
                
                elif prediction == "Safe":
                    combined_records.append(CombinedCheck(
                        user_id=user_id,
                        age=age,
                        bmi=bmi,
                        hemoglobin=float(row["hemoglobin"]),
                        cycle_length=row.get("cycle_length", 28),
                        result="Safe",
                        overall_risk="Low",
                        created_at=datetime.utcnow()
                    ))
                    statuses.append("OK - Saved")
            
            df["Status"] = statuses
            
            # BULK INSERT
            try:
                if anemia_records:
                    db.session.bulk_save_objects(anemia_records)
                    saved_count += len(anemia_records)
                
                if pcod_records:
                    db.session.bulk_save_objects(pcod_records)
                    saved_count += len(pcod_records)
                
                if combined_records:
                    db.session.bulk_save_objects(combined_records)
                    saved_count += len(combined_records)
                
                db.session.commit()
                print(f"✅ Bulk insert completed: {saved_count} records saved to database")
                
            except Exception as db_error:
                db.session.rollback()
                print(f"❌ Database bulk insert failed: {db_error}")
                traceback.print_exc()
                return jsonify({
                    "error": "Failed to save records to database",
                    "details": str(db_error)
                }), 500
        
        # Statistics
        success_count = len(df[df["Status"].str.contains("OK")])
        saved_to_db_count = len(df[df["Status"] == "OK - Saved"]) if save_to_db else 0
        
        print(f"📊 Processing complete:")
        print(f"   - Total: {len(df)}")
        print(f"   - Successful: {success_count}")
        print(f"   - Saved to DB: {saved_to_db_count}")
        
        # Create Excel output
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        
        with pd.ExcelWriter(tmp.name, engine='xlsxwriter') as writer:
            summary_df = pd.DataFrame({
                'Metric': [
                    'Total Records',
                    'Successful Predictions',
                    'Failed',
                    'Saved to Database',
                    'Safe',
                    'Anemia',
                    'PCOD'
                ],
                'Count': [
                    len(df),
                    success_count,
                    len(df) - success_count,
                    saved_to_db_count,
                    int(df['Is_Safe'].sum()),
                    int(df['Is_Anemia'].sum()),
                    int(df['Is_PCOD'].sum())
                ]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            df.to_excel(writer, index=False, sheet_name='Results')

        return send_file(
            tmp.name,
            as_attachment=True,
            download_name=f"ml_predictions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in bulk_upload:")
        print(traceback.format_exc())
        db.session.rollback()
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ================= TEST ENDPOINTS =================
@admin_bp.route("/test", methods=["GET"])
def test_admin():
    """Test endpoint - no auth required"""
    return jsonify({
        "status": "Admin routes working!",
        "ml_model_loaded": disease_model is not None,
        "label_encoder_loaded": label_encoder is not None,
        "model_classes": list(label_encoder.classes_) if label_encoder else None,
        "timestamp": datetime.utcnow().isoformat()
    }), 200


@admin_bp.route("/test-auth", methods=["GET"])
@token_required
@admin_required
def test_auth(current_user):
    """Test endpoint with authentication"""
    return jsonify({
        "status": "Admin authentication successful!",
        "user": {
            "id": current_user.id,
            "full_name": current_user.full_name,
            "roll_no": getattr(current_user, 'roll_no', None),
            "role": current_user.role
        },
        "ml_model_loaded": disease_model is not None,
        "label_encoder_loaded": label_encoder is not None,
        "model_classes": list(label_encoder.classes_) if label_encoder else None,
        "timestamp": datetime.utcnow().isoformat()
    }), 200