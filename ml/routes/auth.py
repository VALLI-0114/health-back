import os
import sys
import jwt
from datetime import datetime, timedelta
from functools import wraps
from sqlalchemy import func
from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.exc import IntegrityError
from ml.utils.auth_utils import token_required

# ================= PATH FIX =================
backend_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

from extensions import db
from ml.models.user import User

print("🔥 User columns:", User.__table__.columns.keys())

auth_bp = Blueprint("auth", __name__)

# ================= CONFIG =================
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-jwt-secret")

# ================= TOKEN DECORATOR =================
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")

        if not token:
            return jsonify({
                "success": False,
                "error": "Token missing"
            }), 401

        try:
            if token.startswith("Bearer "):
                token = token[7:]

            payload = jwt.decode(
                token,
                JWT_SECRET_KEY,
                algorithms=["HS256"]
            )

            user_id = payload.get("user_id")
            current_user = User.query.get(user_id)

            if not current_user:
                return jsonify({
                    "success": False,
                    "error": "User not found"
                }), 401

        except jwt.ExpiredSignatureError:
            return jsonify({
                "success": False,
                "error": "Token expired"
            }), 401

        except jwt.InvalidTokenError:
            return jsonify({
                "success": False,
                "error": "Invalid token"
            }), 401

        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 401

        return f(current_user, *args, **kwargs)

    return decorated

# ================= OPTIONAL TOKEN DECORATOR =================
def optional_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        current_user = None

        if token:
            try:
                if token.startswith("Bearer "):
                    token = token[7:]

                payload = jwt.decode(
                    token,
                    JWT_SECRET_KEY,
                    algorithms=["HS256"]
                )

                user_id = payload.get("user_id")
                current_user = User.query.get(user_id)

            except:
                pass

        return f(current_user, *args, **kwargs)

    return decorated

# ================= ROLE DECORATOR =================
def role_required(required_role):
    def decorator(f):
        @wraps(f)
        def decorated_function(current_user, *args, **kwargs):
            if current_user.role != required_role:
                return jsonify({
                    "success": False,
                    "error": f"{required_role.capitalize()} access required"
                }), 403
            return f(current_user, *args, **kwargs)
        return decorated_function
    return decorator

# ================= ADMIN DECORATOR =================
def admin_required(f):
    @wraps(f)
    def decorated_function(current_user, *args, **kwargs):
        if current_user.role != "admin":
            return jsonify({
                "success": False,
                "error": "Admin privileges required"
            }), 403
        return f(current_user, *args, **kwargs)
    return decorated_function

# ================= SIGNUP =================
@auth_bp.route("/signup", methods=["POST", "OPTIONS"])
def signup():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json(force=True, silent=True)

    if not data:
        return jsonify({
            "success": False,
            "error": "Invalid or missing JSON body"
        }), 400

    full_name = data.get("full_name")
    roll_no = data.get("roll_no")
    age = data.get("age")
    password = data.get("password")
    role = data.get("role", "user")

    if not all([full_name, roll_no, age, password]):
        return jsonify({
            "success": False,
            "error": "Missing required fields"
        }), 400

    try:
        user = User(
            full_name=full_name,
            roll_no=roll_no,
            age=age,
            password_hash=generate_password_hash(password),
            role=role,
        )

        db.session.add(user)
        db.session.commit()

        token = jwt.encode(
            {
                "user_id": user.id,
                "role": user.role,
                "exp": datetime.utcnow() + timedelta(days=7),
            },
            JWT_SECRET_KEY,
            algorithm="HS256",
        )

        return jsonify({
            "success": True,
            "token": token,
            "user": {
                "id": user.id,
                "full_name": user.full_name,
                "roll_no": user.roll_no,
                "role": user.role,
            },
        }), 201

    except IntegrityError:
        db.session.rollback()
        return jsonify({
            "success": False,
            "error": "Roll number already exists"
        }), 409

# ================= LOGIN =================
@auth_bp.route("/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json(force=True, silent=True) or {}

    identifier = data.get("identifier")
    password = data.get("password")

    if not identifier or not password:
        return jsonify({
            "success": False,
            "error": "Identifier and password are required"
        }), 400

    identifier = identifier.strip()

    user = User.query.filter(
        User.roll_no == identifier
    ).first()

    if not user:
        user = User.query.filter(
            func.lower(User.full_name) == identifier.lower()
        ).first()

    if not user:
        return jsonify({
            "success": False,
            "error": "Invalid name or roll number"
        }), 401

    if not check_password_hash(user.password_hash, password):
        return jsonify({
            "success": False,
            "error": "Invalid password"
        }), 401

    token = jwt.encode(
        {
            "user_id": user.id,
            "role": user.role,
            "exp": datetime.utcnow() + timedelta(days=7),
        },
        JWT_SECRET_KEY,
        algorithm="HS256",
    )

    return jsonify({
        "success": True,
        "token": token,
        "user": {
            "id": user.id,
            "full_name": user.full_name,
            "roll_no": user.roll_no,
            "role": user.role,
        },
    }), 200



# ================= ADMIN LOGIN =================
@auth_bp.route("/admin-login", methods=["POST", "OPTIONS"])
def admin_login():
    """Admin login endpoint - uses same User table, checks role"""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json(force=True, silent=True) or {}
    
    identifier = data.get("identifier")
    password = data.get("password")

    print(f"🔐 ADMIN LOGIN ATTEMPT")
    print(f"🔍 Identifier: {identifier}")

    if not identifier or not password:
        return jsonify({
            "success": False,
            "error": "Identifier and password required"
        }), 400

    user = User.query.filter(
        User.roll_no == identifier
    ).first()

    if not user:
        print(f"❌ User not found: {identifier}")
        return jsonify({
            "success": False,
            "error": "Invalid credentials"
        }), 401

    print(f"✅ User found: {user.full_name}, Role: {user.role}")

    if not check_password_hash(user.password_hash, password):
        print(f"❌ Password mismatch")
        return jsonify({
            "success": False,
            "error": "Invalid credentials"
        }), 401

    if user.role != "admin":
        print(f"❌ Not admin, role: {user.role}")
        return jsonify({
            "success": False,
            "error": "Admin access required"
        }), 403

    print(f"✅ Admin login successful for {user.full_name}")

    token = jwt.encode(
        {
            "user_id": user.id,
            "role": user.role,
            "exp": datetime.utcnow() + timedelta(days=7),
        },
        JWT_SECRET_KEY,
        algorithm="HS256",
    )

    return jsonify({
        "success": True,
        "token": token,
        "user": {
            "id": user.id,
            "full_name": user.full_name,
            "roll_no": user.roll_no,
            "role": user.role,
        },
    }), 200

# ================= PROFILE =================
@auth_bp.route("/profile", methods=["GET"])
@token_required
def profile(current_user):
    return jsonify({
        "success": True,
        "user": {
            "id": current_user.id,
            "full_name": current_user.full_name,
            "roll_no": current_user.roll_no,
            "role": current_user.role,
        },
    })

# ================= LOGOUT =================
@auth_bp.route("/logout", methods=["POST"])
@token_required
def logout(current_user):
    return jsonify({
        "success": True,
        "message": "Logout successful"
    })