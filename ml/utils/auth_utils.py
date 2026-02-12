import os
import jwt
from functools import wraps
from flask import request, jsonify
from ml.models.user import User

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-jwt-secret")

def token_required(f):
    """
    Authentication decorator that properly handles CORS preflight requests.
    Place this BEFORE @token_required in your route definitions.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        # ✅ CRITICAL: Allow CORS preflight (OPTIONS) without authentication
        if request.method == "OPTIONS":
            response = jsonify({"status": "ok"})
            response.headers.add("Access-Control-Allow-Origin", request.headers.get("Origin", "*"))
            response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
            response.headers.add("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
            response.headers.add("Access-Control-Allow-Credentials", "true")
            return response, 200
        
        # For all other methods, require authentication
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return jsonify({
                "success": False,
                "error": "Token missing"
            }), 401
        
        try:
            token = auth_header.replace("Bearer ", "")
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
            user_id = payload.get("user_id")
            
            if not user_id:
                return jsonify({
                    "success": False,
                    "error": "Invalid token"
                }), 401
            
            current_user = User.query.get(user_id)
            if not current_user:
                return jsonify({
                    "success": False,
                    "error": "User not found"
                }), 401
                
        except jwt.ExpiredSignatureError:
            return jsonify({"success": False, "error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"success": False, "error": "Invalid token"}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated