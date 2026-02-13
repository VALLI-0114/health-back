import os
import sys
from datetime import timedelta

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv
from sqlalchemy import text

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

print(f"📁 Backend directory: {BASE_DIR}")

# ================= LOAD ENV =================
load_dotenv()

# ================= APP =================
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret")

# ================= JWT CONFIG =================
app.config["JWT_SECRET_KEY"] = os.getenv(
    "JWT_SECRET_KEY",
    "dev-secret-womens-health-2024-CHANGE-IN-PRODUCTION"
)
app.config["JWT_TOKEN_LOCATION"] = ["headers"]
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)
app.config["JWT_REFRESH_TOKEN_EXPIRES"] = timedelta(days=30)

jwt = JWTManager(app)
print("✅ JWT configured")

# ================= CORS =================
CORS(
    app,
    resources={r"/api/*": {"origins": [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://health-front-umber.vercel.app"
    ]}},
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
)

print("✅ CORS enabled")

# ================= DATABASE CONFIG =================
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("❌ DATABASE_URL missing!")

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

print("✅ Connected using DATABASE_URL")


# ================= EXTENSIONS =================
from extensions import db, migrate

db.init_app(app)
migrate.init_app(app, db)

print("✅ SQLAlchemy & Flask-Migrate initialized")

# ================= STATIC FILES =================
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(
        os.path.join(BASE_DIR, "static"),
        filename
    )

# ================= BASIC ROUTES =================
@app.route("/")
def home():
    return jsonify({"status": "Backend running 🔥"})

@app.route("/api/health/db-check")
def db_check():
    try:
        result = db.session.execute(text("SELECT version();"))
        db_version = result.scalar()

        return jsonify({
            "status": "success",
            "database": "connected",
            "postgres_version": db_version
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "database": "not connected",
            "error": str(e)
        }), 500


@app.route("/routes")
def list_routes():
    return "\n".join(
        sorted(rule.rule for rule in app.url_map.iter_rules())
    )

# ================= LOAD BLUEPRINTS =================
print("📦 Loading blueprints...")

from ml.routes.auth import auth_bp
from ml.routes.anaemia import anemia_bp
from ml.routes.pcod import pcod_bp
from ml.routes.combined import combined_bp
from ml.routes.admin import admin_bp
from ml.routes.notification import notification_bp
from ml.routes.profile import profile_bp

# ================= REGISTER BLUEPRINTS =================
app.register_blueprint(auth_bp, url_prefix="/api/auth")
app.register_blueprint(anemia_bp, url_prefix="/api/anemia")
app.register_blueprint(pcod_bp, url_prefix="/api/pcod")
app.register_blueprint(combined_bp, url_prefix="/api/combined")
app.register_blueprint(admin_bp, url_prefix="/api/admin")
app.register_blueprint(notification_bp, url_prefix="/api/notification")
app.register_blueprint(profile_bp, url_prefix="/api/profile")

print("✅ All blueprints registered")

# ================= PRODUCTION READY =================
# Gunicorn will serve this app in production
# No app.run() needed - Render uses: gunicorn run:app
#
# For local development, use:
# python run.py (will use Gunicorn if installed)
# OR: flask run --host=0.0.0.0 --port=5000

if __name__ == "__main__":
    # Create static directory if it doesn't exist
    os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
    
    # Try to use gunicorn for local development too
    try:
        import gunicorn
        print("\n🚀 For local development, run:")
        print("   gunicorn run:app --bind 0.0.0.0:5000 --reload")
        print("   OR: flask run --host=0.0.0.0 --port=5000\n")
    except ImportError:
        print("\n⚠️  Install gunicorn for production-like local testing:")
        print("   pip install gunicorn")
        print("\n🚀 For now, use: flask run --host=0.0.0.0 --port=5000\n")
