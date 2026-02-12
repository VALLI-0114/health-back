import os
from dotenv import load_dotenv

load_dotenv()

SQLALCHEMY_DATABASE_URI = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

SQLALCHEMY_TRACK_MODIFICATIONS = False
class Config:
    # ... your existing config ...
    
    # JWT Configuration
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this-in-production')
    JWT_TOKEN_LOCATION = ['headers']
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    """
Quick JWT Configuration Patch
Add this code to your run.py BEFORE importing any blueprints that use JWT
"""

from datetime import timedelta

# Add this to your Flask app configuration
JWT_CONFIG = {
    'JWT_SECRET_KEY': 'dev-secret-key-CHANGE-THIS-IN-PRODUCTION-min-32-chars',
    'JWT_TOKEN_LOCATION': ['headers'],
    'JWT_ACCESS_TOKEN_EXPIRES': timedelta(hours=24),
    'JWT_REFRESH_TOKEN_EXPIRES': timedelta(days=30),
    'JWT_HEADER_NAME': 'Authorization',
    'JWT_HEADER_TYPE': 'Bearer',
}

# Example of how to add it to your app:
"""
from flask import Flask
from flask_jwt_extended import JWTManager

app = Flask(__name__)

# Apply JWT configuration
for key, value in JWT_CONFIG.items():
    app.config[key] = value

# Initialize JWT
jwt = JWTManager(app)

# Now you can import and register blueprints that use JWT
from ml.routes.pcod import pcod_bp
app.register_blueprint(pcod_bp, url_prefix='/api/pcod')
"""