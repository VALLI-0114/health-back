"""
Script to create an admin user in your database
Run this from your backend root directory: python create_admin.py
"""

import os
import sys
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash

# Load env variables
load_dotenv()

# Add backend to path
backend_root = os.path.dirname(os.path.abspath(__file__))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

# Import Flask app and database
from core.app import app   
from extensions import db
from ml.models.users import User

def create_admin():
    """Create an admin user"""
    with app.app_context():
        # Check if admin already exists
        admin_email = "pravi@gmail.com"
        admin_password = "admin123"  # ⚠️ Change this to a strong password!
        
        existing = User.query.filter_by(email=admin_email).first()
        if existing:
            print(f"❌ Admin user already exists: {admin_email}")
            return
        
        # Create new admin user
        admin = User(
            email=admin_email,
            full_name="Admin User",
            password_hash=generate_password_hash(admin_password),
            is_admin=True,
            role="admin"  # If your User model has a role field
        )
        
        try:
            db.session.add(admin)
            db.session.commit()
            print(f"✅ Admin user created successfully!")
            print(f"   Email: {admin_email}")
            print(f"   Password: {admin_password}")
            print(f"\n⚠️  Change the password in production!")
        except Exception as e:
            db.session.rollback()
            print(f"❌ Error creating admin: {str(e)}")

if __name__ == "__main__":
    create_admin()