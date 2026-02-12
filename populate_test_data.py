"""
Script to populate test data with multiple users for testing the admin dashboard
Run from backend directory:
python populate_test_data.py
"""

import sys
import os
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash

# Fix path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from run import app              # ✅ FIXED
from extensions import db
from ml.models.user import User
from ml.models.anaemia import AnaemiaCheck
from ml.models.pcod import PCODCheck
from ml.models.combined import CombinedCheck


def create_test_users():
    test_users = [
        {"full_name": "Priya Sharma", "roll_no": "21CS001", "age": 21},
        {"full_name": "Ananya Reddy", "roll_no": "21CS002", "age": 22},
        {"full_name": "Sneha Patel", "roll_no": "21CS003", "age": 20},
        {"full_name": "Kavya Nair", "roll_no": "21CS004", "age": 23},
        {"full_name": "Meera Singh", "roll_no": "21CS005", "age": 21},
    ]

    created_users = []

    with app.app_context():
        for data in test_users:
            user = User.query.filter_by(roll_no=data["roll_no"]).first()
            if user:
                print(f"✓ User exists: {user.full_name}")
                created_users.append(user)
            else:
                user = User(
                    full_name=data["full_name"],
                    roll_no=data["roll_no"],
                    age=data["age"],
                    password_hash=generate_password_hash("password123"),
                    role="user"
                )
                db.session.add(user)
                db.session.flush()
                print(f"✓ Created user: {user.full_name}")
                created_users.append(user)

        db.session.commit()

    return created_users


def create_test_health_checks(users):
    with app.app_context():
        base_time = datetime.utcnow()

        for i, u in enumerate(users):
            db.session.add(
                AnaemiaCheck(
                    user_id=u.id,
                    age=u.age,
                    bmi=18 + i,
                    hemoglobin=9 + i,
                    result="High Risk" if i % 2 == 0 else "Medium Risk",
                    created_at=base_time - timedelta(hours=i * 2)
                )
            )

            db.session.add(
                PCODCheck(
                    user_id=u.id,
                    age=u.age,
                    bmi=22 + i,
                    cycle_length=28 + i,
                    bleeding_days=4,
                    result="Low Risk",
                    created_at=base_time - timedelta(hours=i * 3)
                )
            )

            db.session.add(
                CombinedCheck(
                    user_id=u.id,
                    age=u.age,
                    bmi=24 + i,
                    hemoglobin=11 + i,
                    cycle_length=30 + i,
                    result="Medium Risk (Combined)",
                    overall_risk="Medium Risk",
                    created_at=base_time - timedelta(hours=i * 4)
                )
            )

        db.session.commit()
        print("✅ Health checks created")


if __name__ == "__main__":
    print("\nPOPULATING TEST DATA...\n")
    users = create_test_users()
    create_test_health_checks(users)
    print("\n✅ DONE — refresh admin dashboard\n")
