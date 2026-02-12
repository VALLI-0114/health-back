"""
Quick verification script to test relationship queries
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extensions import db
from ml.models.user import User
from ml.models.anaemia import AnaemiaCheck
from ml.models.pcod import PCODCheck
from ml.models.combined import CombinedCheck
from run import app


with app.app_context():
    print("\n" + "=" * 60)
    print("DATABASE RELATIONSHIP VERIFICATION")
    print("=" * 60)
    
    # Check users
    print("\n📋 USERS IN DATABASE:")
    users = User.query.all()
    for u in users:
        print(f"  ID: {u.id}, Name: {u.full_name}, Roll: {u.roll_no}")
    
    # Check anemia checks with relationships
    print("\n📋 ANEMIA CHECKS (with user relationships):")
    anemia_checks = AnaemiaCheck.query.order_by(AnaemiaCheck.created_at.desc()).limit(5).all()
    for check in anemia_checks:
        print(f"  Check ID: {check.id}")
        print(f"    user_id: {check.user_id}")
        print(f"    user object: {check.user}")
        print(f"    user.full_name: {check.user.full_name if check.user else 'N/A'}")
        print(f"    Result: {check.result}")
        print()
    
    # Check PCOD checks
    print("\n📋 PCOD CHECKS (with user relationships):")
    pcod_checks = PCODCheck.query.order_by(PCODCheck.created_at.desc()).limit(5).all()
    for check in pcod_checks:
        print(f"  Check ID: {check.id}")
        print(f"    user_id: {check.user_id}")
        print(f"    user.full_name: {check.user.full_name if check.user else 'N/A'}")
        print(f"    Result: {check.result}")
        print()
    
    # Check combined checks
    print("\n📋 COMBINED CHECKS (with user relationships):")
    combined_checks = CombinedCheck.query.order_by(CombinedCheck.created_at.desc()).limit(5).all()
    for check in combined_checks:
        print(f"  Check ID: {check.id}")
        print(f"    user_id: {check.user_id}")
        print(f"    user.full_name: {check.user.full_name if check.user else 'N/A'}")
        print(f"    Result: {check.result}")
        print()
    
    print("=" * 60)