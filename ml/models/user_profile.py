import os
import sys
from datetime import datetime

# ================= PATH FIX =================
backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

from extensions import db

class UserProfile(db.Model):
    __tablename__ = 'user_profiles'
    
    # PRIMARY KEY IS user_id, NOT id!
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), primary_key=True, nullable=False)
    
    # NEW FIELDS - name and age
    name = db.Column(db.String(100), nullable=True)
    age = db.Column(db.Integer, nullable=True)
    
    # EXISTING FIELDS
    height = db.Column(db.String(50), nullable=True)
    weight = db.Column(db.String(50), nullable=True)
    college = db.Column(db.String(200), nullable=True)
    district = db.Column(db.String(100), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    profile_photo = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('profile', uselist=False))
    
    def to_dict(self):
        """Convert model to dictionary for JSON serialization"""
        return {
            'user_id': self.user_id,
            'name': self.name or "",
            'age': self.age,
            'height': self.height or "",
            'weight': self.weight or "",
            'college': self.college or "",
            'district': self.district or "",
            'phone': self.phone or "",
            'profile_photo': self.profile_photo,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<UserProfile user_id={self.user_id} name={self.name} age={self.age}>'