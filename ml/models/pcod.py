from extensions import db
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSON


class PCODCheck(db.Model):
    __tablename__ = "pcod_checks"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    age = db.Column(db.Integer, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    cycle_length = db.Column(db.Integer, nullable=False)
    bleeding_days = db.Column(db.Integer, nullable=False)

    result = db.Column(db.String(50), nullable=False)
    risk_score = db.Column(db.Integer)
    symptoms = db.Column(JSON)
    
    # 🆕 NEW: Store calculated risk factors
    risk_factors = db.Column(JSON, nullable=True)

    blockchain_hash = db.Column(db.String(256))
    blockchain_verified = db.Column(db.Boolean, default=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    user = db.relationship("User", back_populates="pcod_checks")

    def __repr__(self):
        return f"<PCODCheck {self.id} - {self.result}>"

    # ✅ ADMIN DASHBOARD SAFE OUTPUT
    def to_dict(self):
        """
        Returns dictionary for admin dashboard display
        """
        return {
            "id": self.id,
            "userId": self.user_id,
            "userName": self.user.full_name if self.user else "Unknown",
            "result": self.result,
            "verified": self.blockchain_verified,
            "timestamp": self.created_at.isoformat() + "Z",
            "type": "pcod"
        }
    
    # 🆕 NEW: Full details for user-facing endpoints
    def to_dict_full(self):
        """
        Returns complete dictionary including all health data and risk factors
        Use this for /api/pcod/check, /api/pcod/history, /api/pcod/latest
        """
        return {
            "id": self.id,
            "user_id": self.user_id,
            "user_name": self.user.full_name if self.user else "Unknown",
            "age": self.age,
            "bmi": round(self.bmi, 2) if self.bmi else None,
            "cycle_length": self.cycle_length,
            "bleeding_days": self.bleeding_days,
            "symptoms": self.symptoms or {},
            "result": self.result,
            "risk_score": self.risk_score,
            "risk_factors": self.risk_factors or [],  # 🆕 Include risk factors
            "blockchain_verified": self.blockchain_verified,
            "blockchain_hash": self.blockchain_hash,
            "created_at": self.created_at.isoformat() + "Z" if self.created_at else None
        }