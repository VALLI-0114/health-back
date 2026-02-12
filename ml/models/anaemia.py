from extensions import db
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSON


class AnaemiaCheck(db.Model):
    __tablename__ = "anaemia_checks"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False,
        index=True
    )

    # ✅ RELATIONSHIP
    user = db.relationship(
        "User",
        back_populates="anaemia_checks",
        lazy="joined"  # 🔥 IMPORTANT: prevents same-name bug
    )

    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    hemoglobin = db.Column(db.Float, nullable=False)

    symptoms = db.Column(JSON, nullable=True)

    result = db.Column(db.String(50), nullable=False)
    risk_level = db.Column(db.String(20), nullable=True)
    
    # 🆕 NEW: Store calculated risk factors
    risk_factors = db.Column(JSON, nullable=True)

    blockchain_hash = db.Column(db.String(256), nullable=True)
    blockchain_verified = db.Column(db.Boolean, default=False)

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )

    def __repr__(self):
        return f"<AnaemiaCheck {self.id} - {self.result}>"

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
            "type": "anemia"
        }
    
    # 🆕 NEW: Full details for user-facing endpoints
    def to_dict_full(self):
        """
        Returns complete dictionary including all health data and risk factors
        Use this for /api/anaemia/check, /api/anaemia/history, /api/anaemia/latest
        """
        return {
            "id": self.id,
            "user_id": self.user_id,
            "user_name": self.user.full_name if self.user else "Unknown",
            "age": self.age,
            "bmi": round(self.bmi, 2) if self.bmi else None,
            "hemoglobin": round(self.hemoglobin, 2) if self.hemoglobin else None,
            "symptoms": self.symptoms or {},
            "result": self.result,
            "risk_level": self.risk_level,
            "risk_factors": self.risk_factors or [],  # 🆕 Include risk factors
            "blockchain_verified": self.blockchain_verified,
            "blockchain_hash": self.blockchain_hash,
            "created_at": self.created_at.isoformat() + "Z" if self.created_at else None
        }