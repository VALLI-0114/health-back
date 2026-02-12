from extensions import db
from datetime import datetime

class CombinedCheck(db.Model):
    __tablename__ = "combined_checks"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False,
        index=True
    )

    # ✅ FIXED: Bidirectional relationship
    user = db.relationship(
        "User",
        back_populates="combined_checks",
        lazy="joined"
    )

    age = db.Column(db.Float, nullable=False)
    gender = db.Column(db.String(10), nullable=True)

    bmi = db.Column(db.Float, nullable=False)
    hemoglobin = db.Column(db.Float, nullable=False)
    cycle_length = db.Column(db.Integer, nullable=False)

    anemia_risk_level = db.Column(db.String(20))
    pcod_risk_level = db.Column(db.String(20))

    result = db.Column(db.String(100), nullable=False)
    overall_risk = db.Column(db.String(100))

    blockchain_hash = db.Column(db.String(256))
    blockchain_verified = db.Column(db.Boolean, default=False)

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )

    def to_dict(self):
        return {
            "id": self.id,
            "userId": self.user_id,
            "userName": self.user.full_name if self.user else "Unknown",
            "result": self.result,
            "verified": self.blockchain_verified,
            "timestamp": self.created_at.isoformat() + "Z",
            "type": "combined"
        }
