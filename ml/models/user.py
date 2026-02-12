from extensions import db
from datetime import datetime

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)

    full_name = db.Column(db.String(100), nullable=False, index=True)
    roll_no = db.Column(db.String(50), unique=True, nullable=False, index=True)
    age = db.Column(db.Integer, nullable=False)

    password_hash = db.Column(db.Text, nullable=False)
    role = db.Column(db.String(20), default="user")

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    # ✅ Anaemia
    anaemia_checks = db.relationship(
        "AnaemiaCheck",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select"
    )

    # ✅ PCOD
    pcod_checks = db.relationship(
        "PCODCheck",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select"
    )

    # ✅ Combined
    combined_checks = db.relationship(
        "CombinedCheck",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select"
    )

    def __repr__(self):
        return f"<User {self.full_name}>"
