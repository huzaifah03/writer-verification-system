"""
Database models and initialization.
Uses SQLite via Flask-SQLAlchemy.
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class Assignment(db.Model):
    """Stores uploaded assignment image metadata."""
    __tablename__ = "assignments"

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Assignment {self.original_name}>"


class VerificationResult(db.Model):
    """Stores the result of each writer verification comparison."""
    __tablename__ = "verification_results"

    id = db.Column(db.Integer, primary_key=True)

    # The two assignments being compared
    assignment1_id = db.Column(db.Integer, db.ForeignKey("assignments.id"), nullable=False)
    assignment2_id = db.Column(db.Integer, db.ForeignKey("assignments.id"), nullable=False)

    # Similarity score between 0 and 1
    similarity_score = db.Column(db.Float, nullable=False)

    # Final decision: "Same Writer" or "Different Writer"
    decision = db.Column(db.String(50), nullable=False)

    # Plagiarism risk level: "Low", "Medium", "High"
    risk_level = db.Column(db.String(20), nullable=False)

    verified_at = db.Column(db.DateTime, default=datetime.utcnow)

    assignment1 = db.relationship("Assignment", foreign_keys=[assignment1_id])
    assignment2 = db.relationship("Assignment", foreign_keys=[assignment2_id])

    def __repr__(self):
        return f"<Result {self.decision} | Score: {self.similarity_score:.2f}>"


def init_db(app):
    """Initialize the database with the Flask app."""
    db.init_app(app)
    with app.app_context():
        db.create_all()
